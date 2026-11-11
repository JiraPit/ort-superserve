//! How a batched model output is divided back into per-request results.

use crate::tensor::{Element, Tensor, map_tensor, visit_tensor};
use anyhow::Result;
use ndarray::{ArrayD, Axis};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// A user-supplied splitting function, see [`Split::Custom`].
pub type SplitFn = Arc<dyn Fn(&Tensor, &SplitContext<'_>) -> Result<Vec<Tensor>> + Send + Sync>;

/// How one model output is divided into one tensor per request.
///
/// Configure it per output with
/// [`OutputSpec::with_split`](crate::OutputSpec::with_split). Whatever the
/// split, rows that belong to padding added to fill a fixed model batch size
/// are never handed to a caller.
///
/// # Example
///
/// A detector with NMS: `boxes` is `[num_dets, 4]`, `scores` is `[num_dets]`,
/// `labels` is an `i64` `[num_dets]`, and `batch_index` says which image each
/// detection belongs to.
///
/// ```rust
/// use ort_superserve::{ServerConfig, Split};
///
/// let by_image = || Split::batch_index_from("batch_index", None);
/// let config = ServerConfig::new()
///     .with_output_split("boxes", by_image())
///     .with_output_split("scores", by_image())
///     .with_output_split("labels", by_image());
/// ```
///
/// A custom split, here giving every request the whole output:
///
/// ```rust
/// use ort_superserve::{ServerConfig, Split};
///
/// let config = ServerConfig::new().with_output_split(
///     "class_names",
///     Split::custom(|output, ctx| Ok(vec![output.clone(); ctx.num_requests()])),
/// );
/// ```
#[derive(Clone, Default)]
pub enum Split {
    /// Axis 0 is the batch axis: request `i` gets row `i`, with the batch axis
    /// removed. This is the default.
    ///
    /// The output's axis 0 must have exactly one entry per row of the batch
    /// that was sent to the model; anything else is an error. When the model's
    /// metadata shows that axis 0 is *not* the batch dimension (a fixed size
    /// other than the model batch size, a dynamic size in a fixed-batch
    /// model, or a dimension with a different name than the inputs' batch
    /// dimension), the default split refuses the output with an error instead
    /// of risking handing one caller another caller's rows. Setting
    /// `Split::BatchAxis` explicitly skips the name and dynamic-size checks
    /// (the runtime row count check always applies).
    ///
    /// Positions along axes that follow a padded input dimension are cut back
    /// to the request's own length (see
    /// [`OutputSpec::with_crop_axis`](crate::OutputSpec::with_crop_axis)).
    #[default]
    BatchAxis,

    /// Rows (axis 0) are assigned to requests by a batch index, as produced by
    /// NMS models: request `i` gets, in order, every row whose batch index is
    /// `i`, with all axes kept. Rows of padded batch entries are dropped.
    ///
    /// The batch index is read from the output `source` (or from this output
    /// itself if `None`): from its column `column` if set (the source must
    /// then be `[rows, columns]`), otherwise from the 1-D source directly. Any
    /// integer type works, as do floats holding whole numbers.
    BatchIndex {
        /// Output holding the batch index; `None` for this output.
        source: Option<String>,
        /// Column of the source holding the batch index; `None` if the source
        /// is 1-D.
        column: Option<usize>,
    },

    /// Split with your own function.
    ///
    /// It receives the full batched output (including any rows padded to the
    /// model batch size) and must return exactly
    /// [`SplitContext::num_requests`] tensors, one per request in order.
    /// No automatic cropping is applied; [`SplitContext`] describes the
    /// padding. [`split_batch_axis`] and [`split_by_batch_index`] can be
    /// reused inside it.
    Custom(SplitFn),
}

impl Split {
    /// Batch index stored in column `column` of this output itself, e.g.
    /// `[batch_id, x1, y1, x2, y2, score, class]` detections.
    pub fn batch_index_column(column: usize) -> Self {
        Self::BatchIndex {
            source: None,
            column: Some(column),
        }
    }

    /// Batch index stored in another output: in its column `column`, or in the
    /// whole (1-D) output if `None`.
    pub fn batch_index_from(output: impl Into<String>, column: Option<usize>) -> Self {
        Self::BatchIndex {
            source: Some(output.into()),
            column,
        }
    }

    /// Split with a custom function, see [`Split::Custom`].
    pub fn custom(
        f: impl Fn(&Tensor, &SplitContext<'_>) -> Result<Vec<Tensor>> + Send + Sync + 'static,
    ) -> Self {
        Self::Custom(Arc::new(f))
    }
}

impl fmt::Debug for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchAxis => write!(f, "BatchAxis"),
            Self::BatchIndex { source, column } => f
                .debug_struct("BatchIndex")
                .field("source", source)
                .field("column", column)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

/// A dynamic input dimension that was padded in a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddedDim {
    /// The dimension's name: its symbolic name in the model (e.g.
    /// `"sequence"`), or `"<input>[<axis>]"` for an unnamed dimension.
    pub name: Arc<str>,
    /// Length every request was padded to.
    pub padded: usize,
    /// Each request's own length, in request order.
    pub lengths: Vec<usize>,
}

/// Information about the batch an output is being split for.
pub struct SplitContext<'a> {
    pub(crate) output_name: &'a str,
    pub(crate) num_requests: usize,
    pub(crate) batch_size: usize,
    pub(crate) padded_dims: &'a [PaddedDim],
    pub(crate) outputs: &'a HashMap<Arc<str>, Tensor>,
}

impl SplitContext<'_> {
    /// Name of the output being split.
    pub fn output_name(&self) -> &str {
        self.output_name
    }

    /// Number of real requests in the batch; the split must return this many
    /// tensors.
    pub fn num_requests(&self) -> usize {
        self.num_requests
    }

    /// Number of rows that were sent to the model. Larger than
    /// [`num_requests`](Self::num_requests) when the batch was padded to a
    /// fixed model batch size; rows `num_requests..batch_size` are padding.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Dynamic input dimensions of this batch, with each request's length.
    pub fn padded_dims(&self) -> &[PaddedDim] {
        self.padded_dims
    }

    /// Length request `request` had along dimension `dim` before padding.
    pub fn request_length(&self, dim: &str, request: usize) -> Option<usize> {
        self.padded_dims
            .iter()
            .find(|d| &*d.name == dim)
            .and_then(|d| d.lengths.get(request).copied())
    }

    /// Another raw (batched, unsplit) output of the same model run.
    pub fn output(&self, name: &str) -> Option<&Tensor> {
        self.outputs.get(name)
    }
}

/// Split along the batch axis (axis 0), as [`Split::BatchAxis`] does, minus
/// the cropping of padded dimensions.
///
/// # Errors
///
/// Returns an error if the tensor is a scalar or its axis 0 does not have
/// exactly [`SplitContext::batch_size`] entries.
pub fn split_batch_axis(tensor: &Tensor, ctx: &SplitContext<'_>) -> Result<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.is_empty() {
        anyhow::bail!(
            "output '{}' is a scalar, so it has no batch axis and cannot be split per request; \
             configure a different Split for it or leave it out of the selected outputs",
            ctx.output_name
        );
    }
    if shape[0] != ctx.batch_size {
        anyhow::bail!(
            "Output count mismatch: output '{}' has {} entries on axis 0 for a batch of {} rows, \
             so axis 0 is not one entry per request; configure a different Split for it \
             (e.g. Split::BatchIndex) or leave it out of the selected outputs",
            ctx.output_name,
            shape[0],
            ctx.batch_size
        );
    }
    Ok((0..ctx.num_requests)
        .map(|i| map_tensor!(tensor, a => a.index_axis(Axis(0), i).to_owned()))
        .collect())
}

/// Split rows by a batch index, as [`Split::BatchIndex`] does.
///
/// `index` holds the batch index of every row of `tensor`: in its column
/// `column` if set, otherwise `index` must be 1-D.
///
/// # Errors
///
/// Returns an error if shapes do not line up, an index is not a whole number,
/// or an index is outside `0..batch_size`.
pub fn split_by_batch_index(
    tensor: &Tensor,
    index: &Tensor,
    column: Option<usize>,
    ctx: &SplitContext<'_>,
) -> Result<Vec<Tensor>> {
    let name = ctx.output_name;
    let indices = visit_tensor!(index, a => batch_indices(a, column))
        .map_err(|e| anyhow::anyhow!("output '{name}': invalid batch index: {e}"))?;
    let rows = tensor.shape().first().copied().ok_or_else(|| {
        anyhow::anyhow!("output '{name}' is a scalar and has no rows to split by batch index")
    })?;
    if rows != indices.len() {
        anyhow::bail!(
            "output '{name}' has {rows} rows but its batch index has {} entries",
            indices.len()
        );
    }

    let mut per_request: Vec<Vec<usize>> = vec![Vec::new(); ctx.num_requests];
    for (row, &batch) in indices.iter().enumerate() {
        let Some(batch) = usize::try_from(batch).ok().filter(|&b| b < ctx.batch_size) else {
            anyhow::bail!(
                "output '{name}' row {row} has batch index {batch}, outside the batch of {} rows",
                ctx.batch_size
            );
        };
        // Rows of padded batch entries (`batch >= num_requests`) are dropped.
        if let Some(rows) = per_request.get_mut(batch) {
            rows.push(row);
        }
    }

    Ok(per_request
        .iter()
        .map(|rows| map_tensor!(tensor, a => a.select(Axis(0), rows)))
        .collect())
}

fn batch_indices<T: Element>(index: &ArrayD<T>, column: Option<usize>) -> Result<Vec<i64>> {
    let to_i64 = |v: T| {
        v.to_index()
            .ok_or_else(|| anyhow::anyhow!("{v:?} is not a whole number"))
    };
    match column {
        Some(col) => {
            if index.ndim() != 2 || col >= index.shape()[1] {
                anyhow::bail!(
                    "expected a [rows, columns] tensor with column {col}, got shape {:?}",
                    index.shape()
                );
            }
            index
                .index_axis(Axis(1), col)
                .iter()
                .map(|&v| to_i64(v))
                .collect()
        }
        None => {
            if index.ndim() != 1 {
                anyhow::bail!(
                    "expected a 1-D tensor (or set a column), got shape {:?}",
                    index.shape()
                );
            }
            index.iter().map(|&v| to_i64(v)).collect()
        }
    }
}

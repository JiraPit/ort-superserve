use crate::io::{Inputs, Outputs};
use anyhow::Result;
use std::future::Future;

/// Trait for defining input types that can be processed by the inference server.
///
/// Implement this trait for your input type (e.g., image, text, audio) to define
/// how raw inputs are turned into model inputs.
///
/// Preprocessing produces two things per request: the [`Preprocessed`](Input::Preprocessed)
/// data, which converts into the request's [`Inputs`] (named tensors of any
/// element type, without the batch axis), and a [`Context`](Input::Context) that
/// bypasses inference and is handed to that same request's
/// [`Output::postprocess`]. Use the context for anything postprocessing needs to
/// know about the original request, such as the original image size.
///
/// The library stacks the inputs of all requests in a batch, padding dynamic
/// dimensions where requests differ and partial batches up to a fixed model
/// batch size (see [`InputSpec`](crate::InputSpec)).
///
/// # Example
///
/// A single `f32` input: any `ndarray` array converts into [`Inputs`].
///
/// ```rust
/// use anyhow::Result;
/// use ndarray::Array3;
/// use ort_superserve::Input;
///
/// struct ImageInput { data: Array3<f32> }
///
/// impl Input for ImageInput {
///     type Preprocessed = Array3<f32>;
///     /// Original (height, width), needed to map results back to the input.
///     type Context = (usize, usize);
///
///     async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
///         let original_size = (self.data.shape()[1], self.data.shape()[2]);
///         // Apply normalization, resizing, etc.
///         Ok((self.data, original_size))
///     }
/// }
/// ```
///
/// Several named inputs of any element type:
///
/// ```rust
/// use anyhow::Result;
/// use ndarray::Array1;
/// use ort_superserve::{Input, Inputs};
///
/// struct TextInput { token_ids: Vec<i64> }
///
/// impl Input for TextInput {
///     type Preprocessed = Inputs;
///     type Context = ();
///
///     async fn preprocess(self) -> Result<(Inputs, ())> {
///         let len = self.token_ids.len();
///         let inputs = Inputs::new()
///             .with("input_ids", Array1::from(self.token_ids))
///             .with("attention_mask", Array1::<i64>::ones(len));
///         Ok((inputs, ()))
///     }
/// }
/// ```
pub trait Input: Send + 'static + Sized {
    /// The type produced by preprocessing.
    ///
    /// It converts into the request's model [`Inputs`]; use `Inputs` itself,
    /// or any `ndarray` array for a single-input model.
    type Preprocessed: Into<Inputs> + Send + 'static;

    /// Per-request data passed from preprocessing to postprocessing.
    ///
    /// It never reaches the model; the library delivers it to the
    /// [`Output::postprocess`] call for the same request. Use `()` if
    /// postprocessing needs nothing from the original request.
    type Context: Send + 'static;

    /// Preprocess the input for inference.
    ///
    /// This method runs in a Tokio task and should handle CPU-intensive work
    /// by spawning blocking tasks internally using `tokio::task::spawn_blocking`.
    ///
    /// # Returns
    ///
    /// The preprocessed data ready for batching, and the context to hand to
    /// this request's postprocessing.
    fn preprocess(self)
    -> impl Future<Output = Result<(Self::Preprocessed, Self::Context)>> + Send;

    /// Turn the preprocessed items of one batch into each request's model inputs.
    ///
    /// Called once per batch with the items in request order; it must return
    /// one [`Inputs`] per item, in the same order. The library then stacks
    /// them into batched tensors. The default converts each item with
    /// [`Into`]; override it to inspect or adjust a whole batch at once.
    /// Returning an error fails every request in the batch.
    fn batch(items: Vec<Self::Preprocessed>) -> Result<Vec<Inputs>> {
        Ok(items.into_iter().map(Into::into).collect())
    }
}

/// Trait for defining output types that can be produced by inference.
///
/// Implement this trait for your output type (e.g., detections, classifications)
/// to define how a request's model [`Outputs`] are converted to your result type.
///
/// The type parameter `C` is the [`Input::Context`] of the input this output is
/// paired with. It defaults to `()` for outputs that need no context.
///
/// # Example
///
/// ```rust
/// use anyhow::Result;
/// use ort_superserve::{Output, Outputs};
///
/// struct ClassificationOutput { class_id: usize, confidence: f32 }
///
/// impl Output for ClassificationOutput {
///     async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
///         // The batch dimension has already been sliced by the library;
///         // `logits` holds the output for this request only.
///         let logits = outputs.first::<f32>()?;
///         let class_id = logits.iter().enumerate()
///             .max_by(|a, b| a.1.total_cmp(b.1))
///             .map(|(i, _)| i)
///             .unwrap_or(0);
///         let confidence = logits[[class_id]];
///         Ok(ClassificationOutput { class_id, confidence })
///     }
/// }
/// ```
///
/// A detector with several outputs of different element types:
///
/// ```rust
/// use anyhow::Result;
/// use ort_superserve::{Output, Outputs};
///
/// struct Detections { boxes: Vec<[f32; 4]>, scores: Vec<f32>, labels: Vec<i64> }
///
/// impl Output for Detections {
///     async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
///         let boxes = outputs.get::<f32>("boxes")?;   // [num_boxes, 4]
///         Ok(Detections {
///             boxes: boxes.outer_iter().map(|b| [b[0], b[1], b[2], b[3]]).collect(),
///             scores: outputs.get::<f32>("scores")?.iter().copied().collect(),
///             labels: outputs.get::<i64>("labels")?.iter().copied().collect(),
///         })
///     }
/// }
/// ```
pub trait Output<C: Send + 'static = ()>: Send + 'static + Sized {
    /// Postprocess one request's model outputs into the final result type.
    ///
    /// This method receives the outputs belonging to a single request (split
    /// per output's [`Split`](crate::Split), with padding removed) and should
    /// convert them to your result type.
    ///
    /// This method runs in a Tokio task and can use `spawn_blocking` for
    /// CPU-intensive postprocessing like NMS.
    ///
    /// # Arguments
    ///
    /// * `outputs` - This request's model outputs.
    /// * `context` - The context produced by [`Input::preprocess`] for this request.
    ///
    /// # Returns
    ///
    /// The postprocessed result.
    fn postprocess(outputs: Outputs, context: C) -> impl Future<Output = Result<Self>> + Send;
}

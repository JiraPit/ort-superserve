use anyhow::Result;
use ndarray::{ArrayD, ArrayViewD};
use std::future::Future;

/// Trait for defining input types that can be processed by the inference server.
///
/// Implement this trait for your input type (e.g., image, text, audio) to define
/// how raw inputs are preprocessed and batched together.
///
/// Preprocessing produces two things per request: the [`Preprocessed`](Input::Preprocessed)
/// data that is stacked into the batch tensor, and a [`Context`](Input::Context) that
/// bypasses inference and is handed to that same request's
/// [`Output::postprocess`]. Use the context for anything postprocessing needs to
/// know about the original request, such as the original image size.
///
/// # Example
///
/// ```rust
/// use anyhow::Result;
/// use ndarray::{ArrayD, Array3, Axis};
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
///
///     fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
///         let views: Vec<_> = items.iter().map(|a| a.view()).collect();
///         Ok(ndarray::stack(Axis(0), &views)?.into_dyn())
///     }
/// }
/// ```
pub trait Input: Send + 'static + Sized {
    /// The type produced by preprocessing.
    ///
    /// This is the intermediate representation that can be stacked into a batch
    /// for ONNX Runtime inference.
    type Preprocessed: Send + 'static;

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

    /// Batch multiple preprocessed items into a single tensor.
    ///
    /// This method stacks individual preprocessed inputs into a batched tensor
    /// along the batch dimension (typically axis 0).
    ///
    /// # Arguments
    ///
    /// * `items` - Vector of preprocessed inputs to batch together.
    ///
    /// # Returns
    ///
    /// A batched tensor with shape `[batch_size, ...]`.
    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>>;
}

/// Trait for defining output types that can be produced by inference.
///
/// Implement this trait for your output type (e.g., detections, classifications)
/// to define how raw ONNX tensor outputs are converted to your result type.
///
/// The type parameter `C` is the [`Input::Context`] of the input this output is
/// paired with. It defaults to `()` for outputs that need no context.
///
/// # Example
///
/// ```rust
/// use anyhow::Result;
/// use ndarray::ArrayViewD;
/// use ort_superserve::Output;
///
/// struct ClassificationOutput { class_id: usize, confidence: f32 }
///
/// impl Output for ClassificationOutput {
///     async fn postprocess(raw: ArrayViewD<'_, f32>, _context: ()) -> Result<Self> {
///         // The batch dimension has already been sliced by the library.
///         // `raw` contains the output for a single inference.
///         let class_id = raw.iter().enumerate()
///             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
///             .map(|(i, _)| i)
///             .unwrap_or(0);
///         let confidence = raw[[class_id]];
///         Ok(ClassificationOutput { class_id, confidence })
///     }
/// }
/// ```
pub trait Output<C: Send + 'static = ()>: Send + 'static + Sized {
    /// Postprocess raw ONNX output into the final result type.
    ///
    /// This method receives a single inference output (batch dimension already
    /// removed by the library) and should convert it to your result type.
    ///
    /// This method runs in a Tokio task and can use `spawn_blocking` for
    /// CPU-intensive postprocessing like NMS.
    ///
    /// # Arguments
    ///
    /// * `raw` - Raw tensor output from ONNX Runtime, with batch dimension removed.
    /// * `context` - The context produced by [`Input::preprocess`] for this request.
    ///
    /// # Returns
    ///
    /// The postprocessed result.
    fn postprocess(
        raw: ArrayViewD<'_, f32>,
        context: C,
    ) -> impl Future<Output = Result<Self>> + Send;
}

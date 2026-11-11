use crate::environment::Environment;
use crate::split::Split;
use ort::session::builder::GraphOptimizationLevel;
use std::sync::Arc;
use std::time::Duration;

/// Type alias for a custom execution provider callback.
///
/// The callback receives a session builder and returns a configured builder.
/// This allows arbitrary customization of the ONNX Runtime session.
///
/// Builder methods return `ort::Error<SessionBuilder>`; the `?` operator
/// converts it into the `ort::Error` this callback returns.
pub type SessionBuilderCallback = Arc<
    dyn Fn(
            ort::session::builder::SessionBuilder,
        ) -> Result<ort::session::builder::SessionBuilder, ort::Error>
        + Send
        + Sync,
>;

/// Execution provider for ONNX Runtime inference.
///
/// Specifies which hardware accelerator to use for running the model.
/// Different providers offer different performance characteristics depending
/// on the hardware available.
#[derive(Default)]
pub enum ExecutionProvider {
    /// CPU execution using the default ONNX Runtime CPU provider.
    ///
    /// This is the default and works on all platforms without additional dependencies.
    #[default]
    Cpu,

    /// NVIDIA CUDA GPU execution.
    ///
    /// Requires CUDA to be installed and the `cuda` feature to be enabled.
    Cuda {
        /// GPU device ID to use (defaults to 0 for first GPU).
        device_id: usize,
    },

    /// NVIDIA TensorRT execution for optimized inference.
    ///
    /// Requires TensorRT to be installed and the `tensorrt` feature to be enabled.
    /// Offers the best performance on NVIDIA GPUs for supported models.
    TensorRT {
        /// GPU device ID to use.
        device_id: usize,
        /// Enable FP16 precision for faster inference.
        fp16: bool,
    },

    /// XNNPACK execution for optimized CPU inference.
    ///
    /// Requires the `xnnpack` feature to be enabled. Good for ARM processors.
    Xnnpack,

    /// Apple CoreML execution for iOS/macOS devices.
    ///
    /// Requires the `coreml` feature to be enabled.
    CoreML,

    /// Custom execution provider with a callback.
    ///
    /// Allows arbitrary configuration of the ONNX Runtime session builder.
    /// Use this for providers not covered by the built-in variants or for
    /// advanced configuration options.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ort_superserve::ExecutionProvider;
    /// use std::sync::Arc;
    ///
    /// let custom = ExecutionProvider::Custom(Arc::new(|builder| {
    ///     Ok(builder.with_memory_pattern(true)?)
    /// }));
    /// ```
    Custom(SessionBuilderCallback),
}

impl Clone for ExecutionProvider {
    fn clone(&self) -> Self {
        match self {
            Self::Cpu => Self::Cpu,
            Self::Cuda { device_id } => Self::Cuda {
                device_id: *device_id,
            },
            Self::TensorRT { device_id, fp16 } => Self::TensorRT {
                device_id: *device_id,
                fp16: *fp16,
            },
            Self::Xnnpack => Self::Xnnpack,
            Self::CoreML => Self::CoreML,
            Self::Custom(callback) => Self::Custom(Arc::clone(callback)),
        }
    }
}

impl std::fmt::Debug for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "Cpu"),
            Self::Cuda { device_id } => f
                .debug_struct("Cuda")
                .field("device_id", device_id)
                .finish(),
            Self::TensorRT { device_id, fp16 } => f
                .debug_struct("TensorRT")
                .field("device_id", device_id)
                .field("fp16", fp16)
                .finish(),
            Self::Xnnpack => write!(f, "Xnnpack"),
            Self::CoreML => write!(f, "CoreML"),
            Self::Custom(_) => write!(f, "Custom(<callback>)"),
        }
    }
}

/// Options for one model input.
///
/// # Example
///
/// ```rust
/// use ort_superserve::{InputSpec, ServerConfig};
///
/// // Token sequences of different lengths: pad ids with 0 (the default) and
/// // the attention mask with 0 so padded positions are ignored.
/// let config = ServerConfig::new()
///     .with_input(InputSpec::new("input_ids").with_pad_value(0.0))
///     .with_input(InputSpec::new("attention_mask").with_pad_value(0.0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct InputSpec {
    /// Model input name.
    pub name: String,

    /// Value used to fill padding of this input, converted to its element
    /// type (non-zero means `true` for `bool`). Defaults to `0`.
    ///
    /// Padding is added when requests in a batch have different lengths along
    /// a dynamic dimension (each is padded to the longest), and to fill whole
    /// rows when a partial batch is padded to a fixed model batch size.
    pub pad_value: f64,

    /// Whether the input carries the batch axis.
    ///
    /// `None` (default) means batched unless the model declares the input as
    /// a scalar. An unbatched input (e.g. an NMS threshold) is passed to the
    /// model as given and must be identical for every request in a batch.
    pub batched: Option<bool>,
}

impl InputSpec {
    /// Options for the model input `name`, with default values.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pad_value: 0.0,
            batched: None,
        }
    }

    /// Set the padding value, see [`InputSpec::pad_value`].
    pub fn with_pad_value(mut self, value: f64) -> Self {
        self.pad_value = value;
        self
    }

    /// Set whether the input carries the batch axis, see [`InputSpec::batched`].
    pub fn with_batched(mut self, batched: bool) -> Self {
        self.batched = Some(batched);
        self
    }
}

/// An output axis that follows a dynamic dimension of an input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CropAxis {
    /// Axis of the (batched) output.
    pub axis: usize,
    /// Input whose dimension the axis follows.
    pub input: String,
    /// Axis of the (batched) input; axis 0 is the batch axis.
    pub input_axis: usize,
}

/// Options for one model output.
///
/// # Example
///
/// ```rust
/// use ort_superserve::{OutputSpec, ServerConfig, Split};
///
/// // NMS model: `detections` rows are `[batch_id, x1, y1, x2, y2, score, class]`.
/// let config = ServerConfig::new()
///     .with_output(OutputSpec::new("detections").with_split(Split::batch_index_column(0)));
/// ```
///
/// # Outputs of models with padded inputs
///
/// When an input has a dynamic dimension besides the batch axis (e.g.
/// `input_ids: [batch, sequence]`), requests of different lengths share a
/// batch: shorter ones are padded to the longest (see
/// [`InputSpec::pad_value`]). An output with one entry per input position,
/// like per-token logits `[batch, sequence, num_labels]`, then also has the
/// padded length, and each request's result is cut back to its own length
/// before it reaches the caller.
///
/// To do that, the server has to know which output axes follow which padded
/// input dimension. It reads this from the symbolic dimension names in the
/// model: an output axis named like an input dimension (both `sequence`)
/// follows it, and an axis with a fixed size or with another name chosen by
/// the model author does not. Exporters often generate names instead: an
/// empty name, `?`, a name starting with `unk__`, or one containing
/// `dynamic_axes`. For such an axis the server cannot tell, so if any input
/// can be padded, [`Server::from_file`](crate::Server::from_file) fails with
/// an error naming the output and axis, until you declare it:
///
/// - [`with_crop_axis`](Self::with_crop_axis) if the axis has one entry per
///   position of an input dimension, so padded positions are cut off;
/// - [`with_uncropped_axis`](Self::with_uncropped_axis) if it does not, so it
///   is returned in full.
///
/// Only selected outputs are checked: if you do not need an output, select
/// the ones you need with [`ServerConfig::with_output`] instead. Outputs with
/// a [`Split::Custom`] are not checked either, since their split function
/// handles padding itself.
///
/// If the model declares no shape for an output, nothing can be checked at
/// startup. The server then fails any batch where an axis of that output has
/// the length an input dimension was padded to, rather than return padded
/// positions to callers; declare the axes to avoid this.
///
/// ```rust
/// use ort_superserve::{InputSpec, OutputSpec, ServerConfig};
///
/// // input_ids: [batch, sequence]
/// // logits:    [batch, unk__0, num_labels]  one row per token
/// // top_ids:   [batch, unk__1]              k best labels, unrelated to the sequence
/// let config = ServerConfig::new()
///     .with_input(InputSpec::new("input_ids").with_pad_value(0.0))
///     .with_output(OutputSpec::new("logits").with_crop_axis(1, "input_ids", 1))
///     .with_output(OutputSpec::new("top_ids").with_uncropped_axis(1));
/// ```
#[derive(Clone, Debug)]
pub struct OutputSpec {
    /// Model output name.
    pub name: String,

    /// How the output is split per request; `None` uses the default
    /// [`Split::BatchAxis`] with its metadata checks.
    pub split: Option<Split>,

    /// Output axes that follow a padded input dimension, in addition to the
    /// ones detected from the model's dimension names.
    pub crop_axes: Vec<CropAxis>,

    /// Output axes that never follow a padded input dimension, even if their
    /// length happens to equal a padded length.
    pub uncropped_axes: Vec<usize>,
}

impl OutputSpec {
    /// Options for the model output `name`, with default values.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            split: None,
            crop_axes: Vec::new(),
            uncropped_axes: Vec::new(),
        }
    }

    /// Set how the output is split per request.
    pub fn with_split(mut self, split: Split) -> Self {
        self.split = Some(split);
        self
    }

    /// Declare that output axis `axis` follows axis `input_axis` of input
    /// `input`, so each request's result is cut back to that request's own
    /// length along it.
    ///
    /// Both axes are counted with the batch axis, so the first axis after it
    /// is `1`: for `logits: [batch, unk__0, num_labels]` following
    /// `input_ids: [batch, sequence]`, use `with_crop_axis(1, "input_ids", 1)`.
    ///
    /// Axes named like the input dimension in the model are linked
    /// automatically; this is for axes whose names do not match, typically
    /// names generated by the exporter. See
    /// [Outputs of models with padded inputs](OutputSpec#outputs-of-models-with-padded-inputs)
    /// for when it is required.
    ///
    /// Startup fails if `axis` is 0, `input` is not a batched model input, or
    /// its axis `input_axis` is the batch axis, does not exist or has a fixed
    /// size. A batch fails if the output axis does not have the length the
    /// input dimension was padded to, since then it does not follow it.
    pub fn with_crop_axis(
        mut self,
        axis: usize,
        input: impl Into<String>,
        input_axis: usize,
    ) -> Self {
        self.crop_axes.push(CropAxis {
            axis,
            input: input.into(),
            input_axis,
        });
        self
    }

    /// Declare that output axis `axis` (counted with the batch axis) does not
    /// follow any padded input dimension, so it is returned in full, even
    /// when its length happens to equal a length an input was padded to.
    ///
    /// Use it for a dynamic output axis with a name generated by the exporter
    /// that is unrelated to the input lengths, e.g. the `k` of a top-k
    /// output. See
    /// [Outputs of models with padded inputs](OutputSpec#outputs-of-models-with-padded-inputs)
    /// for when it is required.
    pub fn with_uncropped_axis(mut self, axis: usize) -> Self {
        self.uncropped_axes.push(axis);
        self
    }
}

/// Configuration for the inference server.
///
/// Controls session pool size, batching behavior, and execution providers.
/// Use the builder methods to customize the configuration.
///
/// # Example
///
/// ```rust
/// use ort_superserve::ServerConfig;
/// use std::time::Duration;
///
/// let config = ServerConfig::new()
///     .with_num_sessions(4)
///     .with_threads_per_session(2)
///     .with_max_batch_size(16)
///     .with_max_wait_time(Duration::from_millis(10));
/// ```
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Number of ONNX Runtime sessions in the pool.
    ///
    /// Multiple sessions allow better hardware utilization on multi-core systems.
    /// Each session runs in its own thread.
    pub num_sessions: usize,

    /// Number of intra-op threads per session.
    ///
    /// Controls parallelism within a single inference call. Lower values
    /// (1-4) work better when using multiple sessions.
    pub threads_per_session: usize,

    /// Maximum number of requests to batch together.
    ///
    /// The batcher will collect up to this many requests before sending
    /// them to inference. Higher values increase throughput but also latency.
    ///
    /// For a model exported with a fixed batch size, batches are capped at
    /// that size, and smaller batches are padded up to it (the padded rows
    /// never reach a caller).
    pub max_batch_size: usize,

    /// Minimum number of requests required to form a batch.
    ///
    /// If fewer requests are collected within `max_wait_time`, the batch
    /// is rejected and errors are returned to the callers.
    pub min_batch_size: usize,

    /// Maximum time to wait for a full batch.
    ///
    /// The batcher will wait up to this duration for more requests before
    /// sending what it has to inference.
    pub max_wait_time: Duration,

    /// Execution providers to use for inference.
    ///
    /// Multiple providers can be specified; ONNX Runtime will use the first
    /// available one.
    pub execution_providers: Vec<ExecutionProvider>,

    /// Graph optimization level for ONNX Runtime.
    ///
    /// Controls how aggressively ONNX Runtime optimizes the computation graph.
    pub optimization_level: GraphOptimizationLevel,

    /// Default input name, receiving the unnamed tensor of an
    /// [`Inputs`](crate::Inputs).
    ///
    /// If not specified, the first input from the model is used.
    pub input_name: Option<String>,

    /// Per-input options (padding value, batching).
    ///
    /// Inputs not listed here use [`InputSpec::new`] defaults.
    pub inputs: Vec<InputSpec>,

    /// Outputs to fetch, in order, with how each is split per request.
    ///
    /// If empty (default), every model output is fetched in model order with
    /// the default [`Split::BatchAxis`].
    pub outputs: Vec<OutputSpec>,

    /// Whether to preprocess inputs immediately in `infer()` before queuing.
    ///
    /// When `true`, preprocessing runs in parallel across all `infer()` calls
    /// using `tokio::spawn`, allowing independent preprocessing but with higher
    /// memory usage risk (potential OOM if preprocessing is memory-intensive).
    ///
    /// When `false` (default), preprocessing is batched and run together using
    /// `JoinSet` in the batcher, which is more memory-efficient but may have
    /// higher latency.
    pub preprocess_on_infer: bool,

    /// ONNX Runtime environment to create sessions in.
    ///
    /// ONNX Runtime allows only one environment per process. If not specified,
    /// the server creates its own, which fails if this process already has one.
    /// Applications with several servers (or that recreate servers) should create
    /// one with [`create_env`](crate::create_env) and pass a clone to each.
    pub environment: Option<Environment>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            num_sessions: 1,
            threads_per_session: 1,
            max_batch_size: 8,
            min_batch_size: 1,
            max_wait_time: Duration::from_millis(10),
            execution_providers: vec![ExecutionProvider::Cpu],
            optimization_level: GraphOptimizationLevel::Level3,
            input_name: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            preprocess_on_infer: false,
            environment: None,
        }
    }
}

impl ServerConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of ONNX Runtime sessions in the pool.
    ///
    /// More sessions allow better utilization of multi-core systems.
    /// Each session runs inference in its own dedicated thread.
    pub fn with_num_sessions(mut self, num: usize) -> Self {
        self.num_sessions = num;
        self
    }

    /// Set the number of intra-op threads per session.
    ///
    /// Controls parallelism within a single inference call.
    /// Lower values (1-4) recommended when using multiple sessions.
    pub fn with_threads_per_session(mut self, threads: usize) -> Self {
        self.threads_per_session = threads;
        self
    }

    /// Set the maximum batch size.
    ///
    /// The batcher will collect up to this many requests before sending
    /// them to inference. It is capped at the model's batch size if the model
    /// has a fixed one.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set the minimum batch size.
    ///
    /// Batches smaller than this will be rejected after `max_wait_time` elapses.
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Set the maximum time to wait for a full batch.
    ///
    /// The batcher will wait up to this duration before sending available
    /// requests to inference.
    pub fn with_max_wait_time(mut self, duration: Duration) -> Self {
        self.max_wait_time = duration;
        self
    }

    /// Add an execution provider.
    ///
    /// Multiple providers can be added; ONNX Runtime will use the first
    /// available one.
    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_providers.push(provider);
        self
    }

    /// Set the graph optimization level.
    ///
    /// Controls how aggressively ONNX Runtime optimizes the computation graph.
    pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set the default input name, which receives the unnamed tensor of an
    /// [`Inputs`](crate::Inputs) (e.g. one converted from an array).
    ///
    /// If not specified, the first input from the model is used.
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = Some(name.into());
        self
    }

    /// Set options for one input. Setting the same input again replaces its
    /// options.
    pub fn with_input(mut self, spec: InputSpec) -> Self {
        self.inputs.retain(|s| s.name != spec.name);
        self.inputs.push(spec);
        self
    }

    /// Set the padding value of one input, see [`InputSpec::pad_value`].
    pub fn with_pad_value(mut self, input: impl Into<String>, value: f64) -> Self {
        let input = input.into();
        match self.inputs.iter_mut().find(|s| s.name == input) {
            Some(spec) => spec.pad_value = value,
            None => self
                .inputs
                .push(InputSpec::new(input).with_pad_value(value)),
        }
        self
    }

    /// Fetch an output, with its options.
    ///
    /// Once any output is configured, only the configured outputs are fetched,
    /// in the order they were added. Adding the same output again replaces its
    /// options.
    pub fn with_output(mut self, spec: OutputSpec) -> Self {
        match self.outputs.iter_mut().find(|s| s.name == spec.name) {
            Some(existing) => *existing = spec,
            None => self.outputs.push(spec),
        }
        self
    }

    /// Fetch the output `name` with default options; shorthand for
    /// `with_output(OutputSpec::new(name))`.
    pub fn with_output_name(self, name: impl Into<String>) -> Self {
        self.with_output(OutputSpec::new(name))
    }

    /// Fetch the output `name`, split with `split`; shorthand for
    /// `with_output(OutputSpec::new(name).with_split(split))`.
    pub fn with_output_split(self, name: impl Into<String>, split: Split) -> Self {
        self.with_output(OutputSpec::new(name).with_split(split))
    }

    /// Set whether to preprocess inputs immediately in `infer()` before queuing.
    ///
    /// When `true`, preprocessing runs in parallel across all `infer()` calls,
    /// allowing independent preprocessing but with higher memory usage risk.
    /// When `false` (default), preprocessing is batched and run together.
    pub fn with_preprocess_on_infer(mut self, enabled: bool) -> Self {
        self.preprocess_on_infer = enabled;
        self
    }

    /// Set the ONNX Runtime environment to create sessions in.
    ///
    /// Create it once with [`create_env`](crate::create_env) and pass a clone
    /// to every server. If not set, the server creates its own environment,
    /// which fails if this process already has one.
    pub fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = Some(environment);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = ServerConfig::default();

        assert_eq!(config.num_sessions, 1);
        assert_eq!(config.threads_per_session, 1);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_wait_time, Duration::from_millis(10));
        assert!(matches!(
            config.execution_providers[..],
            [ExecutionProvider::Cpu]
        ));
        assert_eq!(config.optimization_level, GraphOptimizationLevel::Level3);
        assert_eq!(config.input_name, None);
        assert!(config.outputs.is_empty());
        assert!(!config.preprocess_on_infer);
        assert!(config.environment.is_none());
    }

    #[test]
    fn builder_methods_set_their_fields() {
        let config = ServerConfig::new()
            .with_num_sessions(4)
            .with_threads_per_session(2)
            .with_max_batch_size(16)
            .with_min_batch_size(2)
            .with_max_wait_time(Duration::from_millis(20))
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .with_input_name("input_tensor")
            .with_output_name("output_tensor")
            .with_preprocess_on_infer(true);

        assert_eq!(config.num_sessions, 4);
        assert_eq!(config.threads_per_session, 2);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.min_batch_size, 2);
        assert_eq!(config.max_wait_time, Duration::from_millis(20));
        assert_eq!(config.optimization_level, GraphOptimizationLevel::Level1);
        assert_eq!(config.input_name.as_deref(), Some("input_tensor"));
        let output_names: Vec<_> = config.outputs.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(output_names, ["output_tensor"]);
        assert!(config.preprocess_on_infer);
    }

    #[test]
    fn cloned_custom_provider_shares_the_callback() {
        let provider = ExecutionProvider::Custom(Arc::new(Ok));

        match (&provider, &provider.clone()) {
            (ExecutionProvider::Custom(a), ExecutionProvider::Custom(b)) => {
                assert!(Arc::ptr_eq(a, b))
            }
            _ => panic!("clone changed the provider variant"),
        }
    }

    #[test]
    fn debug_output_hides_custom_callback() {
        let provider = ExecutionProvider::Custom(Arc::new(Ok));
        let tensorrt = ExecutionProvider::TensorRT {
            device_id: 1,
            fp16: true,
        };

        assert_eq!(format!("{provider:?}"), "Custom(<callback>)");
        assert_eq!(
            format!("{tensorrt:?}"),
            "TensorRT { device_id: 1, fp16: true }"
        );
    }
}

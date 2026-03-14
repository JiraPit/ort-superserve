use ort::session::builder::GraphOptimizationLevel;
use std::sync::Arc;
use std::time::Duration;

/// Type alias for a custom execution provider callback.
///
/// The callback receives a session builder and returns a configured builder.
/// This allows arbitrary customization of the ONNX Runtime session.
///
/// Note: The error type is `ort::Error<SessionBuilder>` which allows recovering
/// the builder from errors. Use `.map_err(|e| e.to_string())` if you need
/// to convert errors to strings.
pub type SessionBuilderCallback = Arc<
    dyn Fn(
            ort::session::builder::SessionBuilder,
        ) -> Result<
            ort::session::builder::SessionBuilder,
            ort::Error<ort::session::builder::SessionBuilder>,
        > + Send
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
    /// ```rust,ignore
    /// use ort_superserve::ExecutionProvider;
    /// use std::sync::Arc;
    ///
    /// let custom = ExecutionProvider::Custom(Arc::new(|builder| {
    ///     builder.with_memory_pattern(true)
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
    /// them to inference.
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
}

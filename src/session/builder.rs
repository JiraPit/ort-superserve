use crate::config::{ExecutionProvider, ServerConfig};
use crate::environment::Environment;
use anyhow::Result;
use ort::session::Session;

/// Builder for creating ONNX Runtime sessions with configuration.
///
/// This struct is not meant to be instantiated directly; it provides
/// static methods for building sessions.
pub struct SessionBuilder;

impl SessionBuilder {
    /// Build an ONNX Runtime session from a model file.
    ///
    /// Creates a session with the specified configuration, including
    /// thread counts, optimization level, and execution providers.
    ///
    /// # Arguments
    ///
    /// * `environment` - ONNX Runtime environment to create the session in.
    /// * `model_path` - Path to the ONNX model file.
    /// * `config` - Server configuration containing session settings.
    ///
    /// # Returns
    ///
    /// An initialized ONNX Runtime session.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded or if
    /// the requested execution provider is not available.
    pub fn build(
        environment: &Environment,
        model_path: &str,
        config: &ServerConfig,
    ) -> Result<Session> {
        let mut builder =
            Session::builder(environment).map_err(|e| anyhow::Error::msg(e.to_string()))?;

        builder = builder
            .with_optimization_level(config.optimization_level)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_intra_threads(config.threads_per_session)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        for provider in &config.execution_providers {
            builder = Self::apply_execution_provider(builder, provider)?;
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
    }

    /// Apply an execution provider to the session builder.
    ///
    /// Configures the session to use the specified hardware accelerator.
    fn apply_execution_provider(
        builder: ort::session::builder::SessionBuilder,
        provider: &ExecutionProvider,
    ) -> Result<ort::session::builder::SessionBuilder> {
        match provider {
            ExecutionProvider::Cpu => Ok(builder),
            ExecutionProvider::Cuda { device_id: _ } => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::CUDA;
                    builder
                        .with_execution_providers([CUDA::default().build()])
                        .map_err(|e| anyhow::Error::msg(e.to_string()))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!(
                        "CUDA execution provider requested but not compiled with cuda feature"
                    )
                }
            }
            ExecutionProvider::TensorRT {
                device_id: _,
                fp16: _,
            } => {
                #[cfg(feature = "tensorrt")]
                {
                    use ort::ep::TensorRT;
                    builder
                        .with_execution_providers([TensorRT::default().build()])
                        .map_err(|e| anyhow::Error::msg(e.to_string()))
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    anyhow::bail!(
                        "TensorRT execution provider requested but not compiled with tensorrt feature"
                    )
                }
            }
            ExecutionProvider::Xnnpack => {
                #[cfg(feature = "xnnpack")]
                {
                    use ort::ep::XNNPACK;
                    builder
                        .with_execution_providers([XNNPACK::default().build()])
                        .map_err(|e| anyhow::Error::msg(e.to_string()))
                }
                #[cfg(not(feature = "xnnpack"))]
                {
                    anyhow::bail!(
                        "XNNPACK execution provider requested but not compiled with xnnpack feature"
                    )
                }
            }
            ExecutionProvider::CoreML => {
                #[cfg(feature = "coreml")]
                {
                    use ort::ep::CoreML;
                    builder
                        .with_execution_providers([CoreML::default().build()])
                        .map_err(|e| anyhow::Error::msg(e.to_string()))
                }
                #[cfg(not(feature = "coreml"))]
                {
                    anyhow::bail!(
                        "CoreML execution provider requested but not compiled with coreml feature"
                    )
                }
            }
            ExecutionProvider::Custom(callback) => {
                callback(builder).map_err(|e| anyhow::Error::msg(e.to_string()))
            }
        }
    }
}

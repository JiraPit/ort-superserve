use crate::actor::{BatcherTask, Command, WorkerMessage, WorkerTask};
use crate::config::ServerConfig;
use crate::session::SessionBuilder;
use crate::traits::{Input, Output};
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

/// ONNX Runtime inference server with dynamic batching and session pooling.
///
/// The `Server` orchestrates multiple ONNX Runtime sessions with automatic
/// request batching and parallel preprocessing/postprocessing.
///
/// # Type Parameters
///
/// - `I`: Input type implementing [`Input`] trait
/// - `O`: Output type implementing [`Output`] trait
///
/// # Example
///
/// ```rust,no_run
/// use ort_superserve::{Server, ServerConfig, Input, Output};
/// use ndarray::{ArrayD, ArrayViewD, Array3, Axis};
/// use anyhow::Result;
///
/// struct MyInput { data: Array3<f32> }
/// impl Input for MyInput {
///     type Preprocessed = Array3<f32>;
///     async fn preprocess(self) -> Result<Self::Preprocessed> { Ok(self.data) }
///     fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
///         let views: Vec<_> = items.iter().map(|a| a.view()).collect();
///         Ok(ndarray::stack(Axis(0), &views)?.into_dyn())
///     }
/// }
///
/// struct MyOutput { result: Vec<f32> }
/// impl Output for MyOutput {
///     async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
///         Ok(MyOutput { result: raw.iter().cloned().collect() })
///     }
/// }
///
/// # async fn example() -> Result<()> {
/// let config = ServerConfig::new()
///     .with_num_sessions(4)
///     .with_max_batch_size(16);
///
/// let server = Server::<MyInput, MyOutput>::from_file("model.onnx", config).await?;
///
/// // Use server.clone() for concurrent requests
/// let result = server.infer(MyInput { data: Array3::zeros((3, 224, 224)) }).await?;
///
/// server.shutdown();
/// # Ok(())
/// # }
/// ```
pub struct Server<I: Input, O: Output> {
    /// Channel sender for dispatching inference commands to the batcher.
    command_tx: mpsc::UnboundedSender<Command<I, O>>,
    _phantom: std::marker::PhantomData<O>,
}

impl<I: Input, O: Output> Clone for Server<I, O> {
    fn clone(&self) -> Self {
        Self {
            command_tx: self.command_tx.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I: Input, O: Output> Server<I, O> {
    /// Load a model from a local file and start the inference server.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    /// * `config` - Server configuration.
    ///
    /// # Returns
    ///
    /// A running server instance ready to accept inference requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded or the session
    /// fails to initialize.
    pub async fn from_file<P: AsRef<Path>>(path: P, config: ServerConfig) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let session = SessionBuilder::build(&path_str, &config)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .context("Model has no inputs")?;

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("Model has no outputs")?;

        drop(session);

        Self::start(path_str, config, input_name, output_name)
    }

    /// Download a model from a URL and start the inference server.
    ///
    /// The model is downloaded to a temporary file and loaded into memory.
    ///
    /// # Arguments
    ///
    /// * `url` - URL to download the ONNX model from.
    /// * `config` - Server configuration.
    ///
    /// # Returns
    ///
    /// A running server instance ready to accept inference requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the download fails or the model cannot be loaded.
    pub async fn from_url(url: &str, config: ServerConfig) -> Result<Self> {
        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join(format!(
            "ort-superserve-model-{}.onnx",
            uuid::Uuid::new_v4()
        ));
        let model_path_str = model_path.to_string_lossy().to_string();

        tracing::info!("Downloading model from {}...", url);

        let response = reqwest::get(url)
            .await
            .with_context(|| format!("Failed to download model from {}", url))?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download model: HTTP {}", response.status());
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read model bytes")?;

        std::fs::write(&model_path, &bytes)
            .with_context(|| format!("Failed to write model to {}", model_path_str))?;

        tracing::info!("Model downloaded successfully.");

        let session = SessionBuilder::build(&model_path_str, &config)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .context("Model has no inputs")?;

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("Model has no outputs")?;

        drop(session);

        Self::start(model_path_str, config, input_name, output_name)
    }

    /// Start the server with the given configuration.
    ///
    /// Spawns worker threads for ONNX sessions and a batcher task for
    /// collecting and dispatching requests.
    fn start(
        model_path: String,
        config: ServerConfig,
        input_name: String,
        output_name: String,
    ) -> Result<Self> {
        let config = Arc::new(config);
        let model_path = Arc::new(model_path);

        let (command_tx, command_rx) = mpsc::unbounded_channel::<Command<I, O>>();
        let (worker_tx, worker_rx) = std::sync::mpsc::channel::<WorkerMessage>();
        let worker_rx = Arc::new(std::sync::Mutex::new(worker_rx));

        for worker_id in 0..config.num_sessions {
            WorkerTask::spawn(
                Arc::clone(&worker_rx),
                Arc::clone(&model_path),
                Arc::clone(&config),
                worker_id,
            );
        }

        let _batcher_handle = BatcherTask::spawn::<I, O>(
            command_rx,
            worker_tx,
            Arc::clone(&config),
            input_name,
            output_name,
        );

        Ok(Self {
            command_tx,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Submit an inference request to the server.
    ///
    /// The request will be batched with other pending requests and sent to
    /// an available ONNX session for processing.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data to process.
    ///
    /// # Returns
    ///
    /// The processed output from the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the server channel is closed or inference fails.
    pub async fn infer(&self, input: I) -> Result<O> {
        let (responder, response) = tokio::sync::oneshot::channel();

        self.command_tx
            .send(Command::new(input, responder))
            .map_err(|_| anyhow::Error::msg("Server channel closed"))?;

        response.await.context("Response channel closed")?
    }

    /// Shut down the server.
    ///
    /// Drops the command channel, causing the batcher and workers to
    /// gracefully terminate after completing pending requests.
    pub fn shutdown(self) {
        drop(self.command_tx);
    }
}

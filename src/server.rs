use crate::actor::{BatcherTask, Command, WorkerMessage, WorkerTask};
use crate::config::ServerConfig;
use crate::environment::{Environment, create_env};
use crate::model::ModelPlan;
use crate::session::SessionBuilder;
use crate::traits::{Input, Output};
use anyhow::{Context, Result};
use kanal;
use std::path::Path;
use std::sync::Arc;

/// ONNX Runtime inference server with dynamic batching and session pooling.
///
/// The `Server` orchestrates multiple ONNX Runtime sessions with automatic
/// request batching and parallel preprocessing/postprocessing.
///
/// # Type Parameters
///
/// - `I`: Input type implementing [`Input`] trait
/// - `O`: Output type implementing [`Output<I::Context>`](Output), so its
///   postprocessing receives the context produced by `I`'s preprocessing
///
/// # Example
///
/// ```rust,no_run
/// use ort_superserve::{Server, ServerConfig, Input, Output, Outputs, create_env};
/// use ndarray::Array3;
/// use anyhow::Result;
///
/// struct MyInput { data: Array3<f32> }
/// impl Input for MyInput {
///     type Preprocessed = Array3<f32>;
///     type Context = ();
///     async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> { Ok((self.data, ())) }
/// }
///
/// struct MyOutput { result: Vec<f32> }
/// impl Output for MyOutput {
///     async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
///         Ok(MyOutput { result: outputs.first::<f32>()?.iter().cloned().collect() })
///     }
/// }
///
/// # async fn example() -> Result<()> {
/// // Create the ONNX Runtime environment explicitly (one per process; clone it
/// // to share it between servers).
/// let env = create_env()?;
///
/// let config = ServerConfig::new()
///     .with_num_sessions(4)
///     .with_max_batch_size(16)
///     .with_environment(env);
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
pub struct Server<I: Input, O: Output<I::Context>> {
    command_tx: kanal::AsyncSender<Command<I, O>>,
    preprocess_on_infer: bool,
    _phantom: std::marker::PhantomData<O>,
}

impl<I: Input, O: Output<I::Context>> Clone for Server<I, O> {
    fn clone(&self) -> Self {
        Self {
            command_tx: self.command_tx.clone(),
            preprocess_on_infer: self.preprocess_on_infer,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I: Input, O: Output<I::Context>> Server<I, O> {
    /// Load a model from a local file and start the inference server.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file.
    /// * `config` - Server configuration. If it has no environment (see
    ///   [`ServerConfig::with_environment`]), the server creates its own.
    ///
    /// # Returns
    ///
    /// A running server instance ready to accept inference requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded, the model or the
    /// input options can never be served (e.g. no inputs, conflicting fixed
    /// batch sizes, an [`InputSpec`](crate::InputSpec) for an unknown input),
    /// a selected output has a dynamic axis that may or may not follow a
    /// padded input dimension (see
    /// [Outputs of models with padded inputs](crate::OutputSpec#outputs-of-models-with-padded-inputs)),
    /// any session in the pool fails to initialize, or the config has no
    /// environment and one already exists in this process. Outputs that
    /// cannot be split per request do not prevent startup: they are logged
    /// and every request then fails with an error explaining why.
    pub async fn from_file<P: AsRef<Path>>(path: P, config: ServerConfig) -> Result<Self> {
        let environment = Self::resolve_environment(&config)?;
        let path_str = path.as_ref().to_string_lossy().to_string();

        Self::probe_and_start(environment, path_str, config).await
    }

    /// Download a model from a URL and start the inference server.
    ///
    /// The model is downloaded to a temporary file, loaded into every session,
    /// and the file is then deleted (also when startup fails).
    ///
    /// # Arguments
    ///
    /// * `url` - URL to download the ONNX model from.
    /// * `config` - Server configuration. If it has no environment (see
    ///   [`ServerConfig::with_environment`]), the server creates its own.
    ///
    /// # Returns
    ///
    /// A running server instance ready to accept inference requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the download fails, or for any of the reasons
    /// [`from_file`](Self::from_file) fails.
    pub async fn from_url(url: &str, config: ServerConfig) -> Result<Self> {
        let environment = Self::resolve_environment(&config)?;
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

        if let Err(e) = std::fs::write(&model_path, &bytes) {
            let _ = std::fs::remove_file(&model_path);
            return Err(e).with_context(|| format!("Failed to write model to {}", model_path_str));
        }

        tracing::info!("Model downloaded successfully.");

        // Every session has loaded the model once startup finishes, successfully
        // or not, so the downloaded file is no longer needed.
        let result = Self::probe_and_start(environment, model_path_str, config).await;
        if let Err(e) = std::fs::remove_file(&model_path) {
            tracing::warn!(
                "Failed to remove downloaded model {}: {e}",
                model_path.display()
            );
        }
        result
    }

    /// Inspect the model with a temporary session, then start the server.
    async fn probe_and_start(
        environment: Environment,
        model_path: String,
        config: ServerConfig,
    ) -> Result<Self> {
        let session = SessionBuilder::build(&environment, &model_path, &config)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        let plan = ModelPlan::new(&session, &config)?;
        drop(session);

        Self::start(environment, model_path, config, plan).await
    }

    /// Use the configured environment, or create one if none was given.
    fn resolve_environment(config: &ServerConfig) -> Result<Environment> {
        match &config.environment {
            Some(environment) => Ok(environment.clone()),
            None => create_env(),
        }
    }

    /// Start the server with the given configuration.
    ///
    /// Spawns worker threads for ONNX sessions and a batcher task for
    /// collecting and dispatching requests. Waits until every worker's session
    /// is ready, and fails if any of them cannot be initialized.
    async fn start(
        environment: Environment,
        model_path: String,
        config: ServerConfig,
        plan: ModelPlan,
    ) -> Result<Self> {
        let max_batch_size = match plan.batch_size() {
            Some(model_batch) if model_batch < config.max_batch_size => {
                tracing::warn!(
                    "The model has a fixed batch size of {model_batch}; batches are capped at \
                     {model_batch} instead of max_batch_size {}.",
                    config.max_batch_size
                );
                model_batch
            }
            _ => config.max_batch_size,
        };
        if config.min_batch_size > max_batch_size {
            tracing::warn!(
                "min_batch_size {} exceeds the largest possible batch of {max_batch_size}; \
                 every batch will be rejected.",
                config.min_batch_size
            );
        }
        let config = Arc::new(config);
        let plan = Arc::new(plan);
        let model_path = Arc::new(model_path);

        let (command_tx, command_rx) = kanal::unbounded_async::<Command<I, O>>();

        let mut worker_txs: Vec<kanal::AsyncSender<WorkerMessage>> = Vec::new();
        let mut ready_rxs = Vec::with_capacity(config.num_sessions);

        for worker_id in 0..config.num_sessions {
            let (worker_tx, worker_rx) = kanal::unbounded_async::<WorkerMessage>();
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            worker_txs.push(worker_tx);
            ready_rxs.push(ready_rx);

            WorkerTask::spawn(
                worker_rx,
                ready_tx,
                environment.clone(),
                Arc::clone(&model_path),
                Arc::clone(&config),
                worker_id,
            );
        }

        // On failure, returning drops `worker_txs`, which stops the healthy workers.
        for (worker_id, ready_rx) in ready_rxs.into_iter().enumerate() {
            ready_rx
                .await
                .context("Worker exited before reporting readiness")?
                .with_context(|| format!("Worker {worker_id} failed to initialize its session"))?;
        }

        let _batcher_handle = BatcherTask::spawn::<I, O>(
            command_rx,
            worker_txs,
            Arc::clone(&config),
            plan,
            max_batch_size,
        );

        Ok(Self {
            command_tx,
            preprocess_on_infer: config.preprocess_on_infer,
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
    /// Returns an error if the server channel is closed, the request's inputs
    /// do not match the model, inference fails, an output cannot be split per
    /// request, or postprocessing fails.
    pub async fn infer(&self, input: I) -> Result<O> {
        let (responder, response) = tokio::sync::oneshot::channel();

        if self.preprocess_on_infer {
            let (preprocessed, context) = input.preprocess().await?;
            self.command_tx
                .send(Command::new_preprocessed(preprocessed, context, responder))
                .await
                .map_err(|_| anyhow::Error::msg("Server channel closed"))?;
        } else {
            self.command_tx
                .send(Command::new_raw(input, responder))
                .await
                .map_err(|_| anyhow::Error::msg("Server channel closed"))?;
        }

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

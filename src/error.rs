use thiserror::Error;

/// Errors that can occur during inference operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to initialize an ONNX Runtime session.
    #[error("Session initialization failed: {0}")]
    SessionInit(String),

    /// Failed during model inference.
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Failed to batch inputs together.
    #[error("Batching failed: {0}")]
    Batching(String),

    /// Failed during input preprocessing.
    #[error("Preprocessing failed: {0}")]
    Preprocessing(String),

    /// Failed during output postprocessing.
    #[error("Postprocessing failed: {0}")]
    Postprocessing(String),

    /// Communication channel was closed unexpectedly.
    #[error("Channel closed")]
    ChannelClosed,

    /// Server was shut down.
    #[error("Server shutdown")]
    ServerShutdown,

    /// I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to download model from URL.
    #[error("Download failed: {0}")]
    Download(String),
}

impl From<ort::Error> for Error {
    fn from(err: ort::Error) -> Self {
        Error::SessionInit(err.to_string())
    }
}

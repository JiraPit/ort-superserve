//! ONNX Runtime environment creation.
//!
//! ONNX Runtime allows only one environment per process. Applications that
//! create more than one server (or drop and recreate servers) should call
//! [`create_env`] once and give a clone of the result to every server with
//! [`ServerConfig::with_environment`](crate::ServerConfig::with_environment).

use anyhow::Result;

pub use ort::environment::Environment;

/// Create the ONNX Runtime environment for this process.
///
/// Call this once (e.g. at application startup or lazily on first use) and
/// give a clone of the returned environment to every [`Server`](crate::Server)
/// via [`ServerConfig::with_environment`](crate::ServerConfig::with_environment).
/// Cloning is cheap; all clones refer to the same environment.
///
/// # Errors
///
/// Returns an error if an environment was already created in this process,
/// including one created implicitly by a server whose config has no
/// environment.
///
/// # Example
///
/// ```rust,no_run
/// use ort_superserve::{ServerConfig, create_env};
///
/// # fn main() -> anyhow::Result<()> {
/// let env = create_env()?;
/// let config = ServerConfig::new().with_environment(env.clone());
/// # Ok(())
/// # }
/// ```
pub fn create_env() -> Result<Environment> {
    ort::init().build().map_err(|e| {
        anyhow::anyhow!(
            "Failed to create ONNX Runtime environment: {e}. Only one environment is \
             allowed per process; create it once with `create_env()` and give a clone \
             to every server via `ServerConfig::with_environment`."
        )
    })
}

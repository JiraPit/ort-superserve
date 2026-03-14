use anyhow::Result;
use std::marker::PhantomData;
use tokio::sync::oneshot;

use crate::traits::{Input, Output};

/// Internal command sent from the server to the batcher.
///
/// Contains the input data and a oneshot channel for sending
/// the result back to the caller.
pub struct Command<I: Input, O: Output> {
    /// The input to be processed.
    pub input: I,
    /// Channel for sending the result back to the caller.
    pub responder: oneshot::Sender<Result<O>>,
    _phantom: PhantomData<I>,
}

impl<I: Input, O: Output> Command<I, O> {
    /// Create a new command with the given input and response channel.
    pub fn new(input: I, responder: oneshot::Sender<Result<O>>) -> Self {
        Self {
            input,
            responder,
            _phantom: PhantomData,
        }
    }
}

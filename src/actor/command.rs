use anyhow::Result;
use std::marker::PhantomData;
use tokio::sync::oneshot;

use crate::traits::{Input, Output};

/// Input type for a command, either raw or preprocessed.
///
/// When `preprocess_on_infer` is false, commands carry raw inputs
/// that will be preprocessed by the batcher using `JoinSet`.
/// When `preprocess_on_infer` is true, inputs are preprocessed
/// immediately in `infer()` before being queued.
pub enum CommandInput<I: Input> {
    Raw(I),
    Preprocessed(I::Preprocessed),
}

/// Internal command sent from the server to the batcher.
///
/// Contains the input data and a oneshot channel for sending
/// the result back to the caller.
pub struct Command<I: Input, O: Output> {
    /// The input to be processed (raw or preprocessed).
    pub input: CommandInput<I>,
    /// Channel for sending the result back to the caller.
    pub responder: oneshot::Sender<Result<O>>,
    _phantom: PhantomData<O>,
}

impl<I: Input, O: Output> Command<I, O> {
    /// Create a new command with a raw input.
    pub fn new_raw(input: I, responder: oneshot::Sender<Result<O>>) -> Self {
        Self {
            input: CommandInput::Raw(input),
            responder,
            _phantom: PhantomData,
        }
    }

    /// Create a new command with a preprocessed input.
    pub fn new_preprocessed(input: I::Preprocessed, responder: oneshot::Sender<Result<O>>) -> Self {
        Self {
            input: CommandInput::Preprocessed(input),
            responder,
            _phantom: PhantomData,
        }
    }
}

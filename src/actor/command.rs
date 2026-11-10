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
    Preprocessed(I::Preprocessed, I::Context),
}

/// Internal command sent from the server to the batcher.
///
/// Contains the input data and a oneshot channel for sending
/// the result back to the caller.
pub struct Command<I: Input, O: Output<I::Context>> {
    /// The input to be processed (raw or preprocessed).
    pub input: CommandInput<I>,
    /// Channel for sending the result back to the caller.
    pub responder: oneshot::Sender<Result<O>>,
    _phantom: PhantomData<O>,
}

impl<I: Input, O: Output<I::Context>> Command<I, O> {
    /// Create a new command with a raw input.
    pub fn new_raw(input: I, responder: oneshot::Sender<Result<O>>) -> Self {
        Self {
            input: CommandInput::Raw(input),
            responder,
            _phantom: PhantomData,
        }
    }

    /// Create a new command with a preprocessed input and its context.
    pub fn new_preprocessed(
        input: I::Preprocessed,
        context: I::Context,
        responder: oneshot::Sender<Result<O>>,
    ) -> Self {
        Self {
            input: CommandInput::Preprocessed(input, context),
            responder,
            _phantom: PhantomData,
        }
    }
}

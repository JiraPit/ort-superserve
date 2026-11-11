//! Per-request model inputs and outputs.

use crate::tensor::{Element, Tensor};
use anyhow::Result;
use ndarray::{ArrayBase, ArrayViewD, Data, Dimension};
use std::sync::Arc;

/// The model inputs of a single request, without the batch axis.
///
/// Each entry is a tensor for one named model input. The library stacks the
/// entries of all requests in a batch along a new leading batch axis, padding
/// dynamic dimensions where requests differ (see
/// [`InputSpec`](crate::InputSpec)).
///
/// A single *unnamed* tensor is bound to the default input: the one set with
/// [`ServerConfig::with_input_name`](crate::ServerConfig::with_input_name), or
/// the model's first input. Any `ndarray` array converts into such an
/// `Inputs`, which is all a single-input model needs:
///
/// ```rust
/// use ndarray::{Array1, Array3};
/// use ort_superserve::Inputs;
///
/// // Single-input model: just convert the array.
/// let image: Inputs = Array3::<f32>::zeros((3, 224, 224)).into();
///
/// // Several named inputs of different element types.
/// let tokens = Inputs::new()
///     .with("input_ids", Array1::<i64>::from(vec![101, 2023, 102]))
///     .with("attention_mask", Array1::<i64>::ones(3));
/// assert_eq!(tokens.len(), 2);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inputs {
    entries: Vec<(Option<String>, Tensor)>,
}

impl Inputs {
    /// Create an empty set of inputs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor for the model input `name`.
    pub fn with(mut self, name: impl Into<String>, tensor: impl Into<Tensor>) -> Self {
        self.insert(name, tensor);
        self
    }

    /// Add a tensor for the model input `name`.
    pub fn insert(&mut self, name: impl Into<String>, tensor: impl Into<Tensor>) {
        self.entries.push((Some(name.into()), tensor.into()));
    }

    /// Add a tensor for the default model input.
    pub fn with_default(mut self, tensor: impl Into<Tensor>) -> Self {
        self.entries.push((None, tensor.into()));
        self
    }

    /// The tensor given for the model input `name`, if any.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.entries
            .iter()
            .find(|(n, _)| n.as_deref() == Some(name))
            .map(|(_, t)| t)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(name, tensor)` pairs; the name is `None` for the
    /// default input.
    pub fn iter(&self) -> impl Iterator<Item = (Option<&str>, &Tensor)> {
        self.entries.iter().map(|(n, t)| (n.as_deref(), t))
    }

    pub(crate) fn into_entries(self) -> Vec<(Option<String>, Tensor)> {
        self.entries
    }
}

impl From<Tensor> for Inputs {
    fn from(tensor: Tensor) -> Self {
        Self::new().with_default(tensor)
    }
}

impl<T, S, D> From<ArrayBase<S, D>> for Inputs
where
    T: Element,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn from(array: ArrayBase<S, D>) -> Self {
        Self::new().with_default(Tensor::from(array))
    }
}

impl<K: Into<String>, V: Into<Tensor>> FromIterator<(K, V)> for Inputs {
    fn from_iter<It: IntoIterator<Item = (K, V)>>(iter: It) -> Self {
        let mut inputs = Self::new();
        for (name, tensor) in iter {
            inputs.insert(name, tensor);
        }
        inputs
    }
}

/// The model outputs belonging to a single request.
///
/// Every selected model output (all of them by default, or those configured
/// with [`ServerConfig::with_output`](crate::ServerConfig::with_output)) is
/// split per request according to its [`Split`](crate::Split): padded batch
/// rows and padded positions are already removed, so each tensor only holds
/// this request's data. With the default split the batch axis is removed too.
///
/// ```rust,no_run
/// # use ort_superserve::Outputs;
/// # fn f(outputs: Outputs) -> anyhow::Result<()> {
/// let logits = outputs.first::<f32>()?;           // the first selected output
/// let boxes = outputs.get::<f32>("boxes")?;       // by name
/// let labels = outputs.get::<i64>("labels")?;     // any element type
/// # Ok(()) }
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    entries: Vec<(Arc<str>, Tensor)>,
}

impl Outputs {
    pub(crate) fn from_entries(entries: Vec<(Arc<str>, Tensor)>) -> Self {
        Self { entries }
    }

    /// Borrow the output `name` as an array of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is no such output or its element type is not `T`.
    pub fn get<T: Element>(&self, name: &str) -> Result<ArrayViewD<'_, T>> {
        let tensor = self.tensor(name).ok_or_else(|| {
            anyhow::anyhow!("no output named '{name}' (outputs: {})", self.list())
        })?;
        tensor
            .view()
            .map_err(|e| anyhow::anyhow!("output '{name}': {e}"))
    }

    /// Borrow the first selected output as an array of `T`.
    ///
    /// This is the model's first output unless outputs were configured, in
    /// which case it is the first configured one.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no outputs or the element type is not `T`.
    pub fn first<T: Element>(&self) -> Result<ArrayViewD<'_, T>> {
        let (name, tensor) = self
            .entries
            .first()
            .ok_or_else(|| anyhow::anyhow!("the model produced no outputs"))?;
        tensor
            .view()
            .map_err(|e| anyhow::anyhow!("output '{name}': {e}"))
    }

    /// The tensor of output `name`, if present.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.entries
            .iter()
            .find(|(n, _)| &**n == name)
            .map(|(_, t)| t)
    }

    /// Remove and return the tensor of output `name`, if present.
    pub fn take(&mut self, name: &str) -> Option<Tensor> {
        let idx = self.entries.iter().position(|(n, _)| &**n == name)?;
        Some(self.entries.remove(idx).1)
    }

    /// Names of the outputs, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| &**n)
    }

    /// Iterate over `(name, tensor)` pairs, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.entries.iter().map(|(n, t)| (&**n, t))
    }

    /// Number of outputs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no outputs.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn list(&self) -> String {
        self.names().collect::<Vec<_>>().join(", ")
    }
}

impl IntoIterator for Outputs {
    type Item = (Arc<str>, Tensor);
    type IntoIter = std::vec::IntoIter<(Arc<str>, Tensor)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

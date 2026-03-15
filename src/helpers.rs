//! Helper functions for common operations.
//!
//! This module provides utility functions to simplify implementing the
//! [`Input`](crate::Input) and [`Output`](crate::Output) traits for common use cases.

use anyhow::Result;
use ndarray::{ArrayD, Axis, Dimension};

/// Batches multiple arrays along the first dimension.
///
/// This function takes a slice of arrays and stacks them along a new dimension
/// (axis 0), producing a single batched array with shape `[n, ...]` where `n`
/// is the number of input arrays.
///
/// # Example
///
/// ```rust
/// use ndarray::Array3;
/// use ort_superserve::helpers::batch_array;
///
/// let items = vec![
///     Array3::<f32>::zeros((3, 224, 224)),
///     Array3::<f32>::ones((3, 224, 224)),
/// ];
///
/// let batched = batch_array(&items)?;
/// assert_eq!(batched.shape(), &[2, 3, 224, 224]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn batch_array<A, D>(items: &[ndarray::ArrayBase<A, D>]) -> Result<ArrayD<f32>>
where
    A: ndarray::Data<Elem = f32>,
    D: Dimension,
{
    if items.is_empty() {
        anyhow::bail!("cannot batch empty array list");
    }

    let views: Vec<_> = items.iter().map(|a| a.view()).collect();
    let batched = ndarray::stack(Axis(0), &views)?;
    Ok(batched.into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    #[test]
    fn test_batch_array_single() {
        let items = vec![Array3::<f32>::zeros((3, 224, 224))];
        let batched = batch_array(&items).unwrap();
        assert_eq!(batched.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_batch_array_multiple() {
        let items = vec![
            Array3::<f32>::zeros((3, 224, 224)),
            Array3::<f32>::ones((3, 224, 224)),
        ];
        let batched = batch_array(&items).unwrap();
        assert_eq!(batched.shape(), &[2, 3, 224, 224]);
    }

    #[test]
    fn test_batch_array_empty() {
        let items: Vec<Array3<f32>> = vec![];
        let result = batch_array(&items);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_array_1d() {
        let items = vec![
            Array1::<f32>::from_vec(vec![1.0, 2.0]),
            Array1::<f32>::from_vec(vec![3.0, 4.0]),
        ];
        let batched = batch_array(&items).unwrap();
        assert_eq!(batched.shape(), &[2, 2]);
        assert_eq!(batched[[0, 0]], 1.0);
        assert_eq!(batched[[1, 1]], 4.0);
    }

    #[test]
    fn test_batch_array_2d() {
        let items = vec![Array2::<f32>::zeros((4, 5)), Array2::<f32>::ones((4, 5))];
        let batched = batch_array(&items).unwrap();
        assert_eq!(batched.shape(), &[2, 4, 5]);
    }
}

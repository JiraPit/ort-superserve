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
    use ndarray::{Array1, Array2, array};

    #[test]
    fn stacks_items_along_new_leading_axis_in_order() {
        let items = vec![
            Array2::<f32>::from_elem((2, 3), 1.0),
            Array2::<f32>::from_elem((2, 3), 2.0),
        ];
        let batched = batch_array(&items).unwrap();
        assert_eq!(batched.shape(), &[2, 2, 3]);
        assert_eq!(batched[[0, 1, 2]], 1.0);
        assert_eq!(batched[[1, 0, 0]], 2.0);
    }

    #[test]
    fn rejects_empty_input() {
        let items: Vec<Array1<f32>> = vec![];
        assert!(batch_array(&items).is_err());
    }

    #[test]
    fn rejects_items_with_different_shapes() {
        let items = vec![array![1.0f32, 2.0], array![1.0f32, 2.0, 3.0]];
        assert!(batch_array(&items).is_err());
    }
}

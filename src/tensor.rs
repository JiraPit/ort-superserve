//! Typed tensors exchanged between your code and the model.
//!
//! A [`Tensor`] is an owned n-dimensional array of one of the element types
//! ONNX Runtime models commonly use. Preprocessing produces tensors (usually
//! via `From<ndarray::Array…>`), and postprocessing reads them back with
//! [`Tensor::view`] or [`Tensor::into_array`].

use anyhow::Result;
use ndarray::{ArrayBase, ArrayD, ArrayViewD, Data, Dimension};
use std::fmt;

/// Element type of a [`Tensor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `i8`
    I8,
    /// `i16`
    I16,
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `u8`
    U8,
    /// `u16`
    U16,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `bool`
    Bool,
}

impl ElementType {
    /// Convert an ONNX Runtime element type, if it is supported.
    pub(crate) fn from_ort(ty: ort::value::TensorElementType) -> Option<Self> {
        use ort::value::TensorElementType as T;
        Some(match ty {
            T::Float32 => Self::F32,
            T::Float64 => Self::F64,
            T::Int8 => Self::I8,
            T::Int16 => Self::I16,
            T::Int32 => Self::I32,
            T::Int64 => Self::I64,
            T::Uint8 => Self::U8,
            T::Uint16 => Self::U16,
            T::Uint32 => Self::U32,
            T::Uint64 => Self::U64,
            T::Bool => Self::Bool,
            _ => return None,
        })
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        })
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that can be the element of a [`Tensor`].
///
/// Implemented for `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
/// `u32`, `u64` and `bool`. This trait is sealed.
pub trait Element:
    sealed::Sealed
    + ort::value::PrimitiveTensorElementType
    + Copy
    + PartialEq
    + fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// The [`ElementType`] of this Rust type.
    const TYPE: ElementType;

    #[doc(hidden)]
    fn wrap(array: ArrayD<Self>) -> Tensor;
    #[doc(hidden)]
    fn as_array(tensor: &Tensor) -> Option<&ArrayD<Self>>;
    #[doc(hidden)]
    fn into_array(tensor: Tensor) -> std::result::Result<ArrayD<Self>, Tensor>;
    #[doc(hidden)]
    fn from_f64(value: f64) -> Self;
    #[doc(hidden)]
    fn to_index(self) -> Option<i64>;
}

macro_rules! impl_element {
    ($ty:ty, $variant:ident, |$v:ident| $from:expr, |$s:ident| $index:expr) => {
        impl sealed::Sealed for $ty {}
        impl Element for $ty {
            const TYPE: ElementType = ElementType::$variant;
            fn wrap(array: ArrayD<Self>) -> Tensor {
                Tensor::$variant(array)
            }
            fn as_array(tensor: &Tensor) -> Option<&ArrayD<Self>> {
                match tensor {
                    Tensor::$variant(a) => Some(a),
                    _ => None,
                }
            }
            fn into_array(tensor: Tensor) -> std::result::Result<ArrayD<Self>, Tensor> {
                match tensor {
                    Tensor::$variant(a) => Ok(a),
                    other => Err(other),
                }
            }
            fn from_f64($v: f64) -> Self {
                $from
            }
            fn to_index(self) -> Option<i64> {
                let $s = self;
                $index
            }
        }
    };
}

macro_rules! impl_int_element {
    ($ty:ty, $variant:ident) => {
        impl_element!($ty, $variant, |v| v as $ty, |s| i64::try_from(s).ok());
    };
}

macro_rules! impl_float_element {
    ($ty:ty, $variant:ident) => {
        impl_element!($ty, $variant, |v| v as $ty, |s| {
            (s.fract() == 0.0 && s.is_finite()).then_some(s as i64)
        });
    };
}

impl_float_element!(f32, F32);
impl_float_element!(f64, F64);
impl_int_element!(i8, I8);
impl_int_element!(i16, I16);
impl_int_element!(i32, I32);
impl_int_element!(i64, I64);
impl_int_element!(u8, U8);
impl_int_element!(u16, U16);
impl_int_element!(u32, U32);
impl_int_element!(u64, U64);
impl_element!(bool, Bool, |v| v != 0.0, |_s| None);

/// Run `$body` with `$T` bound to the Rust type of the given [`ElementType`].
macro_rules! with_element_type {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::tensor::ElementType::F32 => {
                type $T = f32;
                $body
            }
            $crate::tensor::ElementType::F64 => {
                type $T = f64;
                $body
            }
            $crate::tensor::ElementType::I8 => {
                type $T = i8;
                $body
            }
            $crate::tensor::ElementType::I16 => {
                type $T = i16;
                $body
            }
            $crate::tensor::ElementType::I32 => {
                type $T = i32;
                $body
            }
            $crate::tensor::ElementType::I64 => {
                type $T = i64;
                $body
            }
            $crate::tensor::ElementType::U8 => {
                type $T = u8;
                $body
            }
            $crate::tensor::ElementType::U16 => {
                type $T = u16;
                $body
            }
            $crate::tensor::ElementType::U32 => {
                type $T = u32;
                $body
            }
            $crate::tensor::ElementType::U64 => {
                type $T = u64;
                $body
            }
            $crate::tensor::ElementType::Bool => {
                type $T = bool;
                $body
            }
        }
    };
}
pub(crate) use with_element_type;

/// Run `$body` with `$a` bound to the typed array inside a tensor, re-wrapping
/// the resulting array in the same variant.
macro_rules! map_tensor {
    ($tensor:expr, $a:ident => $body:expr) => {
        match $tensor {
            Tensor::F32($a) => Tensor::F32($body),
            Tensor::F64($a) => Tensor::F64($body),
            Tensor::I8($a) => Tensor::I8($body),
            Tensor::I16($a) => Tensor::I16($body),
            Tensor::I32($a) => Tensor::I32($body),
            Tensor::I64($a) => Tensor::I64($body),
            Tensor::U8($a) => Tensor::U8($body),
            Tensor::U16($a) => Tensor::U16($body),
            Tensor::U32($a) => Tensor::U32($body),
            Tensor::U64($a) => Tensor::U64($body),
            Tensor::Bool($a) => Tensor::Bool($body),
        }
    };
}
pub(crate) use map_tensor;

/// Run `$body` with `$a` bound to the typed array inside a tensor.
macro_rules! visit_tensor {
    ($tensor:expr, $a:ident => $body:expr) => {
        match $tensor {
            Tensor::F32($a) => $body,
            Tensor::F64($a) => $body,
            Tensor::I8($a) => $body,
            Tensor::I16($a) => $body,
            Tensor::I32($a) => $body,
            Tensor::I64($a) => $body,
            Tensor::U8($a) => $body,
            Tensor::U16($a) => $body,
            Tensor::U32($a) => $body,
            Tensor::U64($a) => $body,
            Tensor::Bool($a) => $body,
        }
    };
}
pub(crate) use visit_tensor;

/// An owned, dynamically shaped tensor of any supported element type.
///
/// Build one from any `ndarray` array with `Tensor::from(array)` (or `.into()`),
/// and read it back with [`view`](Tensor::view) or
/// [`into_array`](Tensor::into_array):
///
/// ```rust
/// use ndarray::array;
/// use ort_superserve::{ElementType, Tensor};
///
/// let labels = Tensor::from(array![3i64, 7, 1]);
/// assert_eq!(labels.dtype(), ElementType::I64);
/// assert_eq!(labels.shape(), &[3]);
/// assert_eq!(labels.view::<i64>()?[1], 7);
/// assert!(labels.view::<f32>().is_err());
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    /// `f32` elements.
    F32(ArrayD<f32>),
    /// `f64` elements.
    F64(ArrayD<f64>),
    /// `i8` elements.
    I8(ArrayD<i8>),
    /// `i16` elements.
    I16(ArrayD<i16>),
    /// `i32` elements.
    I32(ArrayD<i32>),
    /// `i64` elements.
    I64(ArrayD<i64>),
    /// `u8` elements.
    U8(ArrayD<u8>),
    /// `u16` elements.
    U16(ArrayD<u16>),
    /// `u32` elements.
    U32(ArrayD<u32>),
    /// `u64` elements.
    U64(ArrayD<u64>),
    /// `bool` elements.
    Bool(ArrayD<bool>),
}

impl Tensor {
    /// The element type of this tensor.
    pub fn dtype(&self) -> ElementType {
        visit_tensor!(self, a => element_type_of(a))
    }

    /// The shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        visit_tensor!(self, a => a.shape())
    }

    /// The number of axes of this tensor.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Borrow the data as an array of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor's element type is not `T`.
    pub fn view<T: Element>(&self) -> Result<ArrayViewD<'_, T>> {
        T::as_array(self).map(|a| a.view()).ok_or_else(|| {
            anyhow::anyhow!(
                "tensor has element type {}, not the requested {}",
                self.dtype(),
                T::TYPE
            )
        })
    }

    /// Take the data as an array of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor's element type is not `T`.
    pub fn into_array<T: Element>(self) -> Result<ArrayD<T>> {
        T::into_array(self).map_err(|t| {
            anyhow::anyhow!(
                "tensor has element type {}, not the requested {}",
                t.dtype(),
                T::TYPE
            )
        })
    }
}

fn element_type_of<T: Element>(_: &ArrayD<T>) -> ElementType {
    T::TYPE
}

impl<T, S, D> From<ArrayBase<S, D>> for Tensor
where
    T: Element,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn from(array: ArrayBase<S, D>) -> Self {
        T::wrap(array.into_owned().into_dyn())
    }
}

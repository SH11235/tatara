mod sealed {
    pub trait Sealed {}
}

/// Numeric scalars with no padding and a valid value for every bit pattern.
/// Sealing excludes references, constrained integers, and types with unsupported alignment.
///
/// ```compile_fail
/// use cuda_native_runtime::DeviceElement;
/// fn require_element<T: DeviceElement>() {}
/// require_element::<std::num::NonZeroU32>();
/// ```
///
/// ```compile_fail
/// use cuda_native_runtime::DeviceElement;
/// fn require_element<T: DeviceElement>() {}
/// require_element::<&'static u32>();
/// ```
///
/// ```compile_fail
/// use cuda_native_runtime::DeviceElement;
/// #[derive(Clone, Copy)]
/// struct Padded { byte: u8, word: u32 }
/// impl DeviceElement for Padded {}
/// ```
pub trait DeviceElement: Copy + sealed::Sealed {}

macro_rules! impl_elements {
    ($($ty:ty),+ $(,)?) => {
        $(impl sealed::Sealed for $ty {}
        impl DeviceElement for $ty {})+
    };
}

impl_elements!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f16, f32, f64
);

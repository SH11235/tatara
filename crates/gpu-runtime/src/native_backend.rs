use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};

use cuda_native_runtime as native;

use crate::{Error, Result};

const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn for_num_elems(n: u32) -> Self {
        Self {
            grid_dim: (n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

pub struct CudaContext {
    inner: native::Context,
}

impl CudaContext {
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ordinal = i32::try_from(ordinal)
            .map_err(|_| Error::KernelArtifact("CUDA device ordinal exceeds i32".into()))?;
        Ok(Arc::new(Self {
            inner: native::Context::new(ordinal)?,
        }))
    }

    pub fn bind_to_thread(&self) -> Result<()> {
        Ok(self.inner.set_current()?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>> {
        Ok(Arc::new(CudaStream {
            inner: self.inner.create_stream()?,
            context: Arc::clone(self),
        }))
    }

    pub fn new_event(self: &Arc<Self>, _flags: Option<u32>) -> Result<CudaEvent> {
        Ok(CudaEvent {
            inner: self.inner.create_event()?,
            context: Arc::clone(self),
        })
    }

    pub fn device_name(&self) -> Result<String> {
        Ok(self.inner.device_name()?)
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((
            self.inner
                .device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            self.inner
                .device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    pub fn occupancy_attributes(&self) -> Result<(i32, i32)> {
        Ok((
            self.inner
                .device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            self.inner
                .device_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        ))
    }

    pub fn load_module_from_image(self: &Arc<Self>, image: &[u8]) -> Result<Arc<CudaModule>> {
        Ok(Arc::new(CudaModule {
            inner: self.inner.load_module(image)?,
            context: Arc::clone(self),
            functions: Mutex::new(HashMap::new()),
        }))
    }
}

pub struct CudaStream {
    inner: native::Stream,
    context: Arc<CudaContext>,
}

impl CudaStream {
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    pub fn wait(&self, event: &CudaEvent) -> Result<()> {
        Ok(self.inner.wait_event(&event.inner)?)
    }

    pub fn cu_stream(&self) -> *mut c_void {
        self.inner.raw_handle()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

pub struct CudaEvent {
    inner: native::Event,
    #[allow(dead_code)]
    context: Arc<CudaContext>,
}

impl CudaEvent {
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        Ok(self.inner.record(&stream.inner)?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }
}

pub struct CudaModule {
    inner: native::Module,
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    functions: Mutex<HashMap<&'static str, native::Function>>,
}

impl CudaModule {
    /// # Safety
    ///
    /// `args` must encode the exact pointer/length/scalar ABI of `kernel`.
    pub unsafe fn launch(
        &self,
        kernel: &'static str,
        stream: &CudaStream,
        config: LaunchConfig,
        args: &mut KernelArgs,
    ) -> Result<()> {
        let function = {
            let mut functions = self
                .functions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(function) = functions.get(kernel) {
                function.clone()
            } else {
                let name = std::ffi::CString::new(kernel).map_err(|_| {
                    Error::KernelArtifact(format!("invalid CUDA kernel name `{kernel}`"))
                })?;
                let function = self.inner.function(&name)?;
                functions.insert(kernel, function.clone());
                function
            }
        };
        // SAFETY: caller guarantees the encoded ABI and device allocation bounds.
        unsafe {
            function.launch(
                &stream.inner,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                args.pointers_mut(),
            )?;
        }
        Ok(())
    }
}

pub struct DeviceBuffer<T: Copy> {
    inner: native::DeviceBuffer<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn from_host(stream: &CudaStream, values: &[T]) -> Result<Self> {
        let inner = native::DeviceBuffer::from_slice(&stream.context.inner, values)?;
        Ok(Self { inner })
    }

    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self> {
        let inner = native::DeviceBuffer::zeroed(&stream.context.inner, len)?;
        Ok(Self { inner })
    }

    pub fn cu_deviceptr(&self) -> u64 {
        self.inner.device_ptr()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    pub fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<T>>
    where
        T: Default,
    {
        stream.synchronize()?;
        let mut values = vec![T::default(); self.len()];
        self.inner.copy_to(&mut values)?;
        Ok(values)
    }

    pub(crate) fn fill_byte_async(&self, value: u8, stream: &CudaStream) -> Result<()> {
        Ok(self.inner.fill_byte_async(value, &stream.inner)?)
    }

    pub(crate) unsafe fn copy_from_host_async(
        &self,
        stream: &CudaStream,
        values: &[T],
    ) -> Result<()> {
        // SAFETY: caller guarantees the source lifetime through stream completion.
        unsafe { self.inner.copy_from_async(values, &stream.inner)? };
        Ok(())
    }

    pub(crate) unsafe fn copy_to_host_async(
        &self,
        stream: &CudaStream,
        values: &mut [T],
    ) -> Result<()> {
        // SAFETY: caller guarantees exclusive destination access through stream completion.
        unsafe { self.inner.copy_to_async(values, &stream.inner)? };
        Ok(())
    }
}

trait KernelArgStorage {}
impl<T> KernelArgStorage for T {}

#[derive(Default)]
pub struct KernelArgs {
    storage: Vec<Box<dyn KernelArgStorage>>,
    pointers: Vec<*mut c_void>,
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_scalar<T: Copy + 'static>(&mut self, value: T) {
        let mut storage = Box::new(value);
        self.pointers
            .push((&mut *storage as *mut T).cast::<c_void>());
        self.storage.push(storage);
    }

    pub fn push_slice<T: Copy>(&mut self, buffer: &DeviceBuffer<T>) {
        self.push_scalar(buffer.cu_deviceptr());
        self.push_scalar(buffer.len() as u64);
    }

    fn pointers_mut(&mut self) -> &mut [*mut c_void] {
        &mut self.pointers
    }
}

pub unsafe fn alloc_pinned_host(bytes: usize) -> Result<*mut c_void> {
    // SAFETY: ownership of the allocation is transferred to the caller.
    Ok(unsafe { native::alloc_pinned_host(bytes)? })
}

pub unsafe fn free_pinned_host(raw: *mut c_void) -> Result<()> {
    // SAFETY: caller guarantees raw is live and has no in-flight operations.
    unsafe { native::free_pinned_host(raw)? };
    Ok(())
}

pub fn is_out_of_memory(err: &(dyn std::error::Error + 'static)) -> bool {
    err.downcast_ref::<native::NativeCudaError>()
        .is_some_and(|error| error.to_string().contains("CUDA_ERROR_OUT_OF_MEMORY"))
        || err
            .downcast_ref::<Error>()
            .is_some_and(|error| error.to_string().contains("CUDA_ERROR_OUT_OF_MEMORY"))
}

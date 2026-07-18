use std::{ffi::c_void, marker::PhantomData, mem::size_of, ptr, sync::Arc};

type CuDevice = i32;
type CuDevicePtr = u64;
type CuResult = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
const CU_STREAM_NON_BLOCKING: u32 = 1;

#[cfg_attr(target_os = "windows", link(name = "cuda"))]
#[cfg_attr(not(target_os = "windows"), link(name = "cuda"))]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: i32) -> CuResult;
    fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: CuDevice) -> CuResult;
    fn cuDeviceGetName(name: *mut i8, len: i32, device: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease_v2(device: CuDevice) -> CuResult;
    fn cuCtxGetCurrent(context: *mut CuContext) -> CuResult;
    fn cuCtxSetCurrent(context: CuContext) -> CuResult;
    fn cuCtxSynchronize() -> CuResult;
    fn cuStreamCreate(stream: *mut CuStream, flags: u32) -> CuResult;
    fn cuStreamDestroy_v2(stream: CuStream) -> CuResult;
    fn cuStreamSynchronize(stream: CuStream) -> CuResult;
    fn cuStreamWaitEvent(stream: CuStream, event: CuEvent, flags: u32) -> CuResult;
    fn cuEventCreate(event: *mut CuEvent, flags: u32) -> CuResult;
    fn cuEventDestroy_v2(event: CuEvent) -> CuResult;
    fn cuEventRecord(event: CuEvent, stream: CuStream) -> CuResult;
    fn cuEventSynchronize(event: CuEvent) -> CuResult;
    fn cuMemAlloc_v2(device_ptr: *mut CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(device_ptr: CuDevicePtr) -> CuResult;
    fn cuMemHostAlloc(host_ptr: *mut *mut c_void, bytes: usize, flags: u32) -> CuResult;
    fn cuMemFreeHost(host_ptr: *mut c_void) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: CuDevicePtr, src: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemcpyHtoDAsync_v2(
        dst: CuDevicePtr,
        src: *const c_void,
        bytes: usize,
        stream: CuStream,
    ) -> CuResult;
    fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CuDevicePtr,
        bytes: usize,
        stream: CuStream,
    ) -> CuResult;
    fn cuMemsetD8_v2(dst: CuDevicePtr, value: u8, bytes: usize) -> CuResult;
    fn cuMemsetD8Async(dst: CuDevicePtr, value: u8, bytes: usize, stream: CuStream) -> CuResult;
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> CuResult;
    fn cuModuleUnload(module: CuModule) -> CuResult;
    fn cuModuleGetFunction(
        function: *mut CuFunction,
        module: CuModule,
        name: *const i8,
    ) -> CuResult;
    fn cuLaunchKernel(
        function: CuFunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: CuStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
    fn cuGetErrorName(error: CuResult, name: *mut *const i8) -> CuResult;
    fn cuGetErrorString(error: CuResult, message: *mut *const i8) -> CuResult;
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA Driver API error {code}: {name}: {message}")]
pub struct NativeCudaError {
    code: CuResult,
    name: String,
    message: String,
}

pub type Result<T> = std::result::Result<T, NativeCudaError>;

/// Allocates page-locked host memory that may be used by any CUDA context.
///
/// # Safety
///
/// The returned pointer must be released exactly once with [`free_pinned_host`].
pub unsafe fn alloc_pinned_host(bytes: usize) -> Result<*mut c_void> {
    assert!(
        bytes > 0,
        "zero-length pinned allocations are not supported"
    );
    let mut raw = ptr::null_mut();
    // SAFETY: raw points to writable output storage and bytes is non-zero.
    unsafe { check(cuMemHostAlloc(&mut raw, bytes, CU_MEMHOSTALLOC_PORTABLE))? };
    Ok(raw)
}

/// Releases memory returned by [`alloc_pinned_host`].
///
/// # Safety
///
/// `raw` must be a live allocation returned by [`alloc_pinned_host`] with no in-flight transfer.
pub unsafe fn free_pinned_host(raw: *mut c_void) -> Result<()> {
    // SAFETY: caller owns the allocation and guarantees all transfers completed.
    unsafe { check(cuMemFreeHost(raw)) }
}

fn check(code: CuResult) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }

    let mut name = ptr::null();
    let mut message = ptr::null();
    // SAFETY: CUDA writes process-lifetime static C string pointers on success.
    unsafe {
        let _ = cuGetErrorName(code, &mut name);
        let _ = cuGetErrorString(code, &mut message);
    }
    let read = |value: *const i8| {
        if value.is_null() {
            "unknown".to_string()
        } else {
            // SAFETY: non-null values returned by CUDA are NUL-terminated static strings.
            unsafe { std::ffi::CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }
    };
    Err(NativeCudaError {
        code,
        name: read(name),
        message: read(message),
    })
}

struct ContextInner {
    device: CuDevice,
    raw: CuContext,
}

// SAFETY: CUDA Driver API contexts are process resources. The current context is selected per
// calling thread before every operation, and the final release is serialized by Arc ownership.
unsafe impl Send for ContextInner {}
// SAFETY: CUDA Driver API entry points are thread-safe and context selection is thread-local.
unsafe impl Sync for ContextInner {}

pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    pub fn new(ordinal: i32) -> Result<Self> {
        let mut device = 0;
        let mut raw = ptr::null_mut();
        // SAFETY: output pointers refer to initialized local storage and ordinal is caller supplied.
        unsafe {
            check(cuInit(0))?;
            check(cuDeviceGet(&mut device, ordinal))?;
            check(cuDevicePrimaryCtxRetain(&mut raw, device))?;
            let set_current = cuCtxSetCurrent(raw);
            if set_current != CUDA_SUCCESS {
                // Retain succeeded, so balance it before propagating the context-bind error.
                let _ = cuDevicePrimaryCtxRelease_v2(device);
                check(set_current)?;
            }
        }
        Ok(Self {
            inner: Arc::new(ContextInner { device, raw }),
        })
    }

    pub fn set_current(&self) -> Result<()> {
        // SAFETY: raw is retained for self's lifetime.
        unsafe { check(cuCtxSetCurrent(self.inner.raw)) }
    }

    pub fn device_attribute(&self, attribute: i32) -> Result<i32> {
        self.set_current()?;
        let mut value = 0;
        // SAFETY: value points to writable local storage and device is retained by self.
        unsafe {
            check(cuDeviceGetAttribute(
                &mut value,
                attribute,
                self.inner.device,
            ))?
        };
        Ok(value)
    }

    pub fn device_name(&self) -> Result<String> {
        self.set_current()?;
        let mut name = [0_i8; 256];
        // SAFETY: name is writable for the advertised length and device is retained by self.
        unsafe {
            check(cuDeviceGetName(
                name.as_mut_ptr(),
                name.len() as i32,
                self.inner.device,
            ))?
        };
        // SAFETY: CUDA guarantees a NUL-terminated device name within the supplied buffer.
        Ok(unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        // SAFETY: a current retained CUDA context exists.
        unsafe { check(cuCtxSynchronize()) }
    }

    pub fn create_stream(&self) -> Result<Stream> {
        self.set_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: raw points to local output storage and the current context is valid.
        unsafe { check(cuStreamCreate(&mut raw, CU_STREAM_NON_BLOCKING))? };
        Ok(Stream {
            raw,
            context: Arc::clone(&self.inner),
        })
    }

    pub fn create_event(&self) -> Result<Event> {
        self.set_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: raw points to local output storage and the current context is valid.
        unsafe { check(cuEventCreate(&mut raw, 0))? };
        Ok(Event {
            raw,
            context: Arc::clone(&self.inner),
        })
    }

    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        self.set_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: image is a complete fat binary and remains live for the duration of the call.
        unsafe { check(cuModuleLoadData(&mut raw, image.as_ptr().cast()))? };
        Ok(Module {
            inner: Arc::new(ModuleInner {
                raw,
                context: Arc::clone(&self.inner),
            }),
        })
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: raw remains retained until this last shared owner is dropped. If this context is
        // current on the dropping thread, clear it before releasing the final primary-context
        // retain so the thread does not keep a dangling current-context handle.
        unsafe {
            let mut current = ptr::null_mut();
            if cuCtxGetCurrent(&mut current) == CUDA_SUCCESS && current == self.raw {
                let _ = cuCtxSetCurrent(ptr::null_mut());
            }
            let _ = cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

pub struct Stream {
    raw: CuStream,
    context: Arc<ContextInner>,
}

// SAFETY: CUDA stream handles may be submitted from any thread after selecting their context;
// every operation in this wrapper selects the retained context before using the handle.
unsafe impl Send for Stream {}
// SAFETY: CUDA serializes work submitted to one stream, and wrapper methods expose no host alias.
unsafe impl Sync for Stream {}

impl Stream {
    pub fn raw_handle(&self) -> *mut c_void {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        set_current(&self.context)?;
        // SAFETY: raw is a live stream owned by self.
        unsafe { check(cuStreamSynchronize(self.raw)) }
    }

    pub fn wait_event(&self, event: &Event) -> Result<()> {
        ensure_same_context(&self.context, &event.context);
        set_current(&self.context)?;
        // SAFETY: both handles are live and belong to the selected context.
        unsafe { check(cuStreamWaitEvent(self.raw, event.raw, 0)) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the retained context owns raw and stays alive through this call.
        unsafe {
            let _ = cuCtxSetCurrent(self.context.raw);
            let _ = cuStreamDestroy_v2(self.raw);
        }
    }
}

pub struct Event {
    raw: CuEvent,
    context: Arc<ContextInner>,
}

impl Event {
    pub fn record(&self, stream: &Stream) -> Result<()> {
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: both handles are live and belong to the selected context.
        unsafe { check(cuEventRecord(self.raw, stream.raw)) }
    }

    pub fn synchronize(&self) -> Result<()> {
        set_current(&self.context)?;
        // SAFETY: raw is a live event owned by self.
        unsafe { check(cuEventSynchronize(self.raw)) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: the retained context owns raw and stays alive through this call.
        unsafe {
            let _ = cuCtxSetCurrent(self.context.raw);
            let _ = cuEventDestroy_v2(self.raw);
        }
    }
}

struct ModuleInner {
    raw: CuModule,
    context: Arc<ContextInner>,
}

// SAFETY: CUDA modules are immutable after loading and stay alive through the retained context.
unsafe impl Send for ModuleInner {}
// SAFETY: CUDA Driver API permits concurrent function lookup and launch from a loaded module.
unsafe impl Sync for ModuleInner {}

pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    pub fn function(&self, name: &std::ffi::CStr) -> Result<Function> {
        let mut raw = ptr::null_mut();
        set_current(&self.inner.context)?;
        // SAFETY: module is live and name is NUL-terminated.
        unsafe { check(cuModuleGetFunction(&mut raw, self.inner.raw, name.as_ptr()))? };
        Ok(Function {
            raw,
            module: Arc::clone(&self.inner),
        })
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // SAFETY: the retained context owns raw and stays alive through this call.
        unsafe {
            let _ = cuCtxSetCurrent(self.context.raw);
            let _ = cuModuleUnload(self.raw);
        }
    }
}

pub struct Function {
    raw: CuFunction,
    module: Arc<ModuleInner>,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw,
            module: Arc::clone(&self.module),
        }
    }
}

// SAFETY: CUfunction is an immutable module-owned handle and the retained module/context outlives
// every clone. Launch selects the correct context for the calling thread before using the handle.
unsafe impl Send for Function {}
// SAFETY: CUDA permits concurrent launches of one function handle on streams in its context.
unsafe impl Sync for Function {}

impl Function {
    /// # Safety
    ///
    /// Each element of `args` must point to storage containing one kernel argument whose type,
    /// order, and lifetime match the CUDA C++ kernel signature. Device pointers must reference
    /// allocations large enough for every index accessed by the launch configuration.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        ensure_same_context(&self.module.context, &stream.context);
        set_current(&self.module.context)?;
        // SAFETY: caller owns the kernel ABI and allocation invariants described above.
        unsafe {
            check(cuLaunchKernel(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream.raw,
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }
}

pub struct DeviceBuffer<T> {
    raw: CuDevicePtr,
    len: usize,
    context: Arc<ContextInner>,
    marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn uninitialized(context: &Context, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                raw: 0,
                len: 0,
                context: Arc::clone(&context.inner),
                marker: PhantomData,
            });
        }
        context.set_current()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("CUDA allocation byte size overflowed");
        let mut raw = 0;
        // SAFETY: raw points to local output storage and bytes is non-zero.
        unsafe { check(cuMemAlloc_v2(&mut raw, bytes))? };
        Ok(Self {
            raw,
            len,
            context: Arc::clone(&context.inner),
            marker: PhantomData,
        })
    }

    pub fn zeroed(context: &Context, len: usize) -> Result<Self> {
        let buffer = Self::uninitialized(context, len)?;
        if len == 0 {
            return Ok(buffer);
        }
        // SAFETY: allocation is live and its exact byte length was checked in uninitialized.
        unsafe { check(cuMemsetD8_v2(buffer.raw, 0, len * size_of::<T>()))? };
        Ok(buffer)
    }

    pub fn from_slice(context: &Context, values: &[T]) -> Result<Self> {
        let buffer = Self::uninitialized(context, values.len())?;
        buffer.copy_from(values)?;
        Ok(buffer)
    }

    pub fn copy_from(&self, values: &[T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "host and device lengths differ");
        if self.len == 0 {
            return Ok(());
        }
        set_current(&self.context)?;
        // SAFETY: both buffers are valid for len * size_of::<T>() bytes.
        unsafe {
            check(cuMemcpyHtoD_v2(
                self.raw,
                values.as_ptr().cast(),
                self.len * size_of::<T>(),
            ))
        }
    }

    /// Enqueues a host-to-device transfer from page-locked memory.
    ///
    /// # Safety
    ///
    /// The caller must keep `values` alive and must not mutate it until the transfer has completed
    /// on `stream` or an event recorded after this operation has completed.
    pub unsafe fn copy_from_pinned_async(
        &self,
        values: &PinnedBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(values.len, self.len, "host and device lengths differ");
        if self.len == 0 {
            return Ok(());
        }
        ensure_same_context(&self.context, &values.context);
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the caller guarantees that the source remains live and immutable until complete.
        unsafe {
            check(cuMemcpyHtoDAsync_v2(
                self.raw,
                values.raw.cast_const(),
                self.len * size_of::<T>(),
                stream.raw,
            ))
        }
    }

    pub fn copy_to(&self, values: &mut [T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "host and device lengths differ");
        if self.len == 0 {
            return Ok(());
        }
        set_current(&self.context)?;
        // SAFETY: both buffers are valid for len * size_of::<T>() bytes.
        unsafe {
            check(cuMemcpyDtoH_v2(
                values.as_mut_ptr().cast(),
                self.raw,
                self.len * size_of::<T>(),
            ))
        }
    }

    /// Enqueues a device-to-host transfer into page-locked memory.
    ///
    /// # Safety
    ///
    /// The caller must keep `values` alive and must not access it until the transfer has completed
    /// on `stream` or an event recorded after this operation has completed.
    pub unsafe fn copy_to_pinned_async(
        &self,
        values: &mut PinnedBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(values.len, self.len, "host and device lengths differ");
        if self.len == 0 {
            return Ok(());
        }
        ensure_same_context(&self.context, &values.context);
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the caller guarantees exclusive destination access until the transfer completes.
        unsafe {
            check(cuMemcpyDtoHAsync_v2(
                values.raw,
                self.raw,
                self.len * size_of::<T>(),
                stream.raw,
            ))
        }
    }

    pub fn zero_async(&self, stream: &Stream) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the allocation is live and stream belongs to the selected context.
        unsafe {
            check(cuMemsetD8Async(
                self.raw,
                0,
                self.len * size_of::<T>(),
                stream.raw,
            ))
        }
    }

    pub fn device_ptr(&self) -> u64 {
        self.raw
    }

    /// Copies at most the allocation length from host memory without synchronizing.
    ///
    /// # Safety
    ///
    /// `values` must remain alive and immutable until the operation completes on `stream`.
    pub unsafe fn copy_from_async(&self, values: &[T], stream: &Stream) -> Result<()> {
        assert!(
            values.len() <= self.len,
            "host slice exceeds device allocation"
        );
        if values.is_empty() {
            return Ok(());
        }
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the capacity assertion bounds the copy and caller owns the source lifetime.
        unsafe {
            check(cuMemcpyHtoDAsync_v2(
                self.raw,
                values.as_ptr().cast(),
                std::mem::size_of_val(values),
                stream.raw,
            ))
        }
    }

    /// Copies at most the allocation length into host memory without synchronizing.
    ///
    /// # Safety
    ///
    /// `values` must remain alive and inaccessible until the operation completes on `stream`.
    pub unsafe fn copy_to_async(&self, values: &mut [T], stream: &Stream) -> Result<()> {
        assert!(
            values.len() <= self.len,
            "host slice exceeds device allocation"
        );
        if values.is_empty() {
            return Ok(());
        }
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the capacity assertion bounds the copy and caller owns destination access.
        unsafe {
            check(cuMemcpyDtoHAsync_v2(
                values.as_mut_ptr().cast(),
                self.raw,
                std::mem::size_of_val(values),
                stream.raw,
            ))
        }
    }

    pub fn fill_byte_async(&self, value: u8, stream: &Stream) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        ensure_same_context(&self.context, &stream.context);
        set_current(&self.context)?;
        // SAFETY: the allocation is live and the exact byte extent is used.
        unsafe {
            check(cuMemsetD8Async(
                self.raw,
                value,
                self.len * size_of::<T>(),
                stream.raw,
            ))
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.raw == 0 {
            return;
        }
        // SAFETY: the retained context owns raw and stays alive through this call.
        unsafe {
            let _ = cuCtxSetCurrent(self.context.raw);
            let _ = cuMemFree_v2(self.raw);
        }
    }
}

pub struct PinnedBuffer<T: Copy> {
    raw: *mut c_void,
    len: usize,
    context: Arc<ContextInner>,
    marker: PhantomData<T>,
}

impl<T: Copy + Default> PinnedBuffer<T> {
    pub fn new(context: &Context, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                raw: ptr::NonNull::<T>::dangling().as_ptr().cast(),
                len: 0,
                context: Arc::clone(&context.inner),
                marker: PhantomData,
            });
        }
        context.set_current()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("pinned allocation byte size overflowed");
        let mut raw = ptr::null_mut();
        // SAFETY: raw points to local output storage and bytes is non-zero.
        unsafe { check(cuMemHostAlloc(&mut raw, bytes, CU_MEMHOSTALLOC_PORTABLE))? };
        let buffer = Self {
            raw,
            len,
            context: Arc::clone(&context.inner),
            marker: PhantomData,
        };
        for index in 0..len {
            // SAFETY: CUDA allocated storage for len T values and every slot is written once.
            unsafe { buffer.raw.cast::<T>().add(index).write(T::default()) };
        }
        Ok(buffer)
    }

    pub fn from_slice(context: &Context, values: &[T]) -> Result<Self> {
        let mut buffer = Self::new(context, values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }
}

impl<T: Copy> PinnedBuffer<T> {
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: every slot is initialized by new and remains live for self's lifetime.
        unsafe { std::slice::from_raw_parts(self.raw.cast(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: self is exclusively borrowed and every slot is initialized.
        unsafe { std::slice::from_raw_parts_mut(self.raw.cast(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: the retained context stays alive and raw was allocated once by cuMemHostAlloc.
        unsafe {
            let _ = cuCtxSetCurrent(self.context.raw);
            let _ = cuMemFreeHost(self.raw);
        }
    }
}

fn set_current(context: &ContextInner) -> Result<()> {
    // SAFETY: context is retained by an Arc for the duration of the call.
    unsafe { check(cuCtxSetCurrent(context.raw)) }
}

fn ensure_same_context(left: &ContextInner, right: &ContextInner) {
    assert!(
        left.raw == right.raw,
        "CUDA resources belong to different contexts"
    );
}

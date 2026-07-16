//! Allocation-free Tier-1 emergency primitives.
//!
//! Healthy code owns configuration parsing and descriptor replacement. Once the
//! guardian is latched, this module only releases a pre-touched `mmap` reserve
//! and retries a write through an already-open cgroup descriptor.

#![expect(
    unsafe_code,
    reason = "Tier-1 explicitly owns Linux mmap, munmap, and write syscalls with local SAFETY proofs"
)]

use crate::monitor::CgroupTarget;
use std::{fmt, ptr::NonNull};

/// The reserve mapping could not be allocated or pre-touched.
#[derive(Debug)]
pub struct ReserveError;

impl fmt::Display for ReserveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("unable to allocate emergency reserve")
    }
}

impl std::error::Error for ReserveError {}

/// A pre-touched anonymous mapping released before the Tier-1 kill write.
#[derive(Debug)]
pub struct EmergencyReserve {
    data: Option<NonNull<u8>>,
    bytes: usize,
    page_size: usize,
    touched_pages: usize,
}

// SAFETY: `EmergencyReserve` uniquely owns its anonymous mmap. Its mutating
// operations require `&mut self`, so moving it to a Tokio worker transfers that
// ownership without creating concurrent aliases to the mapping.
unsafe impl Send for EmergencyReserve {}

impl EmergencyReserve {
    /// Allocates and faults in every page of the reserve mapping.
    ///
    /// # Errors
    ///
    /// Returns [`ReserveError`] when the system page size cannot be read or the
    /// anonymous mapping cannot be allocated and pre-touched.
    pub fn new(bytes: usize) -> Result<Self, ReserveError> {
        // SAFETY: sysconf reads a process-wide immutable configuration value.
        let configured = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = usize::try_from(configured).map_err(|_| ReserveError)?;
        if page_size == 0 {
            return Err(ReserveError);
        }
        Self::with_page_size(bytes, page_size)
    }

    /// Allocates a reserve with an explicit page size for deterministic tests.
    ///
    /// # Errors
    ///
    /// Returns [`ReserveError`] when either size is zero or the mapping cannot
    /// be allocated and pre-touched.
    pub fn with_page_size(bytes: usize, page_size: usize) -> Result<Self, ReserveError> {
        if bytes == 0 || page_size == 0 {
            return Err(ReserveError);
        }
        // SAFETY: anonymous private mmap has no file backing; bytes is nonzero.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(ReserveError);
        }
        let Some(data) = NonNull::new(mapping.cast::<u8>()) else {
            // SAFETY: mapping was returned by mmap for this exact length.
            unsafe { libc::munmap(mapping, bytes) };
            return Err(ReserveError);
        };
        let mut touched_pages = 0;
        for offset in (0..bytes).step_by(page_size) {
            // SAFETY: the mapping is writable for bytes and offset is in range.
            unsafe { data.as_ptr().add(offset).write_volatile(1) };
            touched_pages += 1;
        }
        Ok(Self {
            data: Some(data),
            bytes,
            page_size,
            touched_pages,
        })
    }

    /// Releases the mapping, retaining ownership if the kernel rejects `munmap`.
    pub fn release(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };
        // SAFETY: data and bytes describe the still-owned mmap created above.
        if unsafe { libc::munmap(data.as_ptr().cast::<libc::c_void>(), self.bytes) } != 0 {
            self.data = Some(data);
        }
    }

    /// Whether the reserve mapping remains resident and owned.
    #[must_use]
    pub const fn is_allocated(&self) -> bool {
        self.data.is_some()
    }

    /// The page size used while pre-touching the mapping.
    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// The number of pages explicitly faulted into the reserve.
    #[must_use]
    pub const fn touched_pages(&self) -> usize {
        self.touched_pages
    }

    #[cfg(test)]
    fn address(&self) -> Option<*mut u8> {
        self.data.map(NonNull::as_ptr)
    }
}

impl Drop for EmergencyReserve {
    fn drop(&mut self) {
        self.release();
    }
}

/// Result of one allocation-free attempt against the retained target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttemptOutcome {
    /// The retry deadline has not arrived or the target was already verified.
    Waiting,
    /// The target became empty after the direct write attempt.
    Verified,
    /// The target is still populated or its state could not be observed.
    Retry,
}

/// Releases the reserve and writes the one-byte command to a retained fd.
///
/// This is intentionally independent from configuration and target discovery:
/// it never opens files, formats errors, parses strings, or allocates.
///
/// # Errors
///
/// Returns the final retained-descriptor write failure after bounded `EINTR`
/// retries.
pub fn kill_direct(
    reserve: &mut EmergencyReserve,
    target: &CgroupTarget,
) -> Result<(), std::io::Error> {
    reserve.release();
    let command = *b"1";
    for _ in 0..3 {
        // SAFETY: the target retains an open writable cgroup.kill descriptor
        // and command points to one initialized byte.
        let written = unsafe {
            libc::write(
                target.kill_fd(),
                command.as_ptr().cast::<libc::c_void>(),
                command.len(),
            )
        };
        if written == 1 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINTR) {
            return Err(error);
        }
    }
    Err(std::io::Error::from_raw_os_error(libc::EINTR))
}

/// Tracks retry timing and target verification while the guardian is latched.
#[derive(Debug)]
pub struct EmergencyController {
    reserve: EmergencyReserve,
    retry_millis: u64,
    next_attempt_millis: u64,
    verified: bool,
}

impl EmergencyController {
    /// Creates a controller around one pre-touched reserve mapping.
    #[must_use]
    pub fn new(reserve: EmergencyReserve, retry_millis: u64) -> Self {
        Self {
            reserve,
            retry_millis,
            next_attempt_millis: 0,
            verified: false,
        }
    }

    /// Releases the reserve even when no target is currently armed.
    pub fn enter_emergency(&mut self) {
        self.reserve.release();
    }

    /// Recreates the reserve only after the rearm watermark has been reached.
    ///
    /// # Errors
    ///
    /// Returns [`ReserveError`] when a replacement mapping cannot be allocated.
    pub fn ensure_reserve(&mut self, bytes: usize) -> Result<bool, ReserveError> {
        if self.reserve.is_allocated() {
            return Ok(false);
        }
        self.reserve = EmergencyReserve::new(bytes)?;
        Ok(true)
    }

    /// Starts a fresh episode after a healthy descriptor replacement.
    pub fn reset_for_target_generation(&mut self) {
        self.next_attempt_millis = 0;
        self.verified = false;
    }

    /// Performs one error-tolerant retained-descriptor attempt.
    #[must_use]
    pub fn attempt(&mut self, now_millis: u64, target: &CgroupTarget) -> AttemptOutcome {
        if self.verified || now_millis < self.next_attempt_millis {
            return AttemptOutcome::Waiting;
        }
        let _write_error = kill_direct(&mut self.reserve, target);
        if matches!(target.is_empty(), Ok(true)) {
            self.verified = true;
            return AttemptOutcome::Verified;
        }
        self.next_attempt_millis = now_millis.saturating_add(self.retry_millis);
        AttemptOutcome::Retry
    }
}

#[cfg(test)]
mod tests {
    use super::EmergencyReserve;
    use std::io;

    #[test]
    fn reserve_release_returns_mapping_to_the_kernel() {
        let mut reserve =
            EmergencyReserve::with_page_size(64 * 1024 * 1024, 4096).expect("reserve");
        let address = reserve.address().expect("allocated reserve");
        let page_address = ((address as usize) & !(reserve.page_size() - 1)) as *mut libc::c_void;
        reserve.release();

        let mut residency = 0_u8;
        // SAFETY: mincore reports ENOMEM for an unmapped page-aligned address
        // without dereferencing the address.
        let result = unsafe { libc::mincore(page_address, 4096, &raw mut residency) };
        assert_eq!(result, -1, "released reserve remained mapped");
        assert_eq!(
            io::Error::last_os_error().raw_os_error(),
            Some(libc::ENOMEM)
        );
    }
}

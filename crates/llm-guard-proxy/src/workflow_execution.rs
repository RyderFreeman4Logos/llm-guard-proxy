//! Opaque ownership token for one admitted workflow execution.

use std::{fmt, sync::Arc};

/// Cloneable lease retained by every owner of workflow execution resources.
///
/// Production leases contain the admission permit. Test-only and direct runtime callers may use
/// the empty default. Capacity is released only after the last clone is dropped.
#[derive(Clone, Default)]
pub(crate) struct WorkflowExecutionLease {
    guard: Option<Arc<dyn Send + Sync>>,
}

impl WorkflowExecutionLease {
    pub(crate) fn new(guard: impl Send + Sync + 'static) -> Self {
        Self {
            guard: Some(Arc::new(guard)),
        }
    }
}

impl fmt::Debug for WorkflowExecutionLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkflowExecutionLease")
            .field("admitted", &self.guard.is_some())
            .finish()
    }
}

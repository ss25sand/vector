use std::{
    alloc::{GlobalAlloc, Layout},
    cmp, ptr,
    sync::atomic::Ordering,
};

use super::{
    token::{AllocationGroupId, try_with_suspended_allocation_group},
    tracer::Tracer,
};
use crate::internal_telemetry::allocations::TRACK_ALLOCATIONS;

/// A tracing allocator that groups allocation events by groups.
///
/// This allocator can only be used when specified via `#[global_allocator]`.
pub struct GroupedTraceableAllocator<A, T> {
    allocator: A,
    tracer: T,
}

impl<A, T> GroupedTraceableAllocator<A, T> {
    /// Creates a new `GroupedTraceableAllocator` that wraps the given allocator and tracer.
    #[must_use]
    pub const fn new(allocator: A, tracer: T) -> Self {
        Self { allocator, tracer }
    }
}

// SAFETY: Allocation and deallocation are delegated to `A` with matching layouts.
unsafe impl<A: GlobalAlloc, T: Tracer> GlobalAlloc for GroupedTraceableAllocator<A, T> {
    #[inline]
    unsafe fn alloc(&self, object_layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe {
            if !TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
                return self.allocator.alloc(object_layout);
            }
            self.alloc_traced(object_layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, object_ptr: *mut u8, object_layout: Layout) {
        // SAFETY: The caller provides a pointer and layout returned by `alloc`.
        unsafe {
            if !TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
                self.allocator.dealloc(object_ptr, object_layout);
                return;
            }
            self.dealloc_traced(object_ptr, object_layout);
        }
    }

    // Without these overrides, `GlobalAlloc`'s defaults would turn every `realloc` into
    // alloc + copy + dealloc and every `alloc_zeroed` into alloc + memset, even when allocation
    // tracing is off. That would bypass the wrapped allocator's in-place resizing and its
    // zeroed-page allocation.
    #[inline]
    unsafe fn alloc_zeroed(&self, object_layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe {
            if !TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
                return self.allocator.alloc_zeroed(object_layout);
            }
            self.alloc_zeroed_traced(object_layout)
        }
    }

    #[inline]
    unsafe fn realloc(
        &self,
        object_ptr: *mut u8,
        object_layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe {
            if !TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
                return self.allocator.realloc(object_ptr, object_layout, new_size);
            }
            self.realloc_traced(object_ptr, object_layout, new_size)
        }
    }
}

impl<A: GlobalAlloc, T: Tracer> GroupedTraceableAllocator<A, T> {
    #[inline]
    unsafe fn alloc_traced(&self, object_layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { self.alloc_traced_with(object_layout, |layout| self.allocator.alloc(layout)) }
    }

    #[inline]
    unsafe fn alloc_zeroed_traced(&self, object_layout: Layout) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::alloc_zeroed`'s contract. The group ID is
        // written past the object, so the object's bytes stay zeroed.
        unsafe {
            self.alloc_traced_with(object_layout, |layout| self.allocator.alloc_zeroed(layout))
        }
    }

    /// Reallocates like `GlobalAlloc`'s default: allocate, copy, then free.
    ///
    /// The group ID trailer sits right after the object, so resizing in place would move it. A
    /// fresh traced allocation also attributes the new size to the current group, the same way
    /// it was attributed before `realloc` was overridden.
    #[inline]
    unsafe fn realloc_traced(
        &self,
        object_ptr: *mut u8,
        object_layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        // SAFETY: The caller upholds `GlobalAlloc::realloc`'s contract: `new_size` is nonzero
        // and does not overflow `isize` once rounded up to the alignment, and `object_ptr` was
        // allocated by this allocator with `object_layout`.
        unsafe {
            let new_layout = Layout::from_size_align_unchecked(new_size, object_layout.align());
            let new_ptr = self.alloc_traced(new_layout);
            if !new_ptr.is_null() {
                ptr::copy_nonoverlapping(
                    object_ptr,
                    new_ptr,
                    cmp::min(object_layout.size(), new_size),
                );
                self.dealloc_traced(object_ptr, object_layout);
            }
            new_ptr
        }
    }

    #[inline]
    unsafe fn alloc_traced_with(
        &self,
        object_layout: Layout,
        alloc: impl FnOnce(Layout) -> *mut u8,
    ) -> *mut u8 {
        // SAFETY: `alloc` upholds `GlobalAlloc::alloc`'s contract for the wrapped layout.
        unsafe {
            // Allocate our wrapped layout and make sure the allocation succeeded.
            let (actual_layout, offset_to_group_id) = get_wrapped_layout(object_layout);
            let actual_ptr = alloc(actual_layout);
            if actual_ptr.is_null() {
                return actual_ptr;
            }

            let group_id_ptr = actual_ptr.add(offset_to_group_id).cast::<u8>();

            let object_size = object_layout.size();

            try_with_suspended_allocation_group(
                #[inline(always)]
                |group_id| {
                    group_id_ptr.write(group_id.as_raw());
                    self.tracer.trace_allocation(object_size, group_id);
                },
            );
            actual_ptr
        }
    }

    #[inline]
    unsafe fn dealloc_traced(&self, object_ptr: *mut u8, object_layout: Layout) {
        // SAFETY: The pointer and layout come from a matching call to `alloc_traced`.
        unsafe {
            // Regenerate the wrapped layout so we know where we have to look, as the pointer we've given relates to the
            // requested layout, not the wrapped layout that was actually allocated.
            let (wrapped_layout, offset_to_group_id) = get_wrapped_layout(object_layout);

            let raw_group_id = object_ptr.add(offset_to_group_id).cast::<u8>().read();

            // Deallocate before tracking, just to make sure we're reclaiming memory as soon as possible.
            self.allocator.dealloc(object_ptr, wrapped_layout);

            let object_size = object_layout.size();
            let source_group_id = AllocationGroupId::from_raw(raw_group_id);

            try_with_suspended_allocation_group(
                #[inline(always)]
                |_| {
                    self.tracer.trace_deallocation(object_size, source_group_id);
                },
            );
        }
    }
}

#[inline(always)]
fn get_wrapped_layout(object_layout: Layout) -> (Layout, usize) {
    static HEADER_LAYOUT: Layout = Layout::new::<u8>();

    // We generate a new allocation layout that gives us a location to store the active allocation group ID ahead
    // of the requested allocation, which lets us always attempt to retrieve it on the deallocation path.
    let (actual_layout, offset_to_group_id) = object_layout
        .extend(HEADER_LAYOUT)
        .expect("wrapping requested layout resulted in overflow");

    (actual_layout.pad_to_align(), offset_to_group_id)
}

#[cfg(test)]
mod tests {
    use std::{alloc::System, cell::Cell, ptr, sync::Mutex, thread};

    use tracing::{Dispatch, Span, subscriber::with_default};
    use tracing_subscriber::{Registry, layer::SubscriberExt as _};

    use super::*;
    use crate::internal_telemetry::allocations::{
        allocator::{
            AllocationLayer, token::try_with_suspended_allocation_group, without_allocation_tracing,
        },
        is_allocation_tracing_enabled,
    };

    #[derive(Default)]
    struct RecordingTracer(Mutex<Vec<(bool, usize, AllocationGroupId)>>);

    impl Tracer for RecordingTracer {
        fn trace_allocation(&self, size: usize, group: AllocationGroupId) {
            try_with_suspended_allocation_group(|_| panic!("tracer must not be reentered"));
            self.0.lock().unwrap().push((true, size, group));
        }

        fn trace_deallocation(&self, size: usize, group: AllocationGroupId) {
            try_with_suspended_allocation_group(|_| panic!("tracer must not be reentered"));
            self.0.lock().unwrap().push((false, size, group));
        }
    }

    struct FailingAllocator;

    // SAFETY: This allocator always reports allocation failure and never returns a pointer.
    unsafe impl GlobalAlloc for FailingAllocator {
        unsafe fn alloc(&self, _: Layout) -> *mut u8 {
            ptr::null_mut()
        }

        unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
            panic!("failed allocations must not be deallocated");
        }
    }

    fn current_group() -> AllocationGroupId {
        let mut group = None;
        try_with_suspended_allocation_group(|id| group = Some(id));
        group.expect("allocation tracing should not be suspended")
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep the allocation lifecycle in one scenario"
    )]
    fn allocation_lifecycle_tracks_span_context_and_original_layout() {
        let root = AllocationGroupId::ROOT;
        let outer_group = AllocationGroupId::register().unwrap();
        let inner_group = AllocationGroupId::register().unwrap();
        let subscriber = Registry::default().with(AllocationLayer::default());
        with_default(subscriber, || {
            let outer = tracing::info_span!("outer");
            outer_group.attach_to_span(&outer);
            let inner = tracing::info_span!("inner");
            inner_group.attach_to_span(&inner);
            outer_group.attach_to_span(&Span::none());

            let allocator = GroupedTraceableAllocator::new(System, RecordingTracer::default());
            assert!(!is_allocation_tracing_enabled());
            // SAFETY: The nonzero layout is used for both allocation and deallocation.
            unsafe {
                let layout = Layout::new::<u64>();
                let ptr = allocator.alloc(layout);
                assert!(!ptr.is_null());
                allocator.dealloc(ptr, layout);
            }
            assert!(allocator.tracer.0.lock().unwrap().is_empty());

            for (size, align) in [(1, 1), (7, 8), (64, 64), (4096, 4096)] {
                let layout = Layout::from_size_align(size, align).unwrap();
                let (wrapped, offset) = get_wrapped_layout(layout);
                assert_eq!(wrapped.align(), align);
                assert_eq!(wrapped.size() % align, 0);
                assert!(wrapped.size() > size);
                assert_eq!(offset, size);

                // Call the traced paths directly: toggling the global allocator mid-test would
                // change the layout used to free allocations made by the test harness.
                // SAFETY: Every successful allocation is accessed within its requested size and
                // deallocated once with its original layout through the same traced allocator.
                unsafe {
                    let root_ptr = allocator.alloc_traced(layout);
                    assert!(!root_ptr.is_null());
                    assert_eq!(root_ptr.align_offset(align), 0);
                    root_ptr.write_bytes(0xa5, size);

                    outer.in_scope(|| {
                        assert_eq!(current_group(), outer_group);
                        let outer_ptr = allocator.alloc_traced(layout);
                        assert!(!outer_ptr.is_null());
                        let untagged = tracing::info_span!("untagged");
                        untagged.in_scope(|| assert_eq!(current_group(), outer_group));

                        inner.in_scope(|| {
                            assert_eq!(current_group(), inner_group);
                            let inner_ptr = allocator.alloc_traced(layout);
                            assert!(!inner_ptr.is_null());
                            // Free under a different group from the one that allocated.
                            allocator.dealloc_traced(outer_ptr, layout);
                            allocator.dealloc_traced(inner_ptr, layout);
                        });
                        assert_eq!(current_group(), outer_group);
                    });
                    assert_eq!(current_group(), root);
                    assert!((0..size).all(|idx| root_ptr.add(idx).read() == 0xa5));
                    allocator.dealloc_traced(root_ptr, layout);
                }

                assert_eq!(
                    *allocator.tracer.0.lock().unwrap(),
                    [
                        (true, size, root),
                        (true, size, outer_group),
                        (true, size, inner_group),
                        (false, size, outer_group),
                        (false, size, inner_group),
                        (false, size, root),
                    ]
                );
                allocator.tracer.0.lock().unwrap().clear();
            }

            outer.in_scope(|| {
                let ran = Cell::new(false);
                without_allocation_tracing(|| {
                    try_with_suspended_allocation_group(|_| panic!("tracing must be suspended"));
                    without_allocation_tracing(|| ran.set(true));
                });
                assert!(ran.get());
                assert_eq!(current_group(), outer_group);

                let dispatch = Dispatch::new(Registry::default().with(AllocationLayer::new()));
                thread::scope(|scope| {
                    scope.spawn(|| {
                        tracing::dispatcher::with_default(&dispatch, || {
                            assert_eq!(current_group(), root);
                            outer.in_scope(|| assert_eq!(current_group(), outer_group));
                            assert_eq!(current_group(), root);
                        });
                    });
                });
                assert_eq!(current_group(), outer_group);
            });

            let failing =
                GroupedTraceableAllocator::new(FailingAllocator, RecordingTracer::default());
            // SAFETY: The layout is nonzero; the allocator deliberately returns null.
            let failed = unsafe { failing.alloc_traced(Layout::new::<u64>()) };
            assert!(failed.is_null());
            assert!(failing.tracer.0.lock().unwrap().is_empty());
        });

        // Attaching without the allocation layer should be harmless.
        with_default(Registry::default(), || {
            outer_group.attach_to_span(&tracing::info_span!("without_layer"));
            assert_eq!(current_group(), root);
        });
    }

    #[test]
    fn realloc_and_alloc_zeroed_keep_contents_and_tracking() {
        let allocator = GroupedTraceableAllocator::new(System, RecordingTracer::default());
        let layout = Layout::from_size_align(24, 8).unwrap();
        assert!(!is_allocation_tracing_enabled());

        // Untraced: delegated straight to the wrapped allocator, nothing is recorded.
        // SAFETY: Every pointer is accessed within its current size and freed once with the
        // layout it currently has.
        unsafe {
            let zeroed = allocator.alloc_zeroed(layout);
            assert!(!zeroed.is_null());
            assert!((0..24).all(|idx| zeroed.add(idx).read() == 0));
            zeroed.write_bytes(0x5a, 24);

            let grown = allocator.realloc(zeroed, layout, 4096);
            assert!(!grown.is_null());
            assert!((0..24).all(|idx| grown.add(idx).read() == 0x5a));
            let grown_layout = Layout::from_size_align(4096, 8).unwrap();

            let shrunk = allocator.realloc(grown, grown_layout, 8);
            assert!(!shrunk.is_null());
            assert!((0..8).all(|idx| shrunk.add(idx).read() == 0x5a));
            allocator.dealloc(shrunk, Layout::from_size_align(8, 8).unwrap());
        }
        assert!(allocator.tracer.0.lock().unwrap().is_empty());

        // Traced: called directly for the same reason as in the lifecycle test above.
        let outer_group = AllocationGroupId::register().unwrap();
        let inner_group = AllocationGroupId::register().unwrap();
        with_default(Registry::default().with(AllocationLayer::default()), || {
            let outer = tracing::info_span!("outer");
            outer_group.attach_to_span(&outer);
            let inner = tracing::info_span!("inner");
            inner_group.attach_to_span(&inner);

            // SAFETY: As above, through the traced paths only.
            unsafe {
                let zeroed = outer.in_scope(|| allocator.alloc_zeroed_traced(layout));
                assert!(!zeroed.is_null());
                assert!((0..24).all(|idx| zeroed.add(idx).read() == 0));
                zeroed.write_bytes(0x5a, 24);

                // A traced realloc frees from the group that allocated and allocates in the
                // current one, and both trailers stay readable.
                let grown = inner.in_scope(|| allocator.realloc_traced(zeroed, layout, 4096));
                assert!(!grown.is_null());
                assert!((0..24).all(|idx| grown.add(idx).read() == 0x5a));
                let grown_layout = Layout::from_size_align(4096, 8).unwrap();

                let shrunk = outer.in_scope(|| allocator.realloc_traced(grown, grown_layout, 8));
                assert!(!shrunk.is_null());
                assert!((0..8).all(|idx| shrunk.add(idx).read() == 0x5a));
                allocator.dealloc_traced(shrunk, Layout::from_size_align(8, 8).unwrap());
            }
        });

        assert_eq!(
            *allocator.tracer.0.lock().unwrap(),
            [
                (true, 24, outer_group),
                (true, 4096, inner_group),
                (false, 24, outer_group),
                (true, 8, outer_group),
                (false, 4096, inner_group),
                (false, 8, outer_group),
            ]
        );
    }
}

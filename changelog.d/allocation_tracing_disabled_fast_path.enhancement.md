When `--allocation-tracing` is off (the default), Vector's allocator now passes `realloc` and
zeroed allocations straight to jemalloc, so buffers can grow in place instead of being copied,
and the allocation tracing layer is no longer installed, so component tasks no longer look up
their span on every poll.

authors: armleth

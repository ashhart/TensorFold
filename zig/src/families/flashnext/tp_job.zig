//! Host queue records shared by the pair owner and its startup memory budget.

/// Bytes the host copies into the peer's window once the GPU has posted the job's sequence.
pub const Write = struct { src: [*]const u8, len: usize, dst: usize };

/// A posted sequence's work: a decode exchange, or prefill writes and their flag.
pub const Job = struct {
    kind: enum { exchange, reduce, pick, send, wait } = .send,
    writes: [8]Write = undefined,
    n: usize = 0,
    flag: usize = 0,
    value: u64 = 0,
    bytes: usize = 0, // decode payload size, fixed from this job's row count when queued
    local: usize = 0,
    remote: usize = 0,
};

//! Host queue records shared by the pair owner and its startup memory budget.

/// Bytes the host copies into the peer's window once the GPU has posted the job's sequence.
pub const Write = struct { src: [*]const u8, len: usize, dst: usize };

/// A posted sequence's work: a decode exchange, or prefill writes and their flag.
pub const Job = struct {
    kind: enum { exchange, reduce, pick, send } = .send,
    writes: [8]Write = undefined,
    n: usize = 0,
    flag: usize = 0,
    value: u64 = 0,
    slot: u32 = 0, // decode: the message's send and receive slot (tp_seq.zig)
    bytes: usize = 0, // decode: its size, fixed when queued (the GPU's post word may already be a later message's)
};

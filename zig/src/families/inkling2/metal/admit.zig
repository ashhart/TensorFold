//! Inkling 2 Mini on Metal: the bytes a serve's weights, scratch and streams ask for, as the startup line reports them.
const st = @import("state.zig");
const dspark = @import("dspark.zig");

/// What a serve asks for: the fixed part beside the streams, what the streams share, and the state of one stream.
pub const Demand = struct {
    /// Weights, dense layers and scratch, DSpark's head: held whatever the stream count.
    fixed: u64,
    /// Held once for every stream: DSpark's attention scores at the window.
    shared: u64,
    /// One stream at the window: a cache slot for each of the 48 visits, logits and taps, and its DSpark lane.
    stream: u64,

    /// The bytes a serve of `n` streams asks for in all.
    pub fn of(d: Demand, n: u64) u64 {
        return d.fixed + d.shared + d.stream * n;
    }
};

/// A serve's demand at `window` tokens beside `fixed` bytes of weights, dense layers and scratch.
pub fn demandAt(fixed: u64, window: usize, drafts: bool) Demand {
    return .{
        .fixed = fixed,
        .shared = if (drafts) dspark.Drafter.scoreBytes(window) - dspark.Drafter.scoreBytes(0) else 0,
        .stream = st.cacheBytes(window) + if (drafts) dspark.Drafter.streamBytes(window) else 0,
    };
}

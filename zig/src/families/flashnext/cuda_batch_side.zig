//! A decode or verify pass's shared expert on its own stream beside the routed experts (TF_FLASHNEXT_SHARED_SIDE).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");

const dims = wt.dims;
const slots = wt.slots;
const moe_w = wt.moe_w;

/// On by default: its slot rows of y are apart from the routed experts', so the bits match; =0 runs it after them.
pub fn on() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SHARED_SIDE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The shared expert's stream and the two points it forks and joins at.
pub const Side = struct {
    stream: cuda.Stream,
    fork: cuda.Event,
    join: cuda.Event,

    pub fn init(d: *const cuda.Driver) !Side {
        var st = try cuda.Stream.init(d, true);
        errdefer st.deinit();
        var f = try cuda.Event.init(d, false);
        errdefer f.deinit();
        return .{ .stream = st, .fork = f, .join = try cuda.Event.init(d, false) };
    }

    pub fn deinit(x: *Side) void {
        x.join.deinit();
        x.fork.deinit();
        x.stream.deinit();
    }
};

/// The shared expert over the pass's `n` rows on `st`: gate and up from the MoE input, SwiGLU, down, into y's slot.
pub fn run(pass: anytype, st: cuda.Stream, native: anytype, n: usize) !void {
    const s = pass.s;
    const glue = pass.tri.own orelse return error.NativeGlueRequired;
    try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part, .bf16);
    try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part, .bf16);
    try glue.sharedSwiGLU(st, s.shared_gate, s.shared_up, s.shared_act, n, moe_w);
    try pass.k.nativeMatmul(st, s.shared_act, moe_w, n, native.shared[2].native.?, s.shared_out, s.native_part, .f32);
    try glue.sharedScatter(st, s.shared_out, s.y, n, dims, slots);
}

/// After y is cleared: the shared expert forked onto the side stream (one GPU, not a prompt fill), or null.
pub fn fork(pass: anytype, native: anytype, n: usize) !?Side {
    const x = pass.s.side orelse return null;
    if (native.rank != 0 or native.world != 1 or pass.hooks.prompt) return null;
    try x.fork.record(pass.stream);
    try x.stream.wait(x.fork);
    try run(pass, x.stream, native, n);
    try x.join.record(x.stream);
    return x;
}

/// After the routed experts: the forked shared expert joined, or run here on rank 0 as before.
pub fn join(pass: anytype, side: ?Side, native: anytype, n: usize) !void {
    if (side) |x| return pass.stream.wait(x.join);
    if (native.rank == 0) try run(pass, pass.stream, native, n);
}

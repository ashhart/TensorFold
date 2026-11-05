//! cuda_batch_gdn_lanes.once with a driver that counts launches: which rounds it takes, and none launched on a refusal.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const env = @import("sys").env;
const parts = @import("cuda_batch_state.zig");
const lanes = @import("cuda_batch_gdn_lanes.zig");
const Layer = @import("cuda_prompt_load.zig").Layer;

const Mock = struct {
    var calls: usize = 0;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        calls += 1;
        return 0;
    }
};

const Scratch = struct { proj: u64, gq: u64 = 0x10000, gk: u64 = 0x20000, gv: u64 = 0x30000, ggb: u64 = 0x40000, ys: u64 = 0x50000 };
const Kernels = struct { lanes_keep: cuda.Function, lanes_front: cuda.Function, lanes_chain: cuda.Function, lanes_shift: cuda.Function };
const Collective = struct { id: u32 = 0 };

/// What once() reads of a batch.Pass: the round's lanes, the projection rows, the kernels, the stream and the model.
const Pass = struct {
    lanes: []const parts.Lane,
    s: Scratch,
    k: Kernels,
    stream: cuda.Stream,
    m: struct { collective: ?*const Collective = null } = .{},
};

var history: [2]i64 = .{ 0, 0 };

fn buf(d: *const cuda.Driver, ptr: u64) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = 0 };
}

/// out.len lanes of `rows` rows each, every lane with its own conv window and state at 16-byte aligned addresses.
fn round(d: *const cuda.Driver, conv: []cuda.DeviceBuffer, state: []cuda.DeviceBuffer, out: []parts.Lane, rows: usize, keep: ?*const parts.Keep) []parts.Lane {
    for (out, 0..) |*l, i| {
        conv[i] = buf(d, 0x100000 + i * 0x1000);
        state[i] = buf(d, 0x200000 + i * 0x1000);
        l.* = .{ .seq = .{ .conv = conv[i .. i + 1], .state = state[i .. i + 1], .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = buf(d, 0), .history = &history, .cap = 4096, .scores = 0 }, .pos = 0, .rows = rows, .keep = keep };
    }
    return out;
}

fn pass(d: *const cuda.Driver, ls: []const parts.Lane, proj: u64) Pass {
    const f: cuda.Function = .{ .d = d, .handle = @ptrFromInt(1) };
    return .{ .lanes = ls, .s = .{ .proj = proj }, .k = .{ .lanes_keep = f, .lanes_front = f, .lanes_chain = f, .lanes_shift = f }, .stream = .{ .d = d, .handle = null } };
}

/// once() on `p` answers `took` and launches `launches` kernels.
fn expectOnce(p: *const Pass, lw: *const Layer, took: bool, launches: usize) !void {
    Mock.calls = 0;
    try t.expectEqual(took, try lanes.once(p, lw, 0, 0));
    try t.expectEqual(launches, Mock.calls);
}

test "lanes' DeltaNet steps: one launch a step when the round fits, nothing launched when it falls back" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var lw: Layer = undefined;
    @memset(std.mem.asBytes(&lw), 0);
    lw.heads = .full;
    var conv: [lanes.gdn_lanes_max + 1]cuda.DeviceBuffer = undefined;
    var state: [lanes.gdn_lanes_max + 1]cuda.DeviceBuffer = undefined;
    var ls: [lanes.gdn_lanes_max + 1]parts.Lane = undefined;
    _ = env.unsetenv("TF_FLASHNEXT_GDN_LANES");

    // controls: the counter sees the launches, from two lanes up to the most the lane table holds
    for ([_]usize{ 2, 3, lanes.gdn_lanes_max }) |n| {
        const p = pass(&d, round(&d, &conv, &state, ls[0..n], 9, null), 0x400000);
        try expectOnce(&p, &lw, true, 3); // front, chain, shift: nothing to keep
    }
    var keep: parts.Keep = .{};
    keep.layers[0] = .{ .proj = 0x600000, .hist = 0x700000, .state_out = 0x800000 };
    const kept = pass(&d, round(&d, &conv, &state, ls[0..2], 9, &keep), 0x400000);
    try expectOnce(&kept, &lw, true, 4); // and the rollback copies

    // fallbacks: false and no launch, so the caller's lane loop runs the round as before
    const one = pass(&d, round(&d, &conv, &state, ls[0..1], 9, null), 0x400000);
    try expectOnce(&one, &lw, false, 0); // one lane (a single stream's round) keeps the per-lane launches
    const one_kept = pass(&d, round(&d, &conv, &state, ls[0..1], 9, &keep), 0x400000);
    try expectOnce(&one_kept, &lw, false, 0); // with its rollback copies too
    const many = pass(&d, round(&d, &conv, &state, ls[0 .. lanes.gdn_lanes_max + 1], 9, null), 0x400000);
    try expectOnce(&many, &lw, false, 0); // 17 lanes
    const wide = pass(&d, round(&d, &conv, &state, ls[0..2], 64, null), 0x400000);
    try expectOnce(&wide, &lw, false, 0); // a lane of 64 rows (its chain takes fn_gdn_chain_tpr)
    const odd = pass(&d, round(&d, &conv, &state, ls[0..2], 9, null), 0x400008);
    try expectOnce(&odd, &lw, false, 0); // projection rows not 16-byte aligned
    var odd_keep = keep;
    odd_keep.layers[0].hist = 0x700004;
    const odd_hist = pass(&d, round(&d, &conv, &state, ls[0..2], 9, &odd_keep), 0x400000);
    try expectOnce(&odd_hist, &lw, false, 0); // a keep buffer not 16-byte aligned
    const collective: Collective = .{};
    var tp = pass(&d, round(&d, &conv, &state, ls[0..2], 9, null), 0x400000);
    tp.m.collective = &collective;
    try expectOnce(&tp, &lw, false, 0); // tensor parallel keeps the lane loop
    var half = lw;
    half.heads = .half;
    const halved = pass(&d, round(&d, &conv, &state, ls[0..2], 9, null), 0x400000);
    try expectOnce(&halved, &half, false, 0); // a rank's half of the heads: these kernels have no twin

    _ = env.setenv("TF_FLASHNEXT_GDN_LANES", "0", 1);
    defer _ = env.unsetenv("TF_FLASHNEXT_GDN_LANES");
    const off = pass(&d, round(&d, &conv, &state, ls[0..2], 9, null), 0x400000);
    try expectOnce(&off, &lw, false, 0); // switched off
}

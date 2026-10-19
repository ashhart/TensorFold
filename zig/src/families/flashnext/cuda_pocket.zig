//! Living Weights' pockets in Flash Next's CUDA forward: each layer's SwiGLU neurons beside its shared expert.
const std = @import("std");
const cuda = @import("cuda");
const fb = @import("cuda_fn_back.zig");
const wt = @import("cuda_prompt_wt.zig");

const layers_n = wt.layers_n;
const dims = wt.dims;

/// A layer's pocket on the device: fp32 [cap, dims] gate rows, up rows and outputs, the first `used` neurons live.
pub const Rows = struct { gate: cuda.DeviceBuffer, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, cap: usize, used: usize = 0 };

/// Every layer's pocket (null: none) and the kernels that run them, loaded with the first.
pub const Pockets = struct {
    layers: [layers_n]?Rows = @splat(null),
    k: ?fb.Kernels = null,
    /// fp32 a row of a pass with taught and untaught streams: the pocket's scale on each (made on first use).
    scales: ?cuda.DeviceBuffer = null,

    pub fn deinit(p: *Pockets) void {
        for (&p.layers) |*slot| if (slot.*) |*pk| {
            pk.gate.free();
            pk.up.free();
            pk.down.free();
            slot.* = null;
        };
        if (p.k) |*k| k.deinit();
        p.k = null;
        if (p.scales) |*b| b.free();
        p.scales = null;
    }

    /// A pass's row scales on the device (Pocket.scale), stream-ordered after the passes that read the last ones.
    pub fn rowScales(p: *Pockets, d: *cuda.Driver, s: *cuda.Stream, scales: []const f32) !u64 {
        try s.synchronize();
        const next = try staged(d, scales);
        if (p.scales) |*old| old.free();
        p.scales = next;
        return p.scales.?.ptr;
    }

    /// Layer `layer`'s neurons: gate, up and output rows fp32 [n, dims], the first `used` live; made on first use.
    pub fn set(p: *Pockets, d: *cuda.Driver, s: *cuda.Stream, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        if (layer >= layers_n or gate.len % dims != 0) return error.UnexpectedTensor;
        const n = gate.len / dims;
        if (n == 0 or n > fb.max_pocket or up.len != gate.len or down.len != gate.len or used > n) return error.UnexpectedTensor;
        if (p.k == null) p.k = try fb.Kernels.load(d);
        if (p.layers[layer]) |pk| if (pk.cap != n) return error.PocketOfAnotherSize;
        // Old readers finish before publication; all fallible copies write fresh destinations.
        try s.synchronize();
        var g = try staged(d, gate);
        errdefer g.free();
        var u = try staged(d, up);
        errdefer u.free();
        const v = try staged(d, down);
        var old = p.layers[layer];
        p.layers[layer] = .{ .gate = g, .up = u, .down = v, .cap = n, .used = used };
        if (old) |*x| {
            x.gate.free();
            x.up.free();
            x.down.free();
        }
    }

    /// Layer `layer`'s pocket into gate, up and down ([n, dims] each); the neurons in use (0: none).
    pub fn read(p: *const Pockets, s: *cuda.Stream, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
        if (layer >= layers_n) return error.UnexpectedTensor;
        const pk = p.layers[layer] orelse return 0;
        if (gate.len != pk.cap * dims or up.len != gate.len or down.len != gate.len) return error.PocketOfAnotherSize;
        try s.synchronize();
        try pk.gate.download(0, std.mem.sliceAsBytes(gate));
        try pk.up.download(0, std.mem.sliceAsBytes(up));
        try pk.down.download(0, std.mem.sliceAsBytes(down));
        return pk.used;
    }

    /// Layer `layer`'s pocket as the kernels take it (activations in `act`), or null with no neuron in use.
    pub fn of(p: *const Pockets, layer: usize, act: u64) ?fb.Ops.Pocket {
        const pk = p.layers[layer] orelse return null;
        if (pk.used == 0) return null;
        return .{ .gate = pk.gate.ptr, .up = pk.up.ptr, .down = pk.down.ptr, .n = pk.used, .act = act };
    }

    /// The layers with a neuron in use, a bit each: a pass adds a pocket there alone.
    pub fn mask(p: *const Pockets) u64 {
        var m: u64 = 0;
        for (p.layers, 0..) |slot, l| if (slot) |pk| if (pk.used > 0) {
            m |= @as(u64, 1) << @intCast(l);
        };
        return m;
    }
};

/// Which rows of a pass the pocket acts on: every one, none (every stream untaught), or some (by row scales).
pub const Reads = enum { all, none, some };

/// The rows the pocket acts on from each row's stream's untaught flag; for `.some`, `scales` 1 on a taught row, else 0.
pub fn reads(untaught: []const bool, scales: []f32) Reads {
    if (std.mem.indexOfScalar(bool, untaught, true) == null) return .all;
    if (std.mem.indexOfScalar(bool, untaught, false) == null) return .none;
    for (scales, untaught) |*v, u| v.* = if (u) 0 else 1;
    return .some;
}

/// The neurons in use of a pocket read from the shards: the first ones, up to the last with a nonzero input row.
pub fn usedOf(up: []const f32, n: usize) usize {
    var used: usize = 0;
    for (0..n) |j| {
        if (std.mem.indexOfNone(f32, up[j * dims ..][0..dims], &.{0}) != null) used = j + 1;
    }
    return used;
}

test "a pocket's neurons in use end at its last nonzero input row" {
    var up: [3 * dims]f32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 0), usedOf(&up, 3));
    up[dims + 7] = 0.5;
    try std.testing.expectEqual(@as(usize, 2), usedOf(&up, 3));
    var p: Pockets = .{};
    try std.testing.expectEqual(@as(u64, 0), p.mask());
    try std.testing.expect(p.of(5, 0) == null);
}

test "an untaught stream's rows scale the pocket by 0; a pass of untaught streams alone reads none of it" {
    var scales: [4]f32 = @splat(7);
    try std.testing.expectEqual(Reads.all, reads(&.{ false, false }, scales[0..2]));
    try std.testing.expectEqual(Reads.none, reads(&.{ true, true, true }, scales[0..3]));
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 7 }, &scales);
    try std.testing.expectEqual(Reads.some, reads(&.{ false, true, true, false }, &scales));
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &scales);
}

test {
    _ = @import("cuda_pocket_test.zig");
}

// Synchronous HtoD from pinned host memory completes before return; no async source escapes an error path.
fn staged(d: *cuda.Driver, values: []const f32) !cuda.DeviceBuffer {
    const bytes = std.mem.sliceAsBytes(values);
    var host = try cuda.HostBuffer.alloc(d, bytes.len);
    defer host.free();
    @memcpy(host.bytes, bytes);
    var out = try cuda.DeviceBuffer.alloc(d, bytes.len);
    errdefer out.free();
    try d.check(d.api.cuMemcpyHtoD_v2(out.ptr, host.bytes.ptr, host.bytes.len), "cuMemcpyHtoD pinned pocket");
    return out;
}

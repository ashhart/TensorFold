//! Inkling 2's routed experts: the family's grouping and grouped projections, main's rotation and combine.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("ops.zig");

const ex = cuda.exl3_experts;

/// The two SwiGLU softcaps (config.json mlp_gate_softcap, mlp_up_softcap).
pub const Caps = struct { gate: f32, up: f32 };

/// The least and most half-bit widths (k2) among a layer's expert tensors, as the loader read them.
pub const Bits = struct { lo: u32, hi: u32 };

/// Calls of at least this many rows take an item list (runs of one expert's members), not one item an expert.
pub const listed_from = 17;
/// Calls of at least this many rows take the prompt kernel (32 members by 64 columns a block).
pub const wide_from = 512;

/// main's experts module beside the family's grouped projections.
pub const Kernels = struct {
    main: ex.Experts,
    mod: cuda.Module,
    group: cuda.Function,
    rows: cuda.Function,
    wide: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var main = try ex.Experts.load(d);
        errdefer main.unload();
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_experts);
        errdefer mod.unload();
        return .{
            .main = main,
            .mod = mod,
            .group = try mod.function("tf_ink2x_group"),
            .rows = try mod.function("tf_ink2x_rows"),
            .wide = try mod.function("tf_ink2x_wide"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.main.unload();
    }
};

fn i32d(v: usize) i32 {
    return @intCast(v);
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// Items at most a call of `rows` rows lists: runs of 16 members, plus one partial run a distinct expert.
fn maxItems(l: ex.Layer, rows: usize, slots: usize) usize {
    const p = rows * slots;
    return (p + 15) / 16 + @min(p, l.experts);
}

/// main's scratch, then the item count and the item list (three int32 an item).
pub fn scratchBytes(l: ex.Layer, rows: usize, slots: usize) !usize {
    return try ex.scratchBytes(l, rows, slots) + 256 + align256(maxItems(l, rows, slots) * 12);
}

/// Where the item count sits in scratch laid out by ex.scratch over scratchBytes(); the list follows 256 bytes on.
fn itemsAt(l: ex.Layer, sc: ex.Scratch) u64 {
    return sc.members + align256(@min(sc.rows * sc.slots, l.experts) * sc.rows * 4);
}

/// exl3_experts.Experts.run's launches in its order: the family's grouped projections, ops.cu's tf_ink2_gateup.
pub fn run(e: *const Kernels, k: *const ops.Kernels, s: cuda.Stream, l: ex.Layer, caps: Caps, bits: Bits, x: u64, x_stride: usize, x_dtype: ex.DType, pick: u64, wts: u64, rows: usize, out: u64, sc: ex.Scratch) !void {
    const d = l.dims;
    const w = l.width;
    const slots = sc.slots;
    if (rows < 1 or rows > sc.rows or slots < 1 or slots > 32) return error.BadExperts;
    if (l.experts < 1 or l.experts > 512 or l.codebook != .mul1 or l.gu_stride != 0) return error.BadExperts;
    if (bits.lo < 2 or bits.hi > 8 or bits.lo > bits.hi) return error.BadExpertBits; // the kernels decode 1 to 4 bits
    const g_sk = try ex.plan(d, w, true);
    const d_sk = try ex.plan(w, d, false);
    const p = rows * slots;
    const maxu = @min(p, l.experts);

    const listed = rows >= listed_from;
    const wide = rows >= wide_from;
    const per: usize = if (wide) 32 else 16;
    const at = itemsAt(l, sc);
    var a: cuda.Args = .{};
    for ([_]u64{ pick, sc.ids, sc.count, sc.members, at + 256, at }) |v| a.add(v);
    for ([_]usize{ rows, slots, l.experts, rows, if (listed) per else 0, @intFromBool(!listed) }) |v| a.add(i32d(v));
    const shared: u32 = @intCast(l.experts * (32 * 2 + 2 * 4)); // per-warp uint16 counts, then counts and places
    try cuda.launch.launch(e.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = shared }, s, &a);
    const items: u64 = if (listed) at + 256 else 0;
    const count: u64 = if (listed) at else sc.count;
    const gy: u32 = @intCast(if (listed) (p + per - 1) / per + maxu else maxu);
    const f = if (wide) e.wide else e.rows;
    const cols: usize = if (wide) 64 else 128;

    var r: cuda.Args = .{};
    r.add(x);
    r.add(i32d(x_stride));
    r.add(@as(i32, @backingInt(x_dtype)));
    for ([_]u64{ pick, l.suh_g, l.suh_u, sc.xg, sc.xu }) |v| r.add(v);
    for ([_]usize{ d, slots, l.experts }) |v| r.add(i32d(v));
    const rot: cuda.Dim3 = .{ .x = @intCast(p), .y = @intCast(d / 128), .z = 2 };
    try cuda.launch.launch(e.main.rot_in, .{ .grid = rot, .block = .{ .x = 32 } }, s, &r);

    var g: cuda.Args = .{};
    for ([_]u64{ sc.xg, sc.xu, l.gate, l.up, l.gate_k2, l.up_k2, sc.ids }) |v| g.add(v);
    for ([_]u64{ items, count, sc.members, sc.z }) |v| g.add(v);
    for ([_]usize{ d, w, p, g_sk, rows, slots }) |v| g.add(i32d(v));
    const gu: cuda.Dim3 = .{ .x = @intCast(w / cols), .y = gy, .z = @intCast(2 * g_sk) };
    try cuda.launch.launch(f, .{ .grid = gu, .block = .{ .x = 128 } }, s, &g);

    try ops.gateup(k, s, .{ .z = sc.z, .pick = pick, .svh_g = l.svh_g, .svh_u = l.svh_u, .suh_d = l.suh_d, .xd = sc.xd, .p = i32d(p), .n = i32d(w), .sk = i32d(g_sk), .experts = i32d(l.experts), .gate_cap = caps.gate, .up_cap = caps.up });

    var dn: cuda.Args = .{};
    for ([_]u64{ sc.xd, sc.xd, l.down, l.down, l.down_k2, l.down_k2, sc.ids }) |v| dn.add(v);
    for ([_]u64{ items, count, sc.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ w, d, p, d_sk, rows, slots }) |v| dn.add(i32d(v));
    const down: cuda.Dim3 = .{ .x = @intCast(d / cols), .y = gy, .z = @intCast(d_sk) };
    try cuda.launch.launch(f, .{ .grid = down, .block = .{ .x = 128 } }, s, &dn);

    var cb: cuda.Args = .{};
    for ([_]u64{ sc.z, pick, l.svh_d, sc.y, wts, out }) |v| cb.add(v);
    for ([_]usize{ p, d, d_sk, l.experts, slots }) |v| cb.add(i32d(v));
    const comb: cuda.Dim3 = .{ .x = @intCast(rows), .y = @intCast(d / 128) };
    try cuda.launch.launch(e.main.down_combine, .{ .grid = comb, .block = .{ .x = @intCast(32 * slots) } }, s, &cb);
}

test "experts: Inkling's expert shapes have built tiles (gate/up 4608 -> 2304 whole and a rank's 1152, down back)" {
    try std.testing.expectEqual(@as(u32, 4), try ex.plan(4608, 2304, true));
    try std.testing.expectEqual(@as(u32, 4), try ex.plan(4608, 1152, true));
    _ = try ex.plan(2304, 4608, false);
    _ = try ex.plan(1152, 4608, false);
}

test "experts: the item list sits past main's scratch and holds a 2,048-row call's items" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const bytes = try scratchBytes(l, 2048, 8);
    const sc = try ex.scratch(0, l, 2048, 8);
    const at = itemsAt(l, sc);
    try std.testing.expectEqual(@as(u64, try ex.scratchBytes(l, 2048, 8)), at);
    try std.testing.expectEqual(@as(usize, 16384 / 16 + 384), maxItems(l, 2048, 8));
    try std.testing.expect(at + 256 + maxItems(l, 2048, 8) * 12 <= bytes);
    // the kernels' column blocks: the rows kernel 128 columns, the wide one 64, of a rank's 1152 and the 4608 width
    try std.testing.expect(1152 % 128 == 0 and 4608 % 128 == 0);
}

test "experts: run refuses expert widths outside the kernels' 1 to 4 bits on the host" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    for ([_]Bits{ .{ .lo = 1, .hi = 8 }, .{ .lo = 2, .hi = 9 }, .{ .lo = 6, .hi = 4 } }) |b| {
        try std.testing.expectError(error.BadExpertBits, run(undefined, undefined, undefined, l, caps, b, 0, 0, .f16, 0, 0, 4, 0, sc));
    }
}

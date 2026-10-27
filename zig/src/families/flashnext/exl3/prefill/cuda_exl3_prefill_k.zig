//! The Flash Next EXL3 prefill experts (flashnext_exl3_prefill.cu): 64-member transposed prompt tiles on
const std = @import("std");
const cuda = @import("cuda");

const exl3e = cuda.exl3_experts;

pub const tile: usize = 64;

pub const Kernels = struct {
    mod: cuda.Module,
    rotate: cuda.Function,
    gu: cuda.Function,
    down: cuda.Function,
    /// Expert parallel: a rank's combined branch over its own slots, and the ranks' sum placed for the write-back.
    tp_combine: cuda.Function,
    tp_place: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_prefill);
        errdefer mod.unload();
        return .{ .mod = mod, .rotate = try mod.function("tf_exl3pf_rotate"), .gu = try mod.function("tf_exl3pf_gu"), .down = try mod.function("tf_exl3pf_down"), .tp_combine = try mod.function("tf_exl3pf_tp_combine"), .tp_place = try mod.function("tf_exl3pf_tp_place") };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

/// Members and items of one pass, host side (int32, the layouts the kernels read).
pub const Plan = struct {
    members: []i32,
    items: []i32, // 3 a item
    n_members: usize = 0,
    n_items: usize = 0,

    /// Room for `pairs` members and their items over `experts` table entries.
    pub fn init(gpa: std.mem.Allocator, pairs: usize, experts: usize) !Plan {
        const members = try gpa.alloc(i32, pairs);
        errdefer gpa.free(members);
        return .{ .members = members, .items = try gpa.alloc(i32, 3 * maxItems(pairs, experts)) };
    }

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.members);
        gpa.free(p.items);
        p.* = undefined;
    }

    pub fn itemBytes(p: *const Plan) []const u8 {
        return std.mem.sliceAsBytes(p.items[0 .. 3 * p.n_items]);
    }

    pub fn memberBytes(p: *const Plan) []const u8 {
        return std.mem.sliceAsBytes(p.members[0..p.n_members]);
    }
};

pub fn maxItems(pairs: usize, experts: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

/// Counting sort of `picks` (int32 [pairs], -1 or ids outside the table skipped) by expert, keeping only experts with
pub fn build(picks: []const i32, take: ?[]const bool, counts: []u32, p: *Plan) void {
    @memset(counts, 0);
    const e_n = counts.len;
    for (picks) |pk| {
        if (pk < 0 or @as(usize, @intCast(pk)) >= e_n) continue;
        const id: usize = @intCast(pk);
        if (take) |t| if (!t[id]) continue;
        counts[id] += 1;
    }
    // exclusive prefix into counts (the next free member slot of each expert), items as runs of `tile`
    var at: u32 = 0;
    var items: usize = 0;
    for (counts, 0..) |*c, id| {
        const n = c.*;
        c.* = at;
        var off: u32 = 0;
        while (off < n) : (off += tile) {
            p.items[3 * items] = @intCast(id);
            p.items[3 * items + 1] = @intCast(at + off);
            p.items[3 * items + 2] = @intCast(@min(@as(u32, tile), n - off));
            items += 1;
        }
        at += n;
    }
    for (picks, 0..) |pk, i| {
        if (pk < 0 or @as(usize, @intCast(pk)) >= e_n) continue;
        const id: usize = @intCast(pk);
        if (take) |t| if (!t[id]) continue;
        p.members[counts[id]] = @intCast(i);
        counts[id] += 1;
    }
    p.n_members = at;
    p.n_items = items;
}

/// Device scratch a pass of `members` members needs: xg, xu [members, dims] and xd [members, width] fp16, 256-aligned.
pub fn scratchBytes(members: usize, dims: usize, width: usize) usize {
    return 2 * align256(members * dims * 2) + align256(members * width * 2);
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

pub const Device = struct {
    /// Device copies of the plan (members, items) and the compact scratch base (scratchBytes of the plan's members).
    members: u64,
    items: u64,
    scratch: u64,
};

/// One pass: rotate every member's input, gate/up + SwiGLU, down into y [pairs, dims] fp32 (y[p] for member p).
pub fn run(k: *const Kernels, s: cuda.Stream, l: exl3e.Layer, x: u64, stride: usize, picks: u64, slots: usize, n_members: usize, n_items: usize, dv: Device, y: u64) !void {
    if (n_members == 0) return;
    const d = l.dims;
    const w = l.width;
    if (d % 256 != 0 or w % 128 != 0 or (l.act_mode != 1 and l.act_mode != 0)) return error.Exl3Shape;
    const xg = dv.scratch;
    const xu = xg + align256(n_members * d * 2);
    const xd = xu + align256(n_members * d * 2);
    var r: cuda.Args = .{};
    r.add(x);
    r.add(@as(i32, @intCast(stride)));
    r.add(picks);
    r.add(dv.members);
    r.add(@as(i32, @intCast(slots)));
    r.add(l.suh_g);
    r.add(l.suh_u);
    r.add(xg);
    r.add(xu);
    r.add(@as(i32, @intCast(d)));
    try cuda.launch.launch(k.rotate, .{ .grid = .{ .x = @intCast(n_members), .y = @intCast(d / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &r);
    var g: cuda.Args = .{};
    g.add(xg);
    g.add(xu);
    g.add(l.gate);
    g.add(l.up);
    g.add(l.gate_k2);
    g.add(dv.items);
    g.add(l.svh_g);
    g.add(l.svh_u);
    g.add(l.suh_d);
    g.add(xd);
    g.add(@as(i32, @intCast(d)));
    g.add(@as(i32, @intCast(w)));
    g.add(l.limit);
    g.add(l.act_mode);
    try cuda.launch.launch(k.gu, .{ .grid = .{ .x = @intCast(w / 128), .y = @intCast(n_items) }, .block = .{ .x = 256 } }, s, &g);
    var dn: cuda.Args = .{};
    dn.add(xd);
    dn.add(l.down);
    dn.add(l.down_k2);
    dn.add(dv.items);
    dn.add(dv.members);
    dn.add(l.svh_d);
    dn.add(y);
    dn.add(@as(i32, @intCast(w)));
    dn.add(@as(i32, @intCast(d)));
    try cuda.launch.launch(k.down, .{ .grid = .{ .x = @intCast(d / 256), .y = @intCast(n_items) }, .block = .{ .x = 256 } }, s, &dn);
}

/// Expert parallel: sum[r] = the fp32 fmaf chain over row r's slots whose pick is in [lo, hi), in slot order.
pub fn tpCombine(k: *const Kernels, s: cuda.Stream, y: u64, wts: u64, picks: u64, sum: u64, rows: usize, slots: usize, d: usize, lo: usize, hi: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(y);
    a.add(wts);
    a.add(picks);
    a.add(sum);
    a.add(@as(i32, @intCast(slots)));
    a.add(@as(i32, @intCast(d)));
    a.add(@as(i32, @intCast(lo)));
    a.add(@as(i32, @intCast(hi)));
    try cuda.launch.launch(k.tp_combine, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
}

/// Expert parallel: y[r, 0] = sum[r] and weights {1, 0, ...} (the write-back then writes the sum's bf16).
pub fn tpPlace(k: *const Kernels, s: cuda.Stream, sum: u64, y: u64, wts: u64, rows: usize, slots: usize, d: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(sum);
    a.add(y);
    a.add(wts);
    a.add(@as(i32, @intCast(slots)));
    a.add(@as(i32, @intCast(d)));
    try cuda.launch.launch(k.tp_place, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
}

test "exl3: prefill plan groups pairs by expert into items of at most 64" {
    const gpa = std.testing.allocator;
    // 3 rows of 3 slots: experts 2, 5, -1 / 5, 2, 7 / 2, 2, 513 (out of a 9-entry table: skipped)
    const picks = [_]i32{ 2, 5, -1, 5, 2, 7, 2, 2, 513 };
    var counts: [9]u32 = undefined;
    var p = try Plan.init(gpa, picks.len, 9);
    defer p.deinit(gpa);
    build(&picks, null, &counts, &p);
    try std.testing.expectEqual(@as(usize, 7), p.n_members);
    try std.testing.expectEqualSlices(i32, &.{ 0, 4, 6, 7, 1, 3, 5 }, p.members[0..7]);
    try std.testing.expectEqual(@as(usize, 3), p.n_items);
    try std.testing.expectEqualSlices(i32, &.{ 2, 0, 4, 5, 4, 2, 7, 6, 1 }, p.items[0..9]);
    // a pass of expert 5 alone
    var take: [9]bool = @splat(false);
    take[5] = true;
    build(&picks, &take, &counts, &p);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3 }, p.members[0..p.n_members]);
    try std.testing.expectEqualSlices(i32, &.{ 5, 0, 2 }, p.items[0 .. 3 * p.n_items]);
    // 150 pairs of one expert: items of 64, 64, 22
    var many: [150]i32 = @splat(1);
    var p2 = try Plan.init(gpa, many.len, 9);
    defer p2.deinit(gpa);
    many[0] = -1;
    build(&many, null, &counts, &p2);
    try std.testing.expectEqual(@as(usize, 149), p2.n_members);
    try std.testing.expectEqualSlices(i32, &.{ 1, 0, 64, 1, 64, 64, 1, 128, 21 }, p2.items[0 .. 3 * p2.n_items]);
}

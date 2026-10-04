//! Tree windows: each row's ancestors by depth, attention by logical key position, the kept path's cache rows.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const fwd = @import("forward.zig");

const Buffer = mtl.Buffer;
const Enc = fwd.Enc;
const Forward = fwd.Forward;

/// Ancestors kept a row (the tail kernel's MAXD).
pub const max_depth = 64;

/// Each row's depth and its ancestors' rows by depth into the scratch's tables; the deepest row's depth.
pub fn tables(s: *st.Scratch, parents: []const i32) usize {
    const depths = s.tdepths.slice(i32, st.max_rows);
    const paths = s.tpaths.slice(i32, st.max_rows * max_depth);
    var deepest: usize = 0;
    for (parents, 0..) |p, r| {
        std.debug.assert(p < @as(i32, @intCast(r)));
        const d: usize = if (p < 0) 0 else @as(usize, @intCast(depths[@intCast(p)])) + 1;
        std.debug.assert(d < max_depth);
        depths[r] = @intCast(d);
        if (p >= 0) @memcpy(paths[r * max_depth ..][0..d], paths[@as(usize, @intCast(p)) * max_depth ..][0..d]);
        paths[r * max_depth + d] = @intCast(r);
        deepest = @max(deepest, d);
    }
    return deepest;
}

/// A tree's attention: one shared pass over whole tiles before the window, each row's path tail, merged in chunk order.
pub fn attention(f: Forward, e: *Enc, kv: st.Kv, len: usize, rows: usize, deepest: usize, partial: mtl.Pipeline, first: usize) void {
    const c = f.c;
    const s = f.s;
    const g = c.heads / c.kv_heads;
    const hd = c.head_dim;
    const tile = 64;
    const chunk = 512;
    const pt = len / tile * tile;
    const nca = (pt + chunk - 1) / chunk;
    const ncb = (len + deepest) / chunk - pt / chunk + 1;
    const strides = [4]i64{ @intCast(c.kv_heads * kv.capacity * hd), @intCast(kv.capacity * hd), @intCast(hd), 1 };
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
    if (pt > 0) {
        const tiles: usize = @min(rows, 16);
        e.pipe(partial);
        e.buf(s.qp, 0, 0);
        e.buf(kv.k, 0, 1);
        e.bytes(strides, 2);
        e.buf(kv.v, 0, 3);
        e.bytes(strides, 4);
        e.bytes(scale, 5);
        e.bytes([5]i32{ @intCast(pt), @intCast(nca), @intCast(rows), 0, @intCast(g * rows / 16) }, 6);
        e.buf(s.po, 0, 7);
        e.buf(s.pm, 0, 8);
        e.buf(s.pl, 0, 9);
        e.run(.{ c.kv_heads * 32 * tiles, nca, (rows + tiles - 1) / tiles }, .{ 32 * tiles, 1, 1 });
    }
    const dims = [5]i32{ @intCast(len), @intCast(pt), @intCast(nca), @intCast(ncb), @intCast(rows) };
    e.pipe(f.k.get("tf_tree_tail"));
    e.buf(s.qp, 0, 0);
    e.buf(kv.k, 0, 1);
    e.bytes(strides, 2);
    e.buf(kv.v, 0, 3);
    e.bytes(strides, 4);
    e.bytes(scale, 5);
    e.bytes(dims, 6);
    e.buf(s.tpaths, first * max_depth * 4, 7);
    e.buf(s.tdepths, first * 4, 8);
    e.buf(s.po, 0, 9);
    e.buf(s.pm, 0, 10);
    e.buf(s.pl, 0, 11);
    e.buf(s.tpo, 0, 12);
    e.buf(s.tpm, 0, 13);
    e.buf(s.tpl, 0, 14);
    e.run(.{ c.kv_heads * 32, ncb, rows }, .{ 32, 1, 1 });

    e.pipe(f.k.get("tf_tree_merge"));
    e.buf(s.po, 0, 0);
    e.buf(s.pm, 0, 1);
    e.buf(s.pl, 0, 2);
    e.buf(s.tpo, 0, 3);
    e.buf(s.tpm, 0, 4);
    e.buf(s.tpl, 0, 5);
    e.bytes(dims, 6);
    e.buf(s.att, 0, 7);
    e.run(.{ c.kv_heads * 32, g * rows, 1 }, .{ 32, 1, 1 });
}

/// A GPU round's tree: its dims at byte offsets in `args`, chunks at most; `dual`: chain and tree paths, one gated off.
pub const GpuTree = struct { args: Buffer, tp: usize, tt: usize, nca: usize, ncb: usize, dual: ?Dual = null };

/// Gated threadgroup counts for the two paths: layer a's chain path at entries 8a, 8a + 1, its tree path from 8a + 2.
pub const Dual = struct {
    live: Buffer,
    tmpl: []u32, // this round's recorded counts (3 u32 an entry)

    pub fn gate(d: Dual, layer: usize, path: usize) Enc.Gate {
        const base = 8 * layer + 2 * path;
        return .{ .live = d.live, .base = base, .tmpl = d.tmpl[3 * base ..] };
    }
};

/// attention() with the window's tables (scratch.tpaths, tdepths) and dims written on the GPU.
pub fn attentionGpu(f: Forward, e: *Enc, kv: st.Kv, rows: usize, partial: mtl.Pipeline, gt: GpuTree) void {
    const c = f.c;
    const s = f.s;
    const g = c.heads / c.kv_heads;
    const hd = c.head_dim;
    const strides = [4]i64{ @intCast(c.kv_heads * kv.capacity * hd), @intCast(kv.capacity * hd), @intCast(hd), 1 };
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
    const tiles: usize = @min(rows, 16);
    e.pipe(partial);
    e.buf(s.qp, 0, 0);
    e.buf(kv.k, 0, 1);
    e.bytes(strides, 2);
    e.buf(kv.v, 0, 3);
    e.bytes(strides, 4);
    e.bytes(scale, 5);
    e.buf(gt.args, gt.tp, 6);
    e.buf(s.po, 0, 7);
    e.buf(s.pm, 0, 8);
    e.buf(s.pl, 0, 9);
    e.run(.{ c.kv_heads * 32 * tiles, gt.nca, (rows + tiles - 1) / tiles }, .{ 32 * tiles, 1, 1 });
    e.pipe(f.k.get("tf_tree_tail"));
    e.buf(s.qp, 0, 0);
    e.buf(kv.k, 0, 1);
    e.bytes(strides, 2);
    e.buf(kv.v, 0, 3);
    e.bytes(strides, 4);
    e.bytes(scale, 5);
    e.buf(gt.args, gt.tt, 6);
    e.buf(s.tpaths, 0, 7);
    e.buf(s.tdepths, 0, 8);
    e.buf(s.po, 0, 9);
    e.buf(s.pm, 0, 10);
    e.buf(s.pl, 0, 11);
    e.buf(s.tpo, 0, 12);
    e.buf(s.tpm, 0, 13);
    e.buf(s.tpl, 0, 14);
    e.run(.{ c.kv_heads * 32, gt.ncb, rows }, .{ 32, 1, 1 });
    e.pipe(f.k.get("tf_tree_merge"));
    e.buf(s.po, 0, 0);
    e.buf(s.pm, 0, 1);
    e.buf(s.pl, 0, 2);
    e.buf(s.tpo, 0, 3);
    e.buf(s.tpm, 0, 4);
    e.buf(s.tpl, 0, 5);
    e.buf(gt.args, gt.tt, 6);
    e.buf(s.att, 0, 7);
    e.run(.{ c.kv_heads * 32, g * rows, 1 }, .{ 32, 1, 1 });
}

/// A kept tree path's key and value rows moved to the cache's next rows (row len + path[j] to len + j).
pub fn compact(f: Forward, e: *Enc, kv: st.Kv, len: usize, path: []const u32) void {
    var rows: [st.max_rows]i32 = @splat(0);
    for (path, 0..) |r, j| rows[j] = @intCast(r);
    e.pipe(f.k.get("tf_kv_compact"));
    e.buf(kv.k, 0, 0);
    e.buf(kv.v, 0, 1);
    e.e.setBytes(std.mem.sliceAsBytes(rows[0..@max(path.len, 8)]), 2);
    e.bytes([4]u32{ @intCast(len), @intCast(path.len), @intCast(kv.capacity), @intCast(f.c.head_dim) }, 3);
    e.run(.{ f.c.head_dim, f.c.kv_heads, 1 }, .{ f.c.head_dim, 1, 1 });
}

/// Rows of D bf16 gathered by index into `dst` (dst row j = src row from + rows[j]).
pub fn gather(f: Forward, e: *Enc, src: Buffer, from: usize, rows: []const u32, dst: Buffer) void {
    var idx: [st.max_rows]i32 = @splat(0);
    for (rows, 0..) |r, j| idx[j] = @intCast(from + r);
    e.pipe(f.k.get("tf_gather_rows"));
    e.buf(src, 0, 0);
    e.buf(dst, 0, 1);
    e.e.setBytes(std.mem.sliceAsBytes(idx[0..@max(rows.len, 8)]), 2);
    e.run(.{ f.c.hidden, rows.len, 1 }, .{ 256, 1, 1 });
}

test "tree tables: depths and ancestors by depth" {
    const parents = [_]i32{ -1, 0, 0, 1, 3, 2 };
    var depths: [6]usize = undefined;
    var at: [6][8]i32 = undefined;
    for (parents, 0..) |p, r| {
        depths[r] = if (p < 0) 0 else depths[@intCast(p)] + 1;
        if (p >= 0) @memcpy(at[r][0..depths[r]], at[@intCast(p)][0..depths[r]]);
        at[r][depths[r]] = @intCast(r);
    }
    try std.testing.expectEqual(@as(usize, 3), depths[4]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 3, 4 }, at[4][0..4]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 2, 5 }, at[5][0..3]);
}

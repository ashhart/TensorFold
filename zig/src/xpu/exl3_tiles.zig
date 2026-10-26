// SPDX-License-Identifier: Apache-2.0

//! Prompt-row routed MLP via exl3_moe_sort, exl3_moe_rot, and exl3_moe_tile.

const std = @import("std");
const ex = @import("exl3.zig");
const launch = @import("launch.zig").launch;
const variants = @import("exl3_variants.zig");

pub const MoeTiles = struct {
    route: ex.MoeRoute,
    order: u64,
    tiles: u64,
    ntiles: u64,
};

pub fn tileBound(slots: u32, tables: u32) u32 {
    return slots / variants.moe.tile_rows + tables;
}

const TileScratch = struct { rot: usize, gu: usize, act: usize, size: usize };

fn tileScratch(hidden: usize, ff: usize, slots: usize) TileScratch {
    const rot = std.mem.alignForward(usize, 2 * slots * hidden * 4, 256);
    const gu = std.mem.alignForward(usize, slots * 2 * ff * 4, 256);
    return .{ .rot = 0, .gu = rot, .act = rot + gu, .size = rot + gu + slots * ff * 4 };
}

pub fn moeTileScratchBytes(hidden: usize, ff: usize, slots: usize) usize {
    return tileScratch(hidden, ff, slots).size;
}

fn setU64(k: @import("module.zig").Kernel, i: u32, v: u64) ex.Error!void {
    try k.setBytes(i, std.mem.asBytes(&v));
}

pub fn encodeMoeTiles(p: *const ex.Pipelines, s: ex.Stream, gate: *const ex.Plan, up: *const ex.Plan, down: *const ex.Plan, t: MoeTiles) ex.Error!void {
    const r = t.route;
    const hidden = gate.k_max;
    const ff = gate.n_max;
    const m = variants.moe;
    if (up.k_max != hidden or up.n_max != ff or down.k_max != ff or down.n_max != hidden) return error.Invalid;
    if (gate.count != up.count or gate.count != down.count or gate.bits != up.bits or gate.bits != down.bits) return error.Invalid;
    if (hidden % 128 != 0 or ff % 128 != 0 or ff % (16 * m.tile_ns) != 0 or hidden % (16 * m.tile_ns) != 0) return error.Invalid;
    if (r.shared) |e| if (e >= gate.count) return error.Invalid;
    if (r.x_stride < hidden or r.y_stride < hidden) return error.Invalid;
    const mi = std.mem.indexOfScalar(u32, &m.bits, gate.bits) orelse return error.Invalid;
    const k = p.moe[mi];
    const slots = r.slots();
    if (slots == 0) return error.Invalid;
    const pairs = r.tokens * r.topk;
    const shared = r.shared orelse 0;
    const sc = tileScratch(hidden, ff, slots);
    const ids: ?*anyopaque = @ptrFromInt(r.ids);
    const order: ?*anyopaque = @ptrFromInt(t.order);
    const tiles: ?*anyopaque = @ptrFromInt(t.tiles);
    const ntiles: ?*anyopaque = @ptrFromInt(t.ntiles);
    const tile_bound = tileBound(slots, gate.count);

    try k.sort.setPtr(0, ids);
    try k.sort.setPtr(1, order);
    try k.sort.setPtr(2, tiles);
    try k.sort.setPtr(3, ntiles);
    try k.sort.setU32(4, if (r.shared) |e| e else gate.count);
    try k.sort.setU32(5, pairs);
    try k.sort.setU32(6, if (r.shared != null) r.tokens else 0);
    try launch(k.sort, s, .{ 1, 1, 1 });

    const tile_x = struct {
        fn f(n: u32) u32 {
            return std.math.divCeil(u32, n / 16 / m.tile_ns, 8) catch unreachable;
        }
    }.f;

    try k.rot.setPtr(0, null);
    try k.rot.setPtr(1, gate.table.ptr);
    try k.rot.setPtr(2, up.table.ptr);
    try k.rot.setPtr(3, ids);
    try k.rot.setU32(4, r.topk);
    try k.rot.setU32(5, pairs);
    try k.rot.setU32(6, shared);
    try setU64(k.rot, 7, r.x);
    try k.rot.setU32(8, r.x_stride);
    try k.rot.setU32(9, @backingInt(r.input));
    try setU64(k.rot, 10, r.scratch + sc.rot);
    try k.rot.setU32(11, slots);
    try launch(k.rot, s, .{ hidden / 128, slots, 2 });

    try k.tile.setPtr(0, null);
    try k.tile.setPtr(1, gate.table.ptr);
    try k.tile.setPtr(2, up.table.ptr);
    try k.tile.setPtr(3, order);
    try k.tile.setPtr(4, tiles);
    try k.tile.setPtr(5, ntiles);
    try setU64(k.tile, 6, r.scratch + sc.rot);
    try setU64(k.tile, 7, @as(u64, slots) * hidden);
    try setU64(k.tile, 8, r.scratch + sc.gu);
    try k.tile.setU32(9, 2 * ff);
    try k.tile.setU32(10, ff);
    try launch(k.tile, s, .{ tile_x(ff), tile_bound, 2 });

    try k.act.setPtr(0, null);
    try k.act.setPtr(1, gate.table.ptr);
    try k.act.setPtr(2, up.table.ptr);
    try k.act.setPtr(3, down.table.ptr);
    try k.act.setPtr(4, ids);
    try k.act.setU32(5, pairs);
    try k.act.setU32(6, shared);
    try setU64(k.act, 7, r.scratch + sc.gu);
    try setU64(k.act, 8, r.scratch + sc.act);
    try k.act.setU32(9, 1);
    try launch(k.act, s, .{ ff / 128, slots, 1 });

    try k.tile.setPtr(1, down.table.ptr);
    try k.tile.setPtr(2, down.table.ptr);
    try setU64(k.tile, 6, r.scratch + sc.act);
    try setU64(k.tile, 7, 0);
    try setU64(k.tile, 8, r.scratch + sc.rot);
    try k.tile.setU32(9, hidden);
    try k.tile.setU32(10, 0);
    try launch(k.tile, s, .{ tile_x(hidden), tile_bound, 1 });

    try k.out_rows.setPtr(0, null);
    try k.out_rows.setPtr(1, down.table.ptr);
    try k.out_rows.setPtr(2, ids);
    try k.out_rows.setU32(3, pairs);
    try k.out_rows.setU32(4, shared);
    try setU64(k.out_rows, 5, r.scratch + sc.rot);
    try setU64(k.out_rows, 6, r.y);
    try k.out_rows.setU32(7, r.y_stride);
    try k.out_rows.setU32(8, @backingInt(r.output));
    try launch(k.out_rows, s, .{ hidden / 128, slots, 1 });
}

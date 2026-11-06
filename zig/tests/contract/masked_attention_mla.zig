//! The scalar latent-attention seam, including shared K/V and the production split/merge arguments.
const std = @import("std");
const metal = @import("metal");
const rig = @import("rig.zig");
const c = @import("contract").spec;
const KimiArgs = extern struct { cache: u64, slot_keys: u32, heads: u32, head0: u32 = 0, rows: u32, q_stride: u32, kv_stride: u32 = 576, gate_stride: u32, out_stride: u32, eps: f32 = 1e-6, scale: f32 = 0.0625 };
const DsaArgs = extern struct { rows: u32, heads: u32, head0: u32 = 0, slot_keys: u32, proj_stride: u32 = 0, q_stride: u32 = 0, pad2: u32 = 0, kvb_in_blocks: u32 = 4, eps: f32 = 1e-6, scale: f32 = 0.0625, pad0: f32 = 0, pad1: f32 = 0, kvb: u64 = 0, kvb_scale: u64 = 0 };
pub fn prepare(p: anytype) !void {
    const a = p.arena.?.allocator();
    const d = p.fixture;
    const g = d.geo;
    const cache = try a.alloc(u16, g.cap * 576);
    @memset(cache, 0);
    for (0..g.cap) |j| @memcpy(cache[j * 576 ..][0..512], d.v[j * 512 ..][0..512]);
    p.aux[2] = try p.rig.input(std.mem.sliceAsBytes(cache));
    const qlat = try a.alloc(f32, d.rows * g.heads * 512);
    @memset(qlat, 0);
    p.aux[4] = try p.rig.input(std.mem.sliceAsBytes(qlat));
    p.aux[3] = try p.rig.scratch(d.rows * g.heads * 16 * 514 * 4);
    if (p.kind == .kimi) {
        const rows = try a.alloc([2]u32, d.rows);
        for (rows, 0..) |*r, i| r.* = .{ 0, @intCast(g.first + i) };
        p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(rows));
    } else {
        const rows = try a.alloc([4]u32, d.rows);
        const keys = try a.alloc(u32, d.rows * 64);
        @memset(keys, 0);
        for (rows, 0..) |*r, i| {
            r.* = .{ 0, @intCast(g.first + i), @intCast(i * 64), @intCast(g.first + i + 1) };
            for (0..g.first + i + 1) |j| keys[i * 64 + j] = @intCast(j);
        }
        p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(rows));
        p.aux[1] = try p.rig.input(std.mem.sliceAsBytes(keys));
        const qr = try a.alloc(f32, d.rows * g.heads * 64);
        @memset(qr, 0);
        p.aux[5] = try p.rig.input(std.mem.sliceAsBytes(qr));
    }
}
fn bind(e: metal.ComputeEncoder, ref: rig.Ref, i: usize) void {
    e.setBuffer(ref.buf, ref.off, i);
}
pub fn encode(p: anytype, e: metal.ComputeEncoder) void {
    const d = p.fixture;
    const g = d.geo;
    e.setPipeline(p.pipes[0].?);
    if (p.kind == .kimi) {
        e.useResource(p.aux[2].buf, 1);
        const args = KimiArgs{ .cache = p.aux[2].buf.gpuAddress() + p.aux[2].off, .slot_keys = @intCast(g.cap), .heads = @intCast(g.heads), .rows = @intCast(d.rows), .q_stride = @intCast(g.heads * 192), .gate_stride = @intCast(g.heads * 128), .out_stride = @intCast(g.heads * 128) };
        bind(e, p.q, 0);
        bind(e, p.aux[4], 1);
        bind(e, p.aux[0], 2);
        bind(e, p.aux[3], 3);
        e.setValue(args, 4);
        const fit = @min(@as(usize, 32), p.pipes[0].?.maxThreads() / 32);
        const groups = (g.heads + fit - 1) / fit;
        const per = (g.heads + groups - 1) / groups;
        e.dispatchGroups(.{ .width = groups, .height = 16, .depth = d.rows }, .{ .width = 32 * per });
        e.barrier();
        e.setPipeline(p.pipes[1].?);
        bind(e, p.aux[3], 0);
        bind(e, p.aux[0], 1);
        bind(e, p.out, 2);
        e.setValue(args, 3);
        e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 256 });
    } else {
        const args = DsaArgs{ .rows = @intCast(d.rows), .heads = @intCast(g.heads), .slot_keys = @intCast(g.cap) };
        for ([_]rig.Ref{ p.aux[4], p.aux[5], p.aux[0], p.aux[1], p.aux[2] }, 0..) |r, i| bind(e, r, i);
        e.setValue(args, 5);
        bind(e, p.aux[3], 6);
        e.dispatchGroups(.{ .width = (g.heads + 3) / 4, .height = 16, .depth = d.rows }, .{ .width = 128 });
        e.barrier();
        e.setPipeline(p.pipes[1].?);
        bind(e, p.aux[3], 0);
        bind(e, p.aux[0], 1);
        e.setValue(args, 2);
        bind(e, p.out, 3);
        e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 256 });
    }
}

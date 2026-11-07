//! Actual sparse lists, prompt cuts and page mappings used by the independent masked-value panels.
const std = @import("std");
const metal = @import("metal");
const rig = @import("rig.zig");
const S5Args = extern struct { rows: u32, pos: u32, heads: u32, dim: u32, span: u32 = 0, ring: u32, window: u32, regions: u32 = 0, groups: u32, topk: u32 = 512, first: u32 = 0, norm: u32 = 0, eps: f32 = 1e-5, theta: f32 = 10000, scale: f32 };
const ContextArgs = extern struct { rows: u32, keys: u32, mask_stride: u32, heads: u32, scale_bits: u32 };
const L3Args = extern struct { len: u32, cap: u32, scale: f32, window: u32, chunk_base: u32 = 0 };
const ct = @import("contract");
fn bind(e: metal.ComputeEncoder, r: rig.Ref, i: usize) void {
    e.setBuffer(r.buf, r.off, i);
}
pub fn prepare(p: anytype) !void {
    const a = p.arena.?.allocator();
    const d = p.fixture;
    const g = d.geo;
    switch (p.kind) {
        .glm_dense => for (0..2) |i| {
            p.aux[i] = try p.rig.scratch(g.heads * (g.first + d.rows) * 2);
        },
        .glm_nax64, .glm_nax32, .glm_padding, .glm_split, .glm_indexed64, .glm_indexed32 => {
            const ids = try a.alloc(i32, d.rows * 2051);
            @memset(ids, -1);
            for (0..d.rows) |r| for (0..g.first + r + 1) |j| {
                if (g.skip_zero_first and r == 0) {
                    if (j < g.first) ids[r * 2051 + j] = @intCast(j + 1);
                } else ids[r * 2051 + j] = @intCast(j);
            };
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(ids));
            if (p.kind == .glm_split) {
                p.aux[1] = try p.rig.scratch(d.rows * 17 * g.heads * g.dim * 4);
                for (2..4) |i| p.aux[i] = try p.rig.scratch(d.rows * 17 * g.heads * 4);
            }
        },
        .step_full_nax, .step_sliding_nax, .step_sparse_nax, .step_full_scalar, .step_sliding_scalar, .step_sparse_scalar, .step_padding_nax, .step_padding_scalar => {
            const ids = try a.alloc(i32, d.rows * g.kv * 512);
            for (0..d.rows * g.kv) |r| for (0..512) |j| {
                ids[r * 512 + j] = @intCast(j + (if (r < g.kv and g.selected_shift_first > 0) @as(usize, 1) else 0));
            };
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(ids));
        },
        .qwen4_dense, .qwen4_sparse => {
            const ids = try a.alloc(i32, d.rows * 1026);
            @memset(ids, -1);
            const nk = try a.alloc(i32, d.rows);
            const spr = try a.alloc(i32, d.rows);
            for (0..d.rows) |r| {
                nk[r] = @intCast(g.first + r + 1);
                spr[r] = @intFromBool(p.kind == .qwen4_sparse);
                for (0..g.first + r + 1) |j| ids[r * 1026 + j] = @intCast(j);
            }
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(ids));
            p.aux[1] = try p.rig.input(std.mem.sliceAsBytes(nk));
            p.aux[2] = try p.rig.input(std.mem.sliceAsBytes(spr));
            p.aux[3] = try p.rig.scratch(d.rows * g.heads * 16 * g.dim * 4);
            p.aux[4] = try p.rig.scratch(d.rows * g.heads * 16 * 2 * 4);
        },
        .drama => {
            const zeros = try a.alloc(u16, d.rows * g.heads * g.dim);
            @memset(zeros, 0);
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(zeros));
        },
        .shared, .shared_split => {
            const keys = g.first + d.rows;
            const mask = try a.alloc(u8, d.rows * keys);
            for (0..d.rows) |r| for (0..keys) |j| {
                mask[r * keys + j] = @intFromBool(j <= g.first + r);
            };
            p.aux[0] = try p.rig.input(mask);
            if (p.kind == .shared_split) p.aux[1] = try p.rig.scratch(d.rows * g.heads * (g.dim + 2) * 4);
        },
        .qwen35_pages => {
            const dims = [_]i32{ @intCast(g.first), @intCast(d.rows), @intCast(g.cap), 1, 16, @intCast(d.rows), 0 };
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(&dims));
            for (1..4) |i| p.aux[i] = try p.rig.scratch(d.rows * g.heads * (if (i == 3) g.dim else 1) * 4);
            const page_words = 256 * g.kv * g.dim;
            const keys = try a.alloc(u16, page_words * 2);
            const vals = try a.alloc(u16, keys.len);
            @memset(keys, 0);
            @memset(vals, ct.spec.toBf16(1));
            const pos = g.first + 1;
            if (d.poisoned) vals[page_words + pos * g.dim + 7] = if (d.poison == .inf) 0x7f80 else 0x7fc0;
            p.k = try p.rig.input(std.mem.sliceAsBytes(keys));
            p.v = try p.rig.input(std.mem.sliceAsBytes(vals));
            p.aux[4] = try p.rig.input(std.mem.asBytes(&@as([1]u32, .{1})));
        },
        .lfm3, .lfm3_sliding => {
            for (0..2) |i| p.aux[i] = try p.rig.scratch(d.rows * g.heads * 64 * 4);
            p.aux[2] = try p.rig.scratch(d.rows * g.heads * 64 * g.dim * 4);
            p.aux[3] = try p.rig.scratch(d.rows * g.heads * 64 * 4);
        },
        .deepseek_pre => {
            const indices = try a.alloc(i32, d.rows * 512);
            @memset(indices, -1);
            const meta = try a.alloc([8]i32, d.rows);
            for (meta, 0..) |*m, r| m.* = .{ @intCast(g.first + r), 0, 0, 0, 0, 0, 0, 0 };
            const sink = try a.alloc(f32, g.heads);
            @memset(sink, 0);
            const cs = try a.alloc(f32, (g.first + d.rows) * 64);
            for (cs, 0..) |*x, i| x.* = if (i % 64 < 32) 1 else 0;
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(indices));
            p.aux[1] = try p.rig.input(std.mem.sliceAsBytes(meta));
            p.aux[2] = try p.rig.input(std.mem.sliceAsBytes(sink));
            p.aux[3] = try p.rig.input(std.mem.sliceAsBytes(cs));
        },
        .h3_padding => {
            const query = try a.alloc(u16, g.cap * g.heads * g.dim);
            @memset(query, 0);
            p.q = try p.rig.input(std.mem.sliceAsBytes(query));
            if (d.poisoned) {
                const bytes = p.rig.bytes(p.v, d.v.len * 2);
                std.mem.writeInt(u16, bytes[(g.cap * g.dim + 7) * 2 ..][0..2], if (d.poison == .inf) 0x7f80 else 0x7fc0, .little);
            }
            const sizes = try a.alloc(i32, 16);
            @memset(sizes, 0);
            sizes[0] = 17;
            const rowof = try a.alloc(i32, g.cap);
            @memset(rowof, -1);
            for (0..d.rows) |r| rowof[r] = @intCast(r);
            p.aux[0] = try p.rig.input(std.mem.asBytes(&@as([1]i32, .{0})));
            p.aux[1] = try p.rig.input(std.mem.sliceAsBytes(sizes));
            p.aux[2] = try p.rig.input(std.mem.sliceAsBytes(rowof));
        },
        else => unreachable,
    }
}

pub fn encode(p: anytype, e: metal.ComputeEncoder) void {
    const d = p.fixture;
    const g = d.geo;
    e.setPipeline(p.pipes[0].?);
    switch (p.kind) {
        .glm_dense => for (0..d.rows) |r| {
            const n = g.first + r + 1;
            e.setPipeline(p.pipes[0].?);
            e.setBuffer(p.q.buf, p.q.off + r * g.heads * g.dim * 2, 0);
            bind(e, p.v, 1);
            bind(e, p.aux[0], 2);
            e.setValue([2]i32{ @intCast(n), 1 }, 3);
            e.dispatchGroups(.{ .width = (n + 31) / 32 }, .{ .width = 2 * g.heads });
            e.barrier();
            e.setPipeline(p.pipes[2].?);
            bind(e, p.aux[0], 0);
            e.setValue(@as(i32, @intCast(n)), 1);
            bind(e, p.aux[1], 2);
            e.dispatchGroups(.{ .width = g.heads }, .{ .width = ((n + 3) / 4 + 31) / 32 * 32 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            bind(e, p.aux[1], 0);
            bind(e, p.v, 1);
            e.setBuffer(p.out.buf, p.out.off + r * g.heads * g.dim * 2, 2);
            e.setValue([2]i32{ @intCast(n), 0 }, 3);
            e.dispatchGroups(.{ .width = g.dim / 32 }, .{ .width = 2 * g.heads });
            e.barrier();
        },
        .glm_nax64, .glm_nax32, .glm_padding, .glm_split, .glm_indexed64, .glm_indexed32 => {
            bind(e, p.q, 0);
            bind(e, p.v, 1);
            bind(e, p.aux[0], 2);
            e.setValue(@as(f32, 0.0625), 3);
            if (p.kind == .glm_indexed64 or p.kind == .glm_indexed32) {
                e.setValue(@as(i32, @intCast(g.first + d.rows)), 4);
                bind(e, p.out, 5);
                e.dispatchThreads(.{ .width = 1024, .height = d.rows * g.heads }, .{ .width = 1024 });
            } else if (p.kind == .glm_split) {
                e.setValue([4]i32{ 2051, @intCast(g.first + d.rows), 128, 0 }, 4);
                for (p.aux[1..4], 5..) |r, i| bind(e, r, i);
                e.dispatchGroups(.{ .width = g.heads / 16, .height = 17, .depth = d.rows }, .{ .width = 128 });
                e.barrier();
                e.setPipeline(p.pipes[1].?);
                for (p.aux[1..4], 0..) |r, i| bind(e, r, i);
                e.setValue(@as(i32, 17), 3);
                bind(e, p.out, 4);
                e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 128 });
            } else {
                e.setValue([4]i32{ 2051, @intCast(g.first + d.rows), @intCast(g.first), 0 }, 4);
                bind(e, p.out, 5);
                e.dispatchGroups(.{ .width = g.heads / 16, .height = d.rows }, .{ .width = 128 });
            }
        },
        .step_full_nax, .step_sliding_nax, .step_sparse_nax, .step_full_scalar, .step_sliding_scalar, .step_sparse_scalar, .step_padding_nax, .step_padding_scalar => {
            for ([_]rig.Ref{ p.q, p.k, p.v, p.aux[0] }, 0..) |r, i| bind(e, r, i);
            const args = S5Args{ .rows = @intCast(d.rows), .pos = @intCast(g.first), .heads = @intCast(g.heads), .dim = @intCast(g.dim), .ring = @intCast(g.ring), .window = @intCast(g.window), .groups = @intCast(g.kv), .scale = 1 / @sqrt(@as(f32, @floatFromInt(g.dim))) };
            e.setValue(args, 4);
            bind(e, p.out, 5);
            const nax = std.mem.endsWith(u8, @tagName(p.kind), "_nax");
            e.dispatchGroups(.{ .width = if (nax) g.kv else g.heads, .height = d.rows }, .{ .width = if (nax) 96 else 256 });
        },
        .qwen4_dense, .qwen4_sparse => {
            for ([_]rig.Ref{ p.q, p.k, p.v, p.aux[0], p.aux[1], p.aux[2] }, [_]usize{ 0, 1, 3, 4, 6, 7 }) |r, i| bind(e, r, i);
            e.setValue([4]i32{ 1, 4, @intCast(g.cap), 256 }, 2);
            e.setValue([2]i32{ @intCast(d.rows), 1026 }, 5);
            e.setValue(@as(f32, 0.0625), 8);
            bind(e, p.aux[3], 9);
            bind(e, p.aux[4], 10);
            e.dispatchGroups(.{ .width = g.heads, .height = d.rows, .depth = 16 }, .{ .width = 256 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            bind(e, p.aux[3], 0);
            bind(e, p.aux[4], 1);
            bind(e, p.q, 2);
            bind(e, p.out, 3);
            e.dispatchGroups(.{ .width = 1, .height = g.heads, .depth = d.rows }, .{ .width = 256 });
        },
        .drama => {
            for ([_]rig.Ref{ p.q, p.k, p.v }, 0..) |r, i| bind(e, r, i);
            e.setValue(@as(i32, @intCast(g.first)), 3);
            bind(e, p.aux[0], 4);
            bind(e, p.out, 5);
            e.setValue([3]u32{ @intCast(g.heads * g.dim), 0, @intCast(g.dim) }, 6);
            e.dispatchGroups(.{ .width = d.rows, .height = g.heads }, .{ .width = 256 });
        },
        .shared, .shared_split => {
            for ([_]rig.Ref{ p.q, p.k, p.v, p.aux[0] }, 0..) |r, i| bind(e, r, i);
            const args = ContextArgs{ .rows = @intCast(d.rows), .keys = @intCast(g.first + d.rows), .mask_stride = @intCast(g.first + d.rows), .heads = @intCast(g.heads), .scale_bits = @bitCast(@as(f32, 0.125)) };
            e.setValue(args, 4);
            bind(e, if (p.kind == .shared) p.out else p.aux[1], 5);
            e.dispatchGroups(.{ .width = if (p.kind == .shared) g.heads else g.kv, .height = d.rows }, .{ .width = 32 });
            if (p.kind == .shared_split) {
                e.barrier();
                e.setPipeline(p.pipes[1].?);
                bind(e, p.aux[1], 0);
                e.setValue(args, 1);
                bind(e, p.out, 2);
                e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 32 });
            }
        },
        .qwen35_pages => {
            for ([_]rig.Ref{ p.q, p.k, p.v }, 0..) |r, i| bind(e, r, i);
            e.setValue(@as(f32, 0.0625), 3);
            bind(e, p.aux[0], 4);
            for (p.aux[1..4], 5..) |r, i| bind(e, r, i);
            bind(e, p.aux[4], 8);
            e.setValue(@as(u64, 256 * 2 * 256 * 2), 9);
            e.dispatchGroups(.{ .width = 1, .height = 1, .depth = g.kv * d.rows }, .{ .width = 512 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            for (p.aux[1..4], 0..) |r, i| bind(e, r, i);
            bind(e, p.aux[0], 3);
            bind(e, p.out, 4);
            e.dispatchGroups(.{ .width = 1, .height = g.heads, .depth = d.rows }, .{ .width = 32 });
        },
        .lfm3, .lfm3_sliding => {
            for ([_]rig.Ref{ p.q, p.k, p.v }, 0..) |r, i| bind(e, r, i);
            const args = L3Args{ .len = @intCast(g.first), .cap = @intCast(g.cap), .scale = 1 / @sqrt(@as(f32, 128)), .window = @intCast(g.window) };
            e.setValue(args, 3);
            for (p.aux[0..4], 4..) |r, i| bind(e, r, i);
            e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 128 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            for (p.aux[0..3], 0..) |r, i| bind(e, r, i);
            e.setValue(args, 3);
            bind(e, p.out, 4);
            bind(e, p.q, 5);
            bind(e, p.k, 6);
            bind(e, p.v, 7);
            bind(e, p.aux[3], 8);
            e.dispatchGroups(.{ .width = g.heads, .height = d.rows }, .{ .width = 32 });
        },
        .deepseek_pre => {
            for ([_]rig.Ref{ p.q, p.v, p.v, p.aux[0], p.aux[1], p.aux[2], p.aux[3], p.out }, 0..) |r, i| bind(e, r, i);
            const Args = extern struct { rows: u32, heads: u32, scale: f32, pad: u32 = 0 };
            e.setValue(Args{ .rows = @intCast(d.rows), .heads = @intCast(g.heads), .scale = 1 / @sqrt(@as(f32, 512)) }, 8);
            e.dispatchGroups(.{ .width = (d.rows + 1) / 2, .height = (g.heads + 3) / 4 }, .{ .width = 256 });
        },
        .h3_padding => {
            for ([_]rig.Ref{ p.q, p.k, p.v, p.aux[0], p.aux[1], p.aux[2] }, 0..) |r, i| bind(e, r, i);
            e.setValue([4]i32{ 1, 1, 0, 0 }, 6);
            e.setValue(@as(f32, 1 / @sqrt(@as(f32, 128))), 7);
            bind(e, p.out, 8);
            e.dispatchGroups(.{ .width = 1, .height = g.heads }, .{ .width = 256 });
        },
        else => unreachable,
    }
}

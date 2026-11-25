//! One FastH3 block in bf16: products on h3_gemm_32x64 (fp32 sums, MLX's K order), routed attention in bf16.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3.zig");

const kernels = @import("h3_kernels");

const Model = h3.Model;
const Ref = h3.Ref;

/// Threadgroups walk 2^swizzle row tiles under each weight tile: 64 keep it cached, where MLX's swizzle 2 walks 4.
const swizzle = 6;

/// d (rows, n) = x (rows, k) w^T for w (n, k) in bf16.
fn gemm(m: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, w: Ref, d: mtl.Buffer, n: usize, k: usize) void {
    const pipe = if (m.rows % 64 == 0) m.pipeline(kernels.gemm_32x64) else m.pipeline(kernels.gemm_32x64_ragged);
    product(enc, pipe, swizzle, x, w, d, m.rows, n, k, 128);
}

fn product(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, s: usize, x: mtl.Buffer, w: Ref, d: mtl.Buffer, rows: usize, n: usize, k: usize, comptime threads: usize) void {
    const band = @as(usize, 1) << @intCast(s);
    const tn = n / 128;
    const tm = (rows + 63) / 64;
    enc.setPipeline(pipe);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w.buffer, w.offset, 1);
    var p: [16]i32 = @splat(0);
    const values = [_]usize{ rows, n, k, k, k, n, tn, tm, s, k / 256 };
    for (values, 0..) |v, i| p[i] = @intCast(v);
    enc.setBytes(std.mem.asBytes(&p), 2);
    enc.setBuffer(d, 0, 3);
    enc.dispatchGroups(mtl.Size.of(tn * band, (tm + band - 1) / band, 1), mtl.Size.of(32, 4, threads / 128));
}

/// The stream's modulated norm to bf16 rows in xn; with `gates` it first adds the branch in `y` gated by that part.
fn norm(m: *const Model, enc: mtl.ComputeEncoder, w: Ref, gates: ?Ref, gate: i32, tables: Ref, scale: i32, shift: i32) void {
    const g = gates orelse tables;
    enc.setPipeline(m.pipeline("h3_norm"));
    enc.setBuffer(m.x, 0, 0);
    enc.setBuffer(m.y, 0, 1);
    enc.setBuffer(w.buffer, w.offset, 2);
    enc.setBuffer(g.buffer, g.offset, 3);
    enc.setBuffer(tables.buffer, tables.offset, 4);
    enc.setBuffer(m.line, 0, 5);
    enc.setValue([6]i32{ @intCast(m.rows), @intCast(m.hidden), @intCast(m.lines), if (gates == null) -1 else gate, scale, shift }, 6);
    enc.setValue([1]f32{h3.eps}, 7);
    enc.setBuffer(m.xn, 0, 8);
    enc.dispatchGroups(mtl.Size.of(m.rows, 1, 1), mtl.Size.of(32, 1, 1));
}

fn attend(m: *const Model, enc: mtl.ComputeEncoder, list: mtl.Buffer, queries: usize, keys: usize, first: usize, per_query: bool) void {
    enc.setPipeline(m.pipeline("h3_attention"));
    enc.setBuffer(m.tq, 0, 0);
    enc.setBuffer(m.tk, 0, 1);
    enc.setBuffer(m.tv, 0, 2);
    enc.setBuffer(list, 0, 3);
    enc.setBuffer(m.sizes.buffer, m.sizes.offset, 4);
    enc.setBuffer(m.row_of, 0, 5);
    enc.setValue([4]i32{ @intCast(queries), @intCast(keys), @intCast(first), @intFromBool(per_query) }, 6);
    enc.setValue([1]f32{1.0 / @sqrt(@as(f32, h3.head_dim))}, 7);
    enc.setBuffer(m.qr, 0, 8);
    enc.dispatchGroups(mtl.Size.of(queries, m.heads, 1), mtl.Size.of(256, 1, 1));
}

/// Block `index` of a forward; the block before left its MLP's rows in y, added here with this block's first norm.
pub fn run(m: *Model, pass: *h3.Pass, index: usize) void {
    const b = &m.blocks[index];
    const enc = &pass.enc;
    const heads = m.heads;
    const gates: ?Ref = if (index == 0) null else m.blocks[index - 1].tables;
    norm(m, enc.*, b.norm1, gates, h3.gate_m, b.tables, h3.scale_a, h3.shift_a);
    m.lap(pass, "norms");
    if (m.base) gemm(m, enc.*, m.xn, b.q, m.qkv, 3 * m.inner, m.hidden) else {
        gemm(m, enc.*, m.xn, b.q, m.qr, m.inner, m.hidden);
        gemm(m, enc.*, m.xn, b.k, m.kr, m.inner, m.hidden);
        gemm(m, enc.*, m.xn, b.v, m.vr, m.inner, m.hidden);
        if (!m.dense) gemm(m, enc.*, m.xn, b.compress, m.cr, m.inner, m.hidden);
    }
    m.lap(pass, "q k v gate");
    enc.setPipeline(m.pipeline("h3_heads"));
    // the released layout's rows hold each head's q, k and v together: 384 values a row and head
    enc.setBuffer(if (m.base) m.qkv else m.qr, 0, 0);
    enc.setBuffer(if (m.base) m.qkv else m.kr, if (m.base) 2 * h3.head_dim else 0, 1);
    enc.setBuffer(if (m.base) m.qkv else m.vr, if (m.base) 4 * h3.head_dim else 0, 2);
    enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
    enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
    enc.setBuffer(m.cos.buffer, m.cos.offset, 5);
    enc.setBuffer(m.sin.buffer, m.sin.offset, 6);
    enc.setBuffer(m.slot.buffer, m.slot.offset, 7);
    enc.setValue([5]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.rot), 0, if (m.base) 3 * h3.head_dim else h3.head_dim }, 8);
    enc.setValue([1]f32{h3.eps}, 9);
    enc.setBuffer(m.tq, 0, 10);
    enc.setBuffer(m.tk, 0, 11);
    enc.setBuffer(m.tv, 0, 12);
    enc.dispatchGroups(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(32, 1, 1));
    m.lap(pass, "heads");
    if (m.dense) {
        attend(m, enc.*, m.all_tiles, m.tiles, m.tiles, 0, false);
        m.lap(pass, "attention");
    } else {
        enc.setPipeline(m.pipeline("h3_pool"));
        enc.setBuffer(m.tq, 0, 0);
        enc.setBuffer(m.tk, 0, 1);
        enc.setBuffer(m.tv, 0, 2);
        enc.setBuffer(m.sizes.buffer, m.sizes.offset, 3);
        enc.setBuffer(m.qp, 0, 4);
        enc.setBuffer(m.kp, 0, 5);
        enc.setBuffer(m.vp, 0, 6);
        enc.dispatchGroups(mtl.Size.of(m.tiles, heads, 1), mtl.Size.of(32, 1, 1));
        m.route(enc.*);
        m.lap(pass, "routing");
        // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
        attend(m, enc.*, m.all_tiles, m.prefix_tiles, m.tiles, 0, false);
        attend(m, enc.*, m.chosen, m.tiles - m.prefix_tiles, m.prefix_tiles + m.keep, m.prefix_tiles, true);
        m.lap(pass, "attention");
        // the pooled branch gated into the attention rows in place
        enc.setPipeline(m.pipeline("h3_gate_mix"));
        enc.setBuffer(m.qr, 0, 0);
        enc.setBuffer(m.cr, 0, 1);
        enc.setBuffer(m.coarse, 0, 2);
        enc.setBuffer(m.slot.buffer, m.slot.offset, 3);
        enc.setValue([3]i32{ @intCast(m.rows), @intCast(heads), @intCast(m.tiles) }, 4);
        enc.dispatchThreads(mtl.Size.of(m.rows, heads, 1), mtl.Size.of(64, 4, 1));
        m.lap(pass, "gate mix");
    }
    gemm(m, enc.*, m.qr, b.out, m.y, m.hidden, m.inner);
    m.lap(pass, "attention out");
    norm(m, enc.*, b.norm2, b.tables, h3.gate_a, b.tables, h3.scale_m, h3.shift_m);
    m.lap(pass, "norms");
    // the first projection's rows are [value | gate] ([gate | value] in the released layout), one product
    gemm(m, enc.*, m.xn, b.fc1, m.wide, 2 * m.mlp, m.hidden);
    enc.barrier();
    enc.setPipeline(m.pipeline("h3_swiglu"));
    enc.setBuffer(m.wide, 0, 0);
    enc.setValue([3]i32{ @intCast(m.rows), @intCast(m.mlp), @intFromBool(m.base) }, 1);
    enc.setBuffer(m.act, 0, 2);
    enc.dispatchThreads(mtl.Size.of(m.mlp, m.rows, 1), mtl.Size.of(256, 1, 1));
    m.lap(pass, "mlp in");
    gemm(m, enc.*, m.act, b.fc2, m.y, m.hidden, m.mlp);
    m.lap(pass, "mlp out");
}

test "h3 gemm: 32x64-warp products write every tile as MLX's 32x32 band of 4 does, across a band edge and a ragged tile" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const text = try @import("source.zig").library(a, 56, 18432, 288, "");
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const whole_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_whole, false);
    defer whole_pipe.deinit();
    const ragged_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_ragged, false);
    defer ragged_pipe.deinit();
    const wide_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_32x64, false);
    defer wide_pipe.deinit();
    const wide_ragged_pipe = try mtl.Pipeline.init(device, lib, kernels.gemm_32x64_ragged, false);
    defer wide_ragged_pipe.deinit();
    const n = 256;
    const k = 512;
    const guard = 64 * n; // one more row tile than the product writes, which must stay untouched
    var prng = std.Random.DefaultPrng.init(81);
    // a whole band of 64 tiles, one tile past it, and 65 tiles plus a ragged tail
    for ([_]usize{ 64 * 64 + 24, 64 * 65, 64 * 65 + 24 }) |rows| {
        const whole = rows % 64 == 0;
        const shared = mtl.ResourceOptions.shared;
        const x = try device.buffer(rows * k * 2, shared);
        defer x.deinit();
        const w = try device.buffer(n * k * 2, shared);
        defer w.deinit();
        // bf16 values of magnitude [1, 2): seven random mantissa bits and a random sign
        for ([_][]u16{ x.slice(u16, rows * k), w.slice(u16, n * k) }) |values| {
            for (values) |*v| v.* = 0x3f80 | (prng.random().int(u16) & 0x807f);
        }
        const old = try device.buffer((rows * n + guard) * 2, shared);
        defer old.deinit();
        const wide = try device.buffer((rows * n + guard) * 2, shared);
        defer wide.deinit();
        for ([_]mtl.Buffer{ old, wide }) |d| @memset(d.slice(u16, rows * n + guard), 0xffff); // a NaN no product of these inputs makes
        // MLX's geometry: 32 x 32 a warp, 256 threads; ours: 32 x 64, 128 threads
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.concurrent);
            product(enc, if (whole) whole_pipe else ragged_pipe, 2, x, .{ .buffer = w }, old, rows, n, k, 256);
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.concurrent);
            product(enc, if (whole) wide_pipe else wide_ragged_pipe, swizzle, x, .{ .buffer = w }, wide, rows, n, k, 128);
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
        const after = wide.slice(u16, rows * n + guard);
        try std.testing.expectEqualSlices(u16, old.slice(u16, rows * n + guard), after);
        try std.testing.expect(std.mem.indexOfScalar(u16, after[0 .. rows * n], 0xffff) == null);
        for (after[rows * n ..]) |v| try std.testing.expectEqual(@as(u16, 0xffff), v);
    }
}

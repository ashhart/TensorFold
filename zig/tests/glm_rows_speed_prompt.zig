//! tf-glm-rows-speed's prompt chunk (GLM_SPEED_PROMPT): oQ's per-head, 5/6-bit and split products, ms a chunk.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const speed = @import("glm_rows_speed.zig");
const wide = tf.glm.wide;
const kernels = tf.glm.kernels;
const Pool = speed.Pool;
const Io = speed.Io;
const Bench = speed.Bench;
const buf = speed.buf;
const rf = speed.rf;
const reps = speed.reps;
const H = speed.H;
const W = speed.W;
const PS = speed.PS;
const QR = speed.QR;
const D = speed.D;

const PM = 2048; // a prompt chunk's rows

/// A prompt chunk's per-head product (embed_q: x qp's slices; unembed_out: the latent rows) on the row kernels.
fn headsRows(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io, ldx: u32, kx: u32) void {
    const q = pool.at(i);
    wide.perHead(p, e, q, H, .{ .in = rf(io.x), .ldx = ldx, .out = rf(io.y), .ldo = q.n, .rows = PM }, kx, q.n / H);
}
/// The same as one affine_mm batch (prompt.zig headsMm).
fn headsMm(e: mtl.ComputeEncoder, i: usize, mm: tf.affine_mm.Pipes, pool: Pool, io: Io, ldx: u32, kx: u32) void {
    var head = pool.at(i);
    head.n /= H;
    tf.affine_mm.rowSumsAt(e, mm, 64, rf(io.x), rf(io.z), PM, H * kx, ldx);
    tf.affine_mm.denseBatch(e, mm, rf(io.x), rf(io.z), head, rf(io.y), PM, H, .{
        .x_row = @intCast(ldx),
        .y_row = @intCast(H * head.n),
        .sums_row = H,
        .x_batch = @intCast(kx),
        .y_batch = @intCast(head.n),
        .sums_batch = 1,
        .w_batch = @intCast(head.n),
    });
}
fn ehRows(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, eh: mtl.Buffer, io: Io) void {
    _ = i;
    wide.bf16(p, e, .{ .buf = eh }, D, 2 * D, .{ .in = rf(io.x), .ldx = 2 * D, .out = rf(io.y), .ldo = D, .rows = PM });
}

/// A chunk's dense product: row kernels, or affine_mm (`mm`).
fn chunkRows(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io) void {
    const q = pool.at(i);
    wide.dense(p, e, q, .{ .in = rf(io.x), .ldx = q.k, .out = rf(io.y), .ldo = q.n, .rows = PM });
}
fn chunkMm(e: mtl.ComputeEncoder, i: usize, mm: tf.affine_mm.Pipes, pool: Pool, io: Io) void {
    const q = pool.at(i);
    tf.affine_mm.rowSums(e, mm, 64, rf(io.x), rf(io.z), PM, q.k);
    tf.affine_mm.dense(e, mm, rf(io.x), rf(io.z), q, rf(io.y), PM);
}

/// A prompt chunk's products where oQ's paths differ from uniform's, ms a chunk of 2048 rows.
pub fn prompt(b: *Bench, k: *const kernels.Kernels, p: *const wide.Pipes) !void {
    b.reps = 2;
    defer b.reps = reps;
    std.debug.print("prompt chunk ({d} rows), ms a launch:\n", .{PM});
    const eq = try Pool.init(b.device, H * 512, 256, 8, 1, 2);
    defer eq.deinit();
    std.debug.print("  embed_q q8      per-head rows {d:8.2}  affine_mm batch {d:8.2}  (x11 a chunk)\n", .{
        try b.time(headsRows, .{ p, eq, b.io, @as(u32, QR), @as(u32, 256) }) / 1e3,
        try b.time(headsMm, .{ p.mm8, eq, b.io, @as(u32, QR), @as(u32, 256) }) / 1e3,
    });
    const uo = try Pool.init(b.device, H * 256, 512, 8, 1, 2);
    defer uo.deinit();
    std.debug.print("  unembed_out q8  per-head rows {d:8.2}  affine_mm batch {d:8.2}  (x11 a chunk)\n", .{
        try b.time(headsRows, .{ p, uo, b.io, @as(u32, H * 512), @as(u32, 512) }) / 1e3,
        try b.time(headsMm, .{ p.mm8, uo, b.io, @as(u32, H * 512), @as(u32, 512) }) / 1e3,
    });
    { // the MTP head's bf16 eh_proj [4096, 8192] over the chunk (prompt.zig mtp)
        const eh = try buf(b.device, D * 2 * D * 2);
        defer eh.deinit();
        @memset(eh.slice(u16, 2 * D * D), 0x3c00);
        std.debug.print(
            "  eh_proj bf16 [4096, 8192] rows {d:8.2}  (x1 a chunk with MTP)\n",
            .{try b.time(ehRows, .{ p, eh, b.io }) / 1e3},
        );
    }
    for ([_][3]usize{ .{ PS, D, 9 }, .{ D, W, 10 }, .{ 1536, D, 1 } }) |s| {
        const m8 = try Pool.init(b.device, s[0], s[1], 8, 1, 2);
        defer m8.deinit();
        const m4 = try Pool.init(b.device, s[0], s[1], 4, 1, 2);
        defer m4.deinit();
        const t8 = try b.time(chunkMm, .{ p.mm8, m8, b.io });
        const t4 = try b.time(chunkMm, .{ k.mm_bf16, m4, b.io });
        for ([_]u8{ 5, 6 }) |bits| {
            const m = try Pool.init(b.device, s[0], s[1], bits, 1, 2);
            defer m.deinit();
            const on_mm = if (wide.mm(p, m.at(0))) |mm| try b.time(chunkMm, .{ mm, m, b.io }) else 0;
            std.debug.print(
                "  [{d}, {d}] q{d} rows {d:8.2}  affine_mm {d:8.2}  (affine_mm q8 {d:.2}, q4 {d:.2})\n",
                .{ s[0], s[1], bits, try b.time(chunkRows, .{ p, m, b.io }) / 1e3, on_mm / 1e3, t8 / 1e3, t4 / 1e3 },
            );
        }
    }
}

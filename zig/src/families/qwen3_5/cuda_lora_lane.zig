//! A runtime LoRA's two products on the bf16 lane GEMM the projections use: tensor cores, the base's row rules.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");

const qmmf = cuda.qmmf;
const Weight = qmmf.Weight;

/// A factor pair as lane weights: A an (rank, k) projection, B an (n, rank) one; rank padded to whole 64-groups.
pub const Pair = struct { a: Weight, b: Weight, rank: usize };

/// The rank a lane weight's k takes; the padding rows of A and columns of B are zero, so they add exact zeros.
pub fn padRank(rank: usize) usize {
    return std.mem.alignForward(usize, rank, qmmf.group);
}

/// Device bytes of one packed pair.
pub fn pairBytes(rank: usize, n: usize, k: usize) usize {
    const rp = padRank(rank);
    return aligned(cuda.modelopt.padded(rp) * k * 2) + aligned(cuda.modelopt.padded(n) * rp * 2);
}

/// Staging bytes a pair's unpacked factors take on the device.
pub fn stageBytes(rank: usize, n: usize, k: usize) usize {
    const rp = padRank(rank);
    return @max(rp * k * 2, n * rp * 2);
}

/// Pack A [rank][k] and B [n][rank] (bf16 host bytes) at `dst` in `buf`, staging the padded rows at `stage`.
pub fn pack(gpa: std.mem.Allocator, e: *E.Engine, pk: *const cuda.modelopt.Packer, buf: cuda.DeviceBuffer, dst: usize, stage: usize, a: []const u8, b: []const u8, rank: usize, n: usize, k: usize) !Pair {
    const rp = padRank(rank);
    const host = try gpa.alloc(u8, stageBytes(rank, n, k));
    defer gpa.free(host);
    // A's rows rank..rp stay zero
    @memset(host[0 .. rp * k * 2], 0);
    @memcpy(host[0..a.len], a);
    try buf.upload(stage, host[0 .. rp * k * 2]);
    const wa = dst;
    try pk.bf16Lane(e.stream, buf.ptr + stage, buf.ptr + wa, rp, k);
    try e.stream.synchronize(); // B's staging overwrites A's
    // each of B's n rows takes rank values, then rp - rank zeros
    @memset(host[0 .. n * rp * 2], 0);
    for (0..n) |r| @memcpy(host[r * rp * 2 ..][0 .. rank * 2], b[r * rank * 2 ..][0 .. rank * 2]);
    try buf.upload(stage, host[0 .. n * rp * 2]);
    const wb = dst + aligned(cuda.modelopt.padded(rp) * k * 2);
    try pk.bf16Lane(e.stream, buf.ptr + stage, buf.ptr + wb, n, rp);
    try e.stream.synchronize();
    return .{ .a = cuda.modelopt.bf16Weight(buf.ptr + wa, rp, k), .b = cuda.modelopt.bf16Weight(buf.ptr + wb, n, rp), .rank = rp };
}

/// The slice partials a pair's two products take at any row count up to `rows`.
pub fn partBytes(lane: qmmf.Lane, p: Pair, rows: usize) usize {
    var most: usize = 0;
    for ([_]Weight{ p.a, p.b }) |w| for ([_]usize{ @min(rows, qmmf.fused_rows - 1), rows }) |m| {
        most = @max(most, lane.partBytes(m, w));
    };
    return most;
}

/// y [rows][n] bf16 = bf16(y + bf16(x A^T) B^T): the first product rounds to bf16, the second stays fp32 until the add.
pub fn apply(e: *E.Engine, lane: qmmf.Lane, add: cuda.Function, p: Pair, x: u64, y: u64, rows: usize, t16: u64, d32: u64, part: u64) !void {
    if (rows > E.max_rows) return error.QwenLoraRows; // the scratch holds one max_rows piece
    try lane.matmulOut(e.stream, x, p.a.k, rows, p.a, t16, .bf16, part);
    try lane.matmulOut(e.stream, t16, p.rank, rows, p.b, d32, .f32, part);
    const count = rows * p.b.n;
    try e.run(add, .{ (count + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ y, d32, @as(c_int, @intCast(count)) });
}

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

test "Qwen adapter tensors: lane pairs pad the rank to whole groups and size both packed factors" {
    try std.testing.expectEqual(@as(usize, 64), padRank(32));
    try std.testing.expectEqual(@as(usize, 256), padRank(256));
    // A pads its 64 rows to 128 lane rows; B's 4096 rows are already whole blocks
    try std.testing.expectEqual(aligned(128 * 4096 * 2) + aligned(4096 * 64 * 2), pairBytes(32, 4096, 4096));
    try std.testing.expectEqual(@as(usize, 4096 * 64 * 2), stageBytes(32, 4096, 4096));
}

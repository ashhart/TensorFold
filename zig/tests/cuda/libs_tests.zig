//! cuBLASLt and NCCL through dlopen: one bf16 GEMM against a host reference, one-rank collectives.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

/// Round-to-nearest-even fp32 to bf16 bits (inputs here are finite).
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn cublaslt(gpu: Gpu) !void {
    const d = gpu.d;
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    const m: usize = 16;
    const n: usize = 256;
    const k: usize = 512;
    var rng = std.Random.DefaultPrng.init(1234);
    const r = rng.random();
    const x = try gpu.gpa.alloc(u16, m * k);
    defer gpu.gpa.free(x);
    const w = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(w);
    for (x) |*v| v.* = bf16(r.floatNorm(f32));
    for (w) |*v| v.* = bf16(r.floatNorm(f32) * 0.05);

    var xd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x));
    defer xd.free();
    var wd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(w));
    defer wd.free();
    var out = try cuda.DeviceBuffer.alloc(d, m * n * 4);
    defer out.free();
    const workspace_len: usize = 32 << 20;
    var ws = try cuda.DeviceBuffer.alloc(d, workspace_len);
    defer ws.free();
    var linear = try cuda.cublaslt.Linear.init(&lt, m, n, k, .f32, workspace_len);
    defer linear.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const first = try check.download(gpu, out);
    defer gpu.gpa.free(first);
    try out.fill8(0, null);
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const second = try check.download(gpu, out);
    defer gpu.gpa.free(second);
    try check.sameBytes("cuBLASLt rerun", second, first);

    var worst: f64 = 0;
    const got = std.mem.bytesAsSlice(f32, first);
    for (0..m) |i| for (0..n) |j| {
        var acc: f64 = 0;
        for (0..k) |t| acc += @as(f64, widen(x[i * k + t])) * @as(f64, widen(w[j * k + t]));
        worst = @max(worst, @abs(@as(f64, got[i * n + j]) - acc));
    };
    try expect(worst < 1e-3, "cuBLASLt max abs error {e} vs fp64 reference", .{worst});
    check.pass("cuBLASLt {d}: bf16 x bf16 -> fp32 ({d}x{d}x{d}), reruns bit-equal, max abs error {e} vs fp64, workspace {d} B", .{ lt.api.cublasLtGetVersion(), m, n, k, worst, linear.workspace_need });
}

/// `nccl-pair RANK WORLD ID`: one rank of a multi-host communicator (id as hex in file ID): send/recv, sums, gathers.
pub fn ncclPair(gpu: Gpu, rank: c_int, world: c_int, id_path: []const u8) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        const hex = std.fmt.bytesToHex(id.internal, .lower);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = id_path, .data = &hex });
    } else {
        var hex: [256]u8 = undefined;
        const got = try std.Io.Dir.cwd().readFile(gpu.io, id_path, &hex);
        try expect(got.len == 256, "unique id file {s}: {d} hex digits", .{ id_path, got.len });
        _ = try std.fmt.hexToBytes(&id.internal, &hex);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);
    const count: usize = 1 << 20;
    const W: usize = @intCast(world);
    const R: usize = @intCast(rank);
    const mine = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(mine);
    for (mine, 0..) |*v, i| v.* = @floatFromInt(R * 1000 + i % 977);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(mine));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4 * W);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    for (0..W) |p| if (p != R) {
        try lib.check(lib.api.ncclSend(send.ptr, count, .f32, @intCast(p), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(recv.ptr + p * count * 4, count, .f32, @intCast(p), comm, stream.handle), "ncclRecv");
    };
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(want);
    for (0..W) |p| if (p != R) {
        for (want, 0..) |*v, i| v.* = @floatFromInt(p * 1000 + i % 977);
        try check.sameBytes("send/recv from a peer", got[p * count * 4 ..][0 .. count * 4], std.mem.sliceAsBytes(want));
    };
    try lib.check(lib.api.ncclAllReduce(send.ptr, recv.ptr, count, .f32, .sum, comm, stream.handle), "ncclAllReduce");
    try stream.synchronize();
    try recv.download(0, got[0 .. count * 4]);
    for (want, 0..) |*v, i| {
        var s: f32 = 0;
        for (0..W) |p| s += @floatFromInt(p * 1000 + i % 977);
        v.* = s;
    }
    try check.sameBytes("all-reduce", got[0 .. count * 4], std.mem.sliceAsBytes(want));
    // all-gather: every rank's pattern in rank order
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, count, .f32, comm, stream.handle), "ncclAllGather");
    try stream.synchronize();
    try recv.download(0, got);
    for (0..W) |p| {
        for (want, 0..) |*v, i| v.* = @floatFromInt(p * 1000 + i % 977);
        try check.sameBytes("all-gather", got[p * count * 4 ..][0 .. count * 4], std.mem.sliceAsBytes(want));
    }
    check.pass("NCCL {d}: rank {d} of {d}, send/recv with every peer ({d} MiB each), all-reduce and all-gather exact", .{ try lib.version(), rank, world, count * 4 >> 20 });
}

pub fn nccl(gpu: Gpu) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const version = try lib.version();
    var comms: [1]cuda.nccl.Comm = .{null};
    const devices = [1]c_int{gpu.ctx.device};
    try lib.check(lib.api.ncclCommInitAll(&comms, 1, &devices), "ncclCommInitAll");
    defer _ = lib.api.ncclCommDestroy(comms[0]);
    const count: usize = 4096;
    const src = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(src));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try lib.check(lib.api.ncclAllReduce(send.ptr, recv.ptr, count, .f32, .sum, comms[0], stream.handle), "ncclAllReduce");
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    try check.sameBytes("one-rank all-reduce", got, std.mem.sliceAsBytes(src));
    try recv.fill8(0, null);
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, count, .f32, comms[0], stream.handle), "ncclAllGather");
    try stream.synchronize();
    try recv.download(0, got);
    try check.sameBytes("one-rank all-gather", got, std.mem.sliceAsBytes(src));
    // a send to this rank and its receive, in one group (NCCL pairs them)
    try recv.fill8(0, null);
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    try lib.check(lib.api.ncclSend(send.ptr, count, .f32, 0, comms[0], stream.handle), "ncclSend");
    try lib.check(lib.api.ncclRecv(recv.ptr, count, .f32, 0, comms[0], stream.handle), "ncclRecv");
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try stream.synchronize();
    try recv.download(0, got);
    try check.sameBytes("one-rank send/recv", got, std.mem.sliceAsBytes(src));
    check.pass("NCCL {d}: one-rank communicator, all-reduce, all-gather and send/recv to itself exact", .{version});
}

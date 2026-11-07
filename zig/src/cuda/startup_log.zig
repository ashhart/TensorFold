//! Startup receipts share the same short communicator identity and never change a transfer.
const std = @import("std");
const nccl = @import("nccl.zig");
const cudap = @import("p2p_cuda.zig");
pub const Sink = *const fn ([]const u8) void;

pub fn log(line: []const u8) void {
    std.log.info("{s}", .{line});
}

pub fn idHash(id: *const [128]u8) [8]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(id, &digest, .{});
    return std.fmt.bytesToHex(digest[0..4].*, .lower);
}

pub fn connected(sink: Sink, rank: usize, peers: usize, port: u16, tries: usize) void {
    var buf: [160]u8 = undefined;
    const text = if (rank == 0)
        std.fmt.bufPrint(&buf, "rank 0: liveness lines from {d} ranks on port {d}", .{ peers, port })
    else
        std.fmt.bufPrint(&buf, "rank {d}: liveness line to rank 0 connected after {d} tries", .{ rank, tries });
    sink(text catch return);
}

pub fn handoff(sink: Sink, rank: usize, id: *const [128]u8, sent: bool) void {
    var buf: [120]u8 = undefined;
    sink(std.fmt.bufPrint(&buf, "rank {d}: NCCL id {s} {s}", .{ rank, idHash(id), if (sent) "sent" else "taken" }) catch return);
}

const Clock = struct {
    io: std.Io,
    pub fn now(c: Clock) u64 {
        return @intCast(@divTrunc(std.Io.Clock.awake.now(c.io).toNanoseconds(), std.time.ns_per_ms));
    }
};

/// Device queries are optional diagnostics; a missing query or version never becomes a startup refusal.
pub fn init(io: std.Io, lib: *const nccl.Library, id: nccl.UniqueId, rank: usize, world: usize) !nccl.Comm {
    var device: ?cudap.Dev = null;
    if (world > 1) if (cudap.Library.open()) |cl| {
        device = deviceWith(&cl.api);
    };
    return initWith(lib, id, rank, world, if (device) |*d| d else null, Clock{ .io = io }, log);
}

/// Only the current device's ordinal and bus: no capability query, peer enable or transfer.
pub fn deviceWith(api: *const cudap.Api) ?cudap.Dev {
    var current: c_int = -1;
    if (api.cuCtxGetDevice(&current) != 0) return null;
    var count: c_int = 0;
    if (api.cuDeviceGetCount(&count) != 0) return null;
    for (0..@intCast(@max(count, 0))) |i| {
        var handle: c_int = -1;
        if (api.cuDeviceGet(&handle, @intCast(i)) != 0 or handle != current) continue;
        var device: cudap.Dev = .{ .ordinal = @intCast(i), .handle = handle };
        var bus: [32]u8 = @splat(0);
        if (api.cuDeviceGetPCIBusId(&bus, bus.len, handle) == 0) device.bus = .ofC(&bus);
        return device;
    }
    return null;
}

pub fn initWith(lib: *const nccl.Library, id: nccl.UniqueId, rank: usize, world: usize, device: ?*const cudap.Dev, clock: anytype, sink: Sink) !nccl.Comm {
    if (world > 1) {
        var version: c_int = 0;
        var vb: [32]u8 = undefined;
        const v = if (lib.api.ncclGetVersion(&version) == 0)
            std.fmt.bufPrint(&vb, "{d}.{d}.{d}", .{ @divTrunc(version, 10000), @mod(@divTrunc(version, 100), 100), @mod(version, 100) }) catch "?"
        else
            "?";
        var db: [16]u8 = undefined;
        const ordinal = if (device) |d| std.fmt.bufPrint(&db, "{d}", .{d.ordinal}) catch "?" else "?";
        var buf: [512]u8 = undefined;
        sink(std.fmt.bufPrint(&buf, "rank {d} startup: entering NCCL init ({d} ranks, id {s}, NCCL {s}, CUDA device {s} {s})", .{ rank, world, idHash(&id.internal), v, ordinal, if (device) |d| (if (d.bus.len > 0) d.bus.slice() else "?") else "?" }) catch "NCCL init");
    }
    const since = clock.now();
    var comm: nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
    if (world > 1) {
        var buf: [128]u8 = undefined;
        sink(std.fmt.bufPrint(&buf, "rank {d} startup: NCCL init done in {d} ms", .{ rank, clock.now() -| since }) catch "NCCL init done");
    }
    return comm;
}

test {
    _ = @import("startup_log_test.zig");
}

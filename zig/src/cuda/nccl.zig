//! NCCL opened at run time for tensor-parallel ranks: communicators and the collectives the lane engines use.

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const tp_watch = @import("tp_watch.zig");
const tensors = platform.rank_auth.tensors;

pub const minimum_version: c_int = 23000;

pub fn requireVersion(loaded: c_int) error{NcclVersion}!void {
    if (loaded < minimum_version) return error.NcclVersion;
}

pub const Result = c_int;
pub const Shape = struct { rank: u32, world: u32 };
pub const Comm = ?*opaque {};
pub const UniqueId = extern struct { internal: [128]u8 };

pub const DataType = enum(c_int) { i8 = 0, u8 = 1, i32 = 2, u32 = 3, i64 = 4, u64 = 5, f16 = 6, f32 = 7, f64 = 8, bf16 = 9 };
pub const RedOp = enum(c_int) { sum = 0, prod = 1, max = 2, min = 3, avg = 4 };

/// NCCL 2.30 public ncclConfig_v23000 ABI. New nonblocking rank owners explicitly require 2.30 or later.
pub const Config = extern struct {
    size: usize = @sizeOf(Config),
    magic: u32 = 0xcafebeef,
    version: u32 = 23000,
    blocking: c_int = 0,
    cga_cluster_size: c_int = std.math.minInt(c_int),
    min_ctas: c_int = std.math.minInt(c_int),
    max_ctas: c_int = std.math.minInt(c_int),
    net_name: ?[*:0]const u8 = null,
    split_share: c_int = std.math.minInt(c_int),
    traffic_class: c_int = std.math.minInt(c_int),
    comm_name: ?[*:0]const u8 = null,
    collnet_enable: c_int = std.math.minInt(c_int),
    cta_policy: c_int = std.math.minInt(c_int),
    shrink_share: c_int = std.math.minInt(c_int),
    nvls_ctas: c_int = std.math.minInt(c_int),
    n_channels_per_net_peer: c_int = std.math.minInt(c_int),
    nvlink_centric_sched: c_int = std.math.minInt(c_int),
    graph_usage_mode: c_int = std.math.minInt(c_int),
    num_rma_ctx: c_int = std.math.minInt(c_int),
    max_p2p_peers: c_int = std.math.minInt(c_int),
    graph_stream_ordering: c_int = std.math.minInt(c_int),
};

pub const Error = platform.optional_libraries.Failure || tensors.Error || error{ LibraryUnavailable, MissingSymbol, NcclFailed, NcclVersion, RankTensorsDiffer, RankNeverSentTensors };

pub const in_progress: Result = 7; // ncclInProgress: a non-blocking communicator's call is still running

const R = Result;
const D = abi.DevicePtr;

pub const Api = struct {
    ncclGetVersion: *const fn (*c_int) callconv(.c) R,
    ncclGetUniqueId: *const fn (*UniqueId) callconv(.c) R,
    ncclCommInitRank: *const fn (*Comm, c_int, UniqueId, c_int) callconv(.c) R,
    ncclCommInitAll: *const fn ([*]Comm, c_int, ?[*]const c_int) callconv(.c) R,
    ncclCommCount: *const fn (Comm, *c_int) callconv(.c) R,
    ncclCommUserRank: *const fn (Comm, *c_int) callconv(.c) R,
    ncclCommDestroy: *const fn (Comm) callconv(.c) R,
    ncclGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    ncclAllReduce: *const fn (D, D, usize, DataType, RedOp, Comm, abi.Stream) callconv(.c) R,
    ncclAllGather: *const fn (D, D, usize, DataType, Comm, abi.Stream) callconv(.c) R,
    ncclBroadcast: *const fn (D, D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclSend: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclRecv: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    /// Frees the communicator after ending its pending work; safe from another thread while a kernel waits.
    ncclCommAbort: *const fn (Comm) callconv(.c) R,
    ncclCommGetAsyncError: *const fn (Comm, *Result) callconv(.c) R,
    ncclGroupStart: *const fn () callconv(.c) R,
    ncclGroupEnd: *const fn () callconv(.c) R,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    pub fn open() Error!Library {
        return openPath("libnccl.so.2");
    }

    pub fn openPath(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return platform.optional_libraries.unavailable(.nccl);
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return platform.optional_libraries.unavailable(.nccl);
            };
        }
        // a keyed group: NCCL's sockets run TLS under a key from the group's before its first call, or no rank starts
        var loaded: c_int = 0;
        _ = api.ncclGetVersion(&loaded);
        const mode = try tensors.apply(lib.lookup(tensors.SetEncryption, tensors.symbol), loaded, "NCCL");
        try tp_watch.agreeTensors(@backingInt(mode)); // ranks that differ stop here, by name, not inside NCCL
        return .{ .lib = lib, .api = api };
    }

    pub fn shape(self: *const Library, comm: Comm) !Shape {
        var world: c_int = 0;
        var rank: c_int = -1;
        try self.check(self.api.ncclCommCount(comm, &world), "ncclCommCount");
        try self.check(self.api.ncclCommUserRank(comm, &rank), "ncclCommUserRank");
        if (world < 1 or rank < 0 or rank >= world) return error.BadWorld;
        return .{ .rank = @intCast(rank), .world = @intCast(world) };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, r: Result, what: []const u8) Error!void {
        if (r == 0) return;
        if (@import("builtin").is_test) return error.NcclFailed; // a test's injected failure: the runner fails error logs
        const s = self.api.ncclGetErrorString(r);
        std.log.err("{s}: {s} ({d})", .{ what, if (s) |p| std.mem.span(p) else "?", r });
        if (std.mem.startsWith(u8, what, "ncclCommInit")) if (tensors.hint()) |h| std.log.err("{s}: {s}", .{ what, h });
        tp_watch.fatal(what); // with the ranks' lines open, every rank stops with it; with none, the caller hears it
        return error.NcclFailed;
    }

    /// `post(ctx)`'s sends and receives as one group; a failed post still ends it, so later calls stay out of it.
    pub fn group(self: *const Library, ctx: anytype, comptime post: fn (@TypeOf(ctx)) anyerror!void) anyerror!void {
        try self.check(self.api.ncclGroupStart(), "ncclGroupStart");
        post(ctx) catch |err| {
            _ = self.api.ncclGroupEnd();
            return err;
        };
        try self.check(self.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    pub fn requireSupportedVersion(self: *const Library) Error!void {
        const loaded = try self.version();
        requireVersion(loaded) catch |err| {
            std.log.err("NCCL {d}.{d}.{d} loaded; tensor parallelism requires at least {d}.{d}.{d}", .{
                @divTrunc(loaded, 10000),          @mod(@divTrunc(loaded, 100), 100),          @mod(loaded, 100),
                @divTrunc(minimum_version, 10000), @mod(@divTrunc(minimum_version, 100), 100), @mod(minimum_version, 100),
            });
            return err;
        };
    }

    /// NCCL's version as 10000 * major + 100 * minor + patch.
    pub fn version(self: *const Library) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.ncclGetVersion(&v), "ncclGetVersion");
        return v;
    }
};

test "nonblocking communicator config preserves the public ABI defaults" {
    const c: Config = .{};
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Config));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Config, "net_name"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(Config, "comm_name"));
    try std.testing.expect(c.size == 96 and c.blocking == 0 and c.max_ctas == std.math.minInt(c_int));
}

test "NCCL minimum: 2.28.9 refuses" {
    try std.testing.expectError(error.NcclVersion, requireVersion(22809));
}

test "NCCL minimum: 2.29.7 refuses" {
    try std.testing.expectError(error.NcclVersion, requireVersion(22907));
}

test "NCCL minimum: 2.30.0 passes" {
    try requireVersion(23000);
}

test "NCCL minimum: 2.30.7 passes" {
    try requireVersion(23007);
}

//! RCCL opened at run time: the tensor-parallel forward's collectives on our streams, as hand-declared exports.

const std = @import("std");
const abi = @import("../abi.zig");

pub const Error = error{ RcclUnavailable, MissingSymbol, RcclFailed, Invalid };

/// ncclUniqueId: the opaque 128 bytes rank 0 makes and every rank passes to its init.
pub const UniqueId = extern struct { internal: [128]u8 };

/// ncclDataType_t, RCCL's numbering.
pub const Dtype = enum(c_int) { u8 = 1, i32 = 2, i64 = 4, f16 = 6, f32 = 7, bf16 = 9 };

const Handle = ?*anyopaque;
const C = ?*const anyopaque;
const P = ?*anyopaque;

/// Each field is the library's exact export.
const Api = struct {
    ncclGetErrorString: *const fn (c_int) callconv(.c) [*:0]const u8,
    ncclGetUniqueId: *const fn (*UniqueId) callconv(.c) c_int,
    ncclCommInitRank: *const fn (*Handle, c_int, UniqueId, c_int) callconv(.c) c_int,
    ncclCommDestroy: *const fn (Handle) callconv(.c) c_int,
    ncclAllReduce: *const fn (C, P, usize, c_int, c_int, Handle, abi.Stream) callconv(.c) c_int,
    ncclAllGather: *const fn (C, P, usize, c_int, Handle, abi.Stream) callconv(.c) c_int,
    ncclBroadcast: *const fn (C, P, usize, c_int, c_int, Handle, abi.Stream) callconv(.c) c_int,
};

const paths = [_][]const u8{ "librccl.so.1", "librccl.so", "/opt/rocm/lib/librccl.so.1" };

/// ncclSum: the only reduction the forward uses.
const sum_op: c_int = 0;

pub const Rccl = struct {
    lib: std.DynLib,
    api: Api,

    /// `first` (the policy's `rccl_lib`) is tried before the usual places.
    pub fn open(first: ?[]const u8) Error!Rccl {
        var candidates: [paths.len + 1][]const u8 = undefined;
        var n: usize = 0;
        if (first) |path| {
            candidates[0] = path;
            n = 1;
        }
        for (paths) |path| {
            candidates[n] = path;
            n += 1;
        }
        for (candidates[0..n]) |path| {
            var lib = std.DynLib.open(path) catch continue;
            errdefer lib.close();
            var api: Api = undefined;
            const info = @typeInfo(Api).@"struct";
            inline for (info.field_names, info.field_types) |name, T| {
                @field(api, name) = lib.lookup(T, name) orelse {
                    std.log.err("{s} has no {s}", .{ path, name });
                    return error.MissingSymbol;
                };
            }
            return .{ .lib = lib, .api = api };
        }
        return error.RcclUnavailable;
    }

    pub fn close(r: *Rccl) void {
        r.lib.close();
    }

    fn check(r: *const Rccl, code: c_int, what: []const u8) Error!void {
        if (code == 0) return;
        std.log.err("{s}: RCCL error {d}: {s}", .{ what, code, std.mem.span(r.api.ncclGetErrorString(code)) });
        return error.RcclFailed;
    }

    /// A fresh id; rank 0 makes it and hands it to the others.
    pub fn uniqueId(r: *const Rccl) Error!UniqueId {
        var id: UniqueId = undefined;
        try r.check(r.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        return id;
    }
};

/// One rank's communicator. Collectives queue on the stream they are given and wait for every rank's matching call.
pub const Comm = struct {
    rccl: *const Rccl,
    handle: Handle,
    rank: usize,
    world: usize,

    /// Joins the ranks; blocks until all `world` have called it with the same `id`. The device is current already.
    pub fn init(rccl: *const Rccl, id: UniqueId, rank: usize, world: usize) Error!Comm {
        if (world < 2 or rank >= world) return error.Invalid;
        var h: Handle = null;
        try rccl.check(rccl.api.ncclCommInitRank(&h, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .rccl = rccl, .handle = h, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.rccl.api.ncclCommDestroy(c.handle);
        c.* = undefined;
    }

    /// `recv` = the ranks' `send` summed, `n` values of `dtype` (sum of two ranks is the one add, any order).
    pub fn allReduce(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void {
        if (stream == @import("../launches/util.zig").counting) return;
        try c.rccl.check(c.rccl.api.ncclAllReduce(@ptrFromInt(send), @ptrFromInt(recv), n, @backingInt(dtype), sum_op, c.handle, stream), "ncclAllReduce");
    }

    /// `recv` (world * n values) = every rank's `send` (n values) in rank order.
    pub fn allGather(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void {
        if (stream == @import("../launches/util.zig").counting) return;
        try c.rccl.check(c.rccl.api.ncclAllGather(@ptrFromInt(send), @ptrFromInt(recv), n, @backingInt(dtype), c.handle, stream), "ncclAllGather");
    }

    /// `root`'s `send` into every rank's `recv`, `n` values.
    pub fn broadcast(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, root: usize, stream: abi.Stream) Error!void {
        if (stream == @import("../launches/util.zig").counting) return;
        try c.rccl.check(c.rccl.api.ncclBroadcast(@ptrFromInt(send), @ptrFromInt(recv), n, @backingInt(dtype), @intCast(root), c.handle, stream), "ncclBroadcast");
    }
};

test "the unique id is the 128 bytes RCCL passes by value" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(UniqueId));
}

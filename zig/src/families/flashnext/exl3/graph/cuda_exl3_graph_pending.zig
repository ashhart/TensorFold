//! A captured EXL3 graph instantiated on a worker thread while its step runs eagerly.

const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const wire = @import("../../cuda_rank_wire.zig");

/// A captured graph being instantiated on a worker thread (20-35 ms that the step need not wait for).
pub const Pending = struct {
    thread: std.Thread = undefined,
    done: std.atomic.Value(bool) = .init(false),
    graph: cuda.graph.Graph,
    exec: ?cuda.graph.Exec = null,
    ctx: ?*anyopaque,
    keep_template: bool,
    started_ns: u64 = 0,

    pub fn main(pd: *Pending) void {
        const api = ctxApi() orelse {
            pd.done.store(true, .release);
            return;
        };
        if (api.set(pd.ctx) != 0) {
            pd.done.store(true, .release);
            return;
        }
        pd.exec = pd.graph.instantiate() catch null;
        pd.done.store(true, .release);
    }
};

const CtxApi = struct { get: *const fn (*?*anyopaque) callconv(.c) c_int, set: *const fn (?*anyopaque) callconv(.c) c_int };
pub var ctx_api: ?CtxApi = null;
var ctx_lib: ?platform.DynLib = null;
pub fn ctxApi() ?CtxApi {
    if (ctx_api) |a| return a;
    var lib = platform.DynLib.open(platform.cuda_driver) catch return null;
    const a: CtxApi = .{
        .get = lib.lookup(*const fn (*?*anyopaque) callconv(.c) c_int, "cuCtxGetCurrent") orelse return null,
        .set = lib.lookup(*const fn (?*anyopaque) callconv(.c) c_int, "cuCtxSetCurrent") orelse return null,
    };
    ctx_lib = lib;
    ctx_api = a;
    return a;
}

/// TF_FLASHNEXT_EXL3_GRAPH_ASYNC=0 instantiates on the step's thread.
pub fn asyncInstantiate() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_ASYNC") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Starts instantiating `g` on a worker thread (it owns `g` from here); null when that is not possible.
pub fn startInstantiate(g: cuda.graph.Graph, keep_template: bool) !?*Pending {
    if (!asyncInstantiate()) return null;
    const api = ctxApi() orelse return null;
    var ctx: ?*anyopaque = null;
    if (api.get(&ctx) != 0) return null;
    const pd = try std.heap.c_allocator.create(Pending);
    pd.* = .{ .graph = g, .ctx = ctx, .keep_template = keep_template, .started_ns = wire.now() };
    pd.thread = std.Thread.spawn(.{}, Pending.main, .{pd}) catch |err| {
        std.heap.c_allocator.destroy(pd);
        return err;
    };
    return pd;
}

/// The finished instantiation's exec (the graph template freed), or null while it runs; error when it failed.
pub fn takePending(pd: *Pending) !?cuda.graph.Exec {
    if (!pd.done.load(.acquire)) return null;
    pd.thread.join();
    defer std.heap.c_allocator.destroy(pd);
    pd.graph.deinit();
    return pd.exec orelse error.CudaFailed;
}

//! Two-stream mHC module, checked launches and persistent scratch addresses.
const cuda = @import("cuda");
const sig = cuda.kernels.mhc_sig;

pub const Runtime = struct {
    mod: ?cuda.Module = null,
    enter: cuda.Function = undefined,
    route: cuda.Function = undefined,
    merge: cuda.Function = undefined,
    leave: cuda.Function = undefined,
    streams: u64 = 0,
    post: u64 = 0,
    mix: u64 = 0,

    pub fn load(d: *const cuda.Driver) !Runtime {
        var module = try cuda.Module.load(d, cuda.kernels.qwen35_mhc);
        errdefer module.unload();
        return .{
            .mod = module,
            .enter = try module.function("qwen35_mhc_enter"),
            .route = try module.function("qwen35_mhc_route"),
            .merge = try module.function("qwen35_mhc_merge"),
            .leave = try module.function("qwen35_mhc_leave"),
        };
    }

    pub fn deinit(r: *Runtime) void {
        if (r.mod) |*m| m.unload();
        r.mod = null;
    }
};

fn launch(e: anytype, f: cuda.Function, comptime spec: sig.Sig, rows: usize, args: anytype) !void {
    sig.check(spec, @TypeOf(args));
    try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, args);
}

pub fn enter(e: anytype, pending: bool, rows: usize, params: u64) !void {
    try launch(e, e.mhc.enter, sig.enter, rows, .{ e.h, if (pending) e.r else @as(u64, 0), e.mhc.streams, @as(c_int, @intCast(e.g.hidden)), params });
}

pub fn route(e: anytype, weight: u64, bias: u64, rows: usize, params: u64) !void {
    try launch(e, e.mhc.route, sig.route, rows, .{ e.mhc.streams, weight, bias, e.h, e.mhc.post, e.mhc.mix, @as(c_int, @intCast(e.g.hidden)), params });
}

pub fn merge(e: anytype, rows: usize, params: u64) !void {
    try launch(e, e.mhc.merge, sig.merge, rows, .{ e.mhc.streams, e.r, e.mhc.post, e.mhc.mix, @as(c_int, @intCast(e.g.hidden)), params });
}

pub fn leave(e: anytype, rows: usize, params: u64) !void {
    try launch(e, e.mhc.leave, sig.leave, rows, .{ e.mhc.streams, e.h, @as(c_int, @intCast(e.g.hidden)), params });
}

//! ERNIE 5.5's MoE kernels (ernie5_5/moe.cu): the router's picks and weights, as core's combine takes them.
const cuda = @import("cuda");
pub const route = @import("route.zig");

pub const threads = 64;

pub const Moe = struct {
    mod: cuda.Module,
    route_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Moe {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_moe);
        errdefer mod.unload();
        return .{ .mod = mod, .route_fn = try mod.function("tf_ernie55_route") };
    }

    pub fn unload(x: *Moe) void {
        x.mod.unload();
    }

    /// fp32 logits [rows][2 x 512] (ls apart), f32 bias and scale [512] into int32 ids and fp32 weights [rows][k].
    pub fn picks(x: *const Moe, s: cuda.Stream, logits: u64, bias: u64, scale: u64, ids: u64, wts: u64, sh: route.Shape) !void {
        try launch(x.route_fn, s, logits, bias, scale, ids, wts, sh);
    }
};

/// `f`, tf_ernie55_route or an entry of the image with its arguments, one row a block.
pub fn launch(f: cuda.Function, s: cuda.Stream, logits: u64, bias: u64, scale: u64, ids: u64, wts: u64, sh: route.Shape) !void {
    try route.check(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(sh.ls)));
    inline for (.{ bias, scale, ids, wts }) |v| a.add(v);
    a.add(@as(c_int, @intCast(sh.k)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

//! ERNIE 5.5's head kernels (ernie5_5/head.cu): MultiMax on the fp32 logits in place.
const cuda = @import("cuda");
pub const multimax = @import("multimax.zig");

pub const threads = 256;

pub const Head = struct {
    mod: cuda.Module,
    multimax_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Head {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_head);
        errdefer mod.unload();
        return .{ .mod = mod, .multimax_fn = try mod.function("tf_ernie55_multimax") };
    }

    pub fn unload(h: *Head) void {
        h.mod.unload();
    }

    /// MultiMax over n fp32 logits at x, in place.
    pub fn apply(h: *const Head, s: cuda.Stream, x: u64, n: usize, k: multimax.Coeffs) !void {
        try launch(h.multimax_fn, s, x, n, k);
    }
};

/// `f`, tf_ernie55_multimax or an entry of the image with its arguments, over n logits at x.
pub fn launch(f: cuda.Function, s: cuda.Stream, x: u64, n: usize, k: multimax.Coeffs) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(n)));
    for (k.ranges) |r| a.add(r);
    for (k.ts) |t| a.add(t);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((n + threads - 1) / threads) }, .block = .{ .x = threads } }, s, &a);
}

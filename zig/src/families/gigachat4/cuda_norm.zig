//! GigaChat 4's GatedNorm kernel (gigachat4/gated_norm.cu): load and launch.
const cuda = @import("cuda");
pub const gated_norm = @import("gated_norm.zig");

pub const Norm = struct {
    mod: cuda.Module,
    gated_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Norm {
        var mod = try cuda.Module.load(d, cuda.kernels.gigachat4_norm);
        errdefer mod.unload();
        return .{ .mod = mod, .gated_fn = try mod.function("tf_gc4_gated_norm") };
    }

    pub fn unload(x: *Norm) void {
        x.mod.unload();
    }

    /// `rows` bf16 rows of x (x_ld apart) normed with w [n], up [16, n], down [n, 16]: bf16 rows o_ld apart.
    pub fn gated(x: *const Norm, s: cuda.Stream, rows: usize, n: usize, in: u64, x_ld: usize, w: u64, up: u64, down: u64, out: u64, o_ld: usize, eps: f32, scale: f32) !void {
        try launch(x.gated_fn, null, s, rows, n, in, x_ld, w, up, down, out, o_ld, eps, scale);
    }
};

/// `f`, tf_gc4_gated_norm, or with `mask` the image's control that keeps only those rounding points: one row a block.
pub fn launch(f: cuda.Function, mask: ?u32, s: cuda.Stream, rows: usize, n: usize, in: u64, x_ld: usize, w: u64, up: u64, down: u64, out: u64, o_ld: usize, eps: f32, scale: f32) !void {
    try gated_norm.check(rows, n, x_ld, o_ld);
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(in);
    a.add(@as(i64, @intCast(x_ld)));
    inline for (.{ w, up, down, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(o_ld)));
    a.add(@as(i32, @intCast(n)));
    a.add(eps);
    a.add(scale);
    if (mask) |m| a.add(@as(i32, @intCast(m)));
    const shape: cuda.launch.Config = .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = gated_norm.threads }, .shared = gated_norm.sharedBytes(n) };
    try cuda.launch.launch(f, shape, s, &a);
}

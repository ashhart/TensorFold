//! ERNIE 5.5's mHC kernels (ernie5_5/mhc.cu): the pre side, a row's mixes, Sinkhorn, the collapse and its RMSNorm.
const cuda = @import("cuda");
pub const mhc = @import("mhc.zig");

pub const threads = 256;

pub const Mhc = struct {
    mod: cuda.Module,
    pre_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Mhc {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_mhc);
        errdefer mod.unload();
        return .{ .mod = mod, .pre_fn = try mod.function("tf_ernie55_mhc_pre") };
    }

    pub fn unload(x: *Mhc) void {
        x.mod.unload();
    }

    /// bf16 rows of 4 streams, fn f32 [24][16384], scale [3], base [24], the norm's bf16 w: post, comb, bf16 rows.
    pub fn pre(x: *const Mhc, s: cuda.Stream, rows: u64, fnw: u64, scale: u64, base: u64, w: u64, post: u64, comb: u64, out: u64, sh: mhc.Shape, p: mhc.Params) !void {
        try launch(x.pre_fn, s, rows, fnw, scale, base, w, post, comb, out, sh, p);
    }
};

/// `f`, tf_ernie55_mhc_pre or an entry of the image with its arguments, one row a block.
pub fn launch(f: cuda.Function, s: cuda.Stream, rows: u64, fnw: u64, scale: u64, base: u64, w: u64, post: u64, comb: u64, out: u64, sh: mhc.Shape, p: mhc.Params) !void {
    try mhc.check(sh, p);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(rows);
    a.add(@as(i64, @intCast(sh.xs)));
    inline for (.{ fnw, scale, base, w, post, comb, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    inline for (.{ p.rms_eps, p.hc_eps, p.norm_eps }) |v| a.add(v);
    a.add(@as(c_int, @intCast(p.iters)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

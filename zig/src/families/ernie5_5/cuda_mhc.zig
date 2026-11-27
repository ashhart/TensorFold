//! ERNIE 5.5's mHC kernels (ernie5_5/mhc.cu): the pre side, its broadcast, the post side and hc_head.
const cuda = @import("cuda");
pub const mhc = @import("mhc.zig");

pub const threads = 256;

pub const Mhc = struct {
    mod: cuda.Module,
    pre_fn: cuda.Function,
    bcast_fn: cuda.Function,
    post_fn: cuda.Function,
    head_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Mhc {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_mhc);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .pre_fn = try mod.function("tf_ernie55_mhc_pre"),
            .bcast_fn = try mod.function("tf_ernie55_mhc_pre_broadcast"),
            .post_fn = try mod.function("tf_ernie55_mhc_post"),
            .head_fn = try mod.function("tf_ernie55_hc_head"),
        };
    }

    pub fn unload(x: *Mhc) void {
        x.mod.unload();
    }

    /// bf16 rows of 4 streams, fn f32 [24][16384], scale [3], base [24], the norm's bf16 w: post, comb, bf16 rows.
    pub fn pre(x: *const Mhc, s: cuda.Stream, rows: u64, fnw: u64, scale: u64, base: u64, w: u64, post: u64, comb: u64, out: u64, sh: mhc.Shape, p: mhc.Params) !void {
        try launch(x.pre_fn, s, rows, fnw, scale, base, w, post, comb, out, sh, p);
    }

    /// The first layer's bf16 rows e, fnb f32 [24][4096]: res's 4 streams, post, comb, bf16 rows (mhc.preBroadcast).
    pub fn broadcast(x: *const Mhc, s: cuda.Stream, e: u64, fnb: u64, scale: u64, base: u64, w: u64, res: u64, post: u64, comb: u64, out: u64, sh: mhc.BcastShape, p: mhc.Params) !void {
        try launchBcast(x.bcast_fn, s, e, fnb, scale, base, w, res, post, comb, out, sh, p);
    }

    /// bf16 layer outputs into their rows' 4 streams with pre's post [4] and comb [16]: bf16 streams, as mhc.post.
    pub fn postSide(x: *const Mhc, s: cuda.Stream, out_x: u64, res: u64, post: u64, comb: u64, out: u64, sh: mhc.PostShape) !void {
        try launchPost(x.post_fn, s, out_x, res, post, comb, out, sh);
    }

    /// The last layer's 4 streams into one bf16 row with hc_head's fn [4][16384], scale [1] and base [4], as mhc.head.
    pub fn head(x: *const Mhc, s: cuda.Stream, rows: u64, fnw: u64, scale: u64, base: u64, out: u64, sh: mhc.HeadShape, rms_eps: f32, hc_eps: f32) !void {
        try launchHead(x.head_fn, s, rows, fnw, scale, base, out, sh, rms_eps, hc_eps);
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

/// `f`, tf_ernie55_mhc_pre_broadcast or an entry of the image with its arguments, one row a block.
pub fn launchBcast(f: cuda.Function, s: cuda.Stream, e: u64, fnb: u64, scale: u64, base: u64, w: u64, res: u64, post: u64, comb: u64, out: u64, sh: mhc.BcastShape, p: mhc.Params) !void {
    try mhc.checkBcast(sh, p);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(e);
    a.add(@as(i64, @intCast(sh.es)));
    inline for (.{ fnb, scale, base, w, res }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.rs)));
    inline for (.{ post, comb, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    inline for (.{ p.rms_eps, p.hc_eps, p.norm_eps }) |v| a.add(v);
    a.add(@as(c_int, @intCast(p.iters)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

/// `f`, tf_ernie55_mhc_post or an entry of the image with its arguments, one row a block.
pub fn launchPost(f: cuda.Function, s: cuda.Stream, x: u64, res: u64, post: u64, comb: u64, out: u64, sh: mhc.PostShape) !void {
    try mhc.checkPost(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(sh.xs)));
    a.add(res);
    a.add(@as(i64, @intCast(sh.rs)));
    inline for (.{ post, comb, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

/// `f`, tf_ernie55_hc_head or an entry of the image with its arguments, one row a block.
pub fn launchHead(f: cuda.Function, s: cuda.Stream, x: u64, fnw: u64, scale: u64, base: u64, out: u64, sh: mhc.HeadShape, rms_eps: f32, hc_eps: f32) !void {
    try mhc.checkHead(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(sh.xs)));
    inline for (.{ fnw, scale, base, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    a.add(rms_eps);
    a.add(hc_eps);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

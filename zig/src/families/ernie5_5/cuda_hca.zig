//! ERNIE 5.5's HCA kernels (ernie5_5/hca.cu): the C128 compressor.
const cuda = @import("cuda");
pub const hca = @import("hca.zig");

pub const threads = 256;

pub const Hca = struct {
    mod: cuda.Module,
    compress_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Hca {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_hca);
        errdefer mod.unload();
        return .{ .mod = mod, .compress_fn = try mod.function("tf_ernie55_hca_compress") };
    }

    pub fn unload(x: *Hca) void {
        x.mod.unload();
    }

    /// Blocks of 128 token rows (kv, score fp32), ape f32 [128][512], bf16 w, freq [32], pos: bf16 rows (hca.compress).
    pub fn compress(x: *const Hca, s: cuda.Stream, st: u64, ape: u64, w: u64, freq: u64, pos: u64, out: u64, sh: hca.Shape, eps: f32) !void {
        try launch(x.compress_fn, s, st, ape, w, freq, pos, out, sh, eps);
    }
};

/// `f`, tf_ernie55_hca_compress or an entry of the image with its arguments, one compressed row a block.
pub fn launch(f: cuda.Function, s: cuda.Stream, st: u64, ape: u64, w: u64, freq: u64, pos: u64, out: u64, sh: hca.Shape, eps: f32) !void {
    try hca.check(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(st);
    a.add(@as(i64, @intCast(sh.ts)));
    inline for (.{ ape, w, freq, pos, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    a.add(eps);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

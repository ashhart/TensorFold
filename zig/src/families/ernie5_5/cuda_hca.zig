//! ERNIE 5.5's HCA kernels: the C128 compressor (ernie5_5/hca.cu), the attention's chunk tiles (hca_attn.cu).
const cuda = @import("cuda");
pub const hca = @import("hca.zig");

pub const threads = 256;

/// common/mla.cuh's wide_smem<16>(): 16 heads' q, two 32-key tiles and p, 512 wide, and the reductions' floats.
pub const attn_smem: u32 = (16 * hca.head_dim + 2 * 32 * hca.head_dim + 16 * 32) * 2 + (16 * 4 + 2 * 16) * 4;

/// The attention: hca_attn.cu's chunk tiles, then the mla image's sink merge (core's tf_mla_merge_sink).
pub const Attn = struct {
    mod: cuda.Module,
    chunks_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Attn {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_hca_attn);
        errdefer mod.unload();
        const f = try mod.function("tf_ernie55_hca_chunks");
        try f.allowDynamicShared(attn_smem);
        return .{ .mod = mod, .chunks_fn = f };
    }

    pub fn unload(x: *Attn) void {
        x.mod.unload();
    }

    /// q [rows][heads][512] and 512-wide key rows (bf16), row r's keys tok[r w ..][0..cnt[r]], fp32 sinks [heads].
    pub fn attend(x: *const Attn, m: *const cuda.mla.Mla, s: cuda.Stream, q: u64, keys: u64, tok: u64, cnt: u64, sinks: u64, p: cuda.mla.Partials, out: u64, sh: hca.Attn, scale: f32) !void {
        try attendWith(x.chunks_fn, m, s, q, keys, tok, cnt, sinks, p, out, sh, scale);
    }
};

/// `f`, tf_ernie55_hca_chunks or an entry with its arguments, over nch chunks of listed keys, then the sink merge.
pub fn attendWith(f: cuda.Function, m: *const cuda.mla.Mla, s: cuda.Stream, q: u64, keys: u64, tok: u64, cnt: u64, sinks: u64, p: cuda.mla.Partials, out: u64, sh: hca.Attn, scale: f32) !void {
    try hca.attnCheck(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    inline for (.{ q, keys, @as(u64, 0), tok, cnt, p.po, p.pm, p.pl }) |v| a.add(v);
    inline for (.{ sh.rows, sh.heads, sh.w }) |v| a.add(@as(i64, @intCast(v)));
    a.add(scale);
    a.add(@as(c_int, 1)); // sparse: the keys as `tok` lists them
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows), .y = @intCast((sh.heads + 15) / 16), .z = @intCast(sh.nch) }, .block = .{ .x = threads }, .shared = attn_smem }, s, &a);
    try m.mergeSink(s, p, sinks, out, sh.rows, sh.heads, sh.nch);
}

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

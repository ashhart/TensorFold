//! ERNIE 5.5's indexer kernels (ernie5_5/index.cu): its keys, and the sparse MLA's key lists (picks, then the window).
const cuda = @import("cuda");
pub const window = @import("window.zig");
pub const key = @import("key.zig");

pub const threads = 256;

pub const Index = struct {
    mod: cuda.Module,
    merge_fn: cuda.Function,
    key_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Index {
        var mod = try cuda.Module.load(d, cuda.kernels.ernie5_5_index);
        errdefer mod.unload();
        return .{ .mod = mod, .merge_fn = try mod.function("tf_ernie55_window_merge"), .key_fn = try mod.function("tf_ernie55_index_key") };
    }

    pub fn unload(x: *Index) void {
        x.mod.unload();
    }

    /// topk_rows' out and picked, pos (row r at pos[0] + r) into tok i32 [rows, w] and cnt i32 [rows], as window.merge.
    pub fn merge(x: *const Index, s: cuda.Stream, picks: u64, picked: u64, pos: u64, tok: u64, cnt: u64, sh: window.Shape) !void {
        try launch(x.merge_fn, s, picks, picked, pos, tok, cnt, sh);
    }

    /// bf16 k rows, k_norm's w and b into bf16 keys as key.key: row r at pos[0] + r, inv fp32 theta^(-2i/64).
    pub fn keys(x: *const Index, s: cuda.Stream, k: u64, w: u64, b: u64, out: u64, pos: u64, inv: u64, sh: key.Shape) !void {
        try launchKey(x.key_fn, s, k, w, b, out, pos, inv, sh);
    }
};

/// `f`, tf_ernie55_window_merge or an entry of the image with its arguments, one row a block.
pub fn launch(f: cuda.Function, s: cuda.Stream, picks: u64, picked: u64, pos: u64, tok: u64, cnt: u64, sh: window.Shape) !void {
    try window.check(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    inline for (.{ picks, picked, pos, tok, cnt }) |v| a.add(v);
    a.add(@as(c_int, @intCast(sh.k)));
    a.add(@as(c_int, @intCast(sh.window)));
    a.add(@as(i64, @intCast(sh.w)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = threads } }, s, &a);
}

/// `f`, tf_ernie55_index_key or an entry of the image with its arguments, one row a warp.
pub fn launchKey(f: cuda.Function, s: cuda.Stream, k: u64, w: u64, b: u64, out: u64, pos: u64, inv: u64, sh: key.Shape) !void {
    try key.check(sh);
    if (sh.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(k);
    a.add(@as(i64, @intCast(sh.ks)));
    inline for (.{ w, b, out }) |v| a.add(v);
    a.add(@as(i64, @intCast(sh.os)));
    a.add(pos);
    a.add(key.eps);
    a.add(inv);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(sh.rows) }, .block = .{ .x = 32 } }, s, &a);
}

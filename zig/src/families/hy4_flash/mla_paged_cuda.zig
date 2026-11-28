//! Hy4-flash absorbed gated MLA launchers (`hy4_mla_paged`): absorb, sinked attention, value+gate, post norm.
const std = @import("std");
const cuda = @import("cuda");

pub const c1 = @import("mla_c1_cuda.zig");
pub const c2 = @import("mla_c2_cuda.zig");

const Stream = cuda.Stream;
const Function = cuda.Function;
const Args = cuda.Args;

pub const latent: usize = 512;
pub const rope: usize = 64;
pub const nope: usize = 192;
pub const v_width: usize = 256;
pub const q_head: usize = nope + rope;
pub const kv_head: usize = nope + v_width;
pub const row: usize = latent + rope;
pub const heads_tile: usize = 8;
pub const scale: f32 = 0.0625; // (nope + rope)^-0.5, exact
pub const attn_smem: u32 = heads_tile * row * 4 + 32 * (row + 2) * 2 + heads_tile * 32 * 4 + 32 * 4 + 3 * heads_tile * 4;
pub const post_most: usize = 4096;

pub const Kernels = struct {
    mod: cuda.Module,
    absorb: Function,
    attn: Function,
    attn_rows: Function,
    value: Function,
    post: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_mla_paged);
        errdefer mod.unload();
        const k: Kernels = .{
            .mod = mod,
            .absorb = try mod.function("tf_hy4_mla_absorb"),
            .attn = try mod.function("tf_hy4_mla_attn"),
            .attn_rows = try mod.function("tf_hy4_mla_attn_rows"),
            .value = try mod.function("tf_hy4_mla_value"),
            .post = try mod.function("tf_hy4_mla_post"),
        };
        try k.attn.allowDynamicShared(attn_smem);
        try k.attn_rows.allowDynamicShared(attn_smem);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

fn int(v: usize) c_int {
    return @intCast(v);
}

pub fn absorb(f: Function, s: Stream, q: u64, qs: usize, kvb: u64, ql: u64, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ q, @as(i64, @intCast(qs)), kvb, ql, int(rows), int(heads) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = latent / 128, .z = cdiv(rows, 16) }, .block = .{ .x = 128 } }, s, &a);
}

pub const Keys = struct { pos: u64, window: usize = 0, idx: u64 = 0, nidx: usize = 0, ring: usize = 0 };

pub fn attn(f: Function, s: Stream, ql: u64, qr: u64, cache: u64, keys: Keys, sinks: u64, ol: u64, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    if (heads % heads_tile != 0) return error.MlaHeadsSplit;
    var a: Args = .{};
    inline for (.{ ql, qr, cache, keys.pos, keys.idx, sinks, ol }) |v| a.add(v);
    inline for (.{ heads, keys.window, keys.nidx, keys.ring }) |v| a.add(int(v));
    a.add(scale);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads / heads_tile) }, .block = .{ .x = 256 }, .shared = attn_smem }, s, &a);
}

pub fn value(f: Function, s: Stream, ol: u64, kvb: u64, gate: u64, gs: usize, out: u64, os: usize, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ol, kvb, gate, @as(i64, @intCast(gs)), out, @as(i64, @intCast(os)), int(rows), int(heads) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = v_width / 32, .z = cdiv(rows, 16) }, .block = .{ .x = 128 } }, s, &a);
}

pub fn post(f: Function, s: Stream, parts: u64, world: usize, w: u64, out: u64, rows: usize, width: usize, eps: f32) !void {
    if (rows == 0) return;
    if (width > post_most or world == 0) return error.RowShape;
    var a: Args = .{};
    inline for (.{ parts, @as(i64, @intCast(rows * width)), int(world), w, out, int(width) }) |v| a.add(v);
    a.add(eps);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
}

test "mla_paged_cuda: attention shared bytes match contract record" {
    try std.testing.expectEqual(@as(u32, 56672), attn_smem);
}

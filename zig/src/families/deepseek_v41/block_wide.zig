//! Decode windows of more than 16 rows (TF_DSV41_ROWS_CAP past 16): mHC sites and router config past mhc_cuda's 16.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const moe = @import("block_moe.zig");
const E = block.Emitter;
const Arg = calls.Arg;

/// mhc_cuda / mhc_dec's MAX_ROWS: windows of more rows take this file's path.
pub const cuda_rows: i64 = 16;
/// TF_DSV41_MHC_SPLIT_ROWS (32): split launches and the second streams buffer up to here.
pub const split_rows: i64 = 32;
/// attn_cuda.MAX_ROWS / dtopk.MAX_ROWS / rowtab's largest bucket: the widest decode window.
pub const max_rows: i64 = 64;

pub const Mode = enum { boundary, site1, site2, final };

fn coefs(e: *E, set: u1) ![3]Arg {
    return .{
        try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
        try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
    };
}

/// One mHC site of layer L (`which`: hc_attn / hc_ffn) or the final norm (L null) for a window of more than 16 rows.
pub fn mhc(e: *E, layer: ?u32, which: []const u8, mode: Mode, in_place: bool, norm: ?[]const u8, tap: ?Arg) !void {
    const n = e.n;
    const pf = pfOn(e, mode);
    if ((n <= cuda_rows and !pf) or n > max_rows) return error.WideRows;
    // PFDEC's deferred coefficients (MHC_DEFER): this site reads them (a no-op without a pending launch)
    moe.awaitCoefs(e);
    const D: i64 = e.cfg.hidden;
    const hm: i64 = e.cfg.hc_mult;
    const post = mode == .boundary or mode == .final;
    const mix = mode != .final;
    const x = try e.streams();
    const prev = try coefs(e, e.coef);
    const next = try coefs(e, 1 - e.coef);
    const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
    const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
    // the forward's second streams buffer: a boundary of at most 32 rows writes it; a final writes the streams
    const spare = mode == .boundary and !in_place and n <= split_rows;
    const xout: Arg = if (spare) try e.spare() else x;
    const g: ?Arg = if (post) try e.buf("w.recv.{s}", .{if (std.mem.eql(u8, which, "hc_ffn")) "attn" else "moe"}, .bf16, &.{ e.o.world, n, D }) else null;
    const collapse: i64 = if (mode == .site1) 1 else 2;
    const pre: ?Arg = if (collapse == 2) prev[0] else null;
    const fnw: ?Arg = if (mix) try e.buf("s.L{d}.{s}.fn16", .{ layer.?, which }, .bf16, &.{ 6 * hm, hm * D }) else null;
    if (pf) {
        const L = layer.?;
        try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
            x,                                        if (post) xout else try e.empty(.bf16),
            g orelse try e.empty(.bf16),              if (post) prev[1] else try e.empty(.f32),
            if (post) prev[2] else try e.empty(.f32), pre orelse try e.empty(.f32),
            fnw.?,                                    part,
            c,                                        tap orelse try e.empty(.bf16),
            .{ .i = n },                              .{ .i = @backingInt(mode) },
        });
        try finishK(e, null, part, c, try normOf(e, layer, which, norm));
        if (e.o.mhc_defer) {
            e.df_side = true; // MHC_DEFER: on the deferred stream, the next mHC call joins it
            e.df_pending = true;
        }
        try e.ext("tf_dsv41_mhc_cuda_v1.coef", &.{
            part,                                                       try e.weight("L{d}.{s}.base", .{ L, which }, .f32, &.{6 * hm}),
            try e.weight("L{d}.{s}.scale", .{ L, which }, .f32, &.{3}), next[0],
            next[1],                                                    next[2],
            .{ .i = n },                                                .{ .f = E.dec(e.cfg.eps) },
            .{ .f = E.dec(e.cfg.hc_eps) },                              .{ .f = E.dec(e.cfg.hc_post_alpha) },
            .{ .i = e.cfg.hc_sinkhorn_iters },
        });
        e.coef = 1 - e.coef;
        if (spare) e.cur = 1 - e.cur;
        return;
    }
    const split = mix and n <= split_rows and !(post and !spare);
    try e.triton("_site", .{ std.math.divCeil(i64, n, 16) catch unreachable, 40, if (split) 4 else 1 }, &.{
        .{ .name = "X", .arg = x },                                                    .{ .name = "x_stride", .arg = .{ .i = hm * D } },
        .{ .name = "XOUT", .arg = if (post) xout else x },                             .{ .name = "G", .arg = g orelse x },
        .{ .name = "g_rank", .arg = .{ .i = if (post) n * D else 0 } },                .{ .name = "POST", .arg = if (post) prev[1] else x },
        .{ .name = "COMB", .arg = if (post) prev[2] else x },                          .{ .name = "PRE", .arg = pre orelse x },
        .{ .name = "FN", .arg = fnw orelse x },                                        .{ .name = "PART", .arg = part },
        .{ .name = "C", .arg = c },                                                    .{ .name = "TAP", .arg = tap orelse x },
        .{ .name = "tap_stride", .arg = .{ .i = if (tap) |t| t.t.stride[0] else 0 } }, .{ .name = "R", .arg = .{ .i = n } },
        .{ .name = "D", .arg = .{ .i = D } },                                          .{ .name = "NB", .arg = .{ .i = 40 } },
        .{ .name = "BM", .arg = .{ .i = 16 } },                                        .{ .name = "BK", .arg = .{ .i = 16 } },
        .{ .name = "WORLD", .arg = .{ .i = if (post) e.o.world else 1 } },             .{ .name = "POST_ON", .arg = .{ .b = post } },
        .{ .name = "COLLAPSE", .arg = .{ .i = collapse } },                            .{ .name = "MIX", .arg = .{ .b = mix } },
        .{ .name = "TAP_ON", .arg = .{ .b = tap != null } },                           .{ .name = "SPLIT", .arg = .{ .b = split } },
    });
    try finishK(e, if (mix) .{ layer.?, which, next } else null, part, c, try normOf(e, layer, which, norm));
    if (mix) e.coef = 1 - e.coef;
    if (spare) e.cur = 1 - e.cur;
}

/// TF_DSV41_MHC_PFDEC: this window's mixing sites take mhc_pf + `coef_kernel` (`mhc`'s `pf` branch).
pub fn pfOn(e: *const E, mode: anytype) bool {
    const m = e.o.mhc_pf_rows;
    return m > 0 and e.n >= m and e.n <= max_rows and @backingInt(mode) != @backingInt(Mode.final);
}

/// A site's (or the final's) norm weight: the layer's attn / ffn norm, the model's (or `norm`'s) at the final.
fn normOf(e: *E, layer: ?u32, which: []const u8, norm: ?[]const u8) !Arg {
    const D: i64 = e.cfg.hidden;
    if (layer) |L| return e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D});
    return e.weight("{s}", .{norm orelse "norm"}, .f32, &.{D});
}

/// The mHC finish kernel: with `coef` also the next site's coefficients, else the normed input alone.
fn finishK(e: *E, coef: ?struct { u32, []const u8, [3]Arg }, part: Arg, c: Arg, nw: Arg) !void {
    const n = e.n;
    const D: i64 = e.cfg.hidden;
    const hm: i64 = e.cfg.hc_mult;
    const mix = coef != null;
    try e.triton("_finish_k", .{ n, 1, 1 }, &.{
        .{ .name = "PART", .arg = part },
        .{ .name = "BASE", .arg = if (coef) |k| try e.weight("L{d}.{s}.base", .{ k[0], k[1] }, .f32, &.{6 * hm}) else part },
        .{ .name = "SCALE", .arg = if (coef) |k| try e.weight("L{d}.{s}.scale", .{ k[0], k[1] }, .f32, &.{3}) else part },
        .{ .name = "PRE", .arg = if (coef) |k| k[2][0] else part },
        .{ .name = "POST", .arg = if (coef) |k| k[2][1] else part },
        .{ .name = "COMB", .arg = if (coef) |k| k[2][2] else part },
        .{ .name = "C", .arg = c },
        .{ .name = "NW", .arg = nw },
        .{ .name = "OUT", .arg = try e.buf("w.out", .{}, .bf16, &.{ n, D }) },
        .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
        .{ .name = "hc_eps", .arg = .{ .f = E.dec(e.cfg.hc_eps) } },
        .{ .name = "post_alpha", .arg = .{ .f = E.dec(e.cfg.hc_post_alpha) } },
        .{ .name = "D", .arg = .{ .i = D } },
        .{ .name = "NB", .arg = .{ .i = 40 } },
        .{ .name = "ITERS", .arg = .{ .i = e.cfg.hc_sinkhorn_iters } },
        .{ .name = "COEF", .arg = .{ .b = mix } },
        .{ .name = "BLOCK", .arg = .{ .i = @intCast(std.math.gcd(@as(u64, 1024), @as(u64, @intCast(D)))) } },
        .{ .name = "UNROLL", .arg = .{ .b = true } },
        .{ .name = "NORM_FIRST", .arg = .{ .b = true } },
    });
}

/// block_moe.mhcInto's call for a window of more than 16 rows (`mode`: block_moe.Mode), with its DSpark tap.
pub fn decode(e: *E, layer: ?u32, which: []const u8, mode: anytype, in_place: bool) !void {
    const m = std.meta.stringToEnum(Mode, @tagName(mode)).?;
    var tap: ?Arg = null;
    if (e.o.taps and m == .boundary and std.mem.eql(u8, which, "hc_attn")) if (layer) |L| {
        const t = e.cfg.dspark_targets.items();
        if (std.mem.indexOfScalar(u32, t, L)) |j| {
            const D: i64 = e.cfg.hidden;
            const w: i64 = @as(i64, @intCast(t.len)) * D;
            tap = try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, D }, &.{ w, 1 }, @as(i64, @intCast(j)) * D * 2);
        }
    };
    return mhc(e, layer, which, m, in_place, null, tap);
}

/// router_gemv config at `n` rows: (row tile, warps, experts a warp); past 16 rows the prefill config.
pub fn routerCfg(n: i64) [3]i64 {
    if (n > cuda_rows) return .{ 16, 8, 2 };
    const t = std.math.ceilPowerOfTwo(u64, @intCast(n)) catch unreachable;
    return .{ @intCast(t), 2, 0 };
}

test "wide rows: router config as router_gemv.config / row_tile" {
    try std.testing.expectEqual([3]i64{ 1, 2, 0 }, routerCfg(1));
    try std.testing.expectEqual([3]i64{ 8, 2, 0 }, routerCfg(5));
    try std.testing.expectEqual([3]i64{ 16, 2, 0 }, routerCfg(16));
    try std.testing.expectEqual([3]i64{ 16, 8, 2 }, routerCfg(17));
    try std.testing.expectEqual([3]i64{ 16, 8, 2 }, routerCfg(64));
}

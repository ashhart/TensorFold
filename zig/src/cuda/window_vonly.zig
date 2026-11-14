//! window_vonly.cu: Gemma 4 31B's global layers over V-only pages (TF_GEMMA4_V_ONLY), each key made from its value.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const wa = @import("window_attention.zig");

const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;

/// Keys from values at head dim 512: the k norm's weight w (bf16 [512]) and the global RoPE's inv (f32 [256]).
pub const Vk = struct { w: u64, inv: u64 };

/// Head dim of the only layers V-only serves (Gemma 4's global layers).
pub const head_dim = 512;
/// The decode partials' two staged tiles of 32 keys (window_attention's staged_smem at 512).
const staged_smem: u32 = 2 * 32 * head_dim * 2;

pub const VOnly = struct {
    mod: module.Module,
    part: module.Function,
    merge: module.Function,
    put_fn: module.Function,
    keys_fn: module.Function,

    pub fn load(d: *const Driver) !VOnly {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.window_vonly);
        errdefer mod.unload();
        const v: VOnly = .{
            .mod = mod,
            .part = try mod.function("tf_wrows_part_512_pg_vo"),
            .merge = try mod.function("tf_wrows_merge_512_pg_vo"),
            .put_fn = try mod.function("tf_wrows_put_v"),
            .keys_fn = try mod.function("tf_vonly_keys"),
        };
        try v.part.allowDynamicShared(staged_smem);
        return v;
    }

    pub fn unload(v: *VOnly) void {
        v.mod.unload();
    }

    /// The shapes V-only serves: head dim 512, no window, a page table, at most 16 query heads a key head.
    pub fn shape(w: wa.Window) bool {
        const p = w.pg orelse return false;
        const heads = w.hk > 0 and w.h % w.hk == 0 and w.h / w.hk <= 16;
        const ring = w.ring > 0 and w.ring <= std.math.maxInt(i32);
        const pages = p.tokens > 0 and p.page_elems <= std.math.maxInt(i63);
        return w.d == head_dim and w.win == 0 and heads and ring and pages;
    }

    /// decodeWith's rows over V-only pages: partials a key head and chunk (scratchBytes(w, n)), then their merge.
    pub fn decode(
        v: *const VOnly,
        s: Stream,
        q: u64,
        vc: u64,
        out: u64,
        pos: u64,
        slot: ?u64,
        w: wa.Window,
        vk: Vk,
        scratch: u64,
        n: usize,
    ) !void {
        if (!shape(w)) return error.WindowShape;
        if (n == 0) return;
        const p = w.pg.?;
        const t = wa.WindowAttention.chunks(w);
        const parts = n * w.hk * t * (w.h / w.hk);
        const po = scratch;
        const pm = po + parts * w.d * 4;
        const pl = pm + parts * 4;
        var a: launch_.Args = .{};
        for ([_]u64{ q, vc, pos, slot orelse 0, po, pm, pl }) |x| a.add(x);
        a.add(@as(i32, @intCast(w.h)));
        a.add(@as(i32, @intCast(w.hk)));
        a.add(w.scale);
        tail(&a, p, vk);
        const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = @intCast(t) };
        try launch_.launch(v.part, .{ .grid = grid, .block = .{ .x = 256 }, .shared = staged_smem }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ po, pm, pl, out }) |x| b.add(x);
        b.add(@as(i32, @intCast(w.h)));
        b.add(@as(i32, @intCast(w.hk)));
        b.add(@as(i32, @intCast(t)));
        for ([_]u64{ q, vc, pos, slot orelse 0 }) |x| b.add(x);
        b.add(w.scale);
        tail(&b, p, vk);
        const merged: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = @intCast(w.h / w.hk * w.d / 256) };
        try launch_.launch(v.merge, .{ .grid = merged, .block = .{ .x = 256 } }, s, &b);
    }

    /// The pages' table, elements a page, positions a page, entries a slot, then the keys' w and inv.
    fn tail(a: *launch_.Args, p: wa.Pages, vk: Vk) void {
        a.add(p.table);
        a.add(@as(i64, @intCast(p.page_elems)));
        a.add(@as(i32, @intCast(p.tokens)));
        a.add(@as(i32, @intCast(p.stride)));
        a.add(vk.w);
        a.add(vk.inv);
    }

    /// `put`'s v half: n rows v [n, hk * 512] into V-only pages at pos/slot (int32 [n]).
    pub fn put(v: *const VOnly, s: Stream, rows: u64, vc: u64, pos: u64, slot: u64, w: wa.Window, n: usize) !void {
        if (!shape(w)) return error.WindowShape;
        if (n == 0) return;
        const p = w.pg.?;
        var a: launch_.Args = .{};
        for ([_]u64{ rows, vc, pos, slot, p.table }) |x| a.add(x);
        a.add(@as(i64, @intCast(p.page_elems)));
        a.add(@as(i32, @intCast(p.tokens)));
        a.add(@as(i32, @intCast(p.stride)));
        a.add(@as(i32, @intCast(w.hk * w.d)));
        try launch_.launch(v.put_fn, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// Keys [k0, k0 + n) of one stream (pg: its table row) made from their values into ks, in vc's page layout.
    pub fn keys(v: *const VOnly, s: Stream, x: Keys, k0: usize, n: usize) !void {
        return keysWith(v.keys_fn, s, x, k0, n);
    }

    /// A keys call's caches: values in, keys out (vc's layout), the stream's table row, key heads, the norm.
    pub const Keys = struct { vc: u64, ks: u64, pg: wa.Pages, hk: usize, vk: Vk };

    /// `keys` by `f` (keys_fn, or the contract's control).
    pub fn keysWith(f: module.Function, s: Stream, x: Keys, k0: usize, n: usize) !void {
        const pg = x.pg;
        if (x.hk == 0 or pg.tokens == 0 or pg.page_elems > std.math.maxInt(i63)) return error.WindowShape;
        if (k0 + n > std.math.maxInt(i32)) return error.WindowShape;
        if (n == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ x.vc, x.ks, pg.table }) |p| a.add(p);
        a.add(@as(i64, @intCast(pg.page_elems)));
        a.add(@as(i32, @intCast(pg.tokens)));
        a.add(@as(i32, @intCast(x.hk)));
        a.add(@as(i32, @intCast(k0)));
        a.add(x.vk.w);
        a.add(x.vk.inv);
        const grid: launch_.Dim3 = .{ .x = @intCast(n), .y = @intCast(x.hk) };
        try launch_.launch(f, .{ .grid = grid, .block = .{ .x = head_dim / 2 } }, s, &a);
    }
};

test "V-only serves head dim 512 over pages with no window, and nothing else" {
    const pg: wa.Pages = .{ .table = 1, .page_elems = 256 * 4 * 512, .tokens = 256, .stride = 4 };
    const w: wa.Window = .{ .h = 32, .hk = 4, .d = 512, .win = 0, .ring = 1024, .scale = 1, .pg = pg };
    try std.testing.expect(VOnly.shape(w));
    var x = w;
    x.pg = null;
    try std.testing.expect(!VOnly.shape(x));
    x = w;
    x.d = 256;
    try std.testing.expect(!VOnly.shape(x));
    x = w;
    x.win = 1024;
    try std.testing.expect(!VOnly.shape(x));
    x = w;
    x.h = 33;
    try std.testing.expect(!VOnly.shape(x));
}

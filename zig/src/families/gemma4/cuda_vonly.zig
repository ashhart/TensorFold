//! TF_GEMMA4_V_ONLY: the 31B's global pages hold V alone; attention makes each key from its value (window_vonly).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");

const Vk = cuda.window_vonly.Vk;
const Window = cuda.window_attention.Window;
const environment = @import("sys").environment;

/// The switch's text: exactly 0 or 1 through environment.zig (unset or blank: 0); any other text refused by name.
pub fn asked(raw: ?[]const u8) !bool {
    return environment.value(bool, "TF_GEMMA4_V_ONLY", raw, false) catch error.BadGemmaVOnly;
}

/// TF_GEMMA4_V_ONLY as the process has it.
pub fn read() !bool {
    return environment.read(bool, "TF_GEMMA4_V_ONLY", false) catch error.BadGemmaVOnly;
}

/// V-only pages need pages, keys that are values, several key heads, no TF_KV_SHARE and no drafter that reads keys.
pub fn refusal(paged: bool, keys_are_values: bool, key_heads: usize, kv_share: bool, drafter: bool) ?anyerror {
    if (drafter) return error.GemmaVOnlyDrafter;
    if (kv_share) return error.GemmaVOnlyKvShare;
    if (!paged) return error.GemmaVOnlyUnpaged;
    if (!keys_are_values) return error.GemmaVOnlyKeysNotValues;
    if (key_heads < 2) return error.GemmaVOnlyOneKeyHead; // the 12B: a block a key head is a quarter of the 31B's grid
    return null;
}

/// The switch, then its refusals from the checkpoint's config alone: called before a kernel, weight or stream exists.
pub fn gate(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, paged: bool, kv_share: bool, drafter: bool) !bool {
    if (!try read()) return false;
    const geo = (try cfg.Config.read(gpa, io, dir)).geo;
    if (refusal(paged, geo.keys_are_values, geo.global_kv_heads, kv_share, drafter)) |e| return e;
    return true;
}

pub fn explain(err: anyerror) ?[]const u8 {
    const flag = "TF_GEMMA4_V_ONLY=1";
    return switch (err) {
        error.BadGemmaVOnly => "TF_GEMMA4_V_ONLY: 0 or 1 (1: the global layers keep V alone, attention makes " ++
            "each key from it)",
        error.GemmaVOnlyDrafter => flag ++ ": an assistant or DFlash drafter reads the target's global keys, which " ++
            "V-only pages do not hold: serve --no-drafts, drop --drafter, or unset TF_GEMMA4_V_ONLY",
        error.GemmaVOnlyKvShare => flag ++ ": the TF_KV_SHARE store keeps K and V regions a layer: unset one of them",
        error.GemmaVOnlyUnpaged => flag ++ " needs the global caches in pool pages: unset " ++
            "TENSORFOLD_GEMMA4_KV_PAGES=0 or TF_GEMMA4_V_ONLY",
        error.GemmaVOnlyOneKeyHead => flag ++ ": this model's global layers have one key head, where V-only " ++
            "attention is slower than K+V: unset TF_GEMMA4_V_ONLY",
        error.GemmaVOnlyKeysNotValues => flag ++ ": this checkpoint's global layers have their own v_proj " ++
            "(attention_k_eq_v false), so keys are not made from values: unset TF_GEMMA4_V_ONLY",
        else => null,
    };
}

/// The bytes a stream's global keys take at `context`, which V-only pages do not hold.
pub fn keyBytes(geo: cfg.Geometry, layers: usize, context: usize) usize {
    var total: usize = 0;
    for (0..@min(layers, geo.cacheLayers())) |i| {
        if (cfg.global(i)) total += context * geo.kvHeads(i) * cfg.headDim(i) * 2;
    }
    return total;
}

/// Global layer i's keys from its values: its k norm's weight and the global RoPE's inverse frequencies.
fn keys(m: anytype, i: usize) Vk {
    return .{ .w = m.w.layers[i].k_norm, .inv = m.inv_global };
}

/// `put` on V-only pages: rows' v alone (the keys are made again where they are read).
pub fn put(m: anytype, k: anytype, s: cuda.Stream, x: anytype, i: usize, w: Window, rows: usize) !void {
    const v = &(k.vo orelse return error.GemmaVOnlyNotLoaded);
    try v.put(s, x.vr, m.vc[i], x.pos, x.slot, w, rows);
}

/// Global layer i's attention over V-only pages: chain c's prompt chunk, or decode rows `step` at a launch.
pub fn attend(
    m: anytype,
    k: anytype,
    s: cuda.Stream,
    x: anytype,
    i: usize,
    w: Window,
    c: anytype,
    rows: usize,
    prompt: bool,
    step: usize,
) !void {
    const v = &(k.vo orelse return error.GemmaVOnlyNotLoaded);
    if (prompt) { // the chunk's keys [0, p0 + rows) made once into the scratch; the K+V prompt kernel reads them
        const pg = m.pagesFor(c.slot).?;
        try v.keys(s, .{ .vc = m.vc[i], .ks = m.k_scratch, .pg = pg, .hk = w.hk, .vk = keys(m, i) }, 0, c.p0 + rows);
        const pw: Window = .{ .h = w.h, .hk = w.hk, .d = w.d, .win = 0, .ring = 0, .scale = 1, .p32 = true, .pg = pg };
        return k.win.promptRows(s, x.qr, m.k_scratch, m.vc[i], x.att, c.p0, rows, pw, null);
    }
    const qw = w.h * w.d * 2;
    var r: usize = 0;
    while (r < rows) : (r += step) {
        const n = @min(step, rows - r);
        const q = x.qr + r * qw;
        try v.decode(s, q, m.vc[i], x.att + r * qw, x.pos + r * 4, x.slot + r * 4, w, keys(m, i), x.part, n);
    }
}

test "TF_GEMMA4_V_ONLY takes exactly 0 or 1 through environment.zig; every other text is refused by name" {
    try std.testing.expect(!try asked(null));
    try std.testing.expect(!try asked(""));
    try std.testing.expect(!try asked("0"));
    try std.testing.expect(try asked("1"));
    for ([_][]const u8{ "off", "false", "no", "on", "2", " 0", "1 ", "yes", "true" }) |v|
        try std.testing.expectError(error.BadGemmaVOnly, asked(v));
}

test "TF_GEMMA4_V_ONLY refuses drafters, the TF_KV_SHARE store, linear caches, the E-series and the 12B by name" {
    const t = std.testing;
    try t.expectEqual(@as(?anyerror, null), refusal(true, true, 4, false, false));
    try t.expectEqual(@as(?anyerror, error.GemmaVOnlyDrafter), refusal(true, true, 4, false, true));
    try t.expectEqual(@as(?anyerror, error.GemmaVOnlyKvShare), refusal(true, true, 4, true, false));
    try t.expectEqual(@as(?anyerror, error.GemmaVOnlyUnpaged), refusal(false, true, 4, false, false));
    const e4b = cfg.e4b.keys_are_values;
    try t.expectEqual(@as(?anyerror, error.GemmaVOnlyKeysNotValues), refusal(true, e4b, 2, false, false));
    const heads = cfg.unified12b.global_kv_heads;
    try t.expectEqual(@as(?anyerror, error.GemmaVOnlyOneKeyHead), refusal(true, true, heads, false, false));
    const named = [_]anyerror{
        error.BadGemmaVOnly,
        error.GemmaVOnlyDrafter,
        error.GemmaVOnlyKvShare,
        error.GemmaVOnlyUnpaged,
        error.GemmaVOnlyKeysNotValues,
        error.GemmaVOnlyOneKeyHead,
    };
    for (named) |e|
        try t.expect(std.mem.indexOf(u8, explain(e).?, "TF_GEMMA4_V_ONLY") != null);
}

test "V-only drops the global keys alone: the 31B's 10 global layers, 1280 MiB a stream at 32K" {
    const g = cfg.dense31b;
    var globals: usize = 0;
    for (0..g.cacheLayers()) |i| globals += @intFromBool(cfg.global(i));
    try std.testing.expectEqual(@as(usize, 10), globals);
    try std.testing.expectEqual(globals * 32768 * g.global_kv_heads * cfg.global_dim * 2, keyBytes(g, g.layers, 32768));
    try std.testing.expectEqual(@as(usize, 1280) << 20, keyBytes(g, g.layers, 32768));
}

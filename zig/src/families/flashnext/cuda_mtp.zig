//! Flash Next's MTP drafts on CUDA (Python qwen4_exp/cuda/mtp.py): the head, its draft chains and verify windows.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const vocab = wt.vocab;
const eps = wt.eps;
const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const state_n = wt.state_n;
const out_sk = wt.out_sk;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;

pub const Sampling = lanes.sampling.Sampling;

/// Drafts a verify window can hold beside its pending token.
pub const max_chain: usize = max_rows - 1;
/// Preserve the existing recipe unless an explicit depth is selected.
pub const default_depth: usize = 6;
/// The live recipe's --mtp-confidence (Python's CONFIDENCE default is 0.7).
pub const default_confidence: f64 = 0.6;
/// A verify window's rows at most: the pending token and a chain, or a copied continuation.
pub const max_rows: usize = 16;
/// qmm.split_for(2560, 2560): fc_embedding and fc_hidden sum K in four slices, in order.
pub const fc_sk: usize = 4;
/// Head rows one launch takes at most (a prompt's rows go through in pieces).
pub const piece_rows: usize = 256;
/// Normed-stream partial sums a row (cuda_batch.zig's pss_n).
pub const pss_n = (dims / 256) * streams;
pub const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
pub const proj_bytes = gdn.proj_width * 2;

const vocab_text = @embedFile("draft_vocab_default.txt");

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// The draft vocabulary's ids, sorted (Python draft_token_ids("default")): column j of the draft head is ids[j].
pub fn draftIds(gpa: std.mem.Allocator) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, vocab_text, " \t\r\n,");
    while (it.next()) |part| {
        const id = try std.fmt.parseInt(u32, part, 10);
        if (id >= vocab) continue;
        try out.append(gpa, id);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (out.items, 0..) |id, i| {
        if (i > 0 and id == out.items[n - 1]) continue;
        out.items[n] = id;
        n += 1;
    }
    out.shrinkRetainingCapacity(n);
    return out.toOwnedSlice(gpa);
}

// ---- sampling ------------------------------------------------------------------------------------------------------

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A row's first maximum (argmax's choice): its column.
pub fn argmaxRow(row: []const u16) usize {
    var best = promote(row[0]);
    var at: usize = 0;
    for (row[1..], 1..) |b, i| {
        const v = promote(b);
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return at;
}

/// The keyed draw over one bf16 score row, column j being token ids[j] (or j), as lanes.sampling.choose draws.
pub fn drawRow(gpa: std.mem.Allocator, row: []const u16, ids: ?[]const u32, position: u64, sm: ?Sampling) !u32 {
    const id = struct {
        fn of(map: ?[]const u32, col: usize) u64 {
            return if (map) |m| m[col] else col;
        }
    }.of;
    const s = sm orelse return @intCast(id(ids, argmaxRow(row)));
    if (s.temperature <= 0) return @intCast(id(ids, argmaxRow(row)));
    const n = row.len;
    const k = if (s.top_k == 0 or s.top_k >= n) n else @as(usize, s.top_k);
    const vals = try gpa.alloc(f64, k);
    defer gpa.free(vals);
    const keys = try gpa.alloc(u64, k);
    defer gpa.free(keys);
    if (k == n) {
        for (vals, keys, row, 0..) |*v, *key, b, col| {
            v.* = promote(b);
            key.* = id(ids, col);
        }
    } else {
        // the k best columns, best first; a later column with an equal value ranks below (the id order)
        const top_v = try gpa.alloc(f32, k);
        defer gpa.free(top_v);
        const top_c = try gpa.alloc(usize, k);
        defer gpa.free(top_c);
        var held: usize = 0;
        for (row, 0..) |b, col| {
            const v = promote(b);
            if (held == k and !(v > top_v[k - 1])) continue;
            var at: usize = if (held < k) held else k - 1;
            if (held < k) held += 1;
            while (at > 0 and v > top_v[at - 1]) : (at -= 1) {
                top_v[at] = top_v[at - 1];
                top_c[at] = top_c[at - 1];
            }
            top_v[at] = v;
            top_c[at] = col;
        }
        for (vals, keys, top_v, top_c) |*v, *key, tv, c| {
            v.* = tv;
            key.* = id(ids, c);
        }
    }
    return @intCast(try lanes.sampling.choose(gpa, vals, keys, position, s));
}

// ---- the head's weights --------------------------------------------------------------------------------------------

pub const Head = struct {
    gpa: std.mem.Allocator,
    norm_e: cuda.DeviceBuffer, // [D] fp32
    norm_h: cuda.DeviceBuffer, // [S D] fp32
    fc_e: Q4,
    fc_h: Q4,
    layer: load.Layer,
    mixer: HcW,
    /// lm_head's rows at `ids`, in their order.
    draft: Q4,
    ids: []u32,
    module: cuda.Module,
    best_fn: cuda.Function,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, dir: []const u8) !*Head {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.gpa = gpa;
        var store = try wt.Store.open(gpa, io, dir);
        defer store.deinit();
        try store.detectNorms();
        var buf: [120]u8 = undefined;
        const ne = try store.normScale(gpa, try store.tensor("language_model.mtp.pre_fc_norm_embedding.weight"));
        defer gpa.free(ne);
        const nh = try store.normScale(gpa, try store.tensor("language_model.mtp.pre_fc_norm_hidden.weight"));
        defer gpa.free(nh);
        if (ne.len != dims or nh.len != wide) return error.UnexpectedTensor;
        h.norm_e = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ne));
        errdefer h.norm_e.free();
        h.norm_h = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nh));
        errdefer h.norm_h.free();
        h.fc_e = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_embedding");
        errdefer h.fc_e.free();
        h.fc_h = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_hidden");
        errdefer h.fc_h.free();
        h.mixer = try wt.loadHc(gpa, driver, &store, "language_model.mtp.hyper_connection_mixer", false);
        errdefer h.mixer.free();
        {
            var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
            defer pack_mod.unload();
            const pack_fn = try pack_mod.function(wt.pack_symbol);
            h.layer = try load.loadLayerAt(gpa, driver, stream, &store, pack_fn, "language_model.mtp.layers.0", false, false);
        }
        errdefer h.layer.free(gpa);
        h.ids = try draftIds(gpa);
        errdefer gpa.free(h.ids);
        h.draft = try @import("cuda_mtp_load.zig").draft(gpa, driver, &store, h.ids);
        errdefer h.draft.free();
        h.module = try cuda.Module.load(driver, cuda.kernels.flashnext_mtp);
        errdefer h.module.unload();
        h.best_fn = try h.module.function("fn_row_best");
        return h;
    }

    fn fc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *wt.Store, buf: []u8, name: []const u8) !Q4 {
        _ = buf;
        return @import("cuda_projection_load.zig").load(gpa, driver, store, name, dims, dims);
    }

    pub fn deinit(h: *Head) void {
        h.module.unload();
        h.draft.free();
        h.gpa.free(h.ids);
        h.layer.free(h.gpa);
        h.mixer.free();
        h.fc_h.free();
        h.fc_e.free();
        h.norm_h.free();
        h.norm_e.free();
        h.gpa.destroy(h);
    }

    /// fn_row_best over `rows` rows of `v` scores `ld` apart: [rows, 4] fp32 at `out` (best, log-sum-exp, id bits).
    pub fn best(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.best_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }
};

/// Device bytes the head's weights take beyond the main model's (for the server's admission estimate).
pub fn headBytes() usize {
    const experts = (wt.experts_n + 1) * 3 * wt.moe_w * dims / 2 * 5 / 4;
    const attn = (wt.proj_n + dims) * dims / 2 * 5 / 4;
    const draft = 79_591 * dims / 2 * 5 / 4;
    return experts + attn + draft + 2 * dims * dims / 2 * 5 / 4;
}

pub const Pick = @import("cuda_mtp_draft.zig").Pick;
pub const Chain = @import("cuda_mtp_draft.zig").Chain;
pub const Draft = @import("cuda_mtp_draft.zig").Draft;
pub const Ops = @import("cuda_mtp_ops.zig").Ops;
pub const check = @import("cuda_mtp_check.zig").check;

test "the draft vocabulary is 79,591 sorted ids under the vocabulary" {
    const ids = try draftIds(std.testing.allocator);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 79_591), ids.len);
    for (ids[1..], ids[0 .. ids.len - 1]) |b, a| try std.testing.expect(b > a);
}

test "a keyed draw over the top-k columns is the draw over every column" {
    const gpa = std.testing.allocator;
    var row: [300]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&row) |*v| v.* = @as(u16, 0x3f00) + prng.random().int(u8); // values near 0.5..1, many ties
    const all = struct {
        fn draw(a: std.mem.Allocator, r: []const u16, position: u64, s: Sampling) !u64 {
            const vals = try a.alloc(f64, r.len);
            defer a.free(vals);
            const ids = try a.alloc(u64, r.len);
            defer a.free(ids);
            for (vals, ids, r, 0..) |*v, *id, b, i| {
                v.* = promote(b);
                id.* = i;
            }
            return lanes.sampling.choose(a, vals, ids, position, s);
        }
    }.draw;
    for (0..64) |position| {
        const s: Sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
        try std.testing.expectEqual(try all(gpa, &row, position, s), try drawRow(gpa, &row, null, position, s));
    }
    try std.testing.expectEqual(@as(u32, @intCast(argmaxRow(&row))), try drawRow(gpa, &row, null, 0, null));
}

test "draft chains fit one verify window while the default stays six" {
    try std.testing.expectEqual(@as(usize, 16), max_chain + 1);
    try std.testing.expectEqual(@as(usize, 6), default_depth);
    try std.testing.expect(default_depth <= max_chain);
}

//! The checkpoint's whole inventory spelled the pack's way → the load plan: the EXL3 groups from the
//! widened admission (exl3.zig) plus every passthrough tensor the family reads, header-checked by
//! name, dtype and shape exactly as the device loader takes them (the `L.get` law of glm/cuda_weights.zig).

const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const exl3 = @import("exl3.zig");

const checkpoint = core.checkpoint;
const DType = checkpoint.DType;

/// The routed experts' projections in the engine's order. Measured against the real EAQ v1 pack's
/// shard headers: w1 carries [hidden → moe_inter] (gate), w3 carries [hidden → moe_inter] (up),
/// w2 carries [moe_inter → hidden] (down) — the fork's w2-up/w3-down convention is reversed here.
pub const ExpertProj = enum {
    gate,
    up,
    down,

    /// The checkpoint spelling under `block_sparse_moe.experts.<e>.`, verified against the pack's shard headers.
    pub fn ckpt(p: ExpertProj) []const u8 {
        return switch (p) {
            .gate => "w1",
            .up => "w3",
            .down => "w2",
        };
    }
};

// The tensors' prefix of the language model's per-layer space (the VL wrapper: language_model.model.*).

/// The layers' prefix, verbatim from the pack (the head sits one level up: language_model.lm_head.*).
pub const ckpt_prefix = "language_model.model.layers";

/// The pack's per-module EXL3 suffix words, exactly the core reader's file layout (`core/exl3_load.zig`).
pub const exl3_suffixes = [_][]const u8{ ".trellis", ".suh", ".svh", ".mul1" };

const embed_name = "language_model.model.embed_tokens.weight";
const final_norm_name = "language_model.model.norm.weight";

/// A routed expert `e` of layer `d`: its checkpoint tensor for `proj`, without an EXL3 suffix.
pub fn expertName(buf: []u8, d: u32, e: u32, proj: ExpertProj) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.block_sparse_moe.experts.{d}.{s}", .{ ckpt_prefix, d, e, proj.ckpt() });
}

/// The shared expert's tensors keep the standard spellings (EXL3 here too, at the pack's mixed 5-bit widths).
pub fn sharedName(buf: []u8, d: u32, proj: ExpertProj) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.block_sparse_moe.shared_experts.{s}_proj", .{ ckpt_prefix, d, @tagName(proj) });
}

/// The dense layers' (first dense_layers') MLP tensors under the plain mlp space.
pub fn denseName(buf: []u8, d: u32, proj: ExpertProj) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.mlp.{s}_proj", .{ ckpt_prefix, d, @tagName(proj) });
}

/// The router's score weight (fp32, one row per expert) and its per-expert use_routing_bias.
pub fn routerName(buf: []u8, d: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.block_sparse_moe.gate.weight", .{ ckpt_prefix, d });
}

pub fn routerBiasName(buf: []u8, d: u32) ![]const u8 {
    // measured on the real EAQ v1 pack: the bias sits beside the gate, not under it
    return std.fmt.bufPrint(buf, "{s}.{d}.block_sparse_moe.e_score_correction_bias", .{ ckpt_prefix, d });
}

/// The sparse indexer's projections (index_q/index_k, fp16) and their per-head norms (bf16).
pub fn indexerName(buf: []u8, d: u32, comptime side: []const u8, comptime tail: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.self_attn.index_{s}_{s}", .{ ckpt_prefix, d, side, tail });
}

/// The attention's q/k per-head norms (qk_norm_type "per_head") and the attention projections themselves.
pub fn attnName(buf: []u8, d: u32, comptime part: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.self_attn.{s}", .{ ckpt_prefix, d, part });
}

/// The layer norms' gemma-style pairs (input before attention, post_attention before the MLP).
pub fn layerNormName(buf: []u8, d: u32, comptime which: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{d}.{s}", .{ ckpt_prefix, d, which });
}

/// One load item: a passthrough row or one EXL3 group, with its checkpoint spelling and byte size.
pub const Item = struct {
    name: []u8, // owned; groups carry their `.trellis` prefix (the device loader appends the suffix words)
    layer: u32, // the layer it belongs to; the embed, final norm and head sit at `past`
    kind: Kind,
    bytes: u64, // the checkpoint bytes this item reads (trellis+scales for a group, numel*dtype.size otherwise)

    pub const Kind = union(enum) {
        passthrough: struct { dtype: DType, rows: u32, cols: u32 },
        group: usize, // the index into the widened admission's `exl3.Plan.groups`
    };
};

/// One item past the layers (the embed, the final norm, the head).
pub const past: u32 = std.math.maxInt(u32);

/// The whole load, in checkpoint order: embed, per layer (norms, attention, indexer, MLP/MoE), final norm, head.
pub const Load = struct {
    gpa: std.mem.Allocator,
    items: []Item,
    config: cfg.Config, // the shapes every item was checked against
    head: usize, // the head's EXL3 group item
    routers: []usize, // each MoE layer's router weight item, in layer order (its bias trails at +1)

    pub fn deinit(l: *Load) void {
        for (l.items) |it| l.gpa.free(it.name);
        l.gpa.free(l.items);
        l.gpa.free(l.routers);
        l.* = undefined;
    }

    /// The checkpoint bytes this load reads (the admission's own figure, before any device layout).
    pub fn bytes(l: *const Load) u64 {
        var n: u64 = 0;
        for (l.items) |it| n += it.bytes;
        return n;
    }

    /// This layer's EXL3 group item for `klas`/`proj` (the head matches by class alone).
    pub fn groupAt(l: *const Load, groups: []const exl3.Group, d: u32, klas: exl3.Class, proj: u32) ?usize {
        for (l.items, 0..) |*it, i| switch (it.kind) {
            .group => |gi| {
                const g = groups[gi];
                if (klas == .head) {
                    if (g.class == .head) return i;
                } else if (g.class == klas and g.layer == d and g.proj == proj) return i;
            },
            else => {},
        };
        return null;
    }
};

fn expectRow(ck: *checkpoint.Checkpoint, name: []const u8, dtype: DType, cols: usize) !u64 {
    if (!ck.has(name)) return failMissing(name); // screened without the core's err-level log line
    const t = try ck.get(name);
    if (!t.is(dtype, &.{cols})) return error.UnexpectedTensor;
    return @intCast(t.numel() * dtype.size());
}

fn expectMat(ck: *checkpoint.Checkpoint, name: []const u8, dtype: DType, rows: usize, cols: usize) !u64 {
    if (!ck.has(name)) return failMissing(name);
    const t = try ck.get(name);
    if (!t.is(dtype, &.{ rows, cols })) return error.UnexpectedTensor;
    return @intCast(t.numel() * dtype.size());
}

/// The loud refusal for a passthrough the checkpoint lacks: named at warn (the load refuses with
/// `MissingTensor`; the core checkpoint's err-level line would fail the host test runner on the
/// deliberately-missing fixture variant).
fn failMissing(name: []const u8) error{MissingTensor} {
    std.log.warn("minimax_m3_vl: the checkpoint has no tensor {s}", .{name});
    return error.MissingTensor;
}

const next = struct {
    fn f(consumed: []bool, gs: []const exl3.Group, d: u32, klas: exl3.Class, proj: u32) ?usize {
        for (gs, 0..) |g, gi| {
            if (consumed[gi] or g.class != klas) continue;
            if (klas == .head) return gi; // one group, past the layers
            if (g.layer == d and g.proj == proj) return gi;
        }
        return null;
    }
}.f;

fn row(gpa: std.mem.Allocator, items: *std.ArrayList(Item), name: []const u8, dtype: DType, d: u32, rows: u32, cols: u32, byte_count: u64) !void {
    try items.append(gpa, .{ .name = try gpa.dupe(u8, name), .layer = d, .kind = .{ .passthrough = .{ .dtype = dtype, .rows = rows, .cols = cols } }, .bytes = byte_count });
}

fn grp(gpa: std.mem.Allocator, items: *std.ArrayList(Item), name: []const u8, d: u32, gi: usize, gs: []const exl3.Group) !void {
    try items.append(gpa, .{ .name = try gpa.dupe(u8, name), .layer = d, .kind = .{ .group = gi }, .bytes = exl3.groupBytes(gs[gi]) });
}

/// Every passthrough tensor and EXL3 group the family reads, checked against `dir`'s checkpoint by
/// name, dtype and shape; every admitted group must be consumed (a surplus trellis refuses). TP1:
/// the head and the embed in whole; the vocabulary's rank cut is the device loader's job from this plan.
pub fn plan(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const cfg.Config, groups: *const exl3.Plan) !Load {
    var ck = try checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var consumed = try gpa.alloc(bool, groups.groups.len);
    defer gpa.free(consumed); // the checklist runs on every return path
    @memset(consumed, false);
    var items: std.ArrayList(Item) = .empty;
    errdefer {
        for (items.items) |it| gpa.free(it.name);
        items.deinit(gpa);
    }
    var routers: std.ArrayList(usize) = .empty;
    errdefer routers.deinit(gpa);
    var nb: [320]u8 = undefined;
    var eb: [320]u8 = undefined;

    const head_gi = next(consumed, groups.groups, 0, .head, 0) orelse return error.MissingTrellis;
    consumed[head_gi] = true;

    const embed_bytes = try expectMat(&ck, embed_name, .bf16, c.vocab, c.hidden);
    try row(gpa, &items, embed_name, .bf16, past, c.vocab, c.hidden, embed_bytes);

    for (0..c.layers) |li| {
        const d: u32 = @intCast(li);
        // the gemma-style layer norms and the per-head qk norms, bf16 rows of the Config's dims
        inline for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |which| {
            const n = try expectRow(&ck, try layerNormName(&nb, d, which), .bf16, c.hidden);
            try row(gpa, &items, try layerNormName(&nb, d, which), .bf16, d, 1, c.hidden, n);
        }
        // the indexer: fp16 index_q/index_k projections, their per-head norms and the attention's per-head qk norms
        inline for ([_][]const u8{ "q", "k" }) |side| {
            const qkn = try expectRow(&ck, try attnName(&nb, d, side ++ "_norm.weight"), .bf16, c.head_dim);
            try row(gpa, &items, try attnName(&nb, d, side ++ "_norm.weight"), .bf16, d, 1, c.head_dim, qkn);
            const ikn = try expectRow(&ck, try indexerName(&nb, d, side, "norm.weight"), .bf16, c.i_dim);
            try row(gpa, &items, try indexerName(&nb, d, side, "norm.weight"), .bf16, d, 1, c.i_dim, ikn);
            // index_q projects per head: [i_heads * i_dim, hidden]. index_k is ONE shared row set
            // (the fork's MiniMaxM3Indexer k_proj out_features = head_dim; idx_k is [B, 1, T, D]):
            // [i_dim, hidden] — the side plane every head scores against.
            const ik_rows: u32 = if (std.mem.eql(u8, side, "q")) c.i_heads * c.i_dim else c.i_dim;
            const ikr = try expectMat(&ck, try indexerName(&nb, d, side, "proj.weight"), .f16, ik_rows, c.hidden);
            try row(gpa, &items, try indexerName(&nb, d, side, "proj.weight"), .f16, d, ik_rows, c.hidden, ikr);
        }
        // the attention projections: the 4 EXL3 groups, in q,k,v,o order
        inline for (.{ "q_proj", "k_proj", "v_proj", "o_proj" }, 0..) |p, pi| {
            const gi = next(consumed, groups.groups, d, .attn, @intCast(pi)) orelse return error.MissingTrellis;
            consumed[gi] = true;
            try grp(gpa, &items, try std.fmt.bufPrint(&nb, "{s}.{d}.self_attn.{s}", .{ ckpt_prefix, d, p }), d, gi, groups.groups);
        }
        if (d < c.dense_layers) {
            // the dense MLP's three groups (the pack: 5-bit), under the plain mlp space
            inline for ([_]ExpertProj{ .gate, .up, .down }, 0..) |proj, pi| {
                const gi = next(consumed, groups.groups, d, .dense_mlp, @intCast(pi)) orelse return error.MissingTrellis;
                consumed[gi] = true;
                try grp(gpa, &items, try denseName(&nb, d, proj), d, gi, groups.groups);
            }
        } else {
            // the MoE layers: the shared expert's three groups, both router tensors, all the routed experts
            inline for ([_]ExpertProj{ .gate, .up, .down }, 0..) |proj, pi| {
                const gi = next(consumed, groups.groups, d, .shared, @intCast(pi)) orelse return error.MissingTrellis;
                consumed[gi] = true;
                try grp(gpa, &items, try sharedName(&nb, d, proj), d, gi, groups.groups);
            }
            const rbytes = try expectMat(&ck, try routerName(&nb, d), .f32, c.experts, c.hidden);
            try routers.append(gpa, items.items.len);
            try row(gpa, &items, try routerName(&nb, d), .f32, d, c.experts, c.hidden, rbytes);
            const bbytes = try expectRow(&ck, try routerBiasName(&nb, d), .f32, c.experts);
            try row(gpa, &items, try routerBiasName(&nb, d), .f32, d, 1, c.experts, bbytes);
            for (0..c.experts) |el| {
                inline for ([_]ExpertProj{ .gate, .up, .down }) |proj| {
                    const gi = next(consumed, groups.groups, d, .routed, @intFromEnum(proj)) orelse return error.MissingTrellis;
                    consumed[gi] = true;
                    try grp(gpa, &items, try expertName(&eb, d, @intCast(el), proj), d, gi, groups.groups);
                }
            }
        }
    }
    const fbytes = try expectRow(&ck, final_norm_name, .bf16, c.hidden);
    try row(gpa, &items, final_norm_name, .bf16, past, 1, c.hidden, fbytes);
    try grp(gpa, &items, groups.groups[head_gi].prefix, past, head_gi, groups.groups);

    var leftover: usize = 0;
    for (consumed) |done| if (!done) {
        leftover += 1;
    };
    if (leftover != 0) return error.SurplusTrellis; // an admitted trellis the family never reads
    const taken = try items.toOwnedSlice(gpa);
    return .{
        .gpa = gpa,
        .items = taken,
        .config = c.*,
        .head = taken.len - 1,
        .routers = try routers.toOwnedSlice(gpa),
    };
}

// --- tests: the verbatim spellings first, then the whole plan against a synthetic checkpoint ---

test "the inventory's spellings map w1/w2/w3 onto gate/up/down, prefixed the checkpoint's way" {
    var buf: [320]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.7.block_sparse_moe.experts.0.w1", try expertName(&buf, 7, 0, .gate));
    try std.testing.expectEqualStrings("language_model.model.layers.57.block_sparse_moe.experts.3.w2", try expertName(&buf, 57, 3, .down));
    try std.testing.expectEqualStrings("language_model.model.layers.11.block_sparse_moe.experts.0.w3", try expertName(&buf, 11, 0, .up));
    try std.testing.expectEqualStrings("language_model.model.layers.9.block_sparse_moe.shared_experts.up_proj", try sharedName(&buf, 9, .up));
    try std.testing.expectEqualStrings("language_model.model.layers.0.mlp.gate_proj", try denseName(&buf, 0, .gate));
    try std.testing.expectEqualStrings("language_model.model.layers.29.block_sparse_moe.gate.weight", try routerName(&buf, 29));
    try std.testing.expectEqualStrings("language_model.model.layers.5.block_sparse_moe.e_score_correction_bias", try routerBiasName(&buf, 5)); // use_routing_bias; measured: beside the gate, not under it
    try std.testing.expectEqualStrings("language_model.model.layers.26.self_attn.index_k_proj.weight", try indexerName(&buf, 26, "k", "proj.weight"));
    try std.testing.expectEqualStrings("language_model.model.layers.44.self_attn.index_k_norm.weight", try indexerName(&buf, 44, "k", "norm.weight"));
    try std.testing.expectEqualStrings("language_model.model.layers.47.self_attn.q_norm.weight", try attnName(&buf, 47, "q_norm.weight"));
    try std.testing.expectEqualStrings("language_model.model.layers.59.self_attn.o_proj", try attnName(&buf, 59, "o_proj"));
    try std.testing.expectEqualStrings("language_model.model.layers.3.input_layernorm.weight", try layerNormName(&buf, 3, "input_layernorm.weight"));
    try std.testing.expectEqual(@as(usize, 4), exl3_suffixes.len); // trellis words, both scales, the marker
    try std.testing.expectEqualStrings(".trellis", exl3_suffixes[0]);
    try std.testing.expectEqualStrings(".mul1", exl3_suffixes[3]);
}

/// The passthrough tensors written beside the trellis fixture (a second shard), dtype and shape checked.
const Pass = struct {
    gpa: std.mem.Allocator,
    header: std.ArrayListUnmanaged(u8) = .empty,
    data: std.ArrayListUnmanaged(u8) = .empty,
    first: bool = true,

    fn tensor(p: *Pass, name: []const u8, dtype_text: []const u8, shape_text: []const u8, len: usize) !void {
        if (name.len == 0) return; // the missing-tensor variant names nothing and writes nothing
        const base = p.data.items.len;
        try p.data.appendNTimes(p.gpa, 0, len);
        if (!p.first) try p.header.appendSlice(p.gpa, ",");
        p.first = false;
        try p.header.print(p.gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ name, dtype_text, shape_text, base, base + len });
    }

    fn done(p: *Pass) ![]u8 {
        try p.header.insertSlice(p.gpa, 0, "{"); // the safetensors header is one JSON object of the tensors
        try p.header.appendSlice(p.gpa, "}");
        var file: std.ArrayListUnmanaged(u8) = .empty;
        errdefer file.deinit(p.gpa);
        try file.appendNTimes(p.gpa, 0, 8);
        std.mem.writeInt(u64, file.items[0..8], p.header.items.len, .little);
        try file.appendSlice(p.gpa, p.header.items);
        try file.appendSlice(p.gpa, p.data.items);
        return file.toOwnedSlice(p.gpa);
    }
};

fn writePassthrough(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, skip_index_q: bool) !void {
    const c = exl3.synthConfig();
    var p: Pass = .{ .gpa = gpa };
    defer {
        p.header.deinit(gpa);
        p.data.deinit(gpa);
    }
    var nb: [320]u8 = undefined;
    var eb: [320]u8 = undefined;
    var sh: [64]u8 = undefined;
    var name: []const u8 = undefined;
    var shape: []const u8 = undefined;
    shape = try std.fmt.bufPrint(&sh, "{d},{d}", .{ c.vocab, c.hidden });
    try p.tensor(embed_name, "BF16", shape, c.vocab * c.hidden * 2);
    for ([_]u32{ 0, 1 }) |l| {
        inline for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |which| {
            name = try layerNormName(&nb, l, which);
            shape = try std.fmt.bufPrint(&sh, "{d}", .{c.hidden});
            try p.tensor(name, "BF16", shape, c.hidden * 2);
        }
        name = try attnName(&nb, l, "q_norm.weight");
        shape = try std.fmt.bufPrint(&sh, "{d}", .{c.head_dim});
        try p.tensor(name, "BF16", shape, c.head_dim * 2); // the per-head qk norms, one row per head's dim
        name = try attnName(&nb, l, "k_norm.weight");
        try p.tensor(name, "BF16", shape, c.head_dim * 2);
        name = try indexerName(&nb, l, "q", "norm.weight");
        shape = try std.fmt.bufPrint(&sh, "{d}", .{c.i_dim});
        try p.tensor(name, "BF16", shape, c.i_dim * 2);
        name = try indexerName(&nb, l, "k", "norm.weight");
        try p.tensor(name, "BF16", shape, c.i_dim * 2);
        name = try indexerName(&eb, l, "q", "proj.weight"); // the fp16 projection: [i_heads * i_dim, hidden]
        shape = try std.fmt.bufPrint(&sh, "{d},{d}", .{ c.i_heads * c.i_dim, c.hidden });
        try p.tensor(if (skip_index_q) "" else name, "F16", shape, c.i_heads * c.i_dim * c.hidden * 2);
        name = try indexerName(&eb, l, "k", "proj.weight"); // index_k is ONE shared projection: [i_dim, hidden]
        shape = try std.fmt.bufPrint(&sh, "{d},{d}", .{ c.i_dim, c.hidden });
        try p.tensor(name, "F16", shape, c.i_dim * c.hidden * 2);
        if (l == 1) { // this fixture's one MoE layer: the router's fp32 row per expert plus its bias
            name = try routerName(&nb, l);
            shape = try std.fmt.bufPrint(&sh, "{d},{d}", .{ c.experts, c.hidden });
            try p.tensor(name, "F32", shape, c.experts * c.hidden * 4);
            name = try routerBiasName(&nb, l);
            shape = try std.fmt.bufPrint(&sh, "{d}", .{c.experts});
            try p.tensor(name, "F32", shape, c.experts * 4);
        }
    }
    shape = try std.fmt.bufPrint(&sh, "{d}", .{c.hidden});
    try p.tensor(final_norm_name, "BF16", shape, c.hidden * 2);
    const file = try p.done();
    defer gpa.free(file);
    try dir.writeFile(io, .{ .sub_path = "model-00002.safetensors", .data = file });
}

/// The tests' synthetic checkpoint: the trellis pack, the passthrough tensors beside it, and the
/// config.json admission reads. cuda_native.zig's admission test reuses it. `skip_index_q` drops the
/// fp16 index projection so the same dir exercises a missing-tensor refusal.
pub fn checkpointFixture(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, skip_index_q: bool) !void {
    try exl3.writeFixture(gpa, io, dir, .{});
    try writePassthrough(gpa, io, dir, skip_index_q);
    try exl3.writeSynthConfigJson(gpa, io, dir);
}

test "the whole synthetic checkpoint plans: every passthrough tensor and admitted group, in load order" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try checkpointFixture(gpa, io, tmp.dir, false);
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    const c = exl3.synthConfig(); // the fixture's own shapes, whole 128s at every tile edge
    var p = try exl3.discover(gpa, io, path, &c);
    defer p.deinit();
    var l = try plan(gpa, io, path, &c, &p);
    defer l.deinit();
    try std.testing.expectEqual(@as(usize, 1), l.routers.len); // one MoE layer's router in this fixture
    try std.testing.expectEqual(@as(usize, 47), l.items.len); // 1 embed + 16 rows + 8 attn + 3 + 3 + 2 + 12 + 1 + 1
    try std.testing.expectEqual(p.groups.len, blk: { // every admitted group became exactly one item
        var n: usize = 0;
        for (l.items) |it| switch (it.kind) {
            .group => n += 1,
            .passthrough => {},
        };
        break :blk n;
    });
    try std.testing.expectEqual(@as(u64, 512 * c.hidden * 2), l.items[0].bytes); // the embed's bf16 table
    try std.testing.expectEqual(l.items.len - 1, l.head); // the head is the plan's last item, past the layers
    try std.testing.expect(l.groupAt(p.groups, 0, .attn, 0) != null);
    try std.testing.expect(l.groupAt(p.groups, 0, .dense_mlp, 1) != null);
    try std.testing.expect(l.groupAt(p.groups, 1, .shared, 2) != null);
    try std.testing.expect(l.groupAt(p.groups, 1, .routed, 2) != null);
    try std.testing.expectEqual(@as(usize, l.head), l.groupAt(p.groups, 0, .head, 0).?);
    try std.testing.expect(l.bytes() > 0);
}

test "a missing passthrough tensor refuses by name, and a plan consumes every group" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try checkpointFixture(gpa, io, tmp.dir, true); // no index_q_proj in the checkpoint
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    const c = exl3.synthConfig();
    var p = try exl3.discover(gpa, io, path, &c);
    defer p.deinit();
    try std.testing.expectError(error.MissingTensor, plan(gpa, io, path, &c, &p));
}
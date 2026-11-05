//! Every tensor a Qwen3-ASR checkpoint stores, by name and shape from its config: all bf16, none left unread.
const std = @import("std");
const cfg = @import("config.zig");

pub const tower = "thinker.audio_tower.";
pub const lm = "thinker.model.";
pub const head = "thinker.lm_head.weight";

pub const Want = struct {
    name: []const u8,
    rank: u8,
    shape: [4]usize,
};

fn put(list: *std.ArrayList(Want), arena: std.mem.Allocator, name: []const u8, shape: []const usize) !void {
    var w: Want = .{ .name = name, .rank = @intCast(shape.len), .shape = .{ 0, 0, 0, 0 } };
    @memcpy(w.shape[0..shape.len], shape);
    try list.append(arena, w);
}

/// The audio tower's tensors.
fn towerNames(list: *std.ArrayList(Want), arena: std.mem.Allocator, a: cfg.Audio) !void {
    const k = cfg.conv_taps;
    try put(list, arena, tower ++ "conv2d1.weight", &.{ a.conv, 1, k, k });
    for ([_][]const u8{ "conv2d2.weight", "conv2d3.weight" }) |n| try put(list, arena, try std.mem.concat(arena, u8, &.{ tower, n }), &.{ a.conv, a.conv, k, k });
    for ([_][]const u8{ "conv2d1.bias", "conv2d2.bias", "conv2d3.bias" }) |n| try put(list, arena, try std.mem.concat(arena, u8, &.{ tower, n }), &.{a.conv});
    try put(list, arena, tower ++ "conv_out.weight", &.{ a.d, a.conv * a.convBins() });
    for (0..a.layers) |i| {
        const p = try std.fmt.allocPrint(arena, tower ++ "layers.{d}.", .{i});
        const sq = &[_]usize{ a.d, a.d };
        for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "out_proj" }) |n| {
            try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "self_attn.", n, ".weight" }), sq);
            try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "self_attn.", n, ".bias" }), &.{a.d});
        }
        for ([_][]const u8{ "self_attn_layer_norm", "final_layer_norm" }) |n| for ([_][]const u8{ ".weight", ".bias" }) |s|
            try put(list, arena, try std.mem.concat(arena, u8, &.{ p, n, s }), &.{a.d});
        try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "fc1.weight" }), &.{ a.ffn, a.d });
        try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "fc1.bias" }), &.{a.ffn});
        try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "fc2.weight" }), &.{ a.d, a.ffn });
        try put(list, arena, try std.mem.concat(arena, u8, &.{ p, "fc2.bias" }), &.{a.d});
    }
    try put(list, arena, tower ++ "ln_post.weight", &.{a.d});
    try put(list, arena, tower ++ "ln_post.bias", &.{a.d});
    try put(list, arena, tower ++ "proj1.weight", &.{ a.d, a.d });
    try put(list, arena, tower ++ "proj1.bias", &.{a.d});
    try put(list, arena, tower ++ "proj2.weight", &.{ a.out, a.d });
    try put(list, arena, tower ++ "proj2.bias", &.{a.out});
}

/// The language model's tensors, and the stored head (bit-equal to the tied embedding, checked at load).
fn lmNames(list: *std.ArrayList(Want), arena: std.mem.Allocator, t: cfg.Text) !void {
    const qd = t.heads * t.head_dim;
    const kvd = t.kv_heads * t.head_dim;
    try put(list, arena, lm ++ "embed_tokens.weight", &.{ t.vocab, t.hidden });
    try put(list, arena, head, &.{ t.vocab, t.hidden });
    try put(list, arena, lm ++ "norm.weight", &.{t.hidden});
    for (0..t.layers) |i| {
        const p = try std.fmt.allocPrint(arena, lm ++ "layers.{d}.", .{i});
        const parts = .{
            .{ "input_layernorm.weight", &[_]usize{t.hidden} },        .{ "post_attention_layernorm.weight", &[_]usize{t.hidden} },
            .{ "self_attn.q_proj.weight", &[_]usize{ qd, t.hidden } }, .{ "self_attn.k_proj.weight", &[_]usize{ kvd, t.hidden } },
            .{ "self_attn.v_proj.weight", &[_]usize{ kvd, t.hidden } }, .{ "self_attn.o_proj.weight", &[_]usize{ t.hidden, qd } },
            .{ "self_attn.q_norm.weight", &[_]usize{t.head_dim} },     .{ "self_attn.k_norm.weight", &[_]usize{t.head_dim} },
            .{ "mlp.gate_proj.weight", &[_]usize{ t.intermediate, t.hidden } }, .{ "mlp.up_proj.weight", &[_]usize{ t.intermediate, t.hidden } },
            .{ "mlp.down_proj.weight", &[_]usize{ t.hidden, t.intermediate } },
        };
        inline for (parts) |x| try put(list, arena, try std.mem.concat(arena, u8, &.{ p, x[0] }), x[1]);
    }
}

/// Every tensor `c` admits, names in `arena`.
pub fn all(arena: std.mem.Allocator, c: cfg.Config) ![]Want {
    var list: std.ArrayList(Want) = .empty;
    try towerNames(&list, arena, c.audio);
    try lmNames(&list, arena, c.text);
    return list.items;
}

/// A stored tensor as its header names it.
pub const Have = struct { name: []const u8, bf16: bool, shape: []const usize };

/// Every wanted tensor present once at its shape in bf16, and nothing else stored; the first problem by name.
pub fn validate(arena: std.mem.Allocator, c: cfg.Config, have: []const Have, problem: *[]const u8) !void {
    var by: std.StringHashMapUnmanaged(Have) = .empty;
    for (have) |h| if (try by.fetchPut(arena, h.name, h) != null) {
        problem.* = h.name;
        return error.DuplicateTensor;
    };
    const want = try all(arena, c);
    for (want) |w| {
        problem.* = w.name;
        const h = by.get(w.name) orelse return error.MissingTensor;
        if (!h.bf16) return error.UnexpectedDtype;
        if (!std.mem.eql(usize, h.shape, w.shape[0..w.rank])) return error.UnexpectedShape;
    }
    if (have.len != want.len) {
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (want) |w| try seen.put(arena, w.name, {});
        for (have) |h| if (!seen.contains(h.name)) {
            problem.* = h.name;
            break;
        };
        return error.UnexpectedTensor;
    }
}

test "names: the 0.6B's 612 tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c: cfg.Config = .{
        .audio = .{ .d = 896, .layers = 18, .heads = 14, .ffn = 3584, .mels = 128, .conv = 480, .out = 1024, .window = 50, .window_infer = 800, .positions = 1500, .conv_chunks = 500 },
        .text = .{ .hidden = 1024, .layers = 28, .heads = 16, .kv_heads = 8, .head_dim = 128, .intermediate = 3072, .vocab = 151936, .eps = 1e-6, .theta = 1e6, .context = 65536 },
        .audio_token = 151676,
        .audio_start = 151669,
        .audio_end = 151670,
    };
    const w = try all(arena.allocator(), c);
    try std.testing.expectEqual(@as(usize, 612), w.len);
    try std.testing.expectEqual(@as(usize, 7680), w[6].shape[1]);
    var have = try arena.allocator().alloc(Have, w.len + 1);
    for (w, 0..) |*x, i| have[i] = .{ .name = x.name, .bf16 = true, .shape = x.shape[0..x.rank] };
    var problem: []const u8 = "";
    try validate(arena.allocator(), c, have[0..w.len], &problem);
    have[w.len] = .{ .name = "thinker.extra", .bf16 = true, .shape = &.{1} };
    try std.testing.expectError(error.UnexpectedTensor, validate(arena.allocator(), c, have, &problem));
    try std.testing.expectEqualStrings("thinker.extra", problem);
    have[3].bf16 = false;
    try std.testing.expectError(error.UnexpectedDtype, validate(arena.allocator(), c, have[0..w.len], &problem));
}

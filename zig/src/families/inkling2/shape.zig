//! The engine's view of Inkling 2 Small: the shared config (config.zig) narrowed to what the CUDA kernels run.
const std = @import("std");
const shared = @import("config.zig");

pub const max_layers = 64;

/// A layer's attention: sliding (window, static short convolutions) or global (log scaling, dynamic convolutions).
pub const Kind = enum { sliding, global };

pub const Config = struct {
    hidden: u32 = 4608,
    layers: u32 = 48,
    vocab: u32 = 201024,
    /// Rows of the head that are real tokens; the rest of `vocab` is padding.
    real_vocab: u32 = 200058,
    eps: f32 = 1e-6,
    heads: u32 = 48,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    /// Width of each head's relative-position vector and the distances its bias profile covers.
    d_rel: u32 = 16,
    rel_extent: u32 = 1024,
    window: u32 = 1024,
    qk_clip: f32 = 22.627416997969522,
    rel_clip: f32 = 8.0,
    log_floor: f32 = 128000,
    log_alpha: f32 = 0.1,
    conv: u32 = 4,
    block_layers: u32 = 6,
    experts: u32 = 384,
    topk: u32 = 8,
    intermediate: u32 = 2304,
    gate_softcap: f32 = 4.0,
    up_softcap: f32 = 25.0,
    route_scale: f32 = 8.0,
    logit_softcap: f32 = 30.0,
    context: u32 = 1048576,
    eos: [4]u32 = .{ 200006, 0, 0, 0 },
    eos_count: u32 = 1,
    layer: [max_layers]Kind = undefined,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Query heads that share one key head.
    pub fn group(c: *const Config) u32 {
        return c.heads / c.kv_heads;
    }

    pub fn globals(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.layer[0..c.layers]) |k| n += @intFromBool(k == .global);
        return n;
    }

    /// Key and value bytes one token takes on a rank holding `kv` key heads: the global layers' bf16 keys and values.
    pub fn tokenBytes(c: *const Config, kv: u32) u64 {
        return @as(u64, c.globals()) * kv * c.head_dim * 2 * 2;
    }

    /// The end-of-reply ids config.json names.
    pub fn stops(c: *const Config) []const u32 {
        return c.eos[0..c.eos_count];
    }
};

pub const Error = error{ BadInkling2Config, UnsupportedInkling2Attention, UnsupportedInkling2Experts };

/// config.json through the family's shared parser, then the kernels' limits; a refusal logs the parser's reason.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var why: shared.Why = .{};
    const s = shared.parse(gpa, bytes, &why) catch |err| {
        std.log.err("inkling2: {s}", .{why.text()});
        return err;
    };
    return from(&s);
}

/// The shared config as the kernels read it; one attention shape serves sliding and global layers.
pub fn from(s: *const shared.Config) Error!Config {
    const t = &s.text;
    if (t.layers > max_layers) return error.BadInkling2Config;
    if (!std.meta.eql(t.local, t.global)) return error.UnsupportedInkling2Attention;
    var c: Config = .{
        .hidden = t.hidden,
        .layers = t.layers,
        .vocab = t.vocab,
        .real_vocab = t.unpadded_vocab,
        .eps = t.eps,
        .heads = t.global.q,
        .kv_heads = t.global.kv,
        .head_dim = t.global.dim,
        .d_rel = t.d_rel,
        .rel_extent = t.rel_extent,
        .window = t.window,
        .qk_clip = t.qk_clip,
        .rel_clip = t.rel_clip,
        .log_floor = @floatFromInt(t.log_floor),
        .log_alpha = t.log_alpha,
        .conv = shared.conv_taps,
        .block_layers = t.block,
        .experts = t.experts,
        .topk = t.topk,
        .intermediate = t.expert_inter,
        .gate_softcap = t.gate_cap,
        .up_softcap = t.up_cap,
        .route_scale = t.route_scale,
        .logit_softcap = t.logit_cap,
        .context = t.max_context,
        .eos = .{ s.eos, 0, 0, 0 },
        .eos_count = 1,
    };
    for (c.layer[0..c.layers], 0..) |*k, i| k.* = if (shared.isGlobal(@intCast(i))) .global else .sliding;
    try validate(&c);
    return c;
}

/// What tf_ink2_* and the plan cover: 128-wide heads, a 16-wide rel vector over 1,024 distances, whole key groups.
pub fn validate(c: *const Config) Error!void {
    if (c.head_dim != 128 or c.d_rel != 16 or c.rel_extent != 1024 or c.window != 1024 or c.kv_heads == 0 or
        c.heads % c.kv_heads != 0 or c.kv_heads % 2 != 0 or c.qk_clip <= 0 or c.rel_clip <= 0 or c.log_floor < 1 or
        c.conv != 4 or c.block_layers == 0) return error.UnsupportedInkling2Attention;
    if (c.globals() == c.layers) return error.UnsupportedInkling2Attention; // truncated checks may hold no global layer
    if (c.experts == 0 or c.experts % 8 != 0 or c.topk == 0 or c.topk > 8 or c.intermediate % 128 != 0 or
        c.hidden % 128 != 0) return error.UnsupportedInkling2Experts;
    if (c.real_vocab == 0 or c.real_vocab > c.vocab or c.context == 0) return error.BadInkling2Config;
}

const fixture = @embedFile("fixtures/config.json");

test "inkling2 shape: the release's shape, 40 sliding and 8 global layers, eos from config.json" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(u32, 48), c.layers);
    try std.testing.expectEqual(@as(u32, 8), c.globals());
    for ([_]usize{ 5, 11, 17, 23, 29, 35, 41, 47 }) |i| try std.testing.expectEqual(Kind.global, c.layer[i]);
    try std.testing.expectEqual(Kind.sliding, c.layer[0]);
    try std.testing.expectEqual(Kind.sliding, c.layer[46]);
    try std.testing.expectEqual(@as(u32, 6), c.group());
    try std.testing.expectEqual(@as(u32, 1048576), c.context);
    try std.testing.expectEqualSlices(u32, &.{200006}, c.stops());
    try std.testing.expectEqual(@as(f32, 128000), c.log_floor);
    try std.testing.expectEqual(@as(f32, 22.627416997969522), c.qk_clip);
    // at TP=2 a rank holds 4 key heads: 16 KiB a token, 16 GiB at 1M
    try std.testing.expectEqual(@as(u64, 16384), c.tokenBytes(4));
}

test "inkling2 shape: an EXL3 pack's config reads as the release's" {
    const modelopt = "\"quant_method\": \"modelopt\",\n    \"quant_algo\": \"NVFP4\"";
    const exl3 = "\"quant_method\": \"exl3\",\n    \"version\": \"1.6.6\", \"bits\": 2.2";
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, modelopt, exl3);
    defer std.testing.allocator.free(bytes);
    try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
    const a = try parse(std.testing.allocator, fixture);
    const b = try parse(std.testing.allocator, bytes);
    try std.testing.expect(std.meta.eql(a, b));
}

test "inkling2 shape: shapes the kernels do not run are refused" {
    const Case = struct { from: []const u8, to: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .from = "\"head_dim\": 128", .to = "\"head_dim\": 96", .err = error.UnsupportedInkling2Attention },
        .{ .from = "\"swa_head_dim\": 128", .to = "\"swa_head_dim\": 64", .err = error.UnsupportedInkling2Attention },
        .{ .from = "\"rel_extent\": 1024", .to = "\"rel_extent\": 2048", .err = error.UnsupportedInkling2Attention },
        .{ .from = "\"d_rel\": 16", .to = "\"d_rel\": 32", .err = error.UnsupportedInkling2Attention },
        .{ .from = "\"sliding_window_size\": 1024", .to = "\"sliding_window_size\": 4096", .err = error.UnsupportedInkling2Attention },
        .{ .from = "\"num_experts_per_tok\": 8", .to = "\"num_experts_per_tok\": 9", .err = error.UnsupportedInkling2Experts },
    };
    for (cases) |x| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, x.from, x.to);
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(x.err, parse(std.testing.allocator, bytes));
    }
}

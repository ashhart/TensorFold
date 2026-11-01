//! Idol's image encoder, its backend-free parts: FLUX.2's VAE geometry, the condition rows and the float64 gate.
const std = @import("std");
const ref = @import("reference.zig");

/// The encoder's widths by stage, and the rest of the released geometry.
pub const widths = [4]usize{ 128, 256, 512, 512 };
pub const latent_channels = 32;
pub const resnets = 2;
pub const batch_norm_eps: f32 = 1e-4;

/// GroupNorm's groups and epsilon, and the mid block's attention width.
pub const groups = 32;
pub const eps: f32 = 1e-6;
pub const width = 512;

/// The picture the encoder sees, its latent grid, and the condition rows: 5 zero rows, then a row a 2 x 2 patch.
pub const size = 1024;
pub const latent = size / 8;
pub const pad = 5;
pub const rows = pad + (latent / 2) * (latent / 2);
pub const channels = 4 * latent_channels;

/// Where `encode` copies each stage's output when asked, [rows, columns, channels] fp32 (null: not wanted).
pub const Stages = struct { down: [4]?[]f32 = @splat(null), mid: ?[]f32 = null, moments: ?[]f32 = null };

/// Accepts vae/config.json at FLUX.2's released encoder geometry, else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "AutoencoderKLFlux2")) return error.NotFlux2Vae;
    const want = .{ .{ "in_channels", 3 }, .{ "latent_channels", latent_channels }, .{ "layers_per_block", resnets }, .{ "norm_num_groups", groups } };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("idol vae: {s} is not {d}; the encoder runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const lists = .{ .{ "block_out_channels", [_]i64{ 128, 256, 512, 512 } }, .{ "patch_size", [_]i64{ 2, 2 } } };
    inline for (lists) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .array or v.array.items.len != kv[1].len) return error.UnsupportedGeometry;
        for (v.array.items, kv[1]) |item, n| if (item != .integer or item.integer != n) return error.UnsupportedGeometry;
    }
    inline for (.{ "mid_block_add_attention", "use_quant_conv" }) |f| {
        const v = o.get(f) orelse return error.BadConfig;
        if (v != .bool or !v.bool) return error.UnsupportedGeometry;
    }
    const blocks = o.get("down_block_types") orelse return error.BadConfig;
    if (blocks != .array or blocks.array.items.len != widths.len) return error.UnsupportedGeometry;
    for (blocks.array.items) |b| if (b != .string or !std.mem.eql(u8, b.string, "DownEncoderBlock2D")) return error.UnsupportedGeometry;
    const act = o.get("act_fn") orelse return error.BadConfig;
    if (act != .string or !std.mem.eql(u8, act.string, "silu")) return error.UnsupportedGeometry;
    const bn = o.get("batch_norm_eps") orelse return error.BadConfig;
    const bn_eps: f64 = switch (bn) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => return error.BadConfig,
    };
    if (@as(f32, @floatCast(bn_eps)) != batch_norm_eps) return error.UnsupportedGeometry;
}

/// The bf16 value an fp32 weight rounds to (to nearest, ties to even), as torch's cast to bf16 makes it.
pub fn bf16(v: f32) f32 {
    if (std.math.isNan(v)) return v;
    const bits: u32 = @bitCast(v);
    const rounded = bits +% (0x7fff + ((bits >> 16) & 1));
    return @bitCast(rounded & 0xffff0000);
}

/// The checkpoint's encoder-side tensor names: the encoder's, quant_conv's and the batch norm's two statistics.
pub fn encoderSide(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "encoder.") or std.mem.startsWith(u8, name, "quant_conv.") or
        std.mem.eql(u8, name, "bn.running_mean") or std.mem.eql(u8, name, "bn.running_var");
}

/// The batch norm's sqrt(running_var + eps), as the SDK forms it in bf16, here in float64 rounded once.
pub fn bnStd(v: f32) f32 {
    return @floatCast(@sqrt(@as(f64, v) + 1e-4));
}

/// The encoder's gate against the float64 ruler: worst row and 1 - cosine within twice the SDK's own, rows of 128.
pub const Verdict = struct { ours: ref.Match, sdk: ref.Match, pass: bool };

pub fn gate(ours: []const f32, sdk: []const f32, ruler: []const f32) Verdict {
    const o = ref.match(ours[pad * channels ..], ruler[pad * channels ..], channels);
    const s = ref.match(sdk[pad * channels ..], ruler[pad * channels ..], channels);
    const zeros = for (ours[0 .. pad * channels]) |z| {
        if (z != 0) break false;
    } else true;
    return .{ .ours = o, .sdk = s, .pass = zeros and ref.within(o, s) };
}

test "the rows are the denoiser's condition rows" {
    try std.testing.expectEqual(@as(usize, 4101), rows);
    try std.testing.expectEqual(@as(usize, 128), channels);
}

test "a planted mismatch in one condition row fails the encoder's gate, as does a padding row not zero" {
    const a = std.testing.allocator;
    const n = rows * channels;
    const ruler = try a.alloc(f32, n);
    defer a.free(ruler);
    const sdk = try a.alloc(f32, n);
    defer a.free(sdk);
    const ours = try a.alloc(f32, n);
    defer a.free(ours);
    var rng = std.Random.DefaultPrng.init(11);
    for (ruler, sdk, ours, 0..) |*r, *s, *o, i| {
        const z = i < pad * channels;
        r.* = if (z) 0 else rng.random().floatNorm(f32);
        s.* = r.* * (1 + 4e-3 * rng.random().floatNorm(f32));
        o.* = r.* * (1 + 1e-6 * rng.random().floatNorm(f32));
    }
    try std.testing.expect(gate(ours, sdk, ruler).pass);
    const row = 2000;
    for (ours[row * channels ..][0..channels]) |*o| o.* *= 1.02;
    const v = gate(ours, sdk, ruler);
    try std.testing.expect(!v.pass);
    try std.testing.expect(v.ours.worst_row > 0.019);
    for (ours[row * channels ..][0..channels], ruler[row * channels ..][0..channels]) |*o, r| o.* = r;
    try std.testing.expect(gate(ours, sdk, ruler).pass);
    ours[3 * channels + 7] = 1e-3;
    try std.testing.expect(!gate(ours, sdk, ruler).pass);
}

test "only FLUX.2's released encoder geometry passes" {
    const a = std.testing.allocator;
    const sample =
        \\{"_class_name": "AutoencoderKLFlux2", "act_fn": "silu", "batch_norm_eps": 0.0001, "block_out_channels": [128, 256, 512, 512],
        \\ "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"], "in_channels": 3,
        \\ "latent_channels": 32, "layers_per_block": 2, "mid_block_add_attention": true, "norm_num_groups": 32,
        \\ "patch_size": [2, 2], "use_quant_conv": true}
    ;
    try check(a, sample);
    const narrow = try std.mem.replaceOwned(u8, a, sample, "[128, 256, 512, 512]", "[96, 192, 384, 384]");
    defer a.free(narrow);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, narrow));
    const no_quant = try std.mem.replaceOwned(u8, a, sample, "\"use_quant_conv\": true", "\"use_quant_conv\": false");
    defer a.free(no_quant);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, no_quant));
    const other = try std.mem.replaceOwned(u8, a, sample, "AutoencoderKLFlux2", "AutoencoderKL");
    defer a.free(other);
    try std.testing.expectError(error.NotFlux2Vae, check(a, other));
}

test "an f32 weight rounds to bf16 as torch's cast does: to nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0 + 0x1p-9));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-7), bf16(1.0 + 0x1p-7 + 0x1p-9));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-6), bf16(1.0 + 0x1p-7 + 0x1p-8));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-7), bf16(1.0 + 0x1p-7 + 0x1p-8 - 0x1p-20));
}

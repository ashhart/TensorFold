//! Admission for Cosmos3-Edge's reasoner as tools/cosmos3_edge_to_mlx.py writes it: pinned geometry, MLX affine groups of 64.
const std = @import("std");

// The text model (cosmos3_edge_text): dense GQA, relu^2 MLP, interleaved M-RoPE over [24, 20, 20] pairs.
pub const hidden = 2048;
pub const vocab = 131072;
pub const layers = 28;
pub const intermediate = 9216;
pub const query_heads = 16;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const sections = [3]u32{ 24, 20, 20 };
pub const eps: f32 = 1e-5;
pub const group = 64;

// The image tower (SigLIP2, cosmos3_edge_vision) and the projector: 2 x 2 patches merged into one text row.
pub const v_hidden = 1152;
pub const v_layers = 27;
pub const v_heads = 16;
pub const v_head_dim = 72;
pub const v_intermediate = 4352; // 4304 zero-padded to a multiple of 64 by the converter
pub const v_eps: f32 = 1e-6;
pub const patch = 16;
pub const patch_width = patch * patch * 3;
pub const merge = 2;
pub const grid = 16; // the learned position table's side
pub const projector_hidden = 11520;

pub const bit_widths = [_]u8{ 4, 6, 8 };

pub const Config = struct {
    context: usize,
    theta: f32,
    bits: u8,
    eos: u32,
    image_token: u32,
    vision_start: u32,
    vision_end: u32,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadCosmosConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadCosmosConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadCosmosConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 0) return error.BadCosmosConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedCosmosConfig;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "cosmos3_edge");
    const t = try object(try field(root, "text_config"));
    const v = try object(try field(root, "vision_config"));
    const text = .{
        .{ "hidden_size", hidden },             .{ "vocab_size", vocab },                .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate }, .{ "num_attention_heads", query_heads }, .{ "num_key_value_heads", kv_heads },
        .{ "head_dim", head_dim },
    };
    inline for (text) |d| if (try integer(t, d[0]) != d[1]) return error.UnsupportedCosmosGeometry;
    try string(t, "hidden_act", "relu2");
    if (try number(try field(t, "rms_norm_eps")) != 1e-5) return error.UnsupportedCosmosConfig;
    const vision = .{
        .{ "hidden_size", v_hidden },                    .{ "num_hidden_layers", v_layers }, .{ "num_attention_heads", v_heads },
        .{ "patch_size", patch },                        .{ "spatial_merge_size", merge },   .{ "num_patches", grid * grid },
        .{ "padded_intermediate_size", v_intermediate },
    };
    inline for (vision) |d| if (try integer(v, d[0]) != d[1]) return error.UnsupportedCosmosGeometry;
    if (try integer(root, "projector_hidden_size") != projector_hidden) return error.UnsupportedCosmosGeometry;
    const rope = try object(try field(t, "rope_parameters"));
    try string(rope, "rope_type", "default");
    const sec = try field(rope, "mrope_section");
    if (sec != .array or sec.array.items.len != 3) return error.UnsupportedCosmosRotary;
    for (sec.array.items, sections) |s, want| if (s != .integer or s.integer != want) return error.UnsupportedCosmosRotary;
    const theta = try number(try field(rope, "rope_theta"));
    if (!(theta > 1)) return error.UnsupportedCosmosRotary;
    const q = try object(root.get("quantization") orelse return error.UnsupportedCosmosQuantization);
    const bits = try integer(q, "bits");
    if (try integer(q, "group_size") != group or std.mem.indexOfScalar(u8, &bit_widths, @intCast(@min(bits, 255))) == null)
        return error.UnsupportedCosmosQuantization;
    return .{
        .context = try integer(t, "max_position_embeddings"),
        .theta = @floatCast(theta),
        .bits = @intCast(bits),
        .eos = @intCast(try integer(root, "eos_token_id")),
        .image_token = @intCast(try integer(root, "image_token_id")),
        .vision_start = @intCast(try integer(root, "vision_start_token_id")),
        .vision_end = @intCast(try integer(root, "vision_end_token_id")),
    };
}

pub const fixture =
    \\{"architectures":["Cosmos3EdgeForConditionalGeneration"],"model_type":"cosmos3_edge","image_token_id":19,
    \\"video_token_id":18,"vision_start_token_id":20,"vision_end_token_id":21,"projector_hidden_size":11520,
    \\"text_config":{"hidden_size":2048,"vocab_size":131072,"num_hidden_layers":28,"intermediate_size":9216,
    \\"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"hidden_act":"relu2","rms_norm_eps":1e-05,
    \\"max_position_embeddings":131072,"rope_parameters":{"mrope_section":[24,20,20],"rope_theta":100000000,"rope_type":"default"}},
    \\"vision_config":{"hidden_size":1152,"num_hidden_layers":27,"num_attention_heads":16,"patch_size":16,
    \\"spatial_merge_size":2,"num_patches":256,"intermediate_size":4304,"padded_intermediate_size":4352},
    \\"quantization":{"group_size":64,"bits":8,"mode":"affine"},"eos_token_id":11}
;

test "admit the converted Cosmos3-Edge reasoner and refuse other shapes" {
    const gpa = std.testing.allocator;
    const c = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(u8, 8), c.bits);
    try std.testing.expectEqual(@as(f32, 1e8), c.theta);
    try std.testing.expectEqual([3]u32{ 19, 20, 21 }, [3]u32{ c.image_token, c.vision_start, c.vision_end });
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedCosmosGeometry },
        .{ "\"bits\":8", "\"bits\":5", error.UnsupportedCosmosQuantization },
        .{ "[24,20,20]", "[16,24,24]", error.UnsupportedCosmosRotary },
        .{ "\"padded_intermediate_size\":4352", "\"padded_intermediate_size\":4304", error.UnsupportedCosmosGeometry },
        .{ "\"model_type\":\"cosmos3_edge\"", "\"model_type\":\"qwen3_vl\"", error.UnsupportedCosmosConfig },
    };
    inline for (changes) |change| {
        const bad = try std.mem.replaceOwned(u8, gpa, fixture, change[0], change[1]);
        defer gpa.free(bad);
        try std.testing.expectError(change[2], parse(gpa, bad));
    }
}

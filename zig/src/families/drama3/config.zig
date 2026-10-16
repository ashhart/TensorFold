//! Drama 3's geometry (config.json "fish_omni"): load() refuses a checkpoint unlike the one the kernels are built for.
const std = @import("std");

pub const hidden = 2048;
pub const layers = 40;
pub const vocab = 252288;
pub const heads = 16; // full attention query heads (backbone and tower)
pub const kv_heads = 2;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const rope_theta: f32 = 1.0e7;
pub const eps: f32 = 1.0e-6;
pub const experts = 256;
pub const top_k = 8;
pub const expert_width = 512; // moe_intermediate_size
pub const shared_width = 512;
// Gated DeltaNet
pub const lin_k_heads = 16;
pub const lin_v_heads = 32;
pub const lin_dim = 128;
pub const conv_channels = 8192; // q 2048 + k 2048 + v 4096
pub const conv_kernel = 4;
// fused projection rows
pub const gdn_proj = conv_channels + 4096 + 32 + 32; // qkv | z | b | a = 12352
pub const attn_proj = 2 * heads * head_dim + 2 * kv_heads * head_dim; // 9216
pub const moe_proj = 2 * shared_width + experts + 8; // shared gate | up | router 256 | shared gate 1 | pad 7 = 1288
// tower
pub const codebooks = 10;
pub const codebook_size = 4096;
pub const shared_layers = 4;
pub const tower_layers = shared_layers + codebooks - 1; // 13
pub const tower_ffn = 6144;
// frames: [main token, code 0 .. code 9]
pub const frame_width = 1 + codebooks;

pub const Ids = struct {
    eos: i32,
    pad: i32,
    sem_begin: i32, // <|semantic:0|>
    sem_end: i32, // inclusive
    audio_pad: i32,
};

pub const Config = struct {
    ids: Ids,
    full_attention: [layers]bool, // layer_types: true = full_attention, false = linear_attention

    pub fn candidates(c: Config) usize {
        return @as(usize, @intCast(c.ids.sem_end - c.ids.sem_begin + 1)) + 1;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return parse(gpa, text);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer tree.deinit();
        const root = tree.value.object;
        const tc = (root.get("text_config") orelse return error.BadConfig).object;
        const ac = (root.get("audio_decoder_config") orelse return error.BadConfig).object;
        var c: Config = undefined;
        c.ids = .{
            .eos = @intCast(try int(root, "eos_token_id")),
            .pad = @intCast(try int(root, "pad_token_id")),
            .sem_begin = @intCast(try int(root, "semantic_start_token_id")),
            .sem_end = @intCast(try int(root, "semantic_end_token_id")),
            .audio_pad = @intCast(try int(root, "audio_pad_token_id")),
        };
        const checks = .{
            .{ tc, "hidden_size", hidden },                     .{ tc, "num_hidden_layers", layers },
            .{ tc, "vocab_size", vocab },                       .{ tc, "num_attention_heads", heads },
            .{ tc, "num_key_value_heads", kv_heads },           .{ tc, "head_dim", head_dim },
            .{ tc, "num_experts", experts },                    .{ tc, "num_experts_per_tok", top_k },
            .{ tc, "moe_intermediate_size", expert_width },     .{ tc, "shared_expert_intermediate_size", shared_width },
            .{ tc, "linear_num_key_heads", lin_k_heads },       .{ tc, "linear_num_value_heads", lin_v_heads },
            .{ tc, "linear_key_head_dim", lin_dim },            .{ tc, "linear_value_head_dim", lin_dim },
            .{ tc, "linear_conv_kernel_dim", conv_kernel },     .{ ac, "dim", hidden },
            .{ ac, "n_head", heads },                           .{ ac, "n_local_heads", kv_heads },
            .{ ac, "head_dim", head_dim },                      .{ ac, "intermediate_size", tower_ffn },
            .{ ac, "num_codebooks", codebooks },                .{ ac, "vocab_size", codebook_size },
            .{ ac, "shared_enrichment_layers", shared_layers }, .{ ac, "vq_backbone_codebooks", 1 },
        };
        inline for (checks) |ck| {
            const v = try int(ck[0], ck[1]);
            if (v != ck[2]) {
                std.log.warn("drama3: config {s} = {d}, the kernels need {d}", .{ ck[1], v, ck[2] });
                return error.UnsupportedConfig;
            }
        }
        if (c.ids.sem_end - c.ids.sem_begin + 1 != codebook_size) return error.UnsupportedConfig;
        const lt = (tc.get("layer_types") orelse return error.BadConfig).array;
        if (lt.items.len != layers) return error.BadConfig;
        for (lt.items, 0..) |v, i| {
            const s = v.string;
            c.full_attention[i] = std.mem.eql(u8, s, "full_attention");
            if (!c.full_attention[i] and !std.mem.eql(u8, s, "linear_attention")) return error.BadConfig;
        }
        const rot = (ac.get("rotary_interleaved") orelse return error.BadConfig).bool;
        if (!rot) return error.UnsupportedConfig;
        return c;
    }
};

fn int(o: std.json.ObjectMap, name: []const u8) !usize {
    const v = o.get(name) orelse {
        std.log.err("drama3 config.json has no {s}", .{name});
        return error.BadConfig;
    };
    return switch (v) {
        .integer => |x| std.math.cast(usize, x) orelse error.BadConfig,
        else => error.BadConfig,
    };
}

test "the shipped Drama 3 config parses" {
    const text =
        \\{"eos_token_id": 248046, "pad_token_id": 248044, "semantic_start_token_id": 248099,
        \\ "semantic_end_token_id": 252194, "audio_pad_token_id": 248076,
        \\ "text_config": {"hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 252288,
        \\  "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "num_experts": 256,
        \\  "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        \\  "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        \\  "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        \\  "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention",
        \\   "linear_attention","linear_attention","linear_attention","full_attention"]},
        \\ "audio_decoder_config": {"dim": 2048, "n_head": 16, "n_local_heads": 2, "head_dim": 256,
        \\  "intermediate_size": 6144, "num_codebooks": 10, "vocab_size": 4096, "shared_enrichment_layers": 4,
        \\  "vq_backbone_codebooks": 1, "rotary_interleaved": true}}
    ;
    const c = try Config.parse(std.testing.allocator, text);
    try std.testing.expect(c.full_attention[3] and !c.full_attention[0] and c.full_attention[39]);
    try std.testing.expectEqual(@as(usize, 4097), c.candidates());
}

test "a checkpoint whose shape differs from the kernels' is refused" {
    const text =
        \\{"eos_token_id": 1, "pad_token_id": 2, "semantic_start_token_id": 10, "semantic_end_token_id": 4105,
        \\ "audio_pad_token_id": 3, "text_config": {"hidden_size": 4096}, "audio_decoder_config": {}}
    ;
    try std.testing.expectError(error.UnsupportedConfig, Config.parse(std.testing.allocator, text));
}

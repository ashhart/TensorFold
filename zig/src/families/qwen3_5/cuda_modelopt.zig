//! ModelOpt checkpoints of the Qwen3.5 hybrids (nvidia/Qwen3.8-27B-NVFP4): NVFP4, FP8 and bf16 projections on qlinear.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Geometry = geometry.Geometry;
const Kind = geometry.Kind;
const Kinds = geometry.Kinds;
const Slot = geometry.Slot;
const Engine = engine.Engine;
const Proj = engine.Proj;
const Weight = cuda.qlinear.Weight;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const padded = cuda.modelopt.padded;

/// The language model's tensors under this prefix, the head at `lm_head` (HF's spelling, not MLX's).
pub const prefix = "model.language_model.";

/// A module's recipe as this engine serves it: NVFP4 (W4A16 too), per-tensor FP8, or bf16 where none is declared.
fn kindOf(r: ?core.modelopt.Recipe) !Kind {
    const recipe = r orelse return .bf16;
    return switch (recipe) {
        .nvfp4, .w4a16_nvfp4 => .nvfp4,
        .fp8 => .fp8,
        else => error.UnsupportedQwenQuantization,
    };
}

/// Every module's kind from a MIXED_PRECISION config's quantized_layers; a single-format one names no modules.
pub fn kinds(gpa: std.mem.Allocator, config: []const u8, g: Geometry) !Kinds {
    var why: core.modelopt.Why = .{};
    var r = core.modelopt.admitRecipes(gpa, config, &why) catch |err| {
        std.log.err("{s}", .{why.text()});
        return if (err == error.OutOfMemory) err else error.UnsupportedQwenQuantization;
    };
    defer r.deinit();
    if (r.modules.count() == 0 or g.tied) return error.UnsupportedQwenQuantization;
    if (r.of(prefix ++ "embed_tokens") != null) return error.UnsupportedQwenQuantization; // the rows stay bf16
    var out: Kinds = .{ .modelopt = true, .head = try kindOf(r.of("lm_head")) };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        if (used) out.proj[i][@backingInt(slot)] = try kindOf(r.of(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) })));
    };
    return out;
}

/// The device bytes of an n-by-k projection of `kind` in the lane's layout (rows padded to 128).
pub fn packedBytes(n: usize, k: usize, kind: Kind) usize {
    const np = padded(n);
    return switch (kind) {
        .nvfp4 => aligned(np * k / 2) + aligned(np * k / 16),
        .fp8 => aligned(np * k),
        .bf16 => aligned(np * k * 2),
        .affine => unreachable,
    };
}

/// The checkpoint's bytes of an n-by-k projection of `kind`: codes, then block scales.
pub fn rawBytes(n: usize, k: usize, kind: Kind) usize {
    return switch (kind) {
        .nvfp4 => n * k / 2 + n * k / 16,
        .fp8 => n * k,
        .bf16 => n * k * 2,
        .affine => unreachable,
    };
}

/// The kernels a ModelOpt checkpoint adds: the lane matmul, the prompt GEMM, the packers and the bf16 token rows.
pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    pack: cuda.modelopt.Packer,
    embed: cuda.embed_rows.EmbedRows,
    /// Slice partials where no cluster meets them (before sm_90): the most any projection takes at max_rows.
    part: ?DeviceBuffer = null,

    pub fn load(d: *const cuda.Driver, major: u32) !Kernels {
        var lane = try cuda.qmmf.Lane.load(d, major);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer prompt.unload();
        var pack = try cuda.modelopt.Packer.load(d);
        errdefer pack.unload();
        return .{ .lane = lane, .prompt = prompt, .pack = pack, .embed = try cuda.embed_rows.EmbedRows.load(d) };
    }

    pub fn unload(k: *Kernels) void {
        if (k.part) |*p| p.free();
        k.embed.unload();
        k.pack.unload();
        k.prompt.unload();
        k.lane.unload();
    }

    pub fn linear(k: *const Kernels) cuda.qlinear.Linear {
        return .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt };
    }

    /// The slice partials every projection of `e` takes at up to max_rows rows (none with clusters).
    pub fn reserveParts(k: *Kernels, e: *const Engine) !void {
        var most: usize = 0;
        const Each = struct {
            fn bytes(lane: cuda.qmmf.Lane, p: Proj) usize {
                return switch (p) {
                    .ql => |w| switch (w) {
                        .nvfp4, .fp8, .bf16 => |q| lane.partBytes(engine.max_rows, q),
                        else => 0,
                    },
                    else => 0,
                };
            }
        };
        most = Each.bytes(k.lane, e.head);
        for (e.layers) |l| {
            switch (l.mlp) {
                .dense => |m| for ([_]Proj{ m.gate, m.up, m.down }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
                .moe => {},
            }
            switch (l.mixer) {
                .delta => |m| for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
                .attention => |m| for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
            }
        }
        if (most > 0) k.part = try DeviceBuffer.alloc(e.d, most);
    }
};

/// out (rows, n) = x (rows, k): prompt chunks past 16 rows on the prompt GEMM, else 16-row decode tiles (verify bits).
pub fn project(e: *Engine, w: Weight, x: u64, out: u64, rows: usize, prompt: bool) !void {
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    const l = k.linear();
    const part: u64 = if (k.part) |p| p.ptr else 0;
    if (prompt and rows > 16) return l.prompt(e.stream, w, .{ .x = x, .part = part }, out, rows);
    const kk = w.inputs();
    const n = w.outputs();
    var at: usize = 0;
    while (at < rows) : (at += 16) {
        const m = @min(16, rows - at);
        try l.decode(e.stream, w, .{ .x = x + at * kk * 2, .part = part }, out + at * n * 2, m);
    }
}

/// An fp32 tensor of one element.
fn scalar(ck: *core.Checkpoint, name: []const u8) !f32 {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// `{pre}{module}.{tail}`, a module's tensor.
fn tname(b: []u8, pre: []const u8, module: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "{s}{s}.{s}", .{ pre, module, tail });
}

/// `{pre}{name}` as stored, staged and packed into the lane's layout out of `e`'s arena.
pub fn linear(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer, pre: []const u8, name: []const u8, n: usize, k: usize, kind: Kind) !Proj {
    if (k % 64 != 0) return error.UnsupportedQwenGeometry;
    const p = &(e.mo orelse return error.FormatNotLoaded).pack;
    var buf: [192]u8 = undefined;
    // the static input scale: read for the checkpoint's math (--precision checkpoint), unused by bf16 rows
    const act: f32 = if (kind != .bf16 and ck.has(try tname(&buf, pre, name, "input_scale"))) try scalar(ck, try tname(&buf, pre, name, "input_scale")) else 0;
    const out: Weight = switch (kind) {
        .bf16 => blk: {
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .bf16, &.{ n, k });
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k * 2);
            try p.bf16Lane(e.stream, stage.ptr, w, n, k);
            break :blk .{ .bf16 = cuda.modelopt.bf16Weight(w, n, k) };
        },
        .fp8 => blk: {
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .f8_e4m3, &.{ n, k });
            const scale = try scalar(ck, try tname(&buf, pre, name, "weight_scale"));
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k);
            try p.fp8Lane(e.stream, stage.ptr, w, n, k);
            var q = cuda.modelopt.fp8Weight(w, scale, n, k);
            q.act = act;
            break :blk .{ .fp8 = q };
        },
        .nvfp4 => blk: {
            const codes = try ck.expect(try tname(&buf, pre, name, "weight"), .u8, &.{ n, k / 2 });
            const blocks = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .f8_e4m3, &.{ n, k / 16 });
            const global = try scalar(ck, try tname(&buf, pre, name, "weight_scale_2"));
            try stage.upload(0, codes.bytes);
            try stage.upload(codes.bytes.len, blocks.bytes);
            const words = try e.alloc(padded(n) * k / 2);
            const tiles = try e.alloc(padded(n) * k / 16);
            try p.fp4Lane(e.stream, stage.ptr, stage.ptr + codes.bytes.len, words, tiles, n, k);
            var q = cuda.nvfp4.weight(words, tiles, global, n, k);
            q.act = act;
            break :blk .{ .nvfp4 = q };
        },
        .affine => unreachable,
    };
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return .{ .ql = out };
}

/// Token rows of the bf16 embedding into `out`: an exact copy of each id's row.
pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize) !void {
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    return k.embed.bf16(e.stream, ids, e.embed_w, out, rows, e.g.hidden);
}

test "kinds from nvidia/Qwen3.8-27B-NVFP4's recipes: NVFP4 MLPs and head, FP8 mixers, bf16 a and b" {
    const text =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
        \\   "lm_head":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.linear_attn.in_proj_qkv":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.linear_attn.in_proj_z":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.mlp.gate_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.mlp.up_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.mlp.down_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.self_attn.q_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.k_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.v_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.o_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.mlp.gate_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.mlp.up_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.mlp.down_proj":{"quant_algo":"NVFP4","group_size":16}}},
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":2,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try geometry.parse(std.testing.allocator, text);
    try std.testing.expect(g.kinds.modelopt);
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.head);
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(0, .qkv));
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .a));
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .b));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .down));
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(1, .o));
    try std.testing.expectEqual(Kind.affine, g.kinds.of(1, .qkv)); // not this layer's mixer
    try std.testing.expectEqual(@as(u8, 16), g.bits.embed); // bf16 rows: learning refuses them
    // a recipe the lane does not serve here (block FP8) is refused by name, as is a quantized embedding
    const pb = try std.mem.replaceOwned(u8, std.testing.allocator, text, "layers.0.linear_attn.out_proj\":{\"quant_algo\":\"FP8\"}", "layers.0.linear_attn.out_proj\":{\"quant_algo\":\"FP8_PB_WO\",\"weight_block_size\":[128,128]}");
    defer std.testing.allocator.free(pb);
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(std.testing.allocator, pb));
    try std.testing.expectEqual(@as(usize, 512 * 5120 / 2 + 512 * 5120 / 16), packedBytes(500, 5120, .nvfp4));
    try std.testing.expectEqual(@as(usize, 48 * 5120 * 2), rawBytes(48, 5120, .bf16));
}

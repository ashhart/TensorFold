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
        .nvfp4 => .nvfp4,
        .w4a16_nvfp4 => .nvfp4_w16,
        .fp8 => .fp8,
        else => error.UnsupportedQwenQuantization,
    };
}

/// A single-format NVFP4 config's `ignore` (or `exclude_modules`) names: modules, or prefixes ending in `*`.
const Ignored = struct {
    doc: std.json.Parsed(std.json.Value),
    names: []const std.json.Value,

    fn read(gpa: std.mem.Allocator, config: []const u8) !Ignored {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, config, .{});
        errdefer doc.deinit();
        const q = if (doc.value.object.get("quantization_config")) |v| v.object else return error.UnsupportedQwenQuantization;
        const list = q.get("ignore") orelse q.get("exclude_modules") orelse return .{ .doc = doc, .names = &.{} };
        if (list != .array) return error.UnsupportedQwenQuantization;
        return .{ .doc = doc, .names = list.array.items };
    }

    fn has(i: Ignored, module: []const u8) bool {
        for (i.names) |v| if (v == .string) {
            const n = v.string;
            if (std.mem.endsWith(u8, n, "*") and std.mem.startsWith(u8, module, n[0 .. n.len - 1])) return true;
            if (std.mem.eql(u8, n, module)) return true;
        };
        return false;
    }
};

/// Every module's kind: a MIXED_PRECISION config's quantized_layers, or a single-format NVFP4 one's `ignore` list.
pub fn kinds(gpa: std.mem.Allocator, config: []const u8, g: Geometry) !Kinds {
    var why: core.modelopt.Why = .{};
    var r = core.modelopt.admitRecipes(gpa, config, &why) catch |err| {
        std.log.err("{s}", .{why.text()});
        return if (err == error.OutOfMemory) err else error.UnsupportedQwenQuantization;
    };
    defer r.deinit();
    if (g.tied) return error.UnsupportedQwenQuantization;
    if (r.modules.count() == 0) return single(gpa, config, g);
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

/// quant_algo NVFP4 throughout (Inferact's export): every linear NVFP4 but those `ignore` names, which stay bf16.
fn single(gpa: std.mem.Allocator, config: []const u8, g: Geometry) !Kinds {
    const ig = try Ignored.read(gpa, config);
    defer ig.doc.deinit();
    if (!ig.has(prefix ++ "embed_tokens")) return error.UnsupportedQwenQuantization;
    var out: Kinds = .{ .modelopt = true, .head = if (ig.has("lm_head")) .bf16 else .nvfp4 };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) });
        if (used) out.proj[i][@backingInt(slot)] = if (ig.has(name)) .bf16 else .nvfp4;
    };
    return out;
}

/// The device bytes of an n-by-k projection of `kind` in the lane's layout (rows padded to 128).
pub fn packedBytes(n: usize, k: usize, kind: Kind) usize {
    const np = padded(n);
    return switch (kind) {
        .nvfp4, .nvfp4_w16 => aligned(np * k / 2) + aligned(np * k / 16),
        .fp8 => aligned(np * k),
        .bf16 => aligned(np * k * 2),
        .affine => unreachable,
    };
}

/// The checkpoint's bytes of an n-by-k projection of `kind`: codes, then block scales.
pub fn rawBytes(n: usize, k: usize, kind: Kind) usize {
    return switch (kind) {
        .nvfp4, .nvfp4_w16 => n * k / 2 + n * k / 16,
        .fp8 => n * k,
        .bf16 => n * k * 2,
        .affine => unreachable,
    };
}

/// Whether the checkpoint's math runs: only when asked (the default stays the exact bf16 rows), on Blackwell.
pub fn checkpointMath(ctx: *const cuda.Context, g: Geometry, asked: ?geometry.Precision) !bool {
    const p = asked orelse return false;
    if (p == .full) return false;
    if (!g.kinds.modelopt) return error.PrecisionNotBuilt;
    if (try ctx.attribute(.compute_capability_major) < 12) return error.CheckpointMathNeedsBlackwell;
    return true;
}

/// The kernels a ModelOpt checkpoint adds: the lane, prompt16, the packers, the bf16 token rows; ck's in its math.
pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    pack: cuda.modelopt.Packer,
    embed: cuda.embed_rows.EmbedRows,
    ck: ?cuda.ck.Ck = null, // --precision checkpoint: quant8 and the FP8 lane, quant4 and the NVFP4 packer
    dense4: ?cuda.ck_dense.Dense4 = null,
    /// Slice partials where no cluster meets them (before sm_90), and the checkpoint math's quantized rows.
    part: ?DeviceBuffer = null,
    rows8: ?DeviceBuffer = null,
    rows4: ?DeviceBuffer = null,

    pub fn load(d: *const cuda.Driver, major: u32, sms: usize, math: bool) !Kernels {
        var lane = try cuda.qmmf.Lane.load(d, major);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer prompt.unload();
        var pack = try cuda.modelopt.Packer.load(d);
        errdefer pack.unload();
        var rows = try cuda.embed_rows.EmbedRows.load(d);
        errdefer rows.unload();
        var k: Kernels = .{ .lane = lane, .prompt = prompt, .pack = pack, .embed = rows };
        if (!math) return k;
        k.ck = try cuda.ck.Ck.load(d, major, sms);
        errdefer k.ck.?.unload();
        k.dense4 = try cuda.ck_dense.Dense4.load(d, sms);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        for ([_]*?DeviceBuffer{ &k.part, &k.rows8, &k.rows4 }) |b| if (b.*) |*x| x.free();
        if (k.dense4) |*x| x.unload();
        if (k.ck) |*x| x.unload();
        k.embed.unload();
        k.pack.unload();
        k.prompt.unload();
        k.lane.unload();
    }

    pub fn linear(k: *const Kernels) cuda.qlinear.Linear {
        const ptr = struct {
            fn of(b: ?DeviceBuffer) u64 {
                return if (b) |x| x.ptr else 0;
            }
        }.of;
        return .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt, .ck = if (k.ck) |*c| c else null, .ck_rows = ptr(k.rows8), .ck_part = ptr(k.part), .ck4 = if (k.dense4) |*x| x else null, .ck4_rows = ptr(k.rows4) };
    }

    /// The slice partials and quantized rows every projection of `e` takes at up to max_rows rows.
    pub fn reserve(k: *Kernels, e: *const Engine) !void {
        var most = [3]usize{ 0, 0, 0 }; // partials, e4m3 rows, NVFP4 rows
        const R = engine.max_rows;
        const Each = struct {
            fn add(kk: *const Kernels, m: *[3]usize, p: Proj) void {
                const w = switch (p) {
                    .ql => |x| x,
                    else => return,
                };
                switch (w) {
                    .fp8 => |q| if (q.act != 0 and kk.ck != null) {
                        m[1] = @max(m[1], R * q.k);
                        const sk = cuda.qmmf.splitK(q.n, q.k);
                        if (sk > 1 and !kk.ck.?.clusters) m[0] = @max(m[0], sk * R * q.n * 4);
                    } else {
                        m[0] = @max(m[0], kk.lane.partBytes(R, q));
                    },
                    .nvfp4 => |q| if (q.act != 0 and kk.dense4 != null) {
                        m[2] = @max(m[2], cuda.ck.rowsBytes(R, q.k));
                    } else {
                        m[0] = @max(m[0], kk.lane.partBytes(R, q));
                    },
                    .bf16 => |q| m[0] = @max(m[0], kk.lane.partBytes(R, q)),
                    else => {},
                }
            }
        };
        Each.add(k, &most, e.head);
        for (e.layers) |l| {
            switch (l.mlp) {
                .dense => |m| for ([_]Proj{ m.gate, m.up, m.down }) |p| Each.add(k, &most, p),
                .moe => {},
            }
            switch (l.mixer) {
                .delta => |m| for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| Each.add(k, &most, p),
                .attention => |m| for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| Each.add(k, &most, p),
            }
        }
        for ([_]*?DeviceBuffer{ &k.part, &k.rows8, &k.rows4 }, most) |b, n| if (n > 0) {
            b.* = try DeviceBuffer.alloc(e.d, n);
        };
    }
};

/// out (rows, n) = x (rows, k): prompts past 16 rows on prompt16 (or ck's math), else the lane (row bits at any count).
pub fn project(e: *Engine, w: Weight, x: u64, out: u64, rows: usize, prompt: bool) !void {
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    const l = k.linear();
    const in: cuda.qlinear.Rows = .{ .x = x, .part = if (k.part) |p| p.ptr else 0 };
    if (prompt and rows > 16) return l.prompt(e.stream, w, in, out, rows);
    if (rows >= cuda.qmmf.fused_rows) return error.WindowTooWide; // the fused tile sums its slices in another order
    return l.decode(e.stream, w, in, out, rows);
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
    // the static input scale: the checkpoint's math quantizes the rows with it (--precision checkpoint)
    const math = e.mo.?.ck != null and kind != .nvfp4_w16; // a W4A16 layer's own math is weight-only
    const act: f32 = if (math and kind != .bf16 and ck.has(try tname(&buf, pre, name, "input_scale"))) try scalar(ck, try tname(&buf, pre, name, "input_scale")) else 0;
    if (math and kind != .bf16 and act == 0) return error.NoInputScale;
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
        .nvfp4, .nvfp4_w16 => blk: {
            const codes = try ck.expect(try tname(&buf, pre, name, "weight"), .u8, &.{ n, k / 2 });
            const blocks = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .f8_e4m3, &.{ n, k / 16 });
            const global = try scalar(ck, try tname(&buf, pre, name, "weight_scale_2"));
            try stage.upload(0, codes.bytes);
            try stage.upload(codes.bytes.len, blocks.bytes);
            if (math) { // tf_ck_pack4's tiles of one expert, for the FP4 x FP4 mma
                const c = &e.mo.?.ck.?;
                const words = try e.alloc(n * k / 2);
                const tiles = try e.alloc(n * k / 16);
                try c.pack(e.stream, stage.ptr, stage.ptr + codes.bytes.len, 1, n, k, 0, k, words, tiles);
                break :blk .{ .nvfp4 = cuda.ck_dense.weight(words, tiles, global, act, n, k) };
            }
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
    return k.embed.bf16Copies(e.stream, ids, e.embed_w, out, rows, e.g.hidden, 1); // the registered copy kernel
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

test "kinds from a single-format NVFP4 config (Inferact's): NVFP4 but its ignore list, W4A16 recipes weight-only" {
    const head =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","config_groups":{"group_0":
        \\   {"weights":{"num_bits":4,"type":"float","group_size":16},"targets":["Linear"]}},
        \\   "ignore":["lm_head","model.language_model.embed_tokens","model.language_model.layers.0.linear_attn.in_proj_qkv",
        \\     "model.language_model.layers.0.linear_attn.in_proj_a","model.language_model.layers.0.linear_attn.in_proj_b","mtp*"]},
    ;
    const text_config =
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":2,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try geometry.parse(std.testing.allocator, head ++ text_config);
    try std.testing.expectEqual(Kind.bf16, g.kinds.head);
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .qkv));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .z)); // not in the list
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .out));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(1, .q));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(1, .down));
    // an embedding the list leaves out would be NVFP4 rows: refused
    const quantized_rows = try std.mem.replaceOwned(u8, std.testing.allocator, head ++ text_config, "\"model.language_model.embed_tokens\",", "");
    defer std.testing.allocator.free(quantized_rows);
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(std.testing.allocator, quantized_rows));
    const w16 =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
        \\   "model.language_model.layers.0.mlp.down_proj":{"quant_algo":"W4A16_NVFP4","group_size":16},
        \\   "model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"FP8"}}},
    ;
    const h = try geometry.parse(std.testing.allocator, w16 ++ text_config);
    try std.testing.expectEqual(Kind.nvfp4_w16, h.kinds.of(0, .down));
    try std.testing.expectEqual(Kind.bf16, h.kinds.head);
}

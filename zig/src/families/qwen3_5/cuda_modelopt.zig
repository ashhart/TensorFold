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
    if (try compressedTensors(gpa, config)) return .{ .modelopt = true, .ct = true }; // filled from the index
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

/// Whether quantization_config is compressed-tensors' (unsloth's export): its kinds are read from the tensors.
fn compressedTensors(gpa: std.mem.Allocator, config: []const u8) !bool {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, config, .{});
    defer doc.deinit();
    const q = (doc.value.object.get("quantization_config") orelse return false);
    if (q != .object) return false;
    const m = q.object.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "compressed-tensors");
}

/// A compressed-tensors export's kinds by its index: `weight_packed` NVFP4, a `weight_scale` FP8 (a row's), else bf16.
pub fn ctKinds(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, g: Geometry) !Kinds {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    const map = (doc.value.object.get("weight_map") orelse return error.UnsupportedQwenQuantization).object;
    const K = struct {
        fn of(m: std.json.ObjectMap, module: []const u8) !Kind {
            var b: [192]u8 = undefined;
            if (m.contains(try std.fmt.bufPrint(&b, "{s}.weight_packed", .{module}))) return .nvfp4;
            if (m.contains(try std.fmt.bufPrint(&b, "{s}.weight_scale", .{module}))) return .fp8c;
            return .bf16;
        }
    };
    if (try K.of(map, prefix ++ "embed_tokens") != .bf16) return error.UnsupportedQwenQuantization;
    var out: Kinds = .{ .modelopt = true, .ct = true, .head = try K.of(map, "lm_head") };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        if (used) out.proj[i][@backingInt(slot)] = try K.of(map, try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) }));
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
        .fp8c => aligned(np * k) + aligned(np * (k / 64) * 4),
        .bf16 => aligned(np * k * 2),
        .affine => unreachable,
    };
}

/// The checkpoint's bytes of an n-by-k projection of `kind`: codes, then block scales.
pub fn rawBytes(n: usize, k: usize, kind: Kind) usize {
    return switch (kind) {
        .nvfp4, .nvfp4_w16 => n * k / 2 + n * k / 16,
        .fp8, .fp8c => n * k,
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
                        .nvfp4, .fp8, .bf16, .fp8g => |q| lane.partBytes(engine.max_rows, q),
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

/// out (rows, n) = x (rows, k): prompt chunks past 16 rows on prompt16, else the lane (a row's bits at any count).
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
        .fp8c => blk: { // compressed-tensors FP8, a bf16 scale a row: the FP8G lane, every 64 inputs a row's scale
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .f8_e4m3, &.{ n, k });
            const rs = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .bf16, &.{ n, 1 });
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k);
            try p.fp8Lane(e.stream, stage.ptr, w, n, k);
            const tiles = try e.gpa.alloc(f32, padded(n) * (k / 64));
            defer e.gpa.free(tiles);
            rowScales(tiles, std.mem.bytesAsSlice(u16, rs.bytes), n, k);
            const sd = try e.alloc(tiles.len * 4);
            try e.arena.upload(sd - e.arena.ptr, std.mem.sliceAsBytes(tiles));
            break :blk .{ .fp8g = cuda.fp8.weight(w, sd, n, k) };
        },
        .nvfp4, .nvfp4_w16 => blk: {
            // ModelOpt: weight, weight_scale_2 (multiplies); compressed-tensors: weight_packed, weight_global_scale
            const ct = ck.has(try tname(&buf, pre, name, "weight_packed"));
            const codes = try ck.expect(try tname(&buf, pre, name, if (ct) "weight_packed" else "weight"), .u8, &.{ n, k / 2 });
            const blocks = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .f8_e4m3, &.{ n, k / 16 });
            const global = if (ct) 1.0 / try scalar(ck, try tname(&buf, pre, name, "weight_global_scale")) else try scalar(ck, try tname(&buf, pre, name, "weight_scale_2"));
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

/// A bf16 scale a row as FP8G's tiles [npad/64][k/64][64] of fp32 (fp8.packScales' order), rows past n one.
pub fn rowScales(out: []f32, rows: []align(1) const u16, n: usize, k: usize) void {
    const kg = k / 64;
    for (0..padded(n) / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) @bitCast(@as(u32, rows[row]) << 16) else 1.0;
    };
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

test "a row's FP8 scale fills its column of every 64-input tile, padding rows one" {
    const n = 130;
    const k = 128;
    var rows: [n]u16 = undefined;
    for (&rows, 0..) |*r, i| r.* = @truncate(0x3f80 + i); // 1.0 plus a few ulps
    var out: [256 * 2]f32 = undefined;
    rowScales(&out, &rows, n, k);
    const bf = struct {
        fn f(h: u16) f32 {
            return @bitCast(@as(u32, h) << 16);
        }
    }.f;
    try std.testing.expectEqual(bf(rows[5]), out[(0 * 2 + 1) * 64 + 5]); // tile 0, group 1, row 5
    try std.testing.expectEqual(bf(rows[129]), out[(2 * 2 + 0) * 64 + 1]); // tile 2: rows 128..
    try std.testing.expectEqual(@as(f32, 1.0), out[(2 * 2 + 1) * 64 + 2]); // row 130: padding
}

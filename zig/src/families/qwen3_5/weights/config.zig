//! The HIP loader's configuration: the family's geometry (../config.zig) with the checkpoint's quantization table.

const std = @import("std");
const quant = @import("core").quant;
const modelopt_core = @import("core").modelopt;
const shared = @import("../config.zig");

pub const Error = shared.Error || error{UnsupportedQuantization};

/// One affine width: MLX groups of `group` weights share a scale and a bias, `bits` per weight.
pub const Width = quant.mlx.Width;

pub const Spec = shared.Spec;

/// Whether the checkpoint declares ModelOpt's own formats, at the root's `quantization_config` or in `q`.
fn isModelOpt(root: std.json.ObjectMap, q: std.json.ObjectMap) bool {
    if (root.get("quantization_config")) |qc| if (qc == .object) {
        if (declaresModelOpt(qc.object)) return true;
    };
    return declaresModelOpt(q);
}

/// The checks themselves: `quant_method`, `producer`, `quant_algo`, and the per-module table.
fn declaresModelOpt(q: std.json.ObjectMap) bool {
    if (q.get("quant_method")) |m| if (m == .string and std.mem.eql(u8, m.string, "modelopt")) return true;
    if (q.get("producer")) |p| if (p == .object) if (p.object.get("name")) |n| if (n == .string and std.mem.eql(u8, n.string, "modelopt")) return true;
    if (q.get("quant_algo")) |a| if (a == .string) {
        if (std.mem.eql(u8, a.string, "NVFP4") or std.mem.eql(u8, a.string, "MIXED_PRECISION")) return true;
    };
    if (q.get("quantized_layers")) |l| if (l == .object) {
        var it = l.object.iterator();
        while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("quant_algo")) |a| if (a == .string) {
            if (std.mem.eql(u8, a.string, "NVFP4") or std.mem.eql(u8, a.string, "FP8")) return true;
        };
    };
    return false;
}

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    spec: Spec,
    quant: quant.Config,
    /// ModelOpt's weight-only format, NVFP4 as stored: a chat checkpoint, not a decision backbone.
    modelopt: bool = false,
    /// `tie_word_embeddings`: the output head is the embedding.
    tied: bool,
    /// The end ids of config.json and generation_config.json.
    eos: []const u32,

    pub fn deinit(c: *Config) void {
        c.arena.deinit();
        c.* = undefined;
    }

    /// core/modelopt.zig's admission of the checkpoint's recipes: this backend reads NVFP4 and FP8 alone.
    fn admitted(gpa: std.mem.Allocator, bytes: []const u8) Error!modelopt_core.Recipes {
        var why: modelopt_core.Why = .{};
        var r = modelopt_core.admitRecipes(gpa, bytes, &why) catch {
            std.log.err("{s}", .{why.text()});
            return error.UnsupportedQuantization;
        };
        errdefer r.deinit();
        var it = r.modules.iterator();
        while (it.next()) |e| switch (e.value_ptr.*) {
            .nvfp4, .fp8 => {},
            else => {
                std.log.err("{s}: ModelOpt {s} is not a recipe this backend reads", .{ e.key_ptr.*, @tagName(e.value_ptr.*) });
                return error.UnsupportedQuantization;
            },
        };
        return r;
    }

    /// The config of the checkpoint directory `dir`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gen_path);
        const generation = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => null,
            else => return e,
        };
        defer if (generation) |g| gpa.free(g);
        return parse(gpa, bytes, generation);
    }

    /// `load()`'s reading of config.json (root or `text_config`, qwen3_5 or qwen3_5_moe) and generation_config.json.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, generation: ?[]const u8) Error!Config {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const doc = try shared.declared(a, bytes);
        const tower = try shared.textTower(doc);
        const sources: quant.Sources = .{ .root = doc, .quantization = try shared.pickObject(doc, tower, "quantization") };
        // the family's own order: ModelOpt's NVFP4 as stored, an unquantized BF16 checkpoint, then the shared formats
        const modelopt = isModelOpt(doc, sources.quantization);
        if (modelopt) {
            var r = try admitted(gpa, bytes);
            r.deinit();
        }
        const table: quant.Config = if (modelopt)
            .{ .bf16 = {} }
        else if (try quant.bf16.detect(a, sources)) |b| .{ .bf16 = b } else try quant.detect(a, sources);
        const width: Width = if (table == .bf16) .{ .bits = 16, .group = 0 } else try table.width("");
        const spec = try shared.geometry(tower, width.bits, width.group);
        const gen = if (generation) |g| (try std.json.parseFromSliceLeaky(std.json.Value, a, g, .{})) else null;
        if (gen) |g| if (g != .object) return error.InvalidConfig;
        const ids = try a.create([shared.checks.max_eos]u32);
        const eos = try shared.checks.eos(ids, doc, tower, if (gen) |g| g.object else null, spec.vocab);
        return .{ .arena = arena, .spec = spec, .quant = table, .modelopt = modelopt, .tied = shared.isTied(doc, tower), .eos = eos };
    }
};

const sample =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
    \\ "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\   "language_model.model.layers.0.mlp.gate": {"group_size": 32, "bits": 8},
    \\   "vision_tower.x": {"group_size": 64, "bits": 7}},
    \\ "text_config": {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 8,
    \\   "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    \\   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    \\   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320,
    \\   "rms_norm_eps": 1e-06, "full_attention_interval": 4, "mtp_num_hidden_layers": 1, "eos_token_id": 248044,
    \\   "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\     "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    \\   "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
;

test "the config carries the geometry, the quantization width and the tying" {
    var c = try Config.parse(std.testing.allocator, sample, null);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 4096), c.spec.hidden);
    try std.testing.expectEqual(@as(u8, 4), c.spec.bits);
    try std.testing.expectEqual(@as(u16, 64), c.spec.group);
    try std.testing.expect(!c.tied);
}

test "per-tensor widths override the global one, and a bad entry fails only when used" {
    var c = try Config.parse(std.testing.allocator, sample, null);
    defer c.deinit();
    try std.testing.expectEqual(Width{ .bits = 8, .group = 32 }, try c.quant.width("language_model.model.layers.0.mlp.gate"));
    try std.testing.expectEqual(Width{ .bits = 4, .group = 64 }, try c.quant.width("language_model.model.layers.1.mlp.gate"));
    try std.testing.expectError(error.UnsupportedQuantization, c.quant.width("vision_tower.x"));
}

test "the Python loader's refusals of a quantization" {
    const a = std.testing.allocator;
    const bits = try std.mem.replaceOwned(u8, a, sample, "\"bits\": 4,", "\"bits\": 7,");
    defer a.free(bits);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, bits, null));
    const gptq = try std.mem.replaceOwned(u8, a, sample, "\"mode\": \"affine\"", "\"mode\": \"gptq\"");
    defer a.free(gptq);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, gptq, null));
}

test "a config with no quantization is the bf16 format: 16 bits, no groups, no width table" {
    const a = std.testing.allocator;
    const plain = try std.mem.replaceOwned(u8, a, sample, "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\",", "\"q\": {");
    defer a.free(plain);
    var c = try Config.parse(a, plain, null);
    defer c.deinit();
    try std.testing.expectEqual(quant.Format.bf16, std.meta.activeTag(c.quant));
    try std.testing.expectEqual(@as(u8, 16), c.spec.bits);
    try std.testing.expectEqual(@as(u16, 0), c.spec.group);
    try std.testing.expectError(error.UnsupportedQuantization, c.quant.width(""));
    const named = try std.mem.replaceOwned(u8, a, plain, "\"q\": {", "\"quantization_config\": {\"quant_method\": \"fp8\"}, \"q\": {");
    defer a.free(named);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, named, null));
}

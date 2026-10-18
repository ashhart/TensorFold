//! ModelOpt config checks: NVFP4 (groups of 16; tensors, not ignore, say which linears) and MIXED_PRECISION recipes.

const std = @import("std");

/// The NVFP4 group size ModelOpt writes for this format.
pub const nvfp4_group = 16;

/// A MIXED_PRECISION module's recipe, as this engine reads it (the activations stay bf16 whatever the recipe names).
pub const Recipe = enum {
    /// NVFP4: FP4 weights, an e4m3 scale every 16 inputs, a global scale.
    nvfp4,
    /// NVFP4 whose block scales are stored as csf_32x2_v1 (quant/csf.zig decodes them).
    nvfp4_csf,
    /// W4A16_NVFP4: NVFP4 weights without an input scale.
    w4a16_nvfp4,
    /// MXFP8: e4m3 weights, a UE8M0 scale every 32 inputs.
    mxfp8,
    /// FP8_PB_WO: e4m3 weights, an f32 scale every 128 x 128 block.
    fp8_pb_wo,
};

/// The modules a MIXED_PRECISION checkpoint quantizes and their recipes; none for a single-format NVFP4 one.
pub const Recipes = struct {
    arena: std.heap.ArenaAllocator,
    modules: std.StringHashMapUnmanaged(Recipe) = .empty,

    /// The recipe of `name`, a module or one of its tensors: the longest declared module it is or lies under.
    pub fn of(self: *const Recipes, name: []const u8) ?Recipe {
        var n = name;
        while (true) {
            if (self.modules.get(n)) |r| return r;
            n = n[0 .. std.mem.lastIndexOfScalar(u8, n, '.') orelse return null];
        }
    }

    pub fn deinit(self: *Recipes) void {
        self.arena.deinit();
    }
};

/// Why the check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// config.json's quantization_config (top level, else text_config's) checked as ModelOpt NVFP4.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    const q = block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
    try nvfp4(q, why);
}

/// `admit`'s check, with MIXED_PRECISION read: a quantized_layers entry no `Recipe` names is refused by its module.
pub fn admitRecipes(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !Recipes {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    const q = block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
    var r: Recipes = .{ .arena = .init(gpa) };
    errdefer r.deinit();
    if (!std.ascii.eqlIgnoreCase(text(q.get("quant_algo")) orelse "", "MIXED_PRECISION")) {
        try nvfp4(q, why);
        return r;
    }
    const layers = object(q.get("quantized_layers")) orelse {
        why.set("ModelOpt MIXED_PRECISION without quantized_layers", .{});
        return error.UnsupportedQuant;
    };
    const a = r.arena.allocator();
    var it = layers.iterator();
    while (it.next()) |e| {
        const recipe = recipeOf(object(e.value_ptr.*)) orelse {
            const algo = if (object(e.value_ptr.*)) |o| text(o.get("quant_algo")) orelse "no quant_algo" else "no recipe";
            why.set("ModelOpt MIXED_PRECISION {s}: {s} as declared is not a recipe this engine reads", .{ e.key_ptr.*, algo });
            return error.UnsupportedQuant;
        };
        try r.modules.put(a, try a.dupe(u8, e.key_ptr.*), recipe);
    }
    return r;
}

/// A quantized_layers entry as a recipe this engine reads, its group, block and scale encoding included; else null.
fn recipeOf(entry: ?std.json.ObjectMap) ?Recipe {
    const o = entry orelse return null;
    const algo = text(o.get("quant_algo")) orelse return null;
    const group = o.get("group_size");
    const blocks = o.get("weight_block_size");
    const encoding = o.get("weight_scale_encoding");
    if (std.ascii.eqlIgnoreCase(algo, "NVFP4") or std.ascii.eqlIgnoreCase(algo, "W4A16_NVFP4")) {
        if (int(group) != nvfp4_group or blocks != null) return null;
        if (std.ascii.eqlIgnoreCase(algo, "W4A16_NVFP4")) return if (encoding == null) .w4a16_nvfp4 else null;
        if (encoding == null) return .nvfp4;
        return if (std.mem.eql(u8, text(encoding) orelse "", "csf_32x2_v1")) .nvfp4_csf else null;
    }
    if (encoding != null) return null;
    if (std.ascii.eqlIgnoreCase(algo, "MXFP8")) {
        if (group == null and blocks == null) return null;
        return if ((group == null or int(group) == 32) and (blocks == null or pair(blocks, 1, 32))) .mxfp8 else null;
    }
    if (std.ascii.eqlIgnoreCase(algo, "FP8_PB_WO")) return if (group == null and pair(blocks, 128, 128)) .fp8_pb_wo else null;
    return null;
}

/// `v` is the JSON array [rows, cols].
fn pair(v: ?std.json.Value, rows: i64, cols: i64) bool {
    const x = v orelse return false;
    if (x != .array or x.array.items.len != 2) return false;
    return int(x.array.items[0]) == rows and int(x.array.items[1]) == cols;
}

/// The quantization_config's NVFP4 check: quant_algo NVFP4, every config group 4-bit floats in groups of 16.
fn nvfp4(q: std.json.ObjectMap, why: *Why) !void {
    const algo = text(q.get("quant_algo")) orelse "";
    if (!std.ascii.eqlIgnoreCase(algo, "NVFP4")) {
        const named = if (algo.len > 0) algo else "without quant_algo";
        why.set("ModelOpt {s}: this engine reads ModelOpt NVFP4 (nvidia/GLM-5.3-Flash-NVFP4's layout)", .{named});
        return error.UnsupportedQuant;
    }
    const groups = object(q.get("config_groups")) orelse {
        why.set("ModelOpt NVFP4 without config_groups", .{});
        return error.UnsupportedQuant;
    };
    var it = groups.iterator();
    while (it.next()) |g| {
        const w = if (object(g.value_ptr.*)) |o| object(o.get("weights")) else null;
        const bits = if (w) |o| int(o.get("num_bits")) else null;
        const size = if (w) |o| int(o.get("group_size")) else null;
        const kind = if (w) |o| text(o.get("type")) orelse "" else "";
        if (bits != 4 or size != nvfp4_group or !std.mem.eql(u8, kind, "float")) {
            why.set(
                "ModelOpt NVFP4 group {s}: weights {?d}-bit {s} in groups of {?d}, " ++
                    "expected 4-bit float in groups of 16",
                .{ g.key_ptr.*, bits, kind, size },
            );
            return error.UnsupportedQuant;
        }
    }
}

fn block(root: std.json.Value) ?std.json.ObjectMap {
    const top = object(root) orelse return null;
    for ([_]?std.json.ObjectMap{ top, object(top.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        if (std.ascii.eqlIgnoreCase(text(q.get("quant_method")) orelse "", "modelopt")) return q;
    }
    return null;
}

fn object(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn text(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn int(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return if (x == .integer) x.integer else null;
}

test "nvidia/GLM-5.3-Flash-NVFP4's config is admitted" {
    var why: Why = .{};
    try admit(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
}

/// admit() must refuse `json` with `err`, its reason naming `word`.
fn expectRefused(json: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, admit(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "other ModelOpt formats, and configs without ModelOpt, are refused by name" {
    try expectRefused("{\"text_config\": {}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused("{\"quantization_config\": {\"quant_method\": \"exl3\"}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"FP8\"}}",
        error.UnsupportedQuant,
        "ModelOpt FP8",
    );
    // MIXED_PRECISION (the AXL re-pack) is refused for now
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"}}",
        error.UnsupportedQuant,
        "MIXED_PRECISION",
    );
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}}}}
    , error.UnsupportedQuant, "groups of 32");
    var why: Why = .{};
    try admit(std.testing.allocator,
        \\{"text_config": {"quantization_config": {"quant_method": "modelopt", "quant_algo": "nvfp4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}}}
    , &why);
}

test "nvidia/GLM-5.3-Flash-FP4-Local's recipes: a module's, and its tensors' under it" {
    var why: Why = .{};
    var r = try admitRecipes(std.testing.allocator, @embedFile("fixtures/glm53_flash_fp4_local/config.json"), &why);
    defer r.deinit();
    var count = std.EnumArray(Recipe, usize).initFill(0);
    var it = r.modules.valueIterator();
    while (it.next()) |v| count.getPtr(v.*).* += 1;
    try std.testing.expectEqualSlices(usize, &.{ 0, 42, 77, 205, 170 }, &count.values); // the vision tower's MLPs too
    const at = "model.language_model.layers.";
    const cases = [_]struct { []const u8, ?Recipe }{
        .{ at ++ "3.mlp.experts.17.gate_proj.weight", .nvfp4_csf }, // routed experts: one entry a layer
        .{ at ++ "45.mlp.experts.0.down_proj.weight_scale", .w4a16_nvfp4 }, // MTP's
        .{ at ++ "0.self_attn.q_proj.weight", .mxfp8 }, // KDA, weight_block_size [1, 32]
        .{ at ++ "45.self_attn.kv_b_proj.weight", .mxfp8 }, // MTP's, group_size 32
        .{ "lm_head.weight", .mxfp8 },
        .{ at ++ "3.self_attn.q_a_proj.weight_scale", .fp8_pb_wo }, // MLA
        .{ at ++ "43.mlp.shared_experts.down_proj.weight", .fp8_pb_wo },
        .{ at ++ "3.self_attn.kv_b_proj.weight", null }, // bf16
        .{ at ++ "0.self_attn.f_a_proj.weight", null },
        .{ at ++ "3.mlp.gate.weight", null }, // the router
    };
    for (cases) |c| try std.testing.expectEqual(c[1], r.of(c[0]));
}

test "a recipe covers its module and what lies under it, at a dot" {
    var r: Recipes = .{ .arena = .init(std.testing.allocator) };
    defer r.deinit();
    try r.modules.put(r.arena.allocator(), "m.experts", .nvfp4);
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts"));
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts.1.up_proj.weight"));
    try std.testing.expectEqual(@as(?Recipe, null), r.of("m.experts_bias"));
    try std.testing.expectEqual(@as(?Recipe, null), r.of("m"));
}

test "a single-format NVFP4 config declares no recipes" {
    var why: Why = .{};
    var r = try admitRecipes(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 0), r.modules.count());
}

/// admitRecipes() must refuse a MIXED_PRECISION config of these `layers` as unsupported, its reason naming `word`.
fn expectRecipesRefused(comptime layers: []const u8, word: []const u8) !void {
    const json = "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"" ++
        layers ++ "}}";
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuant, admitRecipes(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "recipes this engine does not read are refused by module" {
    try expectRecipesRefused("", "without quantized_layers");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.q": {"quant_algo": "INT4_AWQ", "group_size": 128}}
    , "m.q: INT4_AWQ");
    try expectRecipesRefused( // the slab encoding (csf) is not read
        \\, "quantized_layers": {"m.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"}}
    , "m.experts: NVFP4");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.e": {"quant_algo": "W4A16_NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1"}}
    , "m.e: W4A16_NVFP4");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.o": {"quant_algo": "MXFP8", "group_size": 16}}
    , "m.o: MXFP8");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.o": {"quant_algo": "MXFP8", "group_size": 32, "weight_block_size": [1, 64]}}
    , "m.o: MXFP8");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.s": {"quant_algo": "FP8_PB_WO", "weight_block_size": [64, 64]}}
    , "m.s: FP8_PB_WO");
    try expectRecipesRefused(
        \\, "quantized_layers": {"m.s": 7}
    , "m.s: no recipe");
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuant, admitRecipes(std.testing.allocator,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
    , &why));
    var r = try admitRecipes(std.testing.allocator,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.experts": {"quant_algo": "NVFP4", "group_size": 16}}}}
    , &why);
    defer r.deinit();
    try std.testing.expectEqual(@as(?Recipe, .nvfp4), r.of("m.experts.0.gate_proj.weight"));
}

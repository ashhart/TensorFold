//! A Nemotron Omni checkpoint's image side from config.json and preprocessor_config.json: tower and projector.
const std = @import("std");
const Value = std.json.Value;
const Object = std.json.ObjectMap;

pub const Config = struct {
    hidden: usize, // the tower's width (vit_hidden_size)
    layers: usize,
    heads: usize,
    ffn: usize, // hidden * mlp_ratio
    eps: f32, // the tower's LayerNorms, its final one too
    patch: usize,
    prefix: usize, // cls and register rows ahead of the patches (num_cls_tokens + num_registers)
    side: usize, // the position table's side: max_img_size / patch
    projector_hidden: usize,
    text_hidden: usize,
    final_norm: bool, // the tower's last LayerNorm (vision_final_layernorm): the language model has an MTP layer
    mean: [3]f32,
    std: [3]f32,
    min_patches: u32,
    max_patches: u32,
    max_model_len: u32,
    image_token: u32, // img_context_token_id

    /// 4 patches (2 x 2) to one token: downsample_ratio 0.5.
    pub const shuffle = 2;
    /// The projector's RMSNorm epsilon (the omni MLP's default; config.json does not carry it).
    pub const rms_eps: f32 = 1e-5;
    pub const channels = 3;

    pub fn dim(c: Config) usize {
        return c.hidden / c.heads;
    }

    pub fn patchWidth(c: Config) usize {
        return channels * c.patch * c.patch;
    }

    /// The config, or null with `problem` (from `a`) naming what the kernels cannot read; NoVisionTower without one.
    pub fn parse(a: std.mem.Allocator, config_json: []const u8, preprocessor_json: ?[]const u8, problem: *[]const u8) !?Config {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const ar = arena.allocator();
        const doc = try std.json.parseFromSliceLeaky(Value, ar, config_json, .{});
        if (doc != .object) return error.BadConfig;
        const root = doc.object;
        const v = object(root.get("vision_config")) orelse return error.NoVisionTower;
        const llm = object(root.get("llm_config")) orelse object(root.get("text_config")) orelse return error.BadConfig;
        if (try refusal(ar, root, v)) |why| {
            problem.* = try a.dupe(u8, why);
            return null;
        }
        const pre = if (preprocessor_json) |text| object(try std.json.parseFromSliceLeaky(Value, ar, text, .{})) else null;
        const hidden = try size(v.get("hidden_size"));
        const patch = try size(v.get("patch_size"));
        const c: Config = .{
            .hidden = hidden,
            .layers = try size(v.get("num_hidden_layers")),
            .heads = try size(v.get("num_attention_heads")),
            .ffn = @intFromFloat(@as(f64, @floatFromInt(hidden)) * try float(v.get("mlp_ratio"))),
            .eps = @floatCast(try float(v.get("layer_norm_eps"))),
            .patch = patch,
            .prefix = try size(v.get("num_cls_tokens")) + try size(v.get("num_registers")),
            .side = try size(v.get("max_img_size")) / patch,
            .projector_hidden = try size(root.get("projector_hidden_size")),
            .text_hidden = try size(llm.get("hidden_size")),
            .final_norm = (if (llm.get("num_nextn_predict_layers")) |n| try size(n) else 0) > 0,
            .mean = try triple(root.get("norm_mean") orelse v.get("norm_mean")),
            .std = try triple(root.get("norm_std") orelse v.get("norm_std")),
            .min_patches = @intCast(try size(field(pre, v, "min_num_patches"))),
            .max_patches = @intCast(try size(field(pre, v, "max_num_patches"))),
            .max_model_len = if (pre) |p| @intCast(try size(p.get("max_model_len"))) else 16384,
            .image_token = @intCast(try size(root.get("img_context_token_id"))),
        };
        if (c.hidden != try size(root.get("vit_hidden_size"))) return refuse(a, problem, "vit_hidden_size differs from the tower's hidden_size");
        if (c.dim() * c.heads != c.hidden or (c.dim() != 72 and c.dim() != 80)) return refuse(a, problem, "the tower's head width is not 72 or 80");
        if (c.side * patch != try size(v.get("max_img_size")) or c.min_patches == 0 or c.max_patches < c.min_patches) return refuse(a, problem, "the processor's patch bounds are inconsistent");
        return c;
    }
};

/// The first reason the tower or projector is outside what the kernels compute, or null.
fn refusal(a: std.mem.Allocator, root: Object, v: Object) !?[]const u8 {
    const want = [_]struct { []const u8, Object, []const u8, []const u8 }{
        .{ "model_type", v, "radio", "the vision tower is not RADIO" },
        .{ "hidden_act", v, "gelu", "the tower's activation is not exact GELU" },
        .{ "ps_version", root, "v2", "the pixel shuffle is not version 2" },
    };
    for (want) |w| {
        const got = w[1].get(w[0]) orelse return w[3];
        if (got != .string or !std.mem.eql(u8, got.string, w[2])) return w[3];
    }
    if (truthy(v.get("use_swiglu_ffn"))) return "the tower's MLP is SwiGLU";
    if (!truthy(v.get("qkv_bias"))) return "the tower's qkv projection has no bias";
    if ((try float(v.get("layerscale_value"))) != 1.0) return "the tower's layer scale is not 1.0";
    if ((try float(root.get("downsample_ratio"))) != 0.5) return try std.fmt.allocPrint(a, "downsample_ratio {d} (0.5 only)", .{try float(root.get("downsample_ratio"))});
    if ((try size(v.get("num_channels"))) != Config.channels) return "the tower does not read three channels";
    return null;
}

fn refuse(a: std.mem.Allocator, problem: *[]const u8, why: []const u8) !?Config {
    problem.* = try a.dupe(u8, why);
    return null;
}

fn object(v: ?Value) ?Object {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

/// The preprocessor's value, else the tower config's.
fn field(pre: ?Object, v: Object, name: []const u8) ?Value {
    if (pre) |p| if (p.get(name)) |x| return x;
    return v.get(name);
}

fn truthy(v: ?Value) bool {
    const x = v orelse return false;
    return x == .bool and x.bool;
}

fn size(v: ?Value) !usize {
    const x = v orelse return error.BadConfig;
    if (x != .integer or x.integer < 0) return error.BadConfig;
    return @intCast(x.integer);
}

fn float(v: ?Value) !f64 {
    const x = v orelse return error.BadConfig;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadConfig,
    };
}

/// Three numbers as torch.tensor reads them: the JSON's double, then float32.
fn triple(v: ?Value) ![3]f32 {
    const x = v orelse return error.BadConfig;
    if (x != .array or x.array.items.len != 3) return error.BadConfig;
    var out: [3]f32 = undefined;
    for (&out, x.array.items) |*o, i| o.* = @floatCast(try float(i));
    return out;
}

test "the public checkpoint's image side, and refusals by name" {
    const a = std.testing.allocator;
    const text =
        \\{"downsample_ratio": 0.5, "img_context_token_id": 18, "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        \\ "norm_std": [0.26862954, 0.26130258, 0.27577711], "projector_hidden_size": 20480, "ps_version": "v2",
        \\ "vit_hidden_size": 1280, "llm_config": {"hidden_size": 4096, "num_nextn_predict_layers": 1},
        \\ "vision_config": {"model_type": "radio", "hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16,
        \\  "mlp_ratio": 4.0, "hidden_act": "gelu", "layer_norm_eps": 1e-06, "layerscale_value": 1.0, "qkv_bias": true,
        \\  "use_swiglu_ffn": false, "patch_size": 16, "num_channels": 3, "num_cls_tokens": 3, "num_registers": 7,
        \\  "max_img_size": 2048, "min_num_patches": 1024, "max_num_patches": 13312}}
    ;
    var problem: []const u8 = "";
    const c = (try Config.parse(a, text, "{\"max_model_len\": 16384}", &problem)).?;
    try std.testing.expectEqual(@as(usize, 80), c.dim());
    try std.testing.expectEqual(@as(usize, 5120), c.ffn);
    try std.testing.expectEqual(@as(usize, 10), c.prefix);
    try std.testing.expectEqual(@as(usize, 128), c.side);
    try std.testing.expectEqual(@as(usize, 768), c.patchWidth());
    try std.testing.expect(c.final_norm);
    try std.testing.expectEqual(@as(u32, 13312), c.max_patches);
    try std.testing.expectEqual(@as(f32, 0.48145466), c.mean[0]);
    const swapped = try std.mem.replaceOwned(u8, a, text, "\"ps_version\": \"v2\"", "\"ps_version\": \"v1\"");
    defer a.free(swapped);
    const refused = try Config.parse(a, swapped, null, &problem);
    defer a.free(problem);
    try std.testing.expect(refused == null);
    try std.testing.expectEqualStrings("the pixel shuffle is not version 2", problem);
}

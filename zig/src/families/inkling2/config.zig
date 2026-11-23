//! Inkling 2 Small and Mini shapes from config.json, refusing every layout the engine does not run.
const std = @import("std");

/// The most backbone layers the plan tables hold (Inkling 2 Small has 48).
pub const max_layers = 128;

/// The most scheduled layer visits the visit-plan table holds.
pub const max_visits = max_layers * 8;

/// Every sixth layer attends over the whole context; the reference derives its layer types the same way.
pub const global_every = 6;

/// The short convolutions' taps, as the checkpoint stores them.
pub const conv_taps = 4;

/// The reference config's default for image_token_id.
pub const default_image_token = 200054;

/// Layer i attends over the whole context; the others over a sliding window.
pub fn isGlobal(i: u32) bool {
    return (i + 1) % global_every == 0;
}

/// Why parse refused the config, one sentence for the caller's log line (tests read it too).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// One attention geometry: sliding layers read the `swa_*` keys, global layers the plain ones.
pub const Heads = struct { q: u32, kv: u32, dim: u32 };

/// The fixed loop schedule: all layers once, then only the core on each later pass.
pub const Loop = struct {
    core_start: u32, // inclusive physical layer id
    core_end: u32, // exclusive physical layer id
    passes: u32,
    gate_rank: u32,
};

pub const Text = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    unpadded_vocab: u32, // rows of the unembedding that hold real tokens
    local: Heads,
    global: Heads,
    window: u32, // sliding_window_size: the local layers' span and their relative-bias extent
    d_rel: u32,
    rel_extent: u32, // the global layers' relative-bias extent
    log_floor: u32,
    log_alpha: f32,
    eps: f32,
    qk_clip: f32,
    rel_clip: f32,
    block: u32, // n_layers_per_attn_res_block
    experts: u32,
    topk: u32,
    expert_inter: u32,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
    logit_cap: f32,
    max_context: u32,
    loop: ?Loop,

    /// The number of scheduled layer visits; a drafter target equal to this is the final norm.
    pub fn visitCount(c: *const Text) u32 {
        const l = c.loop orelse return c.layers;
        return c.layers + (l.passes - 1) * (l.core_end - l.core_start);
    }

    pub fn heads(c: *const Text, i: u32) Heads {
        return if (isGlobal(i)) c.global else c.local;
    }

    /// Layer i's relative-bias extent in backward tokens.
    pub fn extent(c: *const Text, i: u32) u32 {
        return if (isGlobal(i)) c.rel_extent else c.window;
    }
};

/// The draft head's shape (mtp_config); target taps are visit ids, with `visitCount` meaning the final norm.
pub const Mtp = struct {
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    rope_theta: f32,
    window: u32,
    local: [8]bool, // local[i]: draft layer i slides
    block: u32,
    rank: u32,
    taps: [16]u32,
    taps_n: u32,
};

pub const Vision = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    patch: u32,
    merge: u32,
    grid: u32,
    mlp_ratio: u32,
    proj_hidden: u32,
    text_hidden: u32,
    eps: f32, // layer_norm_eps
};

pub const Config = struct {
    text: Text,
    vision: Vision,
    mtp: ?Mtp,
    eos: u32,
    image_token: u32,
};

fn get(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn object(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.ObjectMap {
    const v = try get(o, key, why);
    if (v != .object) {
        why.set("config.json's {s} is not an object", .{key});
        return error.BadConfig;
    }
    return v.object;
}

fn int(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try get(o, key, why);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) {
        why.set("config.json's {s} is not a positive whole number", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn index(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try get(o, key, why);
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) {
        why.set("config.json's {s} is not a non-negative whole number", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn float(o: std.json.ObjectMap, key: []const u8, why: *Why) !f32 {
    const v = try get(o, key, why);
    const x: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
    if (!std.math.isFinite(x) or x <= 0) {
        why.set("config.json's {s} is not a positive number", .{key});
        return error.BadConfig;
    }
    return @floatCast(x);
}

/// A key the engine runs one way only: absent keeps the default, any other value is refused by name.
fn fixed(o: std.json.ObjectMap, key: []const u8, want: bool, why: *Why) !void {
    const v = o.get(key) orelse return;
    if (v == .bool and v.bool == want) return;
    why.set("{s} must be {s}: the engine runs no other layout", .{ key, if (want) "true" else "false" });
    return error.Unsupported;
}

fn text(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn heads(o: std.json.ObjectMap, prefix: []const u8, why: *Why) !Heads {
    var kb: [24]u8 = undefined;
    const q = try int(o, try std.fmt.bufPrint(&kb, "{s}num_attention_heads", .{prefix}), why);
    const kv = try int(o, try std.fmt.bufPrint(&kb, "{s}num_key_value_heads", .{prefix}), why);
    const dim = try int(o, try std.fmt.bufPrint(&kb, "{s}head_dim", .{prefix}), why);
    if (q % kv != 0) {
        why.set("{d} query heads do not split over {d} key/value heads", .{ q, kv });
        return error.Layout;
    }
    return .{ .q = q, .kv = kv, .dim = dim };
}

/// local_layer_ids must list exactly the layers the reference's every-sixth rule leaves sliding.
fn layerTypes(o: std.json.ObjectMap, layers: u32, why: *Why) !void {
    const v = try get(o, "local_layer_ids", why);
    var want: u32 = 0;
    for (0..layers) |i| want += @intFromBool(!isGlobal(@intCast(i)));
    var seen: [max_layers]bool = @splat(false);
    if (v != .array or v.array.items.len != want) {
        why.set("local_layer_ids must list the {d} sliding layers (every {d}th layer is global)", .{ want, global_every });
        return error.Layout;
    }
    for (v.array.items) |e| {
        const ok = e == .integer and e.integer >= 0 and e.integer < layers and !isGlobal(@intCast(e.integer));
        if (!ok or seen[@intCast(e.integer)]) {
            why.set("local_layer_ids names a global, repeated or missing layer: the engine runs the every-{d}th-global layout", .{global_every});
            return error.Layout;
        }
        seen[@intCast(e.integer)] = true;
    }
}

fn parseLoop(o: std.json.ObjectMap, layers: u32, block: u32, why: *Why) !Loop {
    const l: Loop = .{
        .core_start = try index(o, "core_start", why),
        .core_end = try int(o, "core_end", why),
        .passes = try int(o, "num_passes", why),
        .gate_rank = try int(o, "gate_rank", why),
    };
    try fixed(o, "is_varloop", false, why);
    if (l.core_start == 0 or l.core_start >= l.core_end or l.core_end > layers) {
        why.set("loop_config core range {d}..{d} must start after layer 0, be nonempty and fit inside the {d} physical layers", .{ l.core_start, l.core_end, layers });
        return error.Layout;
    }
    if (l.core_start % block != 0 or l.core_end % block != 0 or l.core_start != block or l.core_end + block != layers) {
        why.set("loop_config core_start and core_end must be multiples of the {d}-layer attention-residual block, leaving its first and last blocks outside the loop", .{block});
        return error.Layout;
    }
    const visits = @as(u64, layers) + @as(u64, l.passes - 1) * @as(u64, l.core_end - l.core_start);
    if (visits > max_visits) {
        why.set("loop_config makes {d} layer visits: the plan tables hold {d}", .{ visits, max_visits });
        return error.Layout;
    }
    return l;
}

fn parseText(o: std.json.ObjectMap, why: *Why) !Text {
    var t: Text = undefined;
    t.loop = null;
    t.hidden = try int(o, "hidden_size", why);
    t.layers = try int(o, "num_hidden_layers", why);
    t.vocab = try int(o, "vocab_size", why);
    t.unpadded_vocab = try int(o, "unpadded_vocab_size", why);
    t.global = try heads(o, "", why);
    t.local = try heads(o, "swa_", why);
    t.window = try int(o, "sliding_window_size", why);
    t.d_rel = try int(o, "d_rel", why);
    t.rel_extent = try int(o, "rel_extent", why);
    t.log_floor = try int(o, "log_scaling_n_floor", why);
    t.log_alpha = try float(o, "log_scaling_alpha", why);
    t.eps = try float(o, "rms_norm_eps", why);
    t.qk_clip = try float(o, "qk_l2_clip_threshold", why);
    t.rel_clip = try float(o, "rel_l2_clip_threshold", why);
    t.block = try int(o, "n_layers_per_attn_res_block", why);
    t.experts = try int(o, "n_routed_experts", why);
    t.topk = try int(o, "num_experts_per_tok", why);
    t.expert_inter = try int(o, "intermediate_size", why);
    t.route_scale = try float(o, "route_scale", why);
    t.gate_cap = try float(o, "mlp_gate_softcap", why);
    t.up_cap = try float(o, "mlp_up_softcap", why);
    t.logit_cap = try float(o, "final_logit_softcapping", why);
    t.max_context = try int(o, "model_max_length", why);
    if (t.layers > max_layers) {
        why.set("{d} layers: the plan tables hold {d}", .{ t.layers, max_layers });
        return error.Layout;
    }
    if (t.block > t.layers) {
        why.set("n_layers_per_attn_res_block {d} is past num_hidden_layers {d}", .{ t.block, t.layers });
        return error.Layout;
    }
    if (try int(o, "sconv_kernel_size", why) != conv_taps) {
        why.set("sconv_kernel_size must be {d}: the checkpoint's published short convolutions", .{conv_taps});
        return error.Unsupported;
    }
    if (t.unpadded_vocab > t.vocab) {
        why.set("unpadded_vocab_size {d} is past vocab_size {d}", .{ t.unpadded_vocab, t.vocab });
        return error.Layout;
    }
    if (t.topk > t.experts) {
        why.set("num_experts_per_tok {d} is past n_routed_experts {d}", .{ t.topk, t.experts });
        return error.Layout;
    }
    if (t.hidden % 32 != 0) {
        why.set("hidden_size {d} is not a multiple of 32, the quantized experts' group size", .{t.hidden});
        return error.Layout;
    }
    if (t.expert_inter % 32 != 0) {
        why.set("intermediate_size {d} is not a multiple of 32, the quantized experts' group size", .{t.expert_inter});
        return error.Layout;
    }
    if (o.get("n_shared_experts")) |n| if (n != .integer or n.integer != 0) {
        why.set("n_shared_experts must be 0: Inkling 2 has routed experts only", .{});
        return error.Unsupported;
    };
    try fixed(o, "q_bias", false, why);
    try fixed(o, "o_bias", false, why);
    try fixed(o, "pre_norm_pre_affine", false, why);
    try fixed(o, "final_norm_pre_affine", false, why);
    try fixed(o, "use_value_norm_no_gamma", true, why);
    try fixed(o, "use_gate_bias", true, why);
    try fixed(o, "tie_word_embeddings", false, why);
    if (text(o, "scoring_func")) |s| if (!std.mem.eql(u8, s, "sigmoid")) {
        why.set("scoring_func must be \"sigmoid\"", .{});
        return error.Unsupported;
    };
    try layerTypes(o, t.layers, why);
    if (o.get("loop_config")) |_| t.loop = try parseLoop(try object(o, "loop_config", why), t.layers, t.block, why);
    return t;
}

fn parseVision(o: std.json.ObjectMap, text_hidden: u32, why: *Why) !Vision {
    var v: Vision = undefined;
    v.hidden = try int(o, "hidden_size", why);
    v.layers = try int(o, "num_hidden_layers", why);
    v.heads = try int(o, "num_attention_heads", why);
    v.patch = try int(o, "native_patch_size", why);
    v.merge = try int(o, "adapter_merge_size", why);
    v.grid = try int(o, "posemb_grid_size", why);
    v.proj_hidden = try int(o, "projection_hidden_dim", why);
    v.text_hidden = if (o.get("text_hidden_size") != null) try int(o, "text_hidden_size", why) else text_hidden;
    const ratio = try float(o, "mlp_ratio", why);
    if (ratio != @floor(ratio) or ratio > 1024 or v.hidden % v.heads != 0 or v.text_hidden != text_hidden) {
        why.set("vision_config: a whole mlp_ratio of at most 1024, heads that divide the width and text_hidden_size equal to hidden_size", .{});
        return error.Layout;
    }
    v.mlp_ratio = @intFromFloat(ratio);
    v.eps = if (o.contains("layer_norm_eps")) try float(o, "layer_norm_eps", why) else 1e-5;
    try fixed(o, "projector_bias", false, why);
    return v;
}

fn parseMtp(o: std.json.ObjectMap, t: *const Text, why: *Why) !Mtp {
    var m = std.mem.zeroes(Mtp);
    m.layers = try int(o, "num_hidden_layers", why);
    m.q_heads = try int(o, "num_attention_heads", why);
    m.kv_heads = try int(o, "num_key_value_heads", why);
    m.head_dim = try int(o, "head_dim", why);
    m.inter = try int(o, "intermediate_size", why);
    m.rope_theta = try float(o, "rope_theta", why);
    m.window = try int(o, "sliding_window_size", why);
    m.block = try int(o, "block_size", why);
    m.rank = try int(o, "markov_rank", why);
    const ids = try get(o, "local_layer_ids", why);
    const taps = try get(o, "target_layer_ids", why);
    const bad = m.layers > m.local.len or m.q_heads % m.kv_heads != 0 or ids != .array or taps != .array or
        taps.array.items.len == 0 or taps.array.items.len > m.taps.len or try int(o, "hidden_size", why) != t.hidden;
    if (bad or !std.mem.eql(u8, text(o, "markov_head_type") orelse "", "rnn_hidden")) {
        why.set("mtp_config: a draft head of the backbone's width, with rnn_hidden Markov head and listed layers", .{});
        return error.Layout;
    }
    for (ids.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer >= m.layers) {
            why.set("mtp_config's local_layer_ids names a layer the head does not have", .{});
            return error.Layout;
        }
        m.local[@intCast(e.integer)] = true;
    }
    for (taps.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer > t.visitCount()) {
            why.set("mtp_config's target_layer_ids names a layer the backbone does not have", .{});
            return error.Layout;
        }
        m.taps[m.taps_n] = @intCast(e.integer);
        m.taps_n += 1;
    }
    return m;
}

/// A quantization_config must be ModelOpt NVFP4 (the experts' header dtype refines it per layer) or EXL3.
fn quantization(o: std.json.ObjectMap, why: *Why) !void {
    const v = o.get("quantization_config") orelse return;
    const q = if (v == .object) v.object else {
        why.set("quantization_config is not an object", .{});
        return error.Quant;
    };
    if (std.mem.eql(u8, text(q, "quant_method") orelse "", "exl3")) return; // EXL3 tensors carry their own records
    const ok = std.ascii.eqlIgnoreCase(text(q, "quant_method") orelse "", "modelopt") and
        std.ascii.eqlIgnoreCase(text(q, "quant_algo") orelse "", "NVFP4");
    if (!ok) {
        why.set("quantization_config: the loader reads ModelOpt NVFP4 experts or an EXL3 pack", .{});
        return error.Quant;
    }
}

/// config.json as the reference (configuration_inkling2.py) and the checkpoint read it; a refusal is in `why`.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, why: *Why) !Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        why.set("config.json is not an object", .{});
        return error.BadConfig;
    }
    const o = parsed.value.object;
    const arch = o.get("architectures");
    const arch_ok = arch != null and arch.? == .array and arch.?.array.items.len == 1 and arch.?.array.items[0] == .string;
    const arch_name: []const u8 = if (arch_ok) arch.?.array.items[0].string else "";
    const model_name = text(o, "model_type") orelse "";
    const small = std.mem.eql(u8, arch_name, "Inkling2ForConditionalGeneration") and std.mem.eql(u8, model_name, "inkling2");
    const looped = std.mem.eql(u8, arch_name, "Inkling2LoopForConditionalGeneration") and std.mem.eql(u8, model_name, "inkling2_loop");
    if (!small and !looped) {
        why.set("not an Inkling 2 checkpoint: Small or Mini model_type and architectures must match", .{});
        return error.NotInkling2;
    }
    try fixed(o, "tie_word_embeddings", false, why);
    try quantization(o, why);
    const t = try parseText(try object(o, "text_config", why), why);
    if (looped and t.loop == null) {
        why.set("inkling2_loop requires text_config.loop_config", .{});
        return error.BadConfig;
    }
    if (small and t.loop != null) {
        why.set("loop_config requires the inkling2_loop model and architecture", .{});
        return error.Unsupported;
    }
    const eos = try int(o, "eos_token_id", why);
    if (eos >= t.vocab) {
        why.set("eos_token_id is not a token of the {d}-token vocabulary", .{t.vocab});
        return error.BadConfig;
    }
    const mtp = if (o.get("mtp_config") != null) try parseMtp(try object(o, "mtp_config", why), &t, why) else null;
    const image = if (o.get("image_token_id") != null) try int(o, "image_token_id", why) else default_image_token;
    return .{
        .text = t,
        .vision = try parseVision(try object(o, "vision_config", why), t.hidden, why),
        .mtp = mtp,
        .eos = eos,
        .image_token = image,
    };
}

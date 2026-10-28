//! Ling-3.1-flash's shape from config.json (bailing_hybrid), refusing what the engine does not run.
const std = @import("std");

/// Backbone plus MTP layers the per-layer tables hold (Ling-3.1-flash: 72 + 1).
pub const max_layers = 80;

pub const Kind = enum { kda, mla };

/// Group-limited routing as the CUDA and Metal kernels will take it: n groups, keep of them.
pub const Groups = struct { n: u32, keep: u32 };

/// Layer i's SwiGLU clamps, 0 for none.
pub const Limits = struct { routed: f32, shared: f32, dense: f32 };

/// MLA's rope pairs: rope_interleave true is exllamav3's RopeStyle.GPTJ (adjacent pairs), false NeoX (halves).
pub const RopeStyle = enum { gptj, neox };

/// A linear's format: bf16, FP8 128x128 blocks, ModelOpt NVFP4; EXL3 (#832) adds a variant and a weights_proj reader.
pub const Quant = enum { bf16, fp8_block128, nvfp4 };

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

pub const Config = struct {
    hidden: u32,
    layers: u32, // backbone layers; the MTP layers follow at [layers, layers + mtp)
    vocab: u32,
    eps: f32,
    rope_theta: f64,
    rope_dim: u32, // qk_rope_head_dim: the modeling's rotary embedding takes it whole (partial factor 1)
    rope_style: RopeStyle,
    group: u32, // layer_group_size: MLA on each group's last layer and on the tail past the last full group
    dense_layers: u32, // first_k_dense_replace
    dense_inter: u32,
    kda_heads: u32,
    kda_dim: u32,
    conv: u32,
    lower_bound: f32, // kda_safe_gate's kda_lower_bound
    mla_heads: u32,
    q_lora: u32, // 0: no q_lora_rank, one q_proj
    kv_lora: u32,
    nope: u32,
    v_dim: u32,
    head_gate: bool, // gated_attention_proj_granularity_type head_wise: g_proj gives a sigmoid gate per head
    experts: u32,
    topk: u32,
    n_group: u32,
    topk_group: u32,
    routed_scale: f32,
    norm_topk: bool,
    expert_bias: bool, // moe_router_enable_expert_bias: gate.expert_bias, added for the pick only
    moe_inter: u32,
    shared_inter: u32, // moe_shared_expert_intermediate_size x num_shared_experts, 0 without a shared expert
    expert_limit: [max_layers]f32, // expert_swiglu_limit_list by layer, 0 for none
    shared_limit: [max_layers]f32, // share_expert_swiglu_limit_list by layer, 0 for none
    mtp: u32,
    tie: bool,
    eos: [4]u32,
    eos_n: u32,
    bos: ?u32,
    quant: Quant,
    plain_text: [8192]u8, // modules_to_not_convert (FP8) or ignore / exclude_modules (ModelOpt), packed
    plain_ends: [512]u16, // plain_ends[i] ends entry i
    plain_n: u32,

    /// Layer i's mixer: the modeling's schedule for the backbone; the MTP layers are MLA (mtp_use_kda false).
    pub fn kind(c: *const Config, i: u32) Kind {
        if (i >= c.layers) return .mla;
        const full = c.layers / c.group * c.group;
        return if ((i + 1) % c.group == 0 or i >= full) .mla else .kda;
    }

    /// Layer i has routed experts (the MTP layers always do); the first dense_layers run a dense MLP.
    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }

    pub fn countKind(c: *const Config, k: Kind) u32 {
        var n: u32 = 0;
        for (0..c.layers) |i| n += @intFromBool(c.kind(@intCast(i)) == k);
        return n;
    }

    /// MLA's query and key head width: nope then rope.
    pub fn qkDim(c: *const Config) u32 {
        return c.nope + c.rope_dim;
    }

    /// MLA's softmax scale, qk_head_dim^-0.5 (no rope_scaling mscale: parse refuses rope_scaling).
    pub fn smScale(c: *const Config) f32 {
        return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.qkDim()))));
    }

    /// Group-limited routing: n_group groups, the best topk_group kept (parse holds those bounds).
    pub fn groups(c: *const Config) Groups {
        return .{ .n = c.n_group, .keep = c.topk_group };
    }

    /// Layer i's SwiGLU clamps, 0 for none. The lists cover the backbone; dense and MTP layers run unclamped.
    pub fn layerLimits(c: *const Config, i: u32) Limits {
        if (i >= c.layers or !c.isMoe(i)) return .{ .routed = 0, .shared = 0, .dense = 0 };
        return .{ .routed = c.expert_limit[i], .shared = c.shared_limit[i], .dense = 0 };
    }

    pub fn plain(c: *const Config, i: u32) []const u8 {
        const lo: usize = if (i == 0) 0 else c.plain_ends[i - 1];
        return c.plain_text[lo..c.plain_ends[i]];
    }

    /// A linear module the exception list leaves in bf16: named whole, ended after a dot, or matched by a `*` glob.
    pub fn unconverted(c: *const Config, module: []const u8) bool {
        for (0..c.plain_n) |i| {
            const p = c.plain(@intCast(i));
            if (std.mem.indexOfScalar(u8, p, '*') != null) {
                if (glob(p, module)) return true;
                continue;
            }
            if (std.mem.eql(u8, module, p)) return true;
            if (module.len > p.len and std.mem.endsWith(u8, module, p) and module[module.len - p.len - 1] == '.') return true;
        }
        return false;
    }

    /// The stored format of linear `module`: the checkpoint's quantization unless its exception list names it.
    pub fn format(c: *const Config, module: []const u8) Quant {
        return if (c.unconverted(module)) .bf16 else c.quant;
    }

    /// A linear module carries quantization tensors beside its weight.
    pub fn scaled(c: *const Config, module: []const u8) bool {
        return c.format(module) != .bf16;
    }
};

/// fnmatch with `*` only (ModelOpt's ignore patterns): `*` takes any run of characters, dots included.
pub fn glob(pattern: []const u8, name: []const u8) bool {
    const star = std.mem.indexOfScalar(u8, pattern, '*') orelse return std.mem.eql(u8, pattern, name);
    if (!std.mem.startsWith(u8, name, pattern[0..star])) return false;
    const rest = pattern[star + 1 ..];
    var at = star;
    while (at <= name.len) : (at += 1) if (glob(rest, name[at..])) return true;
    return false;
}

fn get(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn int(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try get(o, key, why);
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) {
        why.set("config.json's {s} is not a whole number", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn number(v: std.json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn float(o: std.json.ObjectMap, key: []const u8, why: *Why) !f64 {
    return number(try get(o, key, why)) orelse {
        why.set("config.json's {s} is not a number", .{key});
        return error.BadConfig;
    };
}

fn flag(o: std.json.ObjectMap, key: []const u8, default: bool, why: *Why) !bool {
    const v = o.get(key) orelse return default;
    if (v != .bool) {
        why.set("config.json's {s} is not true or false", .{key});
        return error.BadConfig;
    }
    return v.bool;
}

fn text(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// A string key that, when present and not null, must read `want`.
fn must(o: std.json.ObjectMap, key: []const u8, want: []const u8, err: anyerror, why: *Why) !void {
    const v = o.get(key) orelse return;
    if (v == .null) return;
    if (v == .string and std.mem.eql(u8, v.string, want)) return;
    why.set("{s} must be \"{s}\": the engine runs no other", .{ key, want });
    return err;
}

fn limits(o: std.json.ObjectMap, key: []const u8, layers: u32, out: *[max_layers]f32, why: *Why) !void {
    out.* = @splat(0);
    const v = o.get(key) orelse return; // absent: no clamp anywhere, as the reference reads it
    if (v == .null) return; // Ling-3.0-tiny writes null: the same, no clamp
    if (v != .array or v.array.items.len != layers) {
        why.set("{s} has {d} entries for {d} layers", .{ key, if (v == .array) v.array.items.len else 0, layers });
        return error.LimitList;
    }
    for (v.array.items, 0..) |e, i| {
        const x = if (e == .null) 0 else number(e) orelse -1;
        if (x < 0 or !std.math.isFinite(x)) {
            why.set("{s}[{d}] is not a limit (0 for none, else positive)", .{ key, i });
            return error.LimitList;
        }
        out[i] = @floatCast(x);
    }
}

/// An exception list's names, packed into the config (modules left in bf16).
fn exceptions(v: ?std.json.Value, c: *Config, why: *Why) !void {
    const list = v orelse return;
    if (list != .array) return;
    var at: usize = if (c.plain_n == 0) 0 else c.plain_ends[c.plain_n - 1];
    for (list.array.items) |m| {
        if (m != .string or c.plain_n == c.plain_ends.len or at + m.string.len > c.plain_text.len) {
            why.set("the quantization's exception list holds more than the loader keeps", .{});
            return error.Quant;
        }
        @memcpy(c.plain_text[at..][0..m.string.len], m.string);
        at += m.string.len;
        c.plain_ends[c.plain_n] = @intCast(at);
        c.plain_n += 1;
    }
}

fn fp8Block(q: std.json.ObjectMap, c: *Config, why: *Why) !void {
    const block = q.get("weight_block_size");
    const b128 = block != null and block.? == .array and block.?.array.items.len == 2 and
        block.?.array.items[0] == .integer and block.?.array.items[0].integer == 128 and
        block.?.array.items[1] == .integer and block.?.array.items[1].integer == 128;
    if (!b128 or !std.mem.eql(u8, text(q, "fmt") orelse "e4m3", "e4m3") or
        !std.mem.eql(u8, text(q, "activation_scheme") orelse "dynamic", "dynamic"))
    {
        why.set("quantization fp8: the loader reads FP8 e4m3 in 128x128 blocks with dynamic activations", .{});
        return error.Quant;
    }
    c.quant = .fp8_block128;
    try exceptions(q.get("modules_to_not_convert"), c, why);
}

fn isInt(v: std.json.Value, want: i64) bool {
    return v == .integer and v.integer == want;
}

fn groups16(why: *Why) error{Quant} {
    why.set("ModelOpt NVFP4 with weights other than 4-bit in groups of 16", .{});
    return error.Quant;
}

/// ModelOpt NVFP4 from config.json's quantization_config or hf_quant_config.json's "quantization" block.
fn modelopt(q: std.json.ObjectMap, c: *Config, why: *Why) !void {
    const algo = text(q, "quant_algo") orelse "";
    if (!std.ascii.eqlIgnoreCase(algo, "NVFP4")) {
        why.set("ModelOpt {s}: the loader reads ModelOpt NVFP4 (e2m1 in groups of 16)", .{if (algo.len > 0) algo else "without quant_algo"});
        return error.Quant;
    }
    if (q.get("group_size")) |v| if (!isInt(v, 16)) return groups16(why);
    if (q.get("config_groups")) |g| if (g == .object) {
        var it = g.object.iterator();
        while (it.next()) |e| {
            const w = if (e.value_ptr.* == .object) e.value_ptr.object.get("weights") else null;
            if (w == null or w.? != .object) return groups16(why);
            const bits = w.?.object.get("num_bits") orelse return groups16(why);
            const size = w.?.object.get("group_size") orelse return groups16(why);
            if (!isInt(bits, 4) or !isInt(size, 16)) return groups16(why);
        }
    };
    c.quant = .nvfp4;
    inline for (.{ "ignore", "exclude_modules", "modules_to_not_convert" }) |key| try exceptions(q.get(key), c, why);
}

fn quantization(o: std.json.ObjectMap, c: *Config, why: *Why) !void {
    c.quant = .bf16;
    c.plain_n = 0;
    const qv = o.get("quantization_config") orelse return;
    if (qv == .null) return;
    if (qv != .object) {
        why.set("quantization_config is not an object", .{});
        return error.Quant;
    }
    const q = qv.object;
    const method = text(q, "quant_method") orelse "";
    if (std.mem.eql(u8, method, "fp8")) return fp8Block(q, c, why);
    if (std.ascii.eqlIgnoreCase(method, "modelopt")) return modelopt(q, c, why);
    if (std.ascii.eqlIgnoreCase(method, "exl3")) {
        why.set("EXL3: Ling's EXL3 weights come with ashhart's EXL3 module (tracker #832), not this loader yet", .{});
        return error.Exl3;
    }
    why.set("quantization {s}: the loader reads bf16, FP8 128x128 blocks or ModelOpt NVFP4", .{if (method.len > 0) method else "without quant_method"});
    return error.Quant;
}

fn tokens(o: std.json.ObjectMap, c: *Config, why: *Why) !void {
    c.eos_n = 0;
    const v = try get(o, "eos_token_id", why);
    const items: []const std.json.Value = if (v == .array) v.array.items else &.{v};
    if (items.len == 0 or items.len > c.eos.len) {
        why.set("eos_token_id holds {d} ids, the engine stops on 1 to {d}", .{ items.len, c.eos.len });
        return error.BadConfig;
    }
    for (items) |e| {
        if (e != .integer or e.integer < 0 or e.integer >= c.vocab) {
            why.set("eos_token_id is not a token of the {d}-token vocabulary", .{c.vocab});
            return error.BadConfig;
        }
        c.eos[c.eos_n] = @intCast(e.integer);
        c.eos_n += 1;
    }
    c.bos = null;
    if (o.get("bos_token_id")) |b| if (b == .integer and b.integer >= 0 and b.integer < c.vocab) {
        c.bos = @intCast(b.integer);
    };
}

/// config.json as the reference (configuration_bailing_moe_v3.py, exllamav3's bailing_hybrid) reads it.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, why: *Why) !Config {
    return parseWith(gpa, json, null, why);
}

/// parse() with ModelOpt's hf_quant_config.json beside it: its NVFP4 block when config.json has no quantization_config.
pub fn parseWith(gpa: std.mem.Allocator, json: []const u8, hf_quant: ?[]const u8, why: *Why) !Config {
    var c = try parseConfig(gpa, json, why);
    const extra = hf_quant orelse return c;
    if (c.quant != .bf16 or c.plain_n != 0) return c;
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, extra, .{}) catch {
        why.set("hf_quant_config.json is not JSON", .{});
        return error.Quant;
    };
    defer parsed.deinit();
    const q = if (parsed.value == .object) parsed.value.object.get("quantization") else null;
    if (q == null or q.? != .object) {
        why.set("hf_quant_config.json has no quantization block", .{});
        return error.Quant;
    }
    try modelopt(q.?.object, &c, why);
    return c;
}

fn parseConfig(gpa: std.mem.Allocator, json: []const u8, why: *Why) !Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const o = parsed.value.object;
    const arch = o.get("architectures");
    const ling = arch != null and arch.? == .array and arch.?.array.items.len == 1 and arch.?.array.items[0] == .string and
        std.mem.eql(u8, arch.?.array.items[0].string, "BailingMoeV3ForCausalLM");
    if (!ling or !std.mem.eql(u8, text(o, "model_type") orelse "", "bailing_hybrid")) {
        why.set("not a Ling-3.1 checkpoint: architectures BailingMoeV3ForCausalLM, model_type bailing_hybrid", .{});
        return error.NotLing;
    }
    var c = std.mem.zeroes(Config);
    c.hidden = try int(o, "hidden_size", why);
    c.layers = try int(o, "num_hidden_layers", why);
    c.vocab = try int(o, "vocab_size", why);
    c.eps = @floatCast(try float(o, "rms_norm_eps", why));
    c.group = try int(o, "layer_group_size", why);
    c.dense_layers = try int(o, "first_k_dense_replace", why);
    c.dense_inter = try int(o, "intermediate_size", why);
    c.mtp = if (o.get("num_nextn_predict_layers")) |_| try int(o, "num_nextn_predict_layers", why) else 0;
    if (c.layers == 0 or c.group == 0 or c.dense_layers > c.layers or c.hidden == 0 or c.vocab == 0) {
        why.set("{d} layers in groups of {d}, {d} dense: not a layer schedule", .{ c.layers, c.group, c.dense_layers });
        return error.BadConfig;
    }
    if (c.layers + c.mtp > max_layers) {
        why.set("{d} layers and {d} MTP layers: the loader holds {d}", .{ c.layers, c.mtp, max_layers });
        return error.TooManyLayers;
    }
    if (c.mtp > 1 or try flag(o, "mtp_use_kda", false, why)) {
        why.set("MTP: the engine runs one MLA MTP layer", .{});
        return error.Mtp;
    }
    try must(o, "hidden_act", "silu", error.Activation, why);
    if (try flag(o, "use_qkv_bias", false, why) or try flag(o, "use_bias", false, why)) {
        why.set("use_qkv_bias/use_bias: the engine's projections have no bias", .{});
        return error.Bias;
    }
    // KDA: one head geometry for q, k and v, full-rank f and g, the safe gate's lower bound
    if (!try flag(o, "no_kda_lora", false, why) or try flag(o, "use_kda_lora", false, why)) {
        why.set("no_kda_lora must be true: the engine runs full-rank f_proj and g_proj", .{});
        return error.KdaLora;
    }
    c.kda_heads = try int(o, "num_attention_heads", why);
    c.kda_dim = try int(o, "head_dim", why);
    c.conv = try int(o, "short_conv_kernel_size", why);
    if (!try flag(o, "kda_safe_gate", false, why) or o.get("kda_lower_bound") == null) {
        why.set("kda_safe_gate with a kda_lower_bound: the engine's KDA gate is the safe one", .{});
        return error.KdaGate;
    }
    c.lower_bound = @floatCast(try float(o, "kda_lower_bound", why));
    // MLA
    c.mla_heads = c.kda_heads;
    const ql = o.get("q_lora_rank") orelse std.json.Value.null;
    c.q_lora = if (ql == .null) 0 else try int(o, "q_lora_rank", why);
    c.kv_lora = try int(o, "kv_lora_rank", why);
    c.nope = try int(o, "qk_nope_head_dim", why);
    c.rope_dim = try int(o, "qk_rope_head_dim", why);
    c.v_dim = try int(o, "v_head_dim", why);
    c.rope_theta = try float(o, "rope_theta", why);
    c.rope_style = if (try flag(o, "rope_interleave", true, why)) .gptj else .neox;
    if (o.get("rope_scaling")) |r| if (r != .null) {
        why.set("rope_scaling: the engine's MLA has plain rope only", .{});
        return error.RopeScaling;
    };
    const gate = o.get("gated_attention_proj_granularity_type") orelse std.json.Value.null;
    c.head_gate = gate == .string and std.mem.eql(u8, gate.string, "head_wise");
    if (gate != .null and !c.head_gate) {
        why.set("gated_attention_proj_granularity_type {s}: the engine gates MLA per head or not at all", .{if (gate == .string) gate.string else "?"});
        return error.GateGranularity;
    }
    // MoE: DeepSeek-V3's sigmoid router with group-limited top-k and a bias for the pick
    try must(o, "score_function", "sigmoid", error.ScoreFunction, why);
    try must(o, "scoring_func", "sigmoid", error.ScoreFunction, why);
    try must(o, "topk_method", "noaux_tc", error.Routing, why);
    c.experts = try int(o, "num_experts", why);
    c.topk = try int(o, "num_experts_per_tok", why);
    c.n_group = if (o.get("n_group")) |_| try int(o, "n_group", why) else 1;
    c.topk_group = if (o.get("topk_group")) |_| try int(o, "topk_group", why) else 1;
    c.routed_scale = @floatCast(if (o.get("routed_scaling_factor")) |_| try float(o, "routed_scaling_factor", why) else 1.0);
    c.norm_topk = try flag(o, "norm_topk_prob", true, why);
    if (!c.norm_topk) {
        why.set("norm_topk_prob must be true: the routed weights are renormalized over the top-k", .{});
        return error.Routing;
    }
    c.expert_bias = try flag(o, "moe_router_enable_expert_bias", true, why);
    // glue.select's bounds: at most 512 experts, top-k under 32, up to 32 equal groups that hold the top-k
    const g_ok = c.n_group >= 1 and c.n_group <= 32 and c.experts % c.n_group == 0 and c.topk_group >= 1 and
        c.topk_group <= c.n_group and c.topk <= c.topk_group * (c.experts / c.n_group);
    if (c.experts > 512 or c.topk == 0 or c.topk >= 32 or c.topk > c.experts or !g_ok) {
        why.set("routing {d} of {d} experts in {d} groups keeping {d}: past glue.select's bounds", .{ c.topk, c.experts, c.n_group, c.topk_group });
        return error.Routing;
    }
    // HF's rule: a group scores its top two (topk(2) fails on a group of one); one pick keeps its weight unnormalized
    if (c.experts / c.n_group < 2) {
        why.set("{d} groups of {d} experts: HF scores a group by its best two, so groups hold two or more", .{ c.n_group, c.experts / c.n_group });
        return error.Routing;
    }
    if (c.topk == 1) {
        why.set("num_experts_per_tok 1: HF leaves one pick's weight unnormalized, the kernel normalizes it", .{});
        return error.Routing;
    }
    c.moe_inter = try int(o, "moe_intermediate_size", why);
    const shared_n = if (o.get("num_shared_experts")) |v| (if (v == .null) 0 else try int(o, "num_shared_experts", why)) else 0;
    const shared_w = if (o.get("moe_shared_expert_intermediate_size")) |_| try int(o, "moe_shared_expert_intermediate_size", why) else c.moe_inter;
    c.shared_inter = shared_n * shared_w;
    try limits(o, "expert_swiglu_limit_list", c.layers, &c.expert_limit, why);
    try limits(o, "share_expert_swiglu_limit_list", c.layers, &c.shared_limit, why);
    for (0..c.dense_layers) |i| if (c.expert_limit[i] != 0 or c.shared_limit[i] != 0) {
        why.set("layer {d} is dense with a SwiGLU limit: neither list says which a dense MLP takes", .{i});
        return error.LimitList;
    };
    c.tie = try flag(o, "tie_word_embeddings", false, why);
    try tokens(o, &c, why);
    try geometry(o, &c, why);
    try quantization(o, &c, why);
    return c;
}

/// The widths the kernels are built for (cuda.zig's tests hold them to glm_kda, glm_glue and the latent cache).
pub const fixed = .{ .kda_dim = 128, .conv = 4, .kv_lora = 512, .nope = 128, .v_dim = 128, .rope = 64, .block = 128 };
/// glue.rmsnorm's instances for q_lora_rank (256 is Ling-3.0-tiny's); the hidden width takes the router's three.
pub const q_lora_widths = [_]u32{ 256, 512, 1536, 4096 };
/// The hidden width: the norms' 512, 1536 and 4096, which the router's slices (8 x 64 inputs) also divide.
pub const norm_widths = [_]u32{ 512, 1536, 4096 };

fn normWidth(w: u32) bool {
    return std.mem.indexOfScalar(u32, &norm_widths, w) != null;
}

/// Every shape the kernels take fixed, refused at load before anything is allocated or launched.
fn geometry(o: std.json.ObjectMap, c: *const Config, why: *Why) !void {
    // KDA: glm_kda.cu's 128-wide heads and 4 taps (a 3-row conv window); one key and value head per query head
    if (c.kda_dim != fixed.kda_dim) {
        why.set("head_dim {d}: the KDA kernels take {d}-wide heads", .{ c.kda_dim, fixed.kda_dim });
        return error.KdaGeometry;
    }
    if (c.conv != fixed.conv) {
        why.set("short_conv_kernel_size {d}: the KDA kernels take {d} taps", .{ c.conv, fixed.conv });
        return error.KdaGeometry;
    }
    const kv_heads = if (o.get("num_key_value_heads")) |_| try int(o, "num_key_value_heads", why) else c.kda_heads;
    if (c.kda_heads == 0 or kv_heads != c.kda_heads) {
        why.set("{d} attention heads, {d} key/value heads: the engine runs one key/value head per query head", .{ c.kda_heads, kv_heads });
        return error.KdaGeometry;
    }
    // MLA: the 512-wide latent, 128-wide nope and value heads, 64 GPT-J roped columns
    const mla = [_]struct { []const u8, u32, u32 }{
        .{ "kv_lora_rank", c.kv_lora, fixed.kv_lora },
        .{ "qk_nope_head_dim", c.nope, fixed.nope },
        .{ "v_head_dim", c.v_dim, fixed.v_dim },
        .{ "qk_rope_head_dim", c.rope_dim, fixed.rope },
    };
    for (mla) |m| if (m[1] != m[2]) {
        why.set("{s} {d}: the MLA kernels take {d}", .{ m[0], m[1], m[2] });
        return error.MlaGeometry;
    };
    if (c.rope_style != .gptj) {
        why.set("rope_interleave false: the MLA rope kernel pairs adjacent columns (GPT-J) only", .{});
        return error.MlaGeometry;
    }
    if (c.q_lora != 0 and std.mem.indexOfScalar(u32, &q_lora_widths, c.q_lora) == null) {
        why.set("q_lora_rank {d}: q_a_layernorm runs at 256, 512, 1536 or 4096", .{c.q_lora});
        return error.MlaGeometry;
    }
    // widths: the norms' instances, and every linear's inputs in whole 128-column blocks (FP8's, the lanes' 64)
    if (!normWidth(c.hidden)) {
        why.set("hidden_size {d}: the norms and the router run at 512, 1536 or 4096", .{c.hidden});
        return error.Width;
    }
    const widths = [_]struct { []const u8, u32, bool }{
        .{ "intermediate_size", c.dense_inter, c.dense_layers > 0 },
        .{ "moe_intermediate_size", c.moe_inter, true },
        .{ "the shared expert's width", c.shared_inter, c.shared_inter > 0 },
    };
    for (widths) |x| if (x[2] and (x[1] == 0 or x[1] % fixed.block != 0)) {
        why.set("{s} {d}: the projections and experts take whole blocks of {d}", .{ x[0], x[1], fixed.block });
        return error.Width;
    };
    if (c.tie) {
        why.set("tie_word_embeddings: the engine reads its own lm_head", .{});
        return error.TiedHead;
    }
}

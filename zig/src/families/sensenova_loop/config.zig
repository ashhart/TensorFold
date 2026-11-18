//! SenseNova-Loop config parsing, validation, 56-step execution plan and skipped-tensor policy.
const std = @import("std");

pub const max_layers: usize = 128;
pub const max_steps: usize = 128;
pub const max_eos: usize = 8;

pub const MhcRole = enum { none, enter, inner, exit };
pub const LoopSettings = struct { start: u8, end: u8, eval_count: u32 };
pub const MhcConfig = struct { start: u8, end: u8, streams: u8, iterations: u32, pre_scale: f32 };
pub const ExecutionStep = struct { exec: u8, phys: u8, state: u8, is_linear: bool, mhc_role: MhcRole };

pub const RopeConfig = struct {
    rope_theta: f64,
    partial_rotary_factor: f64,
    rotary_dim: u32,
    yarn_factor: f64,
    original_max_position_embeddings: u32,
    ceiling: u32,
    mrope_interleaved: bool,
    mrope_section: [3]u32,
};

pub const ExecutionPlan = struct {
    steps: []const ExecutionStep,
    linear_states: u8,
    full_states: u8,
    loops: u32,

    pub fn deinit(self: ExecutionPlan, gpa: std.mem.Allocator) void {
        gpa.free(self.steps);
    }
};

pub const Error = error{
    InvalidConfig,
    UnsupportedModel,
    MissingField,
    BadLoopSettings,
    BadMhcConfig,
    MhcCrossesLoopBoundary,
    BadLayerTypes,
    MissingFullAttention,
    BadRopeConfig,
    MaxPositionExceedsCeiling,
    BadTiedHead,
    BadEos,
    MissingEos,
    TooManyEos,
    InvalidLoopCount,
    TooManyExecutionSteps,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

pub const Config = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u8,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    vocab_size: u32,
    rms_norm_eps: f32,
    max_position_embeddings: u32,
    tie_word_embeddings: bool,
    mtp_num_hidden_layers: u32,
    loop: LoopSettings,
    mhc: ?MhcConfig,
    rope: RopeConfig,
    is_linear: [max_layers]bool,
    eos_buf: [max_eos]u32,
    eos_len: u8,

    pub fn eos(self: *const Config) []const u32 {
        return self.eos_buf[0..self.eos_len];
    }

    pub fn isLinear(self: *const Config, phys: usize) bool {
        return self.is_linear[phys];
    }

    pub fn mhcRole(self: *const Config, phys: u8) MhcRole {
        const m = self.mhc orelse return .none;
        if (phys < m.start or phys >= m.end) return .none;
        if (phys == m.start) return .enter;
        if (phys == m.end - 1) return .exit;
        return .inner;
    }

    /// Fill `buf` with the execution plan for `loops_override` (defaults to `loop.eval_count`).
    pub fn fillPlan(self: *const Config, buf: *[max_steps]ExecutionStep, loops_override: ?u32) Error![]const ExecutionStep {
        const loops = loops_override orelse self.loop.eval_count;
        if (loops == 0) return error.InvalidLoopCount;
        const body_len: usize = self.loop.end - self.loop.start;
        const loop_steps = std.math.mul(usize, body_len, loops) catch return error.TooManyExecutionSteps;
        const total = @as(usize, self.loop.start) + loop_steps + (self.num_hidden_layers - self.loop.end);
        if (total > max_steps) return error.TooManyExecutionSteps;

        var exec: usize = 0;
        var lin: u8 = 0;
        var full: u8 = 0;
        for (0..self.loop.start) |p| appendStep(self, buf, &exec, &lin, &full, @intCast(p));
        for (0..loops) |_| {
            for (self.loop.start..self.loop.end) |p| appendStep(self, buf, &exec, &lin, &full, @intCast(p));
        }
        for (self.loop.end..self.num_hidden_layers) |p| appendStep(self, buf, &exec, &lin, &full, @intCast(p));
        return buf[0..exec];
    }

    pub fn executionPlan(self: *const Config, gpa: std.mem.Allocator, loops_override: ?u32) Error!ExecutionPlan {
        var tmp: [max_steps]ExecutionStep = undefined;
        const view = try self.fillPlan(&tmp, loops_override);
        const steps = try gpa.dupe(ExecutionStep, view);
        var lin: u8 = 0;
        var full: u8 = 0;
        for (steps) |s| {
            if (s.is_linear) lin += 1 else full += 1;
        }
        return .{ .steps = steps, .linear_states = lin, .full_states = full, .loops = loops_override orelse self.loop.eval_count };
    }

    pub fn parse(gpa: std.mem.Allocator, config_bytes: []const u8) Error!Config {
        return parseWithGeneration(gpa, config_bytes, null);
    }

    pub fn parseWithGeneration(gpa: std.mem.Allocator, config_bytes: []const u8, gen_bytes: ?[]const u8) Error!Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, config_bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.InvalidConfig;
        const root = doc.value.object;
        const mt = root.get("model_type") orelse return error.UnsupportedModel;
        if (mt != .string or !std.mem.eql(u8, mt.string, "sensenova_loop")) return error.UnsupportedModel;

        const tower = try textTower(root);
        if (tower.get("model_type")) |tmt| {
            if (tmt != .string) return error.UnsupportedModel;
            const ok = std.mem.eql(u8, tmt.string, "qwen3_5_text") or
                std.mem.eql(u8, tmt.string, "sensenova_loop_text") or
                std.mem.eql(u8, tmt.string, "sensenova_loop");
            if (!ok) return error.UnsupportedModel;
        }

        var gen_doc: ?std.json.Parsed(std.json.Value) = null;
        defer if (gen_doc) |*g| g.deinit();
        var gen_obj: ?std.json.ObjectMap = null;
        if (gen_bytes) |gb| {
            gen_doc = try std.json.parseFromSlice(std.json.Value, gpa, gb, .{});
            if (gen_doc.?.value != .object) return error.InvalidConfig;
            gen_obj = gen_doc.?.value.object;
        }
        return parseObjects(root, tower, gen_obj);
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(cfg_path);
        const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 22));
        defer gpa.free(cfg_bytes);

        const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gen_path);
        const gen_bytes = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        defer if (gen_bytes) |gb| gpa.free(gb);
        return parseWithGeneration(gpa, cfg_bytes, gen_bytes);
    }
};

/// Checkpoint tensors omitted by TensorFold's loader (unused gating logits).
pub fn isSkippedTensor(name: []const u8) bool {
    return std.mem.eql(u8, name, "model.language_model.loop_gate.logits") or
        std.mem.eql(u8, name, "language_model.loop_gate.logits") or
        std.mem.eql(u8, name, "loop_gate.logits");
}

fn appendStep(cfg: *const Config, buf: *[max_steps]ExecutionStep, exec: *usize, lin: *u8, full: *u8, phys: u8) void {
    const is_lin = cfg.is_linear[phys];
    const state = if (is_lin) lin.* else full.*;
    if (is_lin) lin.* += 1 else full.* += 1;
    buf[exec.*] = .{ .exec = @intCast(exec.*), .phys = phys, .state = state, .is_linear = is_lin, .mhc_role = cfg.mhcRole(phys) };
    exec.* += 1;
}

fn textTower(root: std.json.ObjectMap) Error!std.json.ObjectMap {
    const v = root.get("text_config") orelse return root;
    return switch (v) {
        .null => root,
        .object => |o| o,
        else => error.InvalidConfig,
    };
}

fn parseObjects(root: std.json.ObjectMap, tower: std.json.ObjectMap, gen: ?std.json.ObjectMap) Error!Config {
    const hidden = try reqPosU32(tower, root, "hidden_size");
    const intermediate = try reqPosU32(tower, root, "intermediate_size");
    const n_layers_u = try reqPosU32(tower, root, "num_hidden_layers");
    if (n_layers_u > max_layers) return error.InvalidConfig;
    const n_layers: u8 = @intCast(n_layers_u);
    const heads = try reqPosU32(tower, root, "num_attention_heads");
    const kv_heads = try reqPosU32(tower, root, "num_key_value_heads");
    if (heads % kv_heads != 0) return error.InvalidConfig;
    const head_dim = if (optField(tower, root, "head_dim")) |v| try posU32Val(v) else hidden / heads;
    if (head_dim == 0) return error.InvalidConfig;
    const lin_k_heads = try reqPosU32(tower, root, "linear_num_key_heads");
    const lin_v_heads = try reqPosU32(tower, root, "linear_num_value_heads");
    if (lin_v_heads % lin_k_heads != 0) return error.InvalidConfig;
    const eps_f64 = if (optField(tower, root, "rms_norm_eps")) |v| try finiteNum(v) else 1e-6;
    if (eps_f64 <= 0) return error.InvalidConfig;
    const max_pos = try reqPosU32(tower, root, "max_position_embeddings");

    try checkModelFlags(tower);
    var is_linear: [max_layers]bool = @splat(false);
    try parseLayerTypes(tower, n_layers, &is_linear);
    const loop = try parseLoop(root, tower, n_layers);
    const mhc = try parseMhc(root, tower, n_layers, loop);
    const rope = try parseRope(tower, root, head_dim, max_pos);

    var cfg = Config{
        .hidden_size = hidden,
        .intermediate_size = intermediate,
        .num_hidden_layers = n_layers,
        .num_attention_heads = heads,
        .num_key_value_heads = kv_heads,
        .head_dim = head_dim,
        .linear_num_key_heads = lin_k_heads,
        .linear_num_value_heads = lin_v_heads,
        .linear_key_head_dim = try reqPosU32(tower, root, "linear_key_head_dim"),
        .linear_value_head_dim = try reqPosU32(tower, root, "linear_value_head_dim"),
        .linear_conv_kernel_dim = try reqPosU32(tower, root, "linear_conv_kernel_dim"),
        .vocab_size = try reqPosU32(tower, root, "vocab_size"),
        .rms_norm_eps = @floatCast(eps_f64),
        .max_position_embeddings = max_pos,
        .tie_word_embeddings = try parseTied(root, tower),
        .mtp_num_hidden_layers = if (optField(tower, root, "mtp_num_hidden_layers")) |v| try nonNegU32Val(v) else 0,
        .loop = loop,
        .mhc = mhc,
        .rope = rope,
        .is_linear = is_linear,
        .eos_buf = @splat(0),
        .eos_len = 0,
    };
    try collectAllEos(&cfg, root, tower, gen);
    return cfg;
}

fn checkModelFlags(tower: std.json.ObjectMap) Error!void {
    if (tower.get("hidden_act")) |v| if (v != .string or !std.mem.eql(u8, v.string, "silu")) return error.InvalidConfig;
    if (tower.get("attn_output_gate")) |v| if (v != .bool or !v.bool) return error.InvalidConfig;
    if (tower.get("attention_bias")) |v| if (v != .bool or v.bool) return error.InvalidConfig;
    if (tower.get("mamba_ssm_dtype")) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "float32")) return error.InvalidConfig;
    }
}

fn parseTied(root: std.json.ObjectMap, tower: std.json.ObjectMap) Error!bool {
    const inner = try optBool(tower.get("tie_word_embeddings"));
    const outer = try optBool(root.get("tie_word_embeddings"));
    if (inner != null and outer != null and inner.? != outer.?) return error.BadTiedHead;
    return inner orelse outer orelse false;
}

fn optBool(v: ?std.json.Value) Error!?bool {
    const x = v orelse return null;
    return if (x == .bool) x.bool else error.BadTiedHead;
}

fn parseLayerTypes(tower: std.json.ObjectMap, n_layers: u8, out: *[max_layers]bool) Error!void {
    const interval = try parseInterval(tower.get("full_attention_interval"));
    const list_v = tower.get("layer_types") orelse .null;
    var has_full = false;
    if (list_v == .null) {
        const iv = interval orelse 4;
        for (0..n_layers) |i| {
            const is_full = (i + 1) % iv == 0;
            out[i] = !is_full;
            if (is_full) has_full = true;
        }
        if (!has_full) return error.MissingFullAttention;
        return;
    }
    if (list_v != .array or list_v.array.items.len != n_layers) return error.BadLayerTypes;
    for (list_v.array.items, 0..) |item, i| {
        if (item != .string) return error.BadLayerTypes;
        if (std.mem.eql(u8, item.string, "linear_attention")) {
            out[i] = true;
        } else if (std.mem.eql(u8, item.string, "full_attention")) {
            out[i] = false;
            has_full = true;
        } else return error.BadLayerTypes;
        if (interval) |iv| if (out[i] == ((i + 1) % iv == 0)) return error.BadLayerTypes;
    }
    if (!has_full) return error.MissingFullAttention;
}

fn parseInterval(v: ?std.json.Value) Error!?usize {
    const x = v orelse return null;
    return switch (x) {
        .null => null,
        .bool => |b| if (b) error.BadLayerTypes else null,
        .integer => |i| if (i == 0) null else if (i > 0) @as(usize, @intCast(i)) else error.BadLayerTypes,
        else => error.BadLayerTypes,
    };
}

fn parseLoop(root: std.json.ObjectMap, tower: std.json.ObjectMap, n_layers: u8) Error!LoopSettings {
    const lv = optField(root, tower, "loop_settings") orelse return error.BadLoopSettings;
    if (lv != .object) return error.BadLoopSettings;
    const o = lv.object;
    if (o.get("use_gate")) |ug| if (ug != .bool or ug.bool) return error.BadLoopSettings;
    const sv = o.get("start") orelse return error.BadLoopSettings;
    const ev = o.get("end") orelse return error.BadLoopSettings;
    const cv = o.get("eval_count") orelse return error.BadLoopSettings;
    const csv = o.get("counts") orelse return error.BadLoopSettings;
    if (sv != .integer or ev != .integer or cv != .integer) return error.BadLoopSettings;
    if (sv.integer < 0 or ev.integer <= sv.integer or ev.integer > n_layers or cv.integer < 1) {
        return error.BadLoopSettings;
    }
    if (csv != .array or csv.array.items.len != 1) return error.BadLoopSettings;
    const c0 = csv.array.items[0];
    if (c0 != .integer or c0.integer != cv.integer) return error.BadLoopSettings;
    const eval_count = std.math.cast(u32, cv.integer) orelse return error.BadLoopSettings;
    return .{ .start = @intCast(sv.integer), .end = @intCast(ev.integer), .eval_count = eval_count };
}

fn parseMhc(root: std.json.ObjectMap, tower: std.json.ObjectMap, n_layers: u8, loop: LoopSettings) Error!?MhcConfig {
    const mv = optField(root, tower, "mhc") orelse return null;
    if (mv == .null) return null;
    if (mv != .object) return error.BadMhcConfig;
    const o = mv.object;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const known = std.mem.eql(u8, k, "start") or std.mem.eql(u8, k, "end") or
            std.mem.eql(u8, k, "streams") or std.mem.eql(u8, k, "iterations") or
            std.mem.eql(u8, k, "fixed_identity") or std.mem.eql(u8, k, "pre_scale");
        if (!known) return error.BadMhcConfig;
    }
    const sv = o.get("start") orelse return error.BadMhcConfig;
    const ev = o.get("end") orelse return error.BadMhcConfig;
    const stv = o.get("streams") orelse return error.BadMhcConfig;
    const itv = o.get("iterations") orelse return error.BadMhcConfig;
    const fiv = o.get("fixed_identity") orelse return error.BadMhcConfig;
    if (sv != .integer or ev != .integer or stv != .integer or itv != .integer) return error.BadMhcConfig;
    if (fiv != .bool or fiv.bool) return error.BadMhcConfig;
    if (sv.integer < 0 or ev.integer <= sv.integer or ev.integer > n_layers) return error.BadMhcConfig;
    if (stv.integer != 2 and stv.integer != 4) return error.BadMhcConfig;
    if (itv.integer < 1) return error.BadMhcConfig;
    const pre_scale_f64 = if (o.get("pre_scale")) |pv| finiteNum(pv) catch return error.BadMhcConfig else 1.0;
    if (pre_scale_f64 <= 0 or pre_scale_f64 > std.math.floatMax(f32)) return error.BadMhcConfig;

    const ms: u8 = @intCast(sv.integer);
    const me: u8 = @intCast(ev.integer);
    const valid_phase = (0 <= ms and me <= loop.start) or (loop.start <= ms and me <= loop.end) or
        (loop.end <= ms and me <= n_layers);
    if (!valid_phase) return error.MhcCrossesLoopBoundary;
    return .{
        .start = ms,
        .end = me,
        .streams = @intCast(stv.integer),
        .iterations = std.math.cast(u32, itv.integer) orelse return error.BadMhcConfig,
        .pre_scale = @floatCast(pre_scale_f64),
    };
}

fn parseRope(tower: std.json.ObjectMap, root: std.json.ObjectMap, head_dim: u32, max_pos: u32) Error!RopeConfig {
    const rv = optField(tower, root, "rope_parameters") orelse optField(tower, root, "rope_scaling") orelse
        return error.BadRopeConfig;
    if (rv != .object) return error.BadRopeConfig;
    const rope = rv.object;
    const rt = rope.get("rope_type") orelse rope.get("type") orelse return error.BadRopeConfig;
    if (rt != .string or !std.mem.eql(u8, rt.string, "yarn")) return error.BadRopeConfig;

    const factor = finiteNum(rope.get("factor") orelse return error.BadRopeConfig) catch return error.BadRopeConfig;
    if (factor < 1.0) return error.BadRopeConfig;
    const orig = posU32Val(rope.get("original_max_position_embeddings") orelse return error.BadRopeConfig) catch
        return error.BadRopeConfig;
    const ceil_f = @as(f64, @floatFromInt(orig)) * factor;
    if (!std.math.isFinite(ceil_f) or ceil_f > @as(f64, std.math.maxInt(u32))) return error.BadRopeConfig;
    const ceiling: u32 = @intFromFloat(@round(ceil_f));
    if (max_pos > ceiling) return error.MaxPositionExceedsCeiling;

    const theta = finiteNum(rope.get("rope_theta") orelse optField(tower, root, "rope_theta") orelse
        return error.BadRopeConfig) catch return error.BadRopeConfig;
    if (theta <= 0) return error.BadRopeConfig;
    const prf = finiteNum(rope.get("partial_rotary_factor") orelse optField(tower, root, "partial_rotary_factor") orelse
        return error.BadRopeConfig) catch return error.BadRopeConfig;
    if (prf <= 0 or prf > 1.0) return error.BadRopeConfig;
    const rotary_dim: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(head_dim)) * prf));
    if (rotary_dim == 0 or rotary_dim > head_dim or rotary_dim % 2 != 0) return error.BadRopeConfig;

    const mi = rope.get("mrope_interleaved") orelse return error.BadRopeConfig;
    if (mi != .bool or !mi.bool) return error.BadRopeConfig;
    const ms = rope.get("mrope_section") orelse return error.BadRopeConfig;
    if (ms != .array or ms.array.items.len != 3) return error.BadRopeConfig;
    var sec: [3]u32 = undefined;
    var sum: u32 = 0;
    for (ms.array.items, &sec) |item, *dst| {
        dst.* = nonNegU32Val(item) catch return error.BadRopeConfig;
        sum += dst.*;
    }
    if (sum * 2 != rotary_dim) return error.BadRopeConfig;
    return .{
        .rope_theta = theta,
        .partial_rotary_factor = prf,
        .rotary_dim = rotary_dim,
        .yarn_factor = factor,
        .original_max_position_embeddings = orig,
        .ceiling = ceiling,
        .mrope_interleaved = true,
        .mrope_section = sec,
    };
}

fn collectAllEos(cfg: *Config, root: std.json.ObjectMap, tower: std.json.ObjectMap, gen: ?std.json.ObjectMap) Error!void {
    const holders = [_]?std.json.Value{
        tower.get("eos_token_id"),
        root.get("eos_token_id"),
        if (gen) |g| g.get("eos_token_id") else null,
    };
    for (holders) |h| if (h) |v| try collectEos(cfg, v);
    if (cfg.eos_len == 0) return error.MissingEos;
}

fn collectEos(cfg: *Config, v: std.json.Value) Error!void {
    switch (v) {
        .null => {},
        .array => |arr| for (arr.items) |item| try collectEos(cfg, item),
        .integer => |i| {
            if (i < 0 or i >= cfg.vocab_size) return error.BadEos;
            const id: u32 = @intCast(i);
            if (std.mem.indexOfScalar(u32, cfg.eos(), id) != null) return;
            if (cfg.eos_len == max_eos) return error.TooManyEos;
            cfg.eos_buf[cfg.eos_len] = id;
            cfg.eos_len += 1;
        },
        else => return error.BadEos,
    }
}

fn optField(a: std.json.ObjectMap, b: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    return a.get(key) orelse b.get(key);
}

fn reqPosU32(a: std.json.ObjectMap, b: std.json.ObjectMap, key: []const u8) Error!u32 {
    return posU32Val(optField(a, b, key) orelse return error.MissingField);
}

fn posU32Val(v: std.json.Value) Error!u32 {
    if (v != .integer or v.integer <= 0) return error.InvalidConfig;
    return std.math.cast(u32, v.integer) orelse error.InvalidConfig;
}

fn nonNegU32Val(v: std.json.Value) Error!u32 {
    if (v != .integer or v.integer < 0) return error.InvalidConfig;
    return std.math.cast(u32, v.integer) orelse error.InvalidConfig;
}

fn finiteNum(v: std.json.Value) Error!f64 {
    const f: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| x,
        else => return error.InvalidConfig,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfig;
    return f;
}

const sample_layers =
    "[\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"]";

const sample_cfg =
    "{\"model_type\":\"sensenova_loop\",\"eos_token_id\":248046,\"tie_word_embeddings\":false," ++
    "\"loop_settings\":{\"start\":4,\"end\":28,\"eval_count\":2,\"counts\":[2]}," ++
    "\"mhc\":{\"start\":4,\"end\":28,\"streams\":2,\"iterations\":20,\"fixed_identity\":false,\"pre_scale\":1.0}," ++
    "\"text_config\":{\"model_type\":\"qwen3_5_text\",\"hidden_size\":4096,\"intermediate_size\":12288," ++
    "\"num_hidden_layers\":32,\"num_attention_heads\":16,\"num_key_value_heads\":4,\"head_dim\":256," ++
    "\"linear_num_key_heads\":16,\"linear_num_value_heads\":32,\"linear_key_head_dim\":128," ++
    "\"linear_value_head_dim\":128,\"linear_conv_kernel_dim\":4,\"vocab_size\":248320,\"rms_norm_eps\":1e-6," ++
    "\"max_position_embeddings\":524288,\"eos_token_id\":248044,\"mtp_num_hidden_layers\":0," ++
    "\"hidden_act\":\"silu\",\"attn_output_gate\":true,\"attention_bias\":false,\"mamba_ssm_dtype\":\"float32\"," ++
    "\"layer_types\":" ++ sample_layers ++ "," ++
    "\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":262144," ++
    "\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]}}}";

test "sensenova_loop config: parses 9B release config, yarn ceiling and 56-step plan" {
    const a = std.testing.allocator;
    const cfg = try Config.parseWithGeneration(a, sample_cfg, "{\"eos_token_id\":[248044,248046]}");
    try std.testing.expectEqual(@as(u32, 4096), cfg.hidden_size);
    try std.testing.expectEqual(@as(u8, 32), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 524288), cfg.rope.ceiling);
    try std.testing.expectEqual(@as(u32, 64), cfg.rope.rotary_dim);
    try std.testing.expectEqualSlices(u32, &.{ 248044, 248046 }, cfg.eos());
    try std.testing.expectEqual(@as(u32, 0), cfg.mtp_num_hidden_layers);
    try std.testing.expect(!cfg.tie_word_embeddings);

    const plan = try cfg.executionPlan(a, null);
    defer plan.deinit(a);
    try std.testing.expectEqual(@as(usize, 56), plan.steps.len);
    try std.testing.expectEqual(@as(u8, 42), plan.linear_states);
    try std.testing.expectEqual(@as(u8, 14), plan.full_states);
    try std.testing.expectEqual(MhcRole.none, plan.steps[3].mhc_role);
    try std.testing.expectEqual(MhcRole.enter, plan.steps[4].mhc_role);
    try std.testing.expectEqual(@as(u8, 4), plan.steps[4].phys);
    try std.testing.expectEqual(MhcRole.inner, plan.steps[5].mhc_role);
    try std.testing.expectEqual(MhcRole.exit, plan.steps[27].mhc_role);
    try std.testing.expectEqual(MhcRole.enter, plan.steps[28].mhc_role);
    try std.testing.expectEqual(@as(u8, 4), plan.steps[28].phys);
    try std.testing.expectEqual(MhcRole.exit, plan.steps[51].mhc_role);
    try std.testing.expectEqual(@as(u8, 27), plan.steps[51].phys);
    try std.testing.expectEqual(MhcRole.none, plan.steps[52].mhc_role);
    try std.testing.expect(!plan.steps[55].is_linear);
    try std.testing.expectEqual(@as(u8, 13), plan.steps[55].state);
}

test "sensenova_loop config: --loops override, boundary refusals, top-level fallback and skipped tensors" {
    const a = std.testing.allocator;
    const cfg = try Config.parse(a, sample_cfg);
    const p1 = try cfg.executionPlan(a, 1);
    defer p1.deinit(a);
    try std.testing.expectEqual(@as(usize, 32), p1.steps.len);
    const p4 = try cfg.executionPlan(a, 4);
    defer p4.deinit(a);
    try std.testing.expectEqual(@as(usize, 104), p4.steps.len);
    try std.testing.expectError(error.InvalidLoopCount, cfg.executionPlan(a, 0));
    try std.testing.expectError(error.TooManyExecutionSteps, cfg.executionPlan(a, 6));

    const cross = try std.mem.replaceOwned(u8, a, sample_cfg, "\"mhc\":{\"start\":4,\"end\":28", "\"mhc\":{\"start\":3,\"end\":28");
    defer a.free(cross);
    try std.testing.expectError(error.MhcCrossesLoopBoundary, Config.parse(a, cross));

    const gate_on = try std.mem.replaceOwned(u8, a, sample_cfg, "\"counts\":[2]", "\"counts\":[2],\"use_gate\":true");
    defer a.free(gate_on);
    try std.testing.expectError(error.BadLoopSettings, Config.parse(a, gate_on));

    const bad_ceil = try std.mem.replaceOwned(u8, a, sample_cfg, "524288", "524289");
    defer a.free(bad_ceil);
    try std.testing.expectError(error.MaxPositionExceedsCeiling, Config.parse(a, bad_ceil));

    const flat =
        \\{"model_type":"sensenova_loop","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":8,
        \\ "num_attention_heads":8,"num_key_value_heads":2,"head_dim":128,"linear_num_key_heads":8,
        \\ "linear_num_value_heads":16,"linear_key_head_dim":64,"linear_value_head_dim":64,"linear_conv_kernel_dim":4,
        \\ "vocab_size":32000,"max_position_embeddings":8192,"eos_token_id":[1,2],"mtp_num_hidden_layers":1,
        \\ "loop_settings":{"start":2,"end":6,"eval_count":2,"counts":[2],"use_gate":false},
        \\ "mhc":{"start":2,"end":6,"streams":2,"iterations":20,"fixed_identity":false,"pre_scale":0.5},
        \\ "rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":4096,
        \\  "rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[5,5,6]}}
    ;
    const fc = try Config.parse(a, flat);
    try std.testing.expectEqual(@as(u32, 1), fc.mtp_num_hidden_layers);
    try std.testing.expectEqual(@as(f32, 0.5), fc.mhc.?.pre_scale);
    try std.testing.expect(isSkippedTensor("model.language_model.loop_gate.logits"));
    try std.testing.expect(isSkippedTensor("loop_gate.logits"));
    try std.testing.expect(!isSkippedTensor("model.language_model.layers.4.mhc_attn.weight"));
}

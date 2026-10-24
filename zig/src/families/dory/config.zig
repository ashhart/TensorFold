//! Nemotron-Labs Dory's dimensions from config.json: a looped nGPT stack, its loop count, and the layouts refused.
const std = @import("std");
const eos_ids = @import("core").tokenizer.eos;

pub const max_sublayers = 128;
pub const max_loops = 4; // llama.cpp's limit, and the card's largest benchmarked count
pub const max_eos = 4;
pub const max_width = 8192; // the rows kernels' widest row: 4 sixteen-byte loads a thread, 256 threads

/// A sublayer: attention over a sliding window, attention over every key before, or a SwiGLU MLP.
pub const Kind = enum { sliding, full, mlp };

/// How the projections are stored: NVFP4 W4A16 at group 16, compressed-tensors MXFP8 at group 32, or plain bf16.
pub const Quant = enum { nvfp4, mxfp8, bf16 };

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
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
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sublayers: usize, // num_hidden_layers: physical sublayers, input then recurrent then output
    input: usize,
    recurrent: usize,
    output: usize,
    loops: usize, // n_recurrent_loops, or the load option that replaces it
    kinds: [max_sublayers]Kind = @splat(.mlp),
    window: usize, // a sliding sublayer's keys, the row's own included
    theta_sliding: f64,
    theta_full: f64,
    std: f64, // initializer_range: every nGPT scale is stored divided by it
    alpha_init: f64,
    gate_init: f64, // attention_output_gate_logit_scale_init, sqrt(hidden) when null
    positions: usize, // max_position_embeddings
    quant: Quant,
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    pub fn qDim(self: Config) usize {
        return self.heads * self.head_dim;
    }

    pub fn kvDim(self: Config) usize {
        return self.kv_heads * self.head_dim;
    }

    /// Which block sublayer `i` is in: 0 input, 1 recurrent, 2 output.
    pub fn block(self: Config, i: usize) u2 {
        return if (i < self.input) 0 else if (i < self.input + self.recurrent) 1 else 2;
    }

    /// Sublayers of `kind` in [from, to).
    pub fn count(self: Config, kind: Kind, from: usize, to: usize) usize {
        var n: usize = 0;
        for (self.kinds[from..to]) |k| n += @intFromBool(k == kind);
        return n;
    }

    /// The same model run with `loops` passes over the recurrent block (a load option, 1 to 4).
    pub fn withLoops(self: Config, loops: usize) !Config {
        if (loops == 0 or loops > max_loops) return error.BadLoops;
        var c = self;
        c.loops = loops;
        return c;
    }

    /// The nGPT factors in fp32: float64 quotients rounded once, as torch casts a Python float for an fp32 multiply.
    pub fn factors(self: Config) Factors {
        const d: f64 = @floatFromInt(self.head_dim);
        const h: f64 = @floatFromInt(self.hidden);
        return .{
            .alpha = @floatCast(self.alpha_init / self.std),
            .sqk = @floatCast(@sqrt(d) / self.std),
            .gate = @floatCast(@sqrt(self.gate_init) / self.std),
            .suv = @floatCast(@sqrt(h)),
            .logit = 1 / @as(f32, @floatCast(self.std)), // torch divides by a scalar as a multiply by its fp32 reciprocal
            .attn = @floatCast(1 / @sqrt(d)),
        };
    }

    /// End tokens from generation_config.json's text, after config.json's own: replies end on either.
    pub fn addEos(self: *Config, allocator: std.mem.Allocator, generation: []const u8) !void {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, generation, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadEos;
        const ids = try eos_ids.merge(allocator, self.eos[0..self.eos_count], parsed.value, self.vocab);
        defer allocator.free(ids);
        try setEos(self, ids);
    }

    /// `dir`/config.json, checked for the kernels' shapes, and generation_config.json's end tokens when it is there.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        var c = parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        checkShapes(c, &why) catch |e| {
            std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gen_path);
        const gen = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => return c,
            else => return e,
        };
        defer gpa.free(gen);
        try c.addEos(gpa, gen);
        return c;
    }
};

/// The fp32 factors the scale vectors are multiplied by (Config.factors).
pub const Factors = struct { alpha: f32, sqk: f32, gate: f32, suv: f32, logit: f32, attn: f32 };

fn int(obj: std.json.ObjectMap, key: []const u8, why: *Why) !usize {
    const v = obj.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    if (v != .integer or v.integer <= 0) {
        why.set("config.json's {s} is not a positive integer", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn number(obj: std.json.ObjectMap, key: []const u8, why: *Why) !f64 {
    const v = obj.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    const f: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
    if (!(f > 0) or !std.math.isFinite(f)) {
        why.set("config.json's {s} is not a positive number", .{key});
        return error.BadConfig;
    }
    return f;
}

fn flag(obj: std.json.ObjectMap, key: []const u8) bool {
    const v = obj.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn setEos(c: *Config, ids: []const u32) !void {
    if (ids.len > max_eos) return error.TooManyEos;
    @memcpy(c.eos[0..ids.len], ids);
    c.eos_count = ids.len;
}

/// rope_parameters[kind]: a default (unscaled) rotation of every dimension; its theta.
fn rope(o: std.json.ObjectMap, kind: []const u8, why: *Why) !f64 {
    const all = o.get("rope_parameters") orelse {
        why.set("config.json has no rope_parameters", .{});
        return error.BadConfig;
    };
    if (all != .object) return error.BadConfig;
    const p = all.object.get(kind) orelse {
        why.set("rope_parameters has no {s}", .{kind});
        return error.BadConfig;
    };
    if (p != .object) return error.BadConfig;
    if (p.object.get("rope_type")) |t| if (t != .string or !std.mem.eql(u8, t.string, "default")) {
        why.set("{s}'s rope_type is not default; Dory on CUDA rotates without scaling", .{kind});
        return error.UnsupportedRope;
    };
    if (p.object.get("partial_rotary_factor")) |f| {
        const v: f64 = switch (f) {
            .float => |x| x,
            .integer => |x| @floatFromInt(x),
            else => 0,
        };
        if (v != 1) {
            why.set("{s} rotates part of a head; Dory rotates all of it", .{kind});
            return error.UnsupportedRope;
        }
    }
    return number(p.object, "rope_theta", why);
}

/// The quantization config.json declares: none (bf16), ModelOpt's W4A16_NVFP4, or compressed-tensors' mxfp8.
fn quantOf(o: std.json.ObjectMap, why: *Why) !Quant {
    const q = o.get("quantization_config") orelse return .bf16;
    if (q != .object) return error.BadConfig;
    const method = q.object.get("quant_method");
    const algo = q.object.get("quant_algo");
    const modelopt = method != null and method.? == .string and std.mem.eql(u8, method.?.string, "modelopt");
    const w4a16 = algo != null and algo.? == .string and std.mem.eql(u8, algo.?.string, "W4A16_NVFP4");
    if (modelopt and w4a16) return nvfp4Groups(q.object, why);
    const mxfp8 = method != null and method.? == .string and std.mem.eql(u8, method.?.string, "compressed-tensors");
    if (mxfp8) return mxfp8Groups(q.object, why);
    why.set("Dory on CUDA reads ModelOpt's W4A16_NVFP4, compressed-tensors' mxfp8-quantized or plain bf16; this one is quantized otherwise", .{});
    return error.UnsupportedQuantization;
}

/// ModelOpt's groups: 16 weights a scale.
fn nvfp4Groups(q: std.json.ObjectMap, why: *Why) !Quant {
    if (q.get("config_groups")) |groups| if (groups == .object) {
        var it = groups.object.iterator();
        while (it.next()) |g| {
            const w = if (g.value_ptr.* == .object) g.value_ptr.object.get("weights") else null;
            const gs = if (w != null and w.? == .object) w.?.object.get("group_size") else null;
            if (gs == null or gs.? != .integer or gs.?.integer != 16) {
                why.set("NVFP4 group {s} is not 16 weights a scale", .{g.key_ptr.*});
                return error.UnsupportedQuantization;
            }
        }
    };
    return .nvfp4;
}

/// compressed-tensors' groups: mxfp8-quantized at 8 bits, float, 32 weights a scale.
fn mxfp8Groups(q: std.json.ObjectMap, why: *Why) !Quant {
    const groups = q.get("config_groups") orelse {
        why.set("mxfp8's config.json has no config_groups", .{});
        return error.BadConfig;
    };
    if (groups != .object) return error.BadConfig;
    var it = groups.object.iterator();
    while (it.next()) |g| if (g.value_ptr.* == .object) {
        const w = g.value_ptr.object.get("weights");
        const ok = w != null and w.? == .object and intField(w.?.object, "num_bits", 8) and intField(w.?.object, "group_size", 32) and
            stringField(w.?.object, "type", "float") and stringField(g.value_ptr.object, "format", "mxfp8-quantized");
        if (!ok) {
            why.set("mxfp8 group {s} is not mxfp8-quantized at 8-bit float, group 32", .{g.key_ptr.*});
            return error.UnsupportedQuantization;
        }
    };
    return .mxfp8;
}

/// The object's `key` as the integer `want`.
fn intField(o: std.json.ObjectMap, key: []const u8, want: i64) bool {
    const v = o.get(key) orelse return false;
    return v == .integer and v.integer == want;
}

/// The object's `key` as the string `want`.
fn stringField(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, want);
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const o = parsed.value.object;
    const mt = o.get("model_type") orelse return error.BadConfig;
    if (mt != .string or !std.mem.eql(u8, mt.string, "dory")) {
        why.set("model_type {f} is not dory", .{std.json.fmt(mt, .{})});
        return error.NotDory;
    }
    if (flag(o, "tie_word_embeddings") or flag(o, "mlp_bias") or flag(o, "attention_bias")) {
        why.set("Dory has an untied head and no biases; this config ties or adds them", .{});
        return error.UnsupportedLayer;
    }
    if (o.get("hidden_act")) |a| if (a != .string or !std.mem.eql(u8, a.string, "silu")) {
        why.set("hidden_act is not silu", .{});
        return error.UnsupportedLayer;
    };
    // per_loop (vLLM's default): a cache per recurrent attention sublayer and loop; last_loop depends on chunking
    if (o.get("recurrent_kv_cache_mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "per_loop")) {
        why.set("recurrent_kv_cache_mode {f}: only per_loop gives the same tokens however a prompt is cut", .{std.json.fmt(m, .{})});
        return error.UnsupportedKvMode;
    };
    var c = Config{
        .hidden = try int(o, "hidden_size", why),
        .intermediate = try int(o, "intermediate_size", why),
        .vocab = try int(o, "vocab_size", why),
        .heads = try int(o, "num_attention_heads", why),
        .kv_heads = try int(o, "num_key_value_heads", why),
        .head_dim = try int(o, "head_dim", why),
        .sublayers = try int(o, "num_hidden_layers", why),
        .input = try int(o, "n_input_layers", why),
        .recurrent = try int(o, "n_recurrent_layers", why),
        .output = try int(o, "n_output_layers", why),
        .loops = try int(o, "n_recurrent_loops", why),
        .window = try int(o, "sliding_window", why),
        .theta_sliding = try rope(o, "sliding_attention", why),
        .theta_full = try rope(o, "full_attention", why),
        .std = try number(o, "initializer_range", why),
        .alpha_init = try number(o, "ngpt_alpha_init", why),
        .gate_init = 0,
        .positions = try int(o, "max_position_embeddings", why),
        .quant = try quantOf(o, why),
    };
    c.gate_init = if (o.get("attention_output_gate_logit_scale_init")) |g| (if (g == .null) @sqrt(@as(f64, @floatFromInt(c.hidden))) else try number(o, "attention_output_gate_logit_scale_init", why)) else @sqrt(@as(f64, @floatFromInt(c.hidden)));
    if (c.sublayers > max_sublayers or c.input + c.recurrent + c.output != c.sublayers) {
        why.set("{d} sublayers against {d} input, {d} recurrent and {d} output (at most {d})", .{ c.sublayers, c.input, c.recurrent, c.output, max_sublayers });
        return error.BadConfig;
    }
    if (c.loops > max_loops) {
        why.set("{d} recurrent loops; Dory runs 1 to {d}", .{ c.loops, max_loops });
        return error.BadLoops;
    }
    const types = o.get("layer_types") orelse {
        why.set("config.json has no layer_types", .{});
        return error.BadConfig;
    };
    if (types != .array or types.array.items.len != c.sublayers) {
        why.set("layer_types does not name each of the {d} sublayers", .{c.sublayers});
        return error.BadConfig;
    }
    for (types.array.items, 0..) |t, i| {
        if (t != .string) return error.BadConfig;
        const named = std.meta.stringToEnum(enum { sliding_attention, full_attention, mlp }, t.string) orelse {
            why.set("sublayer {d} is {s}; Dory has sliding_attention, full_attention and mlp sublayers", .{ i, t.string });
            return error.UnsupportedLayer;
        };
        c.kinds[i] = switch (named) {
            .sliding_attention => .sliding,
            .full_attention => .full,
            .mlp => .mlp,
        };
    }
    const ids = try eos_ids.resolve(allocator, parsed.value, .null, c.vocab);
    defer allocator.free(ids);
    try setEos(&c, ids);
    return c;
}

/// What the shared kernels take: head dim 256 in whole GQA groups, rows to 8192 wide in 64s, a window of 2 or more.
pub fn checkShapes(c: Config, why: *Why) !void {
    if (c.head_dim != 256 or c.kv_heads == 0 or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 16) {
        why.set("head dim {d}, {d} query heads over {d} key heads: this engine takes head dim 256 and up to 16 query heads a key head", .{ c.head_dim, c.heads, c.kv_heads });
        return error.UnsupportedShapes;
    }
    for ([_]usize{ c.hidden, c.intermediate, c.qDim() }) |w| if (w % 64 != 0 or w > max_width) {
        why.set("hidden {d}, intermediate {d}, attention {d}: the rows kernels take multiples of 64 up to {d}", .{ c.hidden, c.intermediate, c.qDim(), max_width });
        return error.UnsupportedShapes;
    };
    if (c.window < 2) {
        why.set("sliding window {d}: the window kernels take 2 or more keys", .{c.window});
        return error.UnsupportedShapes;
    }
    if (c.count(.mlp, 0, c.sublayers) == c.sublayers) {
        why.set("no attention sublayer", .{});
        return error.UnsupportedShapes;
    }
}

/// The NVFP4 checkpoint's config.json at the pinned revision (41f02335), its layer_types written as the pattern.
pub const pinned =
    \\{"architectures": ["DoryForCausalLM"], "model_type": "dory", "vocab_size": 131072, "hidden_size": 2560,
    \\ "num_hidden_layers": 72, "intermediate_size": 7168, "num_attention_heads": 16, "num_key_value_heads": 4,
    \\ "max_position_embeddings": 262144, "tie_word_embeddings": false, "ngpt_alpha_init": 0.05,
    \\ "attention_output_gate_logit_scale_init": null, "mlp_bias": false, "attention_bias": false,
    \\ "n_input_layers": 16, "n_recurrent_layers": 40, "n_output_layers": 16, "n_recurrent_loops": 2,
    \\ "quantization_config": {"config_groups": {"group_0": {"weights": {"dynamic": false, "num_bits": 4,
    \\  "type": "float", "group_size": 16}, "targets": ["Linear"]}}, "ignore": ["backbone.embeddings", "lm_head"],
    \\  "quant_algo": "W4A16_NVFP4", "producer": {"name": "modelopt", "version": "0.0.0"}, "quant_method": "modelopt"},
    \\ "head_dim": 256, "hidden_act": "silu", "initializer_range": 0.0177, "dtype": "bfloat16", "sliding_window": 1024,
    \\ "layer_types": [S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M,
    \\  S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M, S, M, S, M, S, M, F, M,
    \\  S, M, S, M, S, M, F, M],
    \\ "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    \\  "full_attention": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 1.0}}}
;

/// The pinned generation_config.json: replies end on </s> or <|im_end|>.
pub const pinned_generation =
    \\{"_from_model_config": true, "bos_token_id": 1, "do_sample": true, "eos_token_id": [2, 11], "pad_token_id": 11,
    \\ "top_p": 0.95, "temperature": 1.0, "transformers_version": "4.57.6"}
;

const testing = std.testing;

/// `pinned` with its pattern letters spelled out, as config.json has them.
pub fn pinnedText(a: std.mem.Allocator) ![]u8 {
    const s = try std.mem.replaceOwned(u8, a, pinned, "S,", "\"sliding_attention\",");
    defer a.free(s);
    const f = try std.mem.replaceOwned(u8, a, s, "F,", "\"full_attention\",");
    defer a.free(f);
    const m = try std.mem.replaceOwned(u8, a, f, "M,", "\"mlp\",");
    defer a.free(m);
    return std.mem.replaceOwned(u8, a, m, "M]", "\"mlp\"]");
}

fn replaced(from: []const u8, to: []const u8) ![]u8 {
    const t = try pinnedText(testing.allocator);
    defer testing.allocator.free(t);
    return std.mem.replaceOwned(u8, testing.allocator, t, from, to);
}

fn refused(text: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try testing.expectError(err, parse(testing.allocator, text, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "parse the pinned config: blocks, sublayer kinds, RoPE, nGPT factors and end tokens" {
    const text = try pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: Why = .{};
    var c = try parse(testing.allocator, text, &why);
    try checkShapes(c, &why);
    try testing.expectEqual(@as(usize, 72), c.sublayers);
    try testing.expectEqual(@as(usize, 2), c.loops);
    try testing.expectEqual(Quant.nvfp4, c.quant);
    try testing.expectEqual(Kind.sliding, c.kinds[0]);
    try testing.expectEqual(Kind.mlp, c.kinds[1]);
    for ([_]usize{ 6, 14, 22, 30, 38, 46, 54, 62, 70 }) |i| try testing.expectEqual(Kind.full, c.kinds[i]);
    try testing.expectEqual(@as(usize, 2), c.count(.full, 0, 16));
    try testing.expectEqual(@as(usize, 5), c.count(.full, 16, 56));
    try testing.expectEqual(@as(usize, 15), c.count(.sliding, 16, 56));
    try testing.expectEqual(@as(usize, 36), c.count(.mlp, 0, 72));
    try testing.expectEqual(@as(f64, 1e4), c.theta_sliding);
    try testing.expectEqual(@as(f64, 1e7), c.theta_full);
    try testing.expectEqual(@sqrt(@as(f64, 2560)), c.gate_init);
    const f = c.factors();
    try testing.expectEqual(@as(f32, @floatCast(0.05 / 0.0177)), f.alpha);
    try testing.expectApproxEqRel(@as(f32, 903.9548), f.sqk, 1e-6);
    try testing.expectApproxEqRel(@as(f32, 401.9), f.gate, 1e-4);
    try testing.expectApproxEqRel(@as(f32, 50.596443), f.suv, 1e-7);
    try testing.expectEqual(@as(f32, 1.0 / 16.0), f.attn);
    try testing.expectEqual(@as(usize, 0), c.eos_count); // config.json declares none
    try c.addEos(testing.allocator, pinned_generation);
    try testing.expectEqualSlices(u32, &.{ 2, 11 }, c.eos[0..c.eos_count]);
    try testing.expect(c.isEos(11) and !c.isEos(1));
    try testing.expectEqual(@as(usize, 4), (try c.withLoops(4)).loops);
    try testing.expectError(error.BadLoops, c.withLoops(0));
    try testing.expectError(error.BadLoops, c.withLoops(5));
}

test "refusals: not dory, last_loop KV, other quantization, scaled RoPE, ties, other sublayers, counts" {
    const not = try replaced("\"model_type\": \"dory\"", "\"model_type\": \"llama\"");
    defer testing.allocator.free(not);
    try refused(not, error.NotDory, "dory");
    const last = try replaced("\"n_recurrent_loops\": 2,", "\"n_recurrent_loops\": 2, \"recurrent_kv_cache_mode\": \"last_loop\",");
    defer testing.allocator.free(last);
    try refused(last, error.UnsupportedKvMode, "per_loop");
    const mx = try replaced("\"quant_algo\": \"W4A16_NVFP4\"", "\"quant_algo\": \"MXFP8\"");
    defer testing.allocator.free(mx);
    try refused(mx, error.UnsupportedQuantization, "W4A16_NVFP4");
    const group = try replaced("\"group_size\": 16", "\"group_size\": 32");
    defer testing.allocator.free(group);
    try refused(group, error.UnsupportedQuantization, "16");
    const yarn = try replaced("{\"rope_type\": \"default\", \"rope_theta\": 10000000", "{\"rope_type\": \"yarn\", \"rope_theta\": 10000000");
    defer testing.allocator.free(yarn);
    try refused(yarn, error.UnsupportedRope, "full_attention");
    const partial = try replaced("\"rope_theta\": 10000, \"partial_rotary_factor\": 1.0", "\"rope_theta\": 10000, \"partial_rotary_factor\": 0.5");
    defer testing.allocator.free(partial);
    try refused(partial, error.UnsupportedRope, "part");
    const tied = try replaced("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
    defer testing.allocator.free(tied);
    try refused(tied, error.UnsupportedLayer, "untied");
    const moe = try replaced("\"mlp\"]", "\"moe\"]");
    defer testing.allocator.free(moe);
    try refused(moe, error.UnsupportedLayer, "moe");
    const counts = try replaced("\"n_output_layers\": 16", "\"n_output_layers\": 15");
    defer testing.allocator.free(counts);
    try refused(counts, error.BadConfig, "15 output");
    const loops = try replaced("\"n_recurrent_loops\": 2", "\"n_recurrent_loops\": 5");
    defer testing.allocator.free(loops);
    try refused(loops, error.BadLoops, "5 recurrent loops");
}

test "bf16 checkpoints parse without a quantization_config; shapes outside the kernels are refused" {
    const t = try pinnedText(testing.allocator);
    defer testing.allocator.free(t);
    const at = std.mem.indexOf(u8, t, "\"quantization_config\"").?;
    const end = std.mem.indexOf(u8, t, "\"head_dim\"").?;
    const plain = try std.mem.concat(testing.allocator, u8, &.{ t[0..at], t[end..] });
    defer testing.allocator.free(plain);
    var why: Why = .{};
    const c = try parse(testing.allocator, plain, &why);
    try testing.expectEqual(Quant.bf16, c.quant);
    var narrow = c;
    narrow.head_dim = 128;
    try testing.expectError(error.UnsupportedShapes, checkShapes(narrow, &why));
    var wide = c;
    wide.intermediate = 9216;
    try testing.expectError(error.UnsupportedShapes, checkShapes(wide, &why));
    var odd = c;
    odd.kv_heads = 3;
    try testing.expectError(error.UnsupportedShapes, checkShapes(odd, &why));
}

/// The published MXFP8 quantization_config (targets shortened to "Linear"), spliced in place of NVFP4's.
const mxfp8_config =
    \\{"config_groups": {"dory_matmul_weights": {"format": "mxfp8-quantized",
    \\  "input_activations": null,
    \\  "output_activations": null,
    \\  "targets": ["Linear"],
    \\  "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 32, "num_bits": 8,
    \\   "observer": "mse", "scale_dtype": "torch.uint8", "strategy": "group", "symmetric": true, "type": "float",
    \\   "zp_dtype": null}}},
    \\ "format": "mxfp8-quantized", "global_compression_ratio": null,
    \\ "ignore": ["backbone.embeddings", "lm_head"],
    \\ "kv_cache_scheme": null,
    \\ "quant_method": "compressed-tensors",
    \\ "quantization_status": "compressed"}
;

fn mxfp8Text(a: std.mem.Allocator) ![]u8 {
    const t = try pinnedText(a);
    defer a.free(t);
    const at = std.mem.indexOf(u8, t, "\"quantization_config\"").?;
    const end = std.mem.indexOf(u8, t, "\"head_dim\"").?;
    return std.mem.concat(a, u8, &.{ t[0..at], "\"quantization_config\": ", mxfp8_config, ", ", t[end..] });
}

test "the MXFP8 quantization_config parses to mxfp8; group 16, 4-bit and other compressed-tensors refuse" {
    var why: Why = .{};
    const mx = try mxfp8Text(testing.allocator);
    defer testing.allocator.free(mx);
    const c = try parse(testing.allocator, mx, &why);
    try testing.expectEqual(Quant.mxfp8, c.quant);
    const gs16 = try std.mem.replaceOwned(u8, testing.allocator, mx, "\"group_size\": 32", "\"group_size\": 16");
    defer testing.allocator.free(gs16);
    try refused(gs16, error.UnsupportedQuantization, "mxfp8");
    const bits4 = try std.mem.replaceOwned(u8, testing.allocator, mx, "\"num_bits\": 8", "\"num_bits\": 4");
    defer testing.allocator.free(bits4);
    try refused(bits4, error.UnsupportedQuantization, "mxfp8");
    const int4 = try std.mem.replaceOwned(u8, testing.allocator, mx, "mxfp8-quantized", "int4-quantized");
    defer testing.allocator.free(int4);
    try refused(int4, error.UnsupportedQuantization, "mxfp8");
}

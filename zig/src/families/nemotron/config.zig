//! Nemotron-H dimensions from config.json, one Config for every backend, checked against the shapes our kernels serve.
const std = @import("std");

pub const Kind = enum { mamba, moe, attention };

pub const max_layers = 96;

/// The CUDA latent router's bounds (nemotron_latent.cu's tf_super_topk): one warp of picks, 32 experts a lane.
pub const latent_max_picks = 32;
pub const latent_max_experts = 1024;

/// Conditional MTP acceptance by depth until a stream has its own estimates, from model.py draft_prior.
pub const draft_prior = [_]f64{ 0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5 };

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
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

/// config.json's count of learned neurons at the end of every shared expert (other loaders read them as the expert's).
pub const pocket_key = "tensorfold_learned_neurons";

/// How the checkpoint stores its matrices: MLX affine 4-bit, or NVIDIA ModelOpt (FP8, NVFP4 or bf16, read per tensor).
pub const Format = enum { mlx, modelopt };

/// --precision: a ModelOpt checkpoint's own math (e4m3 and NVFP4 rows), or bf16 rows throughout (full).
pub const Precision = enum { checkpoint, full };

/// One matrix's MLX affine width: `bits` per weight, a scale and a bias for each `group` of them.
pub const Width = struct {
    bits: u8 = 4,
    group: u16 = 64,

    /// The width the 4-bit kernels read (lane tiles, the expert kernels, the embedding); others take the row kernels.
    pub fn four(w: Width) bool {
        return w.bits == 4 and w.group == 64;
    }

    /// A width the core's affine kernels unpack: 2 to 8 bits but 7, in groups of 32, 64 or 128.
    pub fn readable(w: Width) bool {
        return w.bits >= 2 and w.bits <= 8 and w.bits != 7 and (w.group == 32 or w.group == 64 or w.group == 128);
    }
};

/// A matrix by its module name, at its width.
pub const Named = struct { name: []const u8, width: Width };

/// A layer kind's quantized matrices after "backbone.layers.N.mixer.", in Config.widths' order.
pub fn members(kind: Kind) []const []const u8 {
    return switch (kind) {
        .mamba => &.{ "in_proj", "out_proj" },
        .attention => &.{ "q_proj", "k_proj", "v_proj", "o_proj" },
        .moe => &.{ "switch_mlp.fc1", "switch_mlp.fc2", "shared_experts.up_proj", "shared_experts.down_proj" },
    };
}

/// One MoE layer's routed experts and their width.
pub const Moe = struct { experts: usize = 0, width: usize = 0 };

pub const Config = struct {
    hidden: usize,
    vocab: usize,
    layers: usize,
    kinds: [max_layers]Kind = undefined,
    mamba_heads: usize,
    mamba_head_dim: usize,
    groups: usize,
    state: usize,
    conv_kernel: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    experts: usize, // the widest MoE layer's routed experts (every layer's, unless `moe` says otherwise)
    top_k: usize,
    expert_width: usize, // the widest MoE layer's routed expert width
    bolt: bool = false, // puzzle blocks and GGML tables use Bolt's own latent expert path
    moe: [max_layers]Moe = @splat(.{}), // each MoE layer's own experts and width (block_configs); zero: the uniform ones
    shared_width: usize, // the shared expert's own neurons, learned ones (pocket) left out
    pocket: usize = 0, // tensorfold_learned_neurons: learned neurons after the shared expert's own, in the same tensors
    routed_scaling: f32,
    norm_topk: bool = true,
    eps: f32,
    dt_min: f32 = 0, // time_step_limit: the Mamba dt clamp, (0, inf) when the config sets none
    dt_max: f32 = std.math.inf(f32),
    group_size: usize = 64,
    bits: usize = 4,
    format: Format = .mlx,
    eos: [4]u32 = .{ 0, 0, 0, 0 },
    eos_count: usize = 0,
    latent: usize = 0, // moe_latent_size: routed experts read and write this width between two bf16 projections
    omni: bool = false, // nemotron_h_omni: the text model's tensors sit under "language_model."
    shares: usize = 1, // latent MoE: expert widths cut in this many shares, summed in share order on every TP world
    held: ?usize = null, // a split checkpoint (tensorfold_tp): the one share its expert tensors hold
    rank: usize = 0, // this process's tensor-parallel rank of `world`: set at load, never by config.json
    world: usize = 1,
    ck: bool = false, // --precision checkpoint: e4m3 rows into the FP8 layers, NVFP4 rows into the experts
    widths: [max_layers][4]Width = @splat(@splat(.{})), // each layer's matrices (members(kind) order) at their widths
    embed_width: Width = .{},
    head_width: Width = .{},

    /// MoE layer `i`'s routed experts and width.
    pub fn layerMoe(self: Config, i: usize) Moe {
        const m = self.moe[i];
        return if (m.experts == 0) .{ .experts = self.experts, .width = self.expert_width } else m;
    }

    /// The width routed experts read and write: the latent one, else hidden.
    pub fn expertDims(self: Config) usize {
        return if (self.latent > 0) self.latent else self.hidden;
    }

    pub fn inner(self: Config) usize {
        return self.mamba_heads * self.mamba_head_dim;
    }

    pub fn convDim(self: Config) usize {
        return self.inner() + 2 * self.groups * self.state;
    }

    pub fn projDim(self: Config) usize {
        return self.inner() + self.convDim() + self.mamba_heads;
    }

    pub fn qkvDim(self: Config) usize {
        return (self.heads + 2 * self.kv_heads) * self.head_dim;
    }

    /// A token's expert pairs: its routed experts, then the shared expert's two halves (a latent MoE runs it apart).
    pub fn slots(self: Config) usize {
        return if (self.latent > 0) self.top_k else self.top_k + 2;
    }

    /// Where the text model's tensor names start.
    pub fn prefix(self: Config) []const u8 {
        return if (self.omni) "language_model." else "";
    }

    /// The expert width shares this rank holds and sums before the ranks meet.
    pub fn ownShares(self: Config) usize {
        return self.shares / self.world;
    }

    pub fn count(self: Config, kind: Kind) usize {
        var n: usize = 0;
        for (self.kinds[0..self.layers]) |k| n += @intFromBool(k == kind);
        return n;
    }

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// Layer i's matrix `leaf` (one of members(kind)) at its width.
    pub fn widthOf(self: Config, i: usize, leaf: []const u8) Width {
        for (members(self.kinds[i]), 0..) |m, j| if (std.mem.eql(u8, m, leaf)) return self.widths[i][j];
        std.debug.panic("layer {d} has no matrix {s}", .{ i, leaf });
    }

    /// The first matrix at another width than 4-bit in groups of 64 (layer names written into `buf`), else null.
    pub fn other(self: *const Config, buf: []u8) ?Named {
        if (!self.embed_width.four()) return .{ .name = "backbone.embeddings", .width = self.embed_width };
        for (0..self.layers) |i| for (members(self.kinds[i]), 0..) |m, j| if (!self.widths[i][j].four()) {
            const name = std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, m }) catch m;
            return .{ .name = name, .width = self.widths[i][j] };
        };
        if (!self.head_width.four()) return .{ .name = "lm_head", .width = self.head_width };
        return null;
    }

    /// Refuse a checkpoint with any width but 4/64 for `engine`, which reads only those, logging its first such matrix.
    pub fn fourOnly(self: *const Config, engine: []const u8) error{UnsupportedQuantization}!void {
        var buf: [96]u8 = undefined;
        const m = self.other(&buf) orelse return;
        std.log.warn("{s} is {d}-bit in groups of {d}: {s} reads Nemotron MLX checkpoints at 4-bit in groups of 64 only; mixed and 6- or 8-bit (oQ) checkpoints serve on Macs", .{ m.name, m.width.bits, m.width.group, engine });
        return error.UnsupportedQuantization;
    }

    /// 0 when every matrix is 4/64, else a nonzero hash of every width: timed costs keep each mix of widths apart.
    pub fn widthsTag(self: *const Config) u64 {
        var buf: [96]u8 = undefined;
        if (self.other(&buf) == null) return 0;
        var h = std.hash.Wyhash.init(0x6e71);
        const all = [_][]const [4]Width{ &.{@splat(self.embed_width)}, self.widths[0..self.layers], &.{@splat(self.head_width)} };
        for (all) |part| for (part) |l| for (l) |w| h.update(&.{ w.bits, @as(u8, @truncate(w.group)), @as(u8, @truncate(w.group >> 8)) });
        return h.final() | 1;
    }

    /// `dir`/config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 25)); // ModelOpt names each module
        defer gpa.free(text);
        var why: Why = .{};
        return parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
    }
};

fn int(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse {
        std.log.err("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    return @intCast(v.integer);
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .object => |o| inf(o) orelse error.BadConfig, // an infinite limit, as {"__float__": "Infinity"}
        else => error.BadConfig,
    };
}

fn inf(o: std.json.ObjectMap) ?f64 {
    const s = o.get("__float__") orelse return null;
    if (s != .string) return null;
    if (std.mem.eql(u8, s.string, "Infinity")) return std.math.inf(f64);
    if (std.mem.eql(u8, s.string, "-Infinity")) return -std.math.inf(f64);
    return null;
}

fn float(obj: std.json.ObjectMap, key: []const u8) !f32 {
    return @floatCast(try number(obj.get(key) orelse return error.BadConfig));
}

fn kindOf(name: []const u8) !Kind {
    if (std.mem.eql(u8, name, "mamba") or std.mem.eql(u8, name, "M")) return .mamba;
    if (std.mem.eql(u8, name, "moe") or std.mem.eql(u8, name, "E")) return .moe;
    if (std.mem.eql(u8, name, "attention") or std.mem.eql(u8, name, "*")) return .attention;
    return error.UnsupportedBlock;
}

/// The quantization block as MLX reads it: "quantization", else the "quantization_config" that mirrors it.
fn quantBlock(o: std.json.ObjectMap) ?std.json.ObjectMap {
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (o.get(key)) |v| if (v == .object and v.object.count() > 0) return v.object;
    }
    return null;
}

/// ModelOpt's quant_method: "modelopt", or "modelopt_mixed" where layers mix FP8 and NVFP4.
fn modelopt(m: []const u8) bool {
    return std.ascii.eqlIgnoreCase(m, "modelopt") or std.ascii.eqlIgnoreCase(m, "modelopt_mixed");
}

/// A quantization block's own fields; every other key names a module.
fn blockField(key: []const u8) bool {
    for ([_][]const u8{ "bits", "group_size", "mode", "quant_method" }) |f| if (std.mem.eql(u8, key, f)) return true;
    return false;
}

/// MLX's affine mode, the one an entry has when it names none.
fn affine(o: std.json.ObjectMap) bool {
    const m = o.get("mode") orelse return true;
    return m == .null or (m == .string and (m.string.len == 0 or std.ascii.eqlIgnoreCase(m.string, "affine")));
}

/// A positive integer field, `default` when absent or null, 0 when unreadable.
fn width(o: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = o.get(key) orelse return default;
    return switch (v) {
        .null => default,
        .integer => |i| if (i > 0) @intCast(i) else 0,
        else => 0,
    };
}

/// The checkpoint's width and each per-module entry's (oQ quants); a width no kernel reads is refused by its module.
fn quantization(c: *Config, o: std.json.ObjectMap, why: *Why) !void {
    const q = quantBlock(o) orelse {
        why.set("config.json declares no quantization; the native Nemotron kernels read MLX affine checkpoints", .{});
        return error.UnsupportedQuantization;
    };
    if (q.get("quant_method")) |m| if (m == .string and modelopt(m.string)) {
        c.format = .modelopt; // the CUDA loader reads each projection's format from its tensors
        return;
    };
    if (q.get("quant_method")) |m| if (!(m == .null or (m == .string and (std.ascii.eqlIgnoreCase(m.string, "mlx") or std.ascii.eqlIgnoreCase(m.string, "affine"))))) {
        why.set("config.json's quant_method is {f}; the native Nemotron kernels read MLX affine checkpoints", .{std.json.fmt(m, .{})});
        return error.UnsupportedQuantization;
    };
    if (!affine(q)) {
        why.set("config.json quantizes in mode {f}; the native Nemotron kernels read MLX affine checkpoints", .{std.json.fmt(q.get("mode").?, .{})});
        return error.UnsupportedQuantization;
    }
    c.bits = width(q, "bits", 0);
    c.group_size = width(q, "group_size", 64);
    if (c.bits == 0 or c.group_size == 0) {
        why.set("config.json's quantization has no readable bits or group_size", .{});
        return error.UnsupportedQuantization;
    }
    const global = readable(c.bits, c.group_size) orelse {
        why.set("config.json quantizes the checkpoint at {d}-bit in groups of {d}; " ++ kernel_widths, .{ c.bits, c.group_size });
        return error.UnsupportedQuantization;
    };
    c.embed_width = global;
    c.head_width = global;
    c.widths = @splat(@splat(global));
    var it = q.iterator();
    while (it.next()) |e| if (!blockField(e.key_ptr.*)) try module(c, global, e.key_ptr.*, e.value_ptr.*, why);
}

/// What the native Nemotron kernels read, for the refusals that name a width.
const kernel_widths = "the native Nemotron kernels read MLX affine matrices at 2- to 8-bit (not 7) in groups of 32, 64 or 128";

/// The width when the core's affine kernels unpack it, else null.
fn readable(bits: usize, group: usize) ?Width {
    const w: Width = .{ .bits = std.math.cast(u8, bits) orelse return null, .group = std.math.cast(u16, group) orelse return null };
    return if (w.readable()) w else null;
}

/// A per-module entry as nn.quantize reads it: true, a dict at a readable width, or false for a lesson's matrix.
fn module(c: *Config, global: Width, path: []const u8, v: std.json.Value, why: *Why) !void {
    const slot = target(c, path) orelse return; // a module the engine does not read as a quantized matrix
    slot.* = switch (v) {
        .bool => |on| if (on) global else if (learned(path)) Width{} else return unquantized(path, why), // split to 4/64
        .object => |m| if (m.count() == 0) return unquantized(path, why) else blk: {
            // a dict's missing bits or group_size take to_quantized's affine defaults, not the top level's
            const bits = width(m, "bits", 4);
            const group = width(m, "group_size", 64);
            if (affine(m)) if (readable(bits, group)) |w| break :blk w;
            if (!affine(m) or bits == 0 or group == 0) {
                why.set("config.json quantizes {s} as {f}; " ++ kernel_widths, .{ path, std.json.fmt(v, .{}) });
            } else why.set("config.json quantizes {s} at {d}-bit in groups of {d}; " ++ kernel_widths, .{ path, bits, group });
            return error.UnsupportedQuantization;
        },
        .null => return unquantized(path, why),
        else => return,
    };
}

fn unquantized(path: []const u8, why: *Why) error{UnsupportedQuantization} {
    why.set("config.json leaves {s} unquantized; the native Nemotron kernels read it only as an MLX affine matrix", .{path});
    return error.UnsupportedQuantization;
}

/// The width slot of the matrix `path` names (with or without "language_model."), or null for any other module.
fn target(c: *Config, path: []const u8) ?*Width {
    var p = path;
    if (std.mem.startsWith(u8, p, "language_model.")) p = p["language_model.".len..];
    if (std.mem.eql(u8, p, "backbone.embeddings")) return &c.embed_width;
    if (std.mem.eql(u8, p, "lm_head")) return &c.head_width;
    const stem = "backbone.layers.";
    if (!std.mem.startsWith(u8, p, stem)) return null;
    const rest = p[stem.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const i = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    if (i >= c.layers or !std.mem.startsWith(u8, rest[dot..], ".mixer.")) return null;
    for (members(c.kinds[i]), 0..) |m, j| if (std.mem.eql(u8, m, rest[dot + ".mixer.".len ..])) return &c.widths[i][j];
    return null;
}

/// An output projection Sliding Weights wrote in bf16: the loader reads it as codes plus a residual.
fn learned(path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, "backbone.layers.")) return false;
    for ([_][]const u8{ ".mixer.shared_experts.down_proj", ".mixer.out_proj", ".mixer.o_proj" }) |leaf| if (std.mem.endsWith(u8, path, leaf)) return true;
    return false;
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    // nemotron_h_omni (Bolt) wraps the language model's config in llm_config, its quantization at the top
    const wrapped = isType(root, "nemotron_h_omni");
    const o = if (wrapped) (llmConfig(root) orelse return error.BadConfig) else root;
    if (!isType(o, "nemotron_h") and !isType(o, "nemotron_h_puzzle")) return error.NotNemotronH;
    var c = Config{
        .hidden = try int(o, "hidden_size"),
        .vocab = try int(o, "vocab_size"),
        .layers = 0,
        .mamba_heads = try int(o, "mamba_num_heads"),
        .mamba_head_dim = try int(o, "mamba_head_dim"),
        .groups = try int(o, "n_groups"),
        .state = try int(o, "ssm_state_size"),
        .conv_kernel = try int(o, "conv_kernel"),
        .heads = try int(o, "num_attention_heads"),
        .kv_heads = try int(o, "num_key_value_heads"),
        .head_dim = 0,
        // a puzzle config (Bolt) names these per block
        .experts = if (o.get("block_configs") != null) 0 else try int(o, "n_routed_experts"),
        .top_k = if (o.get("block_configs") != null) 0 else try int(o, "num_experts_per_tok"),
        .expert_width = if (o.get("block_configs") != null) 0 else try int(o, "moe_intermediate_size"),
        .shared_width = try int(o, "moe_shared_expert_intermediate_size"),
        .routed_scaling = if (o.get("routed_scaling_factor")) |v| @floatCast(try number(v)) else 1.0,
        .eps = if (o.get("layer_norm_epsilon")) |v| @floatCast(try number(v)) else 1e-5,
    };
    c.head_dim = if (o.get("head_dim")) |v| @intCast(v.integer) else c.hidden / c.heads;
    if (o.get(pocket_key)) |v| {
        if (v != .integer or v.integer < 0 or @as(usize, @intCast(v.integer)) >= c.shared_width or @mod(v.integer, 64) != 0) return error.BadConfig;
        c.pocket = @intCast(v.integer);
        c.shared_width -= c.pocket;
    }
    if (o.get("norm_topk_prob")) |v| c.norm_topk = v.bool;
    c.omni = wrapped;
    if (o.get("moe_latent_size")) |v| if (v == .integer) {
        if (v.integer < 0) return error.BadConfig; // a width: refused before the cast, as the latent checks below would
        c.latent = @intCast(v.integer);
    };
    if (o.get("block_configs")) |v| if (v == .array and v.array.items.len > 0) try puzzle(&c, v.array.items, why);
    // hybrid_override_pattern (one character a block) wins over layers_block_type, as the Python loader reads them
    if (c.layers == 0) if (o.get("hybrid_override_pattern")) |v| if (v == .string and v.string.len > 0) {
        if (v.string.len > max_layers) return error.BadConfig;
        for (v.string, 0..) |ch, i| c.kinds[i] = try kindOf(&.{ch});
        c.layers = v.string.len;
    };
    if (c.layers == 0) {
        const blocks = (o.get("layers_block_type") orelse return error.BadConfig).array.items;
        if (blocks.len > max_layers) return error.BadConfig;
        for (blocks, 0..) |b, i| c.kinds[i] = try kindOf(b.string);
        c.layers = blocks.len;
    }
    // generation_config's [2, 11] is the omni wrapper's eos plus the language model's
    if (wrapped) if (root.get("eos_token_id")) |e| eos(&c, e);
    if (o.get("eos_token_id")) |e| eos(&c, e);
    if (o.get("time_step_limit")) |v| if (v == .array and v.array.items.len == 2) {
        c.dt_min = @floatCast(try number(v.array.items[0]));
        c.dt_max = @floatCast(try number(v.array.items[1]));
    };
    c.omni = wrapped;
    if (o.get("moe_latent_size")) |v| if (v == .integer and v.integer > 0) {
        c.latent = @intCast(v.integer);
        try router(c, o, why);
        if (!c.bolt) c.shares = 2;
        try split(&c, root, why);
    };
    try quantization(&c, root, why);
    return c;
}

/// The latent router's reach: 1 to 1,024 experts, 1 to 32 picks, 64-wide input groups, no expert groups, relu^2.
fn router(c: Config, o: std.json.ObjectMap, why: *Why) !void {
    if (c.experts < 1 or c.experts > latent_max_experts or c.top_k < 1 or c.top_k > @min(c.experts, latent_max_picks) or c.hidden % 64 != 0) {
        why.set("a latent MoE of {d} experts picking {d} over {d} inputs: the router takes 1 to {d} experts, 1 to {d} picks among them and inputs in 64-wide groups", .{ c.experts, c.top_k, c.hidden, latent_max_experts, latent_max_picks });
        return error.BadConfig;
    }
    for ([_][]const u8{ "n_group", "topk_group" }) |k| if (o.get(k)) |v| if (v != .integer or v.integer != 1) {
        why.set("config.json's {s} is {f}: grouped expert selection is not built (the router picks among every expert)", .{ k, std.json.fmt(v, .{}) });
        return error.BadConfig;
    };
    if (o.get("mlp_hidden_act")) |v| if (v != .string or !std.mem.eql(u8, v.string, "relu2")) {
        why.set("config.json's mlp_hidden_act is {f}: the latent experts compute relu2", .{std.json.fmt(v, .{})});
        return error.BadConfig;
    };
}

/// A latent MoE's share widths, and the one share a split checkpoint holds (tensorfold_tp, our own split's record).
fn split(c: *Config, top: std.json.ObjectMap, why: *Why) !void {
    if (c.expert_width % (64 * c.shares) != 0 or c.shared_width % (64 * c.shares) != 0 or c.latent % 64 != 0) {
        why.set("expert widths {d} and {d} do not cut into {d} shares of whole 64-wide groups", .{ c.expert_width, c.shared_width, c.shares });
        return error.BadConfig;
    }
    const v = top.get("tensorfold_tp") orelse return;
    if (v != .object) return error.BadConfig;
    const t = v.object;
    const scheme = t.get("scheme") orelse return error.BadConfig;
    const ok = scheme == .string and std.mem.eql(u8, scheme.string, "moe_width") and width(t, "world", 0) == c.shares and
        width(t, "routed_width", 0) == c.expert_width / c.shares and width(t, "shared_width", 0) == c.shared_width / c.shares;
    const r = t.get("rank") orelse return error.BadConfig;
    if (!ok or r != .integer or r.integer < 0 or r.integer >= c.shares) {
        why.set("tensorfold_tp {f} is not a {d}-share cut of these expert widths", .{ std.json.fmt(v, .{}), c.shares });
        return error.BadConfig;
    }
    c.held = @intCast(r.integer);
}

/// fourOnly on config.json's text: its quantization refusals logged, other errors left to `engine`'s own checks.
pub fn fourOnlyText(gpa: std.mem.Allocator, text: []const u8, engine: []const u8) !void {
    var why: Why = .{};
    const c = parse(gpa, text, &why) catch |err| {
        if (err != error.UnsupportedQuantization) return;
        std.log.warn("{s}", .{why.text()});
        return err;
    };
    if (c.format == .mlx) try c.fourOnly(engine);
}

/// The shapes our kernels serve (Nemotron 3.5 Lightning 30B-A3B at any width parse admits): Metal's and CUDA's.
fn isType(o: std.json.ObjectMap, t: []const u8) bool {
    const v = o.get("model_type") orelse return false;
    return v == .string and std.mem.eql(u8, v.string, t);
}

fn llmConfig(o: std.json.ObjectMap) ?std.json.ObjectMap {
    const v = o.get("llm_config") orelse return null;
    return if (v == .object) v.object else null;
}

/// An eos id, or a list of them, added once each.
fn eos(c: *Config, e: std.json.Value) void {
    const items: []const std.json.Value = switch (e) {
        .integer => &.{e},
        .array => |a| a.items,
        else => &.{},
    };
    for (items) |x| if (x == .integer and !c.isEos(@intCast(x.integer)) and c.eos_count < c.eos.len) {
        c.eos[c.eos_count] = @intCast(x.integer);
        c.eos_count += 1;
    };
}

/// nemotron_h_puzzle's block_configs: a block type per layer, each MoE layer its own experts and width (one top-k).
fn puzzle(c: *Config, blocks: []const std.json.Value, why: *Why) !void {
    if (blocks.len > max_layers) {
        why.set("config.json has {d} layers; the native Nemotron engine reads up to {d}", .{ blocks.len, max_layers });
        return error.BadConfig;
    }
    c.bolt = true;
    c.experts = 0;
    c.expert_width = 0;
    for (blocks, 0..) |b, i| {
        if (b != .object) return error.BadConfig;
        c.kinds[i] = try kindOf((b.object.get("block_type") orelse return error.BadConfig).string);
        if (c.kinds[i] != .moe) continue;
        const m: Moe = .{ .experts = try int(b.object, "n_routed_experts"), .width = try int(b.object, "moe_intermediate_size") };
        const k = try int(b.object, "num_experts_per_tok");
        if (c.top_k == 0) c.top_k = k;
        if (k != c.top_k) {
            why.set("layer {d} routes to a different number of experts than the others; the engine reads one top-k", .{i});
            return error.BadConfig;
        }
        c.moe[i] = m;
        c.experts = @max(c.experts, m.experts);
        c.expert_width = @max(c.expert_width, m.width);
    }
    c.layers = blocks.len;
}

pub fn checkShapes(c: Config) !void {
    const ok = c.hidden == 2688 and c.vocab == 131072 and c.mamba_heads == 64 and c.mamba_head_dim == 64 and
        c.groups == 8 and c.state == 128 and c.conv_kernel == 4 and c.heads == 32 and c.kv_heads == 2 and
        c.head_dim == 128 and c.experts == 128 and c.top_k == 6 and c.expert_width == 1856 and c.shared_width == 3712;
    if (c.format == .modelopt) {
        std.log.err("this is a ModelOpt (FP8 and NVFP4) checkpoint: the CUDA engine serves it; Metal reads MLX checkpoints", .{});
        return error.UnsupportedQuantization;
    }
    if (!ok) {
        std.log.err("this Nemotron-H's shapes differ from the kernels built for Nemotron 3.5 Lightning 30B-A3B", .{});
        return error.UnsupportedShapes;
    }
}

pub const lightning =
    \\{"model_type": "nemotron_h", "hidden_size": 2688, "vocab_size": 131072, "mamba_num_heads": 64,
    \\ "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
    \\ "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": 128, "num_experts_per_tok": 6,
    \\ "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712, "routed_scaling_factor": 2.5,
    \\ "layer_norm_epsilon": 1e-05, "eos_token_id": [2, 11], "layers_block_type": ["mamba", "moe", "attention"],
    \\ "quantization": {"group_size": 64, "bits": 4}}
;

test {
    _ = @import("config_test.zig");
}

test {
    _ = @import("config_bolt_test.zig");
}

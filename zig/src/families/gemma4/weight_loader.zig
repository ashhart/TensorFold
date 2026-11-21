//! Gemma 4 checkpoint staging/packing and layer loading; packed types and projection pipelines live in weights.zig.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");
const plan = @import("plan.zig");
const weights = @import("weights.zig");

const Ref = weights.Ref;
const Linear = weights.Linear;
const Mlx = weights.Mlx;
const Stack = weights.Stack;
const Dense = weights.Dense;
const Weights = weights.Weights;

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Tensors left in their files at load (checkpoint_metal's `later`): the layers' linears, read once through a Stage.
pub fn later(name: []const u8, _: usize) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    const linear = std.mem.eql(u8, leaf, "weight") or std.mem.eql(u8, leaf, "scales") or std.mem.eql(u8, leaf, "biases");
    return linear and std.mem.indexOf(u8, name, ".layers.") != null and std.mem.endsWith(u8, name[0..dot], "_proj");
}

/// The load's reader: kept tensors where they are, the linears staged from their files.
const Src = struct {
    c: *const ckpt.Checkpoint,
    stage: *Stage,

    /// A tensor's dtype and shape, kept or in its file.
    fn meta(r: Src, name: []const u8) !ckpt.Deferred {
        if (r.c.tensors.get(name)) |t| return .{ .file = 0, .at = 0, .bytes = t.bytes, .dtype = t.dtype, .shape = t.shape, .rank = t.rank };
        return r.c.deferred.get(name) orelse {
            std.log.err("checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }
};

/// The linears' staging in host memory, not a Metal buffer: load makes its Metal buffers in main's order, no holes.
pub const Stage = struct {
    gpa: std.mem.Allocator,
    mem: []align(ckpt.alignment) u8 = &.{},
    used: usize = 0,

    /// A staged or kept tensor's bytes, valid until `reset`.
    pub const View = struct {
        base: [*]const u8,
        bytes: usize,
        shape: @FieldType(ckpt.Deferred, "shape"),

        pub fn host(v: View, comptime T: type) []const T {
            return @as([*]const T, @ptrCast(@alignCast(v.base)))[0 .. v.bytes / @sizeOf(T)];
        }
    };

    /// Room for the deferred tensors among `names` (a kept one is read where it is), reserved while the stage is empty.
    pub fn reserve(s: *Stage, ck: *const ckpt.Checkpoint, names: []const []const u8) !void {
        if (s.used != 0) return error.StageInUse;
        var bytes: usize = 0;
        for (names) |name| {
            const d = ck.deferred.get(name) orelse {
                if (!ck.tensors.contains(name)) return error.MissingTensor;
                continue;
            };
            bytes = std.mem.alignForward(usize, bytes, ckpt.alignment) + d.bytes;
        }
        if (s.mem.len >= bytes) return;
        s.gpa.free(s.mem);
        s.mem = &.{};
        s.mem = try s.gpa.alignedAlloc(u8, .fromByteUnits(ckpt.alignment), bytes);
    }

    /// `name`: a kept tensor where it is, a deferred one read into the stage.
    pub fn read(s: *Stage, ck: *const ckpt.Checkpoint, name: []const u8) !View {
        const d = ck.deferred.get(name) orelse {
            const t = try ck.get(name);
            return .{ .base = t.buffer.contents() + t.offset, .bytes = t.bytes, .shape = t.shape };
        };
        const at = std.mem.alignForward(usize, s.used, ckpt.alignment);
        if (at + d.bytes > s.mem.len) return error.StageFull;
        try ck.read(name, 0, s.mem[at..][0..d.bytes]);
        s.used = at + d.bytes;
        return .{ .base = s.mem[at..].ptr, .bytes = d.bytes, .shape = d.shape };
    }

    pub fn reset(s: *Stage) void {
        s.used = 0;
    }

    pub fn deinit(s: *Stage) void {
        s.gpa.free(s.mem);
        s.* = .{ .gpa = s.gpa };
    }
};

/// A member's three tensor names.
fn names3(buf: *[3][160]u8, prefix: []const u8, member: []const u8) ![3][]const u8 {
    var out: [3][]const u8 = undefined;
    for (&out, buf, [_][]const u8{ "weight", "scales", "biases" }) |*o, *b, leaf| o.* = try std.fmt.bufPrint(b, "{s}{s}.{s}", .{ prefix, member, leaf });
    return out;
}

/// A linear's three tensors copied into one aligned buffer: checkpoint tensors can sit at any byte offset in a shard.
fn mlxCopy(gpa: std.mem.Allocator, device: mtl.Device, r: Src, owned: *std.ArrayList(mtl.Buffer), prefix: []const u8, member: []const u8, width: cfg.widths.Width) !Mlx {
    var buf: [3][160]u8 = undefined;
    const ns = try names3(&buf, prefix, member);
    const k = try storedK(r, prefix, member, width);
    try r.stage.reserve(r.c, &ns);
    defer r.stage.reset();
    const w = try r.stage.read(r.c, ns[0]);
    const s = try r.stage.read(r.c, ns[1]);
    const b = try r.stage.read(r.c, ns[2]);
    const sb_at = std.mem.alignForward(usize, w.bytes, 256);
    const bb_at = sb_at + std.mem.alignForward(usize, s.bytes, 256);
    const out = try device.buffer(bb_at + b.bytes, shared);
    errdefer out.deinit();
    const dst = out.slice(u8, bb_at + b.bytes);
    @memcpy(dst[0..w.bytes], w.host(u8)[0..w.bytes]);
    @memcpy(dst[sb_at..][0..s.bytes], s.host(u8)[0..s.bytes]);
    @memcpy(dst[bb_at..][0..b.bytes], b.host(u8)[0..b.bytes]);
    try owned.append(gpa, out);
    return .{ .w = .{ .buf = out }, .s = .{ .buf = out, .off = sb_at }, .b = .{ .buf = out, .off = bb_at }, .n = @intCast(w.shape[0]), .k = @intCast(k), .bits = width.bits };
}

/// The bytes `mlxCopy` allocates for a member: its three tensors at 256-byte offsets in one buffer.
fn copyBytes(r: Src, prefix: []const u8, member: []const u8) !u64 {
    var buf: [3][160]u8 = undefined;
    const ns = try names3(&buf, prefix, member);
    const sb_at = std.mem.alignForward(usize, (try r.meta(ns[0])).bytes, 256);
    return sb_at + std.mem.alignForward(usize, (try r.meta(ns[1])).bytes, 256) + (try r.meta(ns[2])).bytes;
}

/// A member's rows of a row-path pack as core/affine_mm reads it: the same bytes, no copy (no tensor units).
fn mlxAlias(x: Linear, row: usize, n: usize) Mlx {
    const raw = x.raw.?;
    const words = x.k * x.format.bits / 32;
    const groups = x.k / x.format.group;
    return .{ .w = .{ .buf = raw.w, .off = row * words * 4 }, .s = .{ .buf = raw.scales, .off = row * groups * 2 }, .b = .{ .buf = raw.biases, .off = row * groups * 2 }, .n = @intCast(n), .k = @intCast(x.k), .bits = x.format.bits };
}

/// A stack's members as aliases of its runs' rows, in `members`' order (runs of equal widths, as `stack` made them).
fn mlxStack(r: Src, s: *const Stack, prefix: []const u8, members: []const []const u8, ws: []const cfg.widths.Width, out: []Mlx) !void {
    var buf: [3][160]u8 = undefined;
    var run: usize = 0;
    var row: usize = 0;
    for (members, ws, out, 0..) |m, w, *o, i| {
        if (i > 0 and !std.meta.eql(w, ws[i - 1])) {
            run += 1;
            row = 0;
        }
        const n = (try r.meta((try names3(&buf, prefix, m))[0])).shape[0];
        o.* = mlxAlias(s.runs[run], row, n);
        row += n;
    }
}

/// A member's K at its config width (words [N, K * bits / 32], tables [N, K / group]); other storage refused.
fn storedK(r: Src, prefix: []const u8, member: []const u8, width: cfg.widths.Width) !usize {
    var buf: [3][160]u8 = undefined;
    const ns = try names3(&buf, prefix, member);
    const w = try r.meta(ns[0]);
    const s = try r.meta(ns[1]);
    const b = try r.meta(ns[2]);
    const ok = w.dtype == .u32 and w.rank == 2 and s.rank == 2 and b.rank == 2 and s.dtype == .bf16 and b.dtype == .bf16 and
        s.shape[0] == w.shape[0] and b.shape[0] == w.shape[0] and s.shape[1] == b.shape[1] and
        w.bytes == w.shape[0] * w.shape[1] * 4 and s.bytes == s.shape[0] * s.shape[1] * 2 and b.bytes == s.bytes;
    const k = s.shape[1] * width.group;
    if (!ok or w.shape[1] * 32 != k * width.bits) {
        std.log.warn("gemma4: {s}{s} is not stored at its config width, {d}-bit in groups of {d}", .{ prefix, member, width.bits, width.group });
        return error.BadGemmaWeight;
    }
    return k;
}

/// Lean E-series loads also leave the tied table in its file until `pack` reads it once.
pub fn laterLean(name: []const u8, rank: usize) bool {
    if (later(name, rank)) return true;
    for ([_][]const u8{ "embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases" }) |t| {
        if (std.mem.endsWith(u8, name, t) and std.mem.indexOf(u8, name, plan.prefix) != null) return true;
    }
    return false;
}

fn ref(c: *const ckpt.Checkpoint, name: []const u8) !Ref {
    const t = try c.get(name);
    return .{ .buf = t.buffer, .off = t.offset };
}

/// Members of one width stacked along N into the lane layout; `interleave`: two members in alternate 16-row blocks.
fn pack(gpa: std.mem.Allocator, device: mtl.Device, r: Src, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8, width: cfg.widths.Width, interleave: bool) !Linear {
    var buf: [3][160]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (members) |m| {
        const mk = try storedK(r, prefix, m, width);
        if (k != 0 and k != mk) return error.BadGemmaWeight;
        n += (try r.meta((try names3(&buf, prefix, m))[0])).shape[0];
        k = mk;
    }
    const row_words = k * width.bits / 32;
    const groups = k / width.group;
    const words = try gpa.alloc(u32, n * row_words);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    defer gpa.free(biases);
    if (interleave and (members.len != 2 or n % 32 != 0)) return error.BadGemmaWeight;
    var row: usize = 0;
    for (members, 0..) |m, mi| {
        const ns = try names3(&buf, prefix, m);
        try r.stage.reserve(r.c, &ns);
        defer r.stage.reset();
        const w = (try r.stage.read(r.c, ns[0])).host(u32);
        const s = (try r.stage.read(r.c, ns[1])).host(u16);
        const b = (try r.stage.read(r.c, ns[2])).host(u16);
        const rows = w.len / row_words;
        if (interleave and rows != n / 2) return error.BadGemmaWeight;
        for (0..rows) |i| {
            const dst = if (interleave) 32 * (i / 16) + 16 * mi + i % 16 else row + i;
            @memcpy(words[dst * row_words ..][0..row_words], w[i * row_words ..][0..row_words]);
            @memcpy(scales[dst * groups ..][0..groups], s[i * groups ..][0..groups]);
            @memcpy(biases[dst * groups ..][0..groups], b[i * groups ..][0..groups]);
        }
        row += rows;
    }
    return weights.packArrays(gpa, device, owned, geo, words, scales, biases, .{ .w = undefined, .sb = undefined, .n = n, .k = k, .format = .{ .bits = width.bits, .group = width.group } });
}

/// Whether gate|up packs as one run in alternate 16-row blocks (lane path); the row path keeps gate rows, then up's.
pub fn interleaves(tensor_units: bool, runs: usize) bool {
    return tensor_units and runs == 1;
}

/// A q|k|v or gate|up stack: each run of members of one width packed as one linear writing its columns of the rows.
fn stack(gpa: std.mem.Allocator, device: mtl.Device, r: Src, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8, ws: []const cfg.widths.Width, glu: bool) !Stack {
    var s: Stack = .{};
    var first: usize = 0;
    while (first < members.len) {
        var end = first + 1;
        while (end < members.len and std.meta.eql(ws[end], ws[first])) end += 1;
        const whole = first == 0 and end == members.len;
        var x = try pack(gpa, device, r, owned, geo, prefix, members[first..end], ws[first], glu and whole and interleaves(device.tensorUnits(), 1));
        x.col = s.n;
        s.n += x.n;
        s.runs[s.count] = x;
        s.count += 1;
        first = end;
    }
    s.interleaved = glu and interleaves(device.tensorUnits(), s.count);
    if (s.count > 1) for (s.runs[0..s.count]) |*x| {
        x.y_stride = s.n; // each run writes its columns of the stack's rows (the lane words do not depend on it)
    };
    return s;
}

/// A bf16 matrix [n, k] read in place.
fn dense(c: *const ckpt.Checkpoint, name: []const u8, n: usize, k: usize) !Dense {
    const t = try c.get(name);
    if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != n or t.shape[1] != k) {
        std.log.warn("gemma4: {s} is not a bf16 matrix [{d}, {d}] (tools/zig/gemma4_e4b_convert.py --dense-ple keeps it so)", .{ name, n, k });
        return error.BadGemmaWeight;
    }
    return .{ .w = .{ .buf = t.buffer, .off = t.offset }, .n = n, .k = k };
}

/// plan.check's view of a loaded checkpoint: each tensor asked for at its dtype and shape; skipped ones are not read.
const Checker = struct {
    r: Src,

    pub fn expect(k: Checker, name: []const u8, dtype: ckpt.DType, shape: []const usize) !void {
        const t = try k.r.meta(name);
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.warn("gemma4: {s} is not the {s} tensor of shape {any} the plan stores", .{ name, @tagName(dtype), shape });
            return error.UnexpectedTensor;
        }
    }

    pub fn skip(_: Checker, _: []const u8) !void {}
};

/// `lean`: the checkpoint was read with `laterLean`, so its tied table does not stay resident beside the head pack.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, config: cfg.Config, lean: bool) !Weights {
    const geo = config.geo;
    const ws = &config.widths;
    var stage = Stage{ .gpa = gpa };
    defer stage.deinit();
    const r: Src = .{ .c = c, .stage = &stage };
    try plan.check(config, Checker{ .r = r });
    const pm = plan.model(config);
    var w: Weights = undefined;
    w.owned = .empty;
    w.aliased = 0;
    w.ple = null;
    errdefer w.deinit(gpa);
    const rows = !device.tensorUnits();
    var p: [96]u8 = undefined;
    var name: [160]u8 = undefined;
    for (w.layers[0..geo.layers], 0..) |*l, i| {
        const pl = plan.layer(config, i);
        const pre = try std.fmt.bufPrint(&p, plan.prefix ++ "layers.{d}.", .{i});
        const lw = ws.layers[i];
        // the planned members: a layer that reads another's cache has q alone, a global layer with k = v no v_proj
        var members: [3][]const u8 = .{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
        var widths3: [3]cfg.widths.Width = .{ lw[0], lw[1], lw[2] };
        var count: usize = 1;
        if (pl.k != null) count += 1;
        if (pl.v != null) {
            members[count] = "self_attn.v_proj";
            widths3[count] = lw[2];
            count += 1;
        }
        l.qkv = try stack(gpa, device, r, &w.owned, geo, pre, members[0..count], widths3[0..count], false);
        l.o = try pack(gpa, device, r, &w.owned, geo, pre, &.{"self_attn.o_proj"}, lw[3], false);
        l.gate_up = try stack(gpa, device, r, &w.owned, geo, pre, &.{ "mlp.gate_proj", "mlp.up_proj" }, lw[4..6], true);
        l.down = try pack(gpa, device, r, &w.owned, geo, pre, &.{"mlp.down_proj"}, lw[6], false);
        if (geo.layer_input != 0) {
            l.mlx = undefined; // prompt chunks (prompt.zig) read these; the E-series' prompts run in decode windows
        } else if (rows) { // the dense models keep main's row-pack aliases for their prompt chunks
            try mlxStack(r, &l.qkv, pre, members[0..count], widths3[0..count], l.mlx[0..count]);
            l.mlx[3] = mlxAlias(l.o, 0, l.o.n);
            try mlxStack(r, &l.gate_up, pre, &.{ "mlp.gate_proj", "mlp.up_proj" }, lw[4..6], l.mlx[4..6]);
            l.mlx[6] = mlxAlias(l.down, 0, l.down.n);
            if (cfg.global(i)) l.mlx[2] = l.mlx[1]; // global layers have no v_proj: k stands in, never read
            for (cfg.widths.module_names) |member| if (!(cfg.global(i) and std.mem.eql(u8, member, "self_attn.v_proj"))) {
                w.aliased += try copyBytes(r, pre, member);
            };
        } else for (&l.mlx, cfg.widths.module_names, lw) |*x, member, width| {
            // global layers have no v_proj: k stands in, never read
            x.* = if (cfg.global(i) and std.mem.eql(u8, member, "self_attn.v_proj")) l.mlx[1] else try mlxCopy(gpa, device, r, &w.owned, pre, member, width);
        }
        inline for (.{ "q_norm", "input_norm", "post_attention", "pre_feedforward", "post_feedforward", "scalar" }, .{ "self_attn.q_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" }) |field, t|
            @field(l, field) = try ref(c, try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t }));
        l.k_norm = if (geo.layer_input == 0 or pl.k != null) try ref(c, try std.fmt.bufPrint(&name, "{s}self_attn.k_norm.weight", .{pre})) else l.q_norm; // an E-series reader of another's cache has none
        l.ple_gate = null;
        l.ple_projection = null;
        l.ple_norm = null;
        if (pl.layer_input) |li| {
            if (li.gate.width != null or li.project.width != null) return perLayerQuantized();
            l.ple_gate = try dense(c, try std.fmt.bufPrint(&name, "{s}per_layer_input_gate.weight", .{pre}), li.gate.n, li.gate.k);
            l.ple_projection = try dense(c, try std.fmt.bufPrint(&name, "{s}per_layer_projection.weight", .{pre}), li.project.n, li.project.k);
            l.ple_norm = try ref(c, try std.fmt.bufPrint(&name, "{s}post_per_layer_input_norm.weight", .{pre}));
        }
    }
    w.head = try pack(gpa, device, r, &w.owned, geo, plan.prefix, &.{"embed_tokens"}, ws.embed, false);
    w.embed_bits = ws.embed.bits;
    // lean: the embedding gather reads the head's words, scales and biases as the checkpoint stores them
    if (lean) {
        const raw = w.head.raw orelse return error.BadGemmaWeight;
        w.embed_w = .{ .buf = raw.w };
        w.embed_s = .{ .buf = raw.scales };
        w.embed_b = .{ .buf = raw.biases };
    } else {
        w.embed_w = try ref(c, plan.prefix ++ "embed_tokens.weight");
        w.embed_s = try ref(c, plan.prefix ++ "embed_tokens.scales");
        w.embed_b = try ref(c, plan.prefix ++ "embed_tokens.biases");
    }
    w.final_norm = try ref(c, plan.prefix ++ "norm.weight");
    if (pm.layer_table) |table| {
        const mp = pm.model_project.?;
        if (table.width == null or !std.meta.eql(table.width.?, cfg.widths.fast)) { // the per-layer gather reads 4/64 codes
            std.log.warn("gemma4: embed_tokens_per_layer must be 4-bit in groups of 64", .{});
            return error.UnsupportedGemmaQuantization;
        }
        if (mp.width != null) return perLayerQuantized();
        w.ple = .{
            .table_w = try ref(c, plan.prefix ++ "embed_tokens_per_layer.weight"),
            .table_s = try ref(c, plan.prefix ++ "embed_tokens_per_layer.scales"),
            .table_b = try ref(c, plan.prefix ++ "embed_tokens_per_layer.biases"),
            .model_projection = try dense(c, plan.prefix ++ "per_layer_model_projection.weight", mp.n, mp.k),
            .projection_norm = try ref(c, plan.prefix ++ "per_layer_projection_norm.weight"),
        };
    }
    return w;
}

fn perLayerQuantized() error{UnsupportedGemmaQuantization} {
    std.log.warn("gemma4: the per-layer linears must be bf16 (tools/zig/gemma4_e4b_convert.py --dense-ple)", .{});
    return error.UnsupportedGemmaQuantization;
}

test "later: the layers' linears stay in their files; the embed, norms and scalars stay in the shards" {
    const pre = "language_model.model.layers.3.";
    for ([_][]const u8{ pre ++ "self_attn.q_proj.weight", pre ++ "self_attn.k_proj.scales", pre ++ "mlp.gate_proj.biases", pre ++ "mlp.down_proj.weight" }) |t|
        try std.testing.expect(later(t, 2));
    for ([_][]const u8{ "language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales", "language_model.model.norm.weight", pre ++ "input_layernorm.weight", pre ++ "self_attn.q_norm.weight", pre ++ "layer_scalar" }) |t|
        try std.testing.expect(!later(t, 1));
}

test "laterLean defers the tied table too, without changing the dense loader's pick" {
    const pre = "language_model.model.";
    for ([_][]const u8{ pre ++ "embed_tokens.weight", pre ++ "embed_tokens.scales", pre ++ "embed_tokens.biases" }) |name| {
        try std.testing.expect(!later(name, 2));
        try std.testing.expect(laterLean(name, 2));
    }
    try std.testing.expect(!laterLean(pre ++ "norm.weight", 1));
    try std.testing.expect(laterLean(pre ++ "layers.3.self_attn.q_proj.weight", 2));
}

test "gate|up interleaves on tensor units alone: the row path keeps gate rows, then up's, as prompt chunks read them" {
    try std.testing.expect(interleaves(true, 1));
    try std.testing.expect(!interleaves(false, 1));
    try std.testing.expect(!interleaves(true, 2));
}

/// Two 4-bit members (q, k: 16 rows of K 64) and a norm, at odd file offsets as MLX writes them.
fn fixture(gpa: std.mem.Allocator, tmp: *std.testing.TmpDir) ![:0]u8 {
    const p = "language_model.model.layers.0.";
    const header = "{\"pad\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]}," ++
        "\"" ++ p ++ "self_attn.q_proj.weight\":{\"dtype\":\"U32\",\"shape\":[16,8],\"data_offsets\":[3,515]}," ++
        "\"" ++ p ++ "self_attn.q_proj.scales\":{\"dtype\":\"BF16\",\"shape\":[16,1],\"data_offsets\":[515,547]}," ++
        "\"" ++ p ++ "self_attn.q_proj.biases\":{\"dtype\":\"BF16\",\"shape\":[16,1],\"data_offsets\":[547,579]}," ++
        "\"" ++ p ++ "self_attn.k_proj.weight\":{\"dtype\":\"U32\",\"shape\":[16,8],\"data_offsets\":[579,1091]}," ++
        "\"" ++ p ++ "self_attn.k_proj.scales\":{\"dtype\":\"BF16\",\"shape\":[16,1],\"data_offsets\":[1091,1123]}," ++
        "\"" ++ p ++ "self_attn.k_proj.biases\":{\"dtype\":\"BF16\",\"shape\":[16,1],\"data_offsets\":[1123,1155]}," ++
        "\"" ++ p ++ "input_layernorm.weight\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[1155,1171]}}";
    var data: [8 + header.len + 1171]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (data[8 + header.len ..], 0..) |*b, i| b.* = @truncate(i * 7 + 1);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "g.safetensors", .data = &data });
    return std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/g.safetensors", .{tmp.sub_path}, 0);
}

test "load: linears staged once, row-path prompt views alias the pack's bytes, and a freed linear is refused by name" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest; // a recording Metal stand-in has no device
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(gpa, &tmp);
    defer gpa.free(path);
    var c = ckpt.Checkpoint.init(gpa);
    defer c.deinit();
    try c.addFileSplit(device, path, "", "language_model.", later);
    try std.testing.expectEqual(@as(usize, 16), c.residentBytes()); // the norm alone
    const pre = "language_model.model.layers.0.";
    // the planted stale pointer: a Ref into a linear's shard bytes is refused, not read from freed memory
    try std.testing.expectError(error.DeferredTensor, ref(&c, pre ++ "self_attn.q_proj.weight"));
    _ = try ref(&c, pre ++ "input_layernorm.weight");
    var stage = Stage{ .gpa = gpa };
    defer stage.deinit();
    const r: Src = .{ .c = &c, .stage = &stage };
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (owned.items) |b| b.deinit();
        owned.deinit(gpa);
    }
    const members = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj" };
    const ws = [_]cfg.widths.Width{ cfg.widths.fast, cfg.widths.fast };
    const s = try stack(gpa, device, r, &owned, cfg.dense31b, pre, &members, &ws, true);
    try std.testing.expectEqual(device.tensorUnits(), s.interleaved);
    if (!device.tensorUnits()) {
        var views: [2]Mlx = undefined;
        try mlxStack(r, &s, pre, &members, &ws, &views);
        for (members, views) |m, v| {
            const copy = try mlxCopy(gpa, device, r, &owned, pre, m, cfg.widths.fast);
            try std.testing.expectEqual(copy.n, v.n);
            try std.testing.expectEqual(@as(u64, copy.w.buf.length()), try copyBytes(r, pre, m)); // what `aliased` counts
            for ([_][2]Ref{ .{ v.w, copy.w }, .{ v.s, copy.s }, .{ v.b, copy.b } }, [_]usize{ 512, 32, 32 }) |pair, n|
                try std.testing.expectEqualSlices(u8, (pair[1].buf.contents() + pair[1].off)[0..n], (pair[0].buf.contents() + pair[0].off)[0..n]);
        }
    }
    c.closeFiles(); // after load: the files are shut, and nothing reads a linear from them again
    try std.testing.expectError(error.ShardClosed, mlxCopy(gpa, device, r, &owned, pre, "self_attn.q_proj", cfg.widths.fast));
}

//! Borrows native scalar tensors and owns only byte-preserving affine layouts; rows and columns follow config.Shard.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const cfg = @import("config.zig");
const projection = @import("projection.zig");
const lane = @import("core").lane_projection;
const widths = @import("widths.zig");
const Width = widths.Width;
const Ref = projection.Ref;
const Linear = projection.Linear;
pub const Tensor = cp.Tensor;

pub const Gdn = struct { qkv: Linear, zba: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, kv: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct { norm: Tensor, post_norm: Tensor, gu: Linear, down: Linear, mixer: union(@import("config.zig").Kind) { linear: Gdn, attention: Attention } };

pub const Weights = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    embed: projection.Triple,
    norm: Tensor,
    head: Linear,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    runs: std.ArrayList([]projection.Run) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.allocator);
        for (w.runs.items) |r| w.allocator.free(r);
        w.runs.deinit(w.allocator);
        w.allocator.free(w.layers);
    }

    /// Whether every MLP projection is 4-bit g64 in one matrix (the Neural Engine share's format).
    pub fn uniformMlp(w: *const Weights) bool {
        for (w.layers) |l| for ([_]Linear{ l.gu, l.down }) |p| if (!uniform(p)) return false;
        return true;
    }
};

fn uniform(p: Linear) bool {
    return p.bits == 4 and p.group == 64 and p.runs.len == 0;
}

/// Modules the graph reads only through repacked copies: their tensors stay in the files until load stages them.
const repacked_modules = [_][]const u8{ ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.in_proj_b", ".linear_attn.in_proj_a", ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj" };

/// Whether `name` is a weight, scales or biases tensor of a repacked module or the LM head.
pub fn repacked(name: []const u8, _: usize) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    if (!std.mem.eql(u8, leaf, "weight") and !std.mem.eql(u8, leaf, "scales") and !std.mem.eql(u8, leaf, "biases")) return false;
    const module = name[0..dot];
    if (std.mem.eql(u8, module, "lm_head") or std.mem.endsWith(u8, module, ".lm_head")) return true;
    for (repacked_modules) |m| if (std.mem.endsWith(u8, module, m)) return true;
    return false;
}

const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    checkpoint: *const cp.Checkpoint,
    prefix: []const u8,
    owned: *std.ArrayList(mtl.Buffer),
    stage: *cp.Stage,
    runs: *std.ArrayList([]projection.Run),
    widths: *const widths.Widths,

    fn get(b: Builder, suffix: []const u8) !Tensor {
        const name = try std.mem.concat(b.allocator, u8, &.{ b.prefix, suffix });
        defer b.allocator.free(name);
        return b.checkpoint.get(name);
    }

    fn tensor(b: Builder, suffix: []const u8, dtype: cp.DType, shape: []const usize) !Tensor {
        return checked(try b.get(suffix), dtype, shape);
    }

    fn quant(b: Builder, module: []const u8, n: usize, k: usize, w: Width) !projection.Triple {
        if (k * w.bits % 32 != 0 or k % w.group != 0) return error.BadNativeTensor;
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        var tensors: [3]Tensor = undefined;
        for (suffixes, &tensors, 0..) |suffix, *t, i| {
            const name = try std.mem.concat(b.allocator, u8, &.{ module, suffix });
            defer b.allocator.free(name);
            t.* = try b.tensor(name, if (i == 0) .u32 else .bf16, &.{ n, if (i == 0) k * w.bits / 32 else k / w.group });
        }
        return .{ .weight = tensors[0], .scales = tensors[1], .biases = tensors[2], .bits = w.bits, .group = w.group };
    }

    /// One part's staged tensors, shape-checked and viewed at the rank's rows; the stage must outlive these views.
    fn viewed(p: Part, k: usize, tensors: [3]Tensor) !projection.Triple {
        const weight = try checked(tensors[0], .u32, &.{ p.n, k * p.w.bits / 32 });
        const scales = try checked(tensors[1], .bf16, &.{ p.n, k / p.w.group });
        const biases = try checked(tensors[2], .bf16, &.{ p.n, k / p.w.group });
        return .{ .weight = try rowView(weight, p.span), .scales = try rowView(scales, p.span), .biases = try rowView(biases, p.span), .bits = p.w.bits, .group = p.w.group, .gn = p.n };
    }

    /// Parts at their widths and K windows, all staged before any is viewed: the stage is one reused buffer.
    fn linear(b: Builder, parts: []const Part, k: usize) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        if (parts.len > 3) return error.BadProjectionShape;
        for (parts) |p| if (k * p.w.bits % 32 != 0 or k % p.w.group != 0) return error.BadNativeTensor;
        const a = arena.allocator();
        const names = try a.alloc([]const u8, parts.len * 3);
        for (parts, 0..) |p, i| for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, j| {
            names[i * 3 + j] = try std.mem.concat(a, u8, &.{ b.prefix, p.module, suffix });
        };
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        var triples: [3]projection.Triple = undefined;
        for (parts, triples[0..parts.len], 0..) |p, *t, i| {
            t.* = try viewed(p, k, .{
                try b.stage.read(b.checkpoint, names[i * 3]),
                try b.stage.read(b.checkpoint, names[i * 3 + 1]),
                try b.stage.read(b.checkpoint, names[i * 3 + 2]),
            });
        }
        return b.stack(triples[0..parts.len]);
    }

    /// One tensor a rank takes in blocks (GDN q | k | v); `n` is the split's key, `rows` each block's checkpoint rows.
    fn blocks(b: Builder, module: []const u8, n: usize, k: usize, w: Width, spans: []const cfg.Span, rows: []const usize) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        if (k * w.bits % 32 != 0 or k % w.group != 0) return error.BadNativeTensor;
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        const names = try arena.allocator().alloc([]const u8, suffixes.len);
        for (suffixes, 0..) |suffix, j| names[j] = try std.mem.concat(arena.allocator(), u8, &.{ b.prefix, module, suffix });
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        const weight = try checked(try b.stage.read(b.checkpoint, names[0]), .u32, &.{ n, k * w.bits / 32 });
        const scales = try checked(try b.stage.read(b.checkpoint, names[1]), .bf16, &.{ n, k / w.group });
        const biases = try checked(try b.stage.read(b.checkpoint, names[2]), .bf16, &.{ n, k / w.group });
        var triples: [3]projection.Triple = undefined;
        if (spans.len > triples.len or rows.len != spans.len) return error.BadProjectionShape;
        for (spans, rows, triples[0..spans.len]) |span, gn, *t| {
            t.* = .{ .weight = try rowView(weight, span), .scales = try rowView(scales, span), .biases = try rowView(biases, span), .bits = w.bits, .group = w.group, .gn = gn, .whole = n };
        }
        return b.stack(triples[0..spans.len]);
    }

    /// `conv1d.weight`'s row blocks fused in the same q | k | v order, as one bf16 copy (the file's is not repacked).
    fn conv(b: Builder, module: []const u8, kernel: usize, n: usize, spans: []const cfg.Span) !Tensor {
        const t = try b.tensor(module, .bf16, &.{ n, kernel, 1 });
        var at: usize = 0;
        for (spans) |span| {
            if (span.len == 0 or span.from != at) break;
            at = span.end();
        }
        if (at == n) return t; // the blocks are the whole tensor in order: no copy
        var rows: usize = 0;
        for (spans) |span| rows += span.len;
        const out = try b.device.buffer(rows * kernel * @sizeOf(u16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        b.owned.append(b.allocator, out) catch |err| {
            out.deinit();
            return err;
        };
        const from = t.host(u16);
        const to = out.slice(u16, rows * kernel);
        var base: usize = 0;
        for (spans) |span| {
            if (span.end() > n) return error.BadNativeTensor;
            @memcpy(to[base * kernel ..][0 .. span.len * kernel], from[span.from * kernel ..][0 .. span.len * kernel]);
            base += span.len;
        }
        var view = t;
        view.buffer = out;
        view.offset = 0;
        view.bytes = rows * kernel * @sizeOf(u16);
        view.shape[0] = rows;
        return view;
    }

    /// Fused members of one width take one matrix as before; a mixed stack becomes one matrix a member.
    fn stack(b: Builder, parts: []const projection.Triple) !Linear {
        var ws: [3]Width = undefined;
        for (parts, 0..) |p, i| ws[i] = .{ .bits = p.bits, .group = @intCast(p.group) };
        var buf: [3][2]usize = undefined;
        const list = widths.runs(ws[0..parts.len], &buf);
        if (list.len == 1) return projection.prepare(b.device, b.allocator, b.owned, parts, 32);
        const runs = try b.allocator.alloc(projection.Run, list.len);
        b.runs.append(b.allocator, runs) catch |err| {
            b.allocator.free(runs);
            return err;
        };
        var first: u32 = 0;
        for (list, runs) |r, *run| {
            run.* = .{ .linear = try projection.prepare(b.device, b.allocator, b.owned, parts[r[0]..][0..r[1]], 32), .first = first };
            first += run.linear.n;
        }
        var fused = runs[0].linear;
        fused.n = first;
        fused.gn = 0;
        for (runs) |run| fused.gn += run.linear.gn;
        fused.raw = null;
        fused.runs = runs;
        return fused;
    }

    fn layer(b: Builder, s: *const cfg.Shard, i: usize) !Layer {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        const Name = struct {
            fn full(allocator: std.mem.Allocator, p: []const u8, leaf: []const u8) ![]const u8 {
                return std.mem.concat(allocator, u8, &.{ p, leaf });
            }
        };
        const c = s.local;
        const g = s.global;
        const w = b.widths.layers[i];
        const M = widths.Member;
        const qkv = s.qkvSpans();
        const qkvRows = [3]usize{ g.k_heads * g.dk, g.k_heads * g.dk, g.gdnValueDim() };
        const convRows = s.convSpans();
        const mixer: @FieldType(Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.blocks(try Name.full(a, prefix, "linear_attn.in_proj_qkv"), g.gdnQkvDim(), g.hidden, w[@backingInt(M.qkv)], &qkv, &qkvRows),
                .zba = try b.linear(&.{
                    .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_z"), .n = g.gdnValueDim(), .w = w[@backingInt(M.z)], .span = s.zRows() },
                    .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_b"), .n = g.v_heads, .w = w[@backingInt(M.b)], .span = s.abRows() },
                    .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_a"), .n = g.v_heads, .w = w[@backingInt(M.a)], .span = s.abRows() },
                }, g.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.out_proj"), .n = g.hidden, .w = w[@backingInt(M.out)], .span = s.outputRows() }}, g.gdnValueDim()),
                .conv = try b.conv(try Name.full(a, prefix, "linear_attn.conv1d.weight"), g.conv_kernel, g.gdnQkvDim(), &convRows),
                .a_log = try rowView(try decayLog(try b.get(try Name.full(a, prefix, "linear_attn.A_log")), &.{g.v_heads}), s.valueRows()),
                .dt = try rowView(try b.tensor(try Name.full(a, prefix, "linear_attn.dt_bias"), .bf16, &.{g.v_heads}), s.valueRows()),
                .norm = try b.tensor(try Name.full(a, prefix, "linear_attn.norm.weight"), .bf16, &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.q_proj"), .n = g.qDim(), .w = w[@backingInt(M.q)], .span = s.qRows() }}, g.hidden),
                .kv = try b.linear(&.{
                    .{ .module = try Name.full(a, prefix, "self_attn.k_proj"), .n = g.kvDim(), .w = w[@backingInt(M.k)], .span = s.kvRows() },
                    .{ .module = try Name.full(a, prefix, "self_attn.v_proj"), .n = g.kvDim(), .w = w[@backingInt(M.v)], .span = s.kvRows() },
                }, g.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.o_proj"), .n = g.hidden, .w = w[@backingInt(M.o)], .span = s.outputRows() }}, g.heads * g.head_dim),
                .q_norm = try b.tensor(try Name.full(a, prefix, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim}),
                .k_norm = try b.tensor(try Name.full(a, prefix, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim}),
            } },
        };
        const gu = try b.linear(&.{
            .{ .module = try Name.full(a, prefix, "mlp.gate_proj"), .n = g.intermediate, .w = w[@backingInt(M.gate)], .span = s.intermediateRows() },
            .{ .module = try Name.full(a, prefix, "mlp.up_proj"), .n = g.intermediate, .w = w[@backingInt(M.up)], .span = s.intermediateRows() },
        }, g.hidden);
        const down = try b.linear(&.{.{ .module = try Name.full(a, prefix, "mlp.down_proj"), .n = g.hidden, .w = w[@backingInt(M.down)], .span = s.outputRows() }}, g.intermediate);
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = gu, .down = down, .mixer = mixer };
    }
};

const Part = struct {
    /// The module's checkpoint name, without the shard's namespace prefix.
    module: []const u8,
    /// The rows the checkpoint's tensor has, before this rank takes its span.
    n: usize,
    w: Width,
    /// The rows this rank takes: the whole tensor unless the rank splits this matrix.
    span: cfg.Span = .{},
};

/// Rows `[span.from, end)` of `t`: rows are contiguous, so the same buffer at a later offset and a shorter shape.
fn rowView(t: Tensor, span: cfg.Span) !Tensor {
    if (span.len == 0) return t;
    if (span.from == 0 and span.len == t.shape[0]) return t;
    if (span.end() > t.shape[0]) return error.BadNativeTensor;
    var row: usize = 1;
    for (t.shape[1..t.rank]) |d| row *= d;
    const elem = t.dtype.size();
    const at = span.from * row * elem;
    if (at % elem != 0) return error.BadNativeTensor;
    var view = t;
    view.offset = t.offset + at;
    view.bytes = row * span.len * elem;
    view.shape[0] = span.len;
    return view;
}

fn checked(t: Tensor, dtype: cp.DType, shape: []const usize) !Tensor {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

/// A_log as stored: bf16, or f32 (mlx-vlm and mlx_lm conversions); the GDN kernels read either through weight_flags.
fn decayLog(t: Tensor, shape: []const usize) !Tensor {
    return checked(t, if (t.dtype == .f32) .f32 else .bf16, shape);
}

/// Graph from `checkpoint`; each Linear is staged, repacked, dropped. `s` is the rank's share, `local` from `global`.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, s: *const cfg.Shard, ws: *const widths.Widths) !Weights {
    const c = s.local;
    const g = s.global;
    const prefix: []const u8 = if (checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else if (checkpoint.has("model.embed_tokens.weight")) "" else return error.MissingTextNamespace;
    if (checkpoint.has("language_model.model.embed_tokens.weight") and checkpoint.has("model.embed_tokens.weight")) return error.AmbiguousTextNamespace;
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |b| b.deinit();
        owned.deinit(allocator);
    }
    var runs: std.ArrayList([]projection.Run) = .empty;
    errdefer {
        for (runs.items) |r| allocator.free(r);
        runs.deinit(allocator);
    }
    const layers = try allocator.alloc(Layer, c.layers);
    errdefer allocator.free(layers);
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    const b = Builder{ .allocator = allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &owned, .stage = &stage, .runs = &runs, .widths = ws };
    for (layers, 0..) |*l, i| l.* = try b.layer(s, i);
    const embed = try b.quant("model.embed_tokens", g.vocab, g.hidden, ws.embed);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const head = try b.linear(&.{.{ .module = "lm_head", .n = g.vocab, .w = ws.head, .span = s.vocabRows() }}, g.hidden);
    if (device.tensorUnits()) try laneOrder(allocator, device, layers, head);
    return .{ .allocator = allocator, .layers = layers, .embed = embed, .norm = norm, .head = head, .owned = owned, .runs = runs };
}

/// Tensor-unit projections leave load in lane_projection.reg_layout, the order quant_gpu's reg kernel reads.
fn laneOrder(a: std.mem.Allocator, device: mtl.Device, layers: []const Layer, head: Linear) !void {
    var jobs: std.ArrayList(lane.Order) = .empty;
    defer jobs.deinit(a);
    for (layers) |l| {
        const list = switch (l.mixer) {
            .linear => |g| [_]Linear{ g.qkv, g.zba, g.out, l.gu, l.down },
            .attention => |t| [_]Linear{ t.q, t.kv, t.out, l.gu, l.down },
        };
        for (list) |p| try regJobs(a, &jobs, p);
    }
    try regJobs(a, &jobs, head);
    try lane.regOrder(device, jobs.items);
}

/// Only 4-bit g64 matrices take the reg kernel; other widths keep lane_projection.weight_layout.
fn regJobs(a: std.mem.Allocator, jobs: *std.ArrayList(lane.Order), p: Linear) std.mem.Allocator.Error!void {
    for (p.runs) |r| try regJobs(a, jobs, r.linear);
    if (p.runs.len == 0 and p.bits == 4 and p.group == 64 and p.n % 32 == 0) try jobs.append(a, .{ .w = .{ .buf = p.words.buffer, .off = p.words.offset }, .n = p.n, .k = p.k });
}

pub fn ref(t: Tensor) Ref {
    return .{ .buffer = t.buffer, .offset = t.offset };
}

test "repacked names the modules the graph reads only through owned copies" {
    try std.testing.expect(repacked("language_model.model.layers.3.mlp.gate_proj.weight", 2));
    try std.testing.expect(repacked("model.layers.3.self_attn.o_proj.scales", 2));
    try std.testing.expect(repacked("language_model.model.layers.0.linear_attn.out_proj.biases", 2));
    try std.testing.expect(repacked("language_model.lm_head.weight", 2));
    try std.testing.expect(repacked("lm_head.scales", 2));
    try std.testing.expect(!repacked("language_model.model.embed_tokens.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.conv1d.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.3.self_attn.q_norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.A_log", 2));
}

test "A_log loads in bf16 or in f32, as mlx-vlm conversions store it, and refuses other dtypes and shapes" {
    var t: Tensor = .{ .buffer = undefined, .offset = 0, .bytes = 32 * 4, .dtype = .f32, .rank = 1 };
    t.shape[0] = 32;
    try std.testing.expectEqual(cp.DType.f32, (try decayLog(t, &.{32})).dtype);
    t.dtype = .bf16;
    t.bytes = 32 * 2;
    try std.testing.expectEqual(cp.DType.bf16, (try decayLog(t, &.{32})).dtype);
    t.dtype = .f16;
    try std.testing.expectError(error.BadNativeTensor, decayLog(t, &.{32}));
    t.dtype = .f32;
    t.bytes = 32 * 4;
    try std.testing.expectError(error.BadNativeTensor, decayLog(t, &.{48}));
}

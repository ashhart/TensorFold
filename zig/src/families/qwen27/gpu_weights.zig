//! The Metal graph borrows native scalar tensors and owns only byte-preserving affine layouts.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const Config = @import("config.zig").Config;
const projection = @import("projection.zig");
const lane = @import("core").lane_projection;
const row = @import("core").row_projection;
const widths = @import("widths.zig");
const Width = widths.Width;
const Ref = projection.Ref;
const Linear = projection.Linear;
pub const Tensor = cp.Tensor;

pub const Gdn = struct { qkv: Linear, zba: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, kv: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
/// The routed block: the router (experts, shared gate, zero rows), the stacked experts, the shared expert's matrices.
pub const Moe = struct { router: row.Weights, gate: row.Experts, up: row.Experts, down: row.Experts, shared_gate: row.Weights, shared_up: row.Weights, shared_down: row.Weights };
/// With experts, `gu` and `down` are the shared expert's.
pub const Layer = struct { norm: Tensor, post_norm: Tensor, gu: Linear, down: Linear, moe: ?Moe = null, mixer: union(@import("config.zig").Kind) { linear: Gdn, attention: Attention } };

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

    /// `experts` [n, k] matrices stacked in `module` at width `w`, borrowed where the files packed them.
    fn experts(b: Builder, module: []const u8, count: usize, n: usize, k: usize, w: Width, repeat: usize) !row.Experts {
        const t = try b.quantAt(module, &.{count}, n, k, w);
        const out: row.Experts = .{ .one = weightsOf(t, n, k), .experts = count, .repeat = repeat };
        try out.validate();
        return out;
    }

    /// A quantized module's three tensors of `lead` ++ [n, packed k] shape.
    fn quantAt(b: Builder, module: []const u8, lead: []const usize, n: usize, k: usize, w: Width) !projection.Triple {
        if (k * w.bits % 32 != 0 or k % w.group != 0) return error.BadNativeTensor;
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        var tensors: [3]Tensor = undefined;
        var shape: [3]usize = undefined;
        @memcpy(shape[0..lead.len], lead);
        shape[lead.len] = n;
        for (suffixes, &tensors, 0..) |suffix, *t, i| {
            const name = try std.mem.concat(b.allocator, u8, &.{ module, suffix });
            defer b.allocator.free(name);
            shape[lead.len + 1] = if (i == 0) k * w.bits / 32 else k / w.group;
            t.* = try b.tensor(name, if (i == 0) .u32 else .bf16, shape[0 .. lead.len + 2]);
        }
        return .{ .weight = tensors[0], .scales = tensors[1], .biases = tensors[2], .bits = w.bits, .group = w.group };
    }

    /// The router's [experts, k] rows and the shared expert's gate row in one owned matrix of widths.routerRows rows.
    fn router(b: Builder, prefix: []const u8, c: Config, w: Width, shared: Width) !row.Weights {
        // one matrix holds both: widths.admit refuses a shared gate quantized unlike its router
        if (shared.bits != w.bits or shared.group != w.group) return error.UnsupportedRouterFormat;
        const k = c.hidden;
        const router_name = try std.mem.concat(b.allocator, u8, &.{ prefix, widths.leaf(c, .router) });
        defer b.allocator.free(router_name);
        const gate_name = try std.mem.concat(b.allocator, u8, &.{ prefix, widths.leaf(c, .shared_gate) });
        defer b.allocator.free(gate_name);
        const r = try b.quantAt(router_name, &.{}, c.experts, k, w);
        const g = try b.quantAt(gate_name, &.{}, 1, k, shared);
        const n = widths.routerRows(c.experts);
        const words = k * w.bits / 32 * 4;
        const groups = k / w.group * 2;
        const parts = [3][2]usize{ .{ words, 0 }, .{ groups, words * n }, .{ groups, words * n + groups * n } };
        const buffer = try b.device.buffer(parts[2][1] + groups * n, mtl.ResourceOptions.shared);
        errdefer buffer.deinit();
        const bytes = buffer.slice(u8, parts[2][1] + groups * n);
        @memset(bytes, 0);
        for ([_]Tensor{ r.weight, r.scales, r.biases }, [_]Tensor{ g.weight, g.scales, g.biases }, parts) |rt, gt, part| {
            @memcpy(bytes[part[1]..][0 .. part[0] * c.experts], rt.host(u8)[0 .. part[0] * c.experts]);
            @memcpy(bytes[part[1] + part[0] * c.experts ..][0..part[0]], gt.host(u8)[0..part[0]]);
        }
        try b.owned.append(b.allocator, buffer);
        const out: row.Weights = .{ .w = buffer, .w_off = parts[0][1], .scales = buffer, .s_off = parts[1][1], .biases = buffer, .b_off = parts[2][1], .n = n, .k = k, .group = w.group, .bits = w.bits };
        try out.validate();
        return out;
    }

    /// `parts` (stacked in order, `n` rows of `k` each, at their widths) staged from the files and repacked.
    fn linear(b: Builder, parts: []const Part, k: usize) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const names = try arena.allocator().alloc([]const u8, parts.len * 3);
        for (parts, 0..) |p, i| for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, j| {
            names[i * 3 + j] = try std.mem.concat(arena.allocator(), u8, &.{ b.prefix, p.module, suffix });
        };
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        var triples: [3]projection.Triple = undefined;
        if (parts.len > triples.len) return error.BadProjectionShape;
        for (parts, triples[0..parts.len], 0..) |p, *t, i| {
            if (k * p.w.bits % 32 != 0 or k % p.w.group != 0) return error.BadNativeTensor;
            t.* = .{
                .weight = try checked(try b.stage.read(b.checkpoint, names[i * 3]), .u32, &.{ p.n, k * p.w.bits / 32 }),
                .scales = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 1]), .bf16, &.{ p.n, k / p.w.group }),
                .biases = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 2]), .bf16, &.{ p.n, k / p.w.group }),
                .bits = p.w.bits,
                .group = p.w.group,
            };
        }
        return b.stack(triples[0..parts.len]);
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
        fused.raw = null;
        fused.runs = runs;
        return fused;
    }

    fn layer(b: Builder, c: Config, i: usize) !Layer {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        const Name = struct {
            fn full(allocator: std.mem.Allocator, p: []const u8, leaf: []const u8) ![]const u8 {
                return std.mem.concat(allocator, u8, &.{ p, leaf });
            }
        };
        const w = b.widths.layers[i];
        const M = widths.Member;
        const mixer: @FieldType(Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.in_proj_qkv"), .n = c.gdnQkvDim(), .w = w[@backingInt(M.qkv)] }}, c.hidden),
                .zba = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_z"), .n = c.gdnValueDim(), .w = w[@backingInt(M.z)] }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_b"), .n = c.v_heads, .w = w[@backingInt(M.b)] }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_a"), .n = c.v_heads, .w = w[@backingInt(M.a)] } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.out_proj"), .n = c.hidden, .w = w[@backingInt(M.out)] }}, c.gdnValueDim()),
                .conv = try b.tensor(try Name.full(a, prefix, "linear_attn.conv1d.weight"), .bf16, &.{ c.gdnQkvDim(), c.conv_kernel, 1 }),
                .a_log = try b.tensor(try Name.full(a, prefix, "linear_attn.A_log"), .bf16, &.{c.v_heads}),
                .dt = try b.tensor(try Name.full(a, prefix, "linear_attn.dt_bias"), .bf16, &.{c.v_heads}),
                .norm = try b.tensor(try Name.full(a, prefix, "linear_attn.norm.weight"), .bf16, &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.q_proj"), .n = c.qDim(), .w = w[@backingInt(M.q)] }}, c.hidden),
                .kv = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "self_attn.k_proj"), .n = c.kvDim(), .w = w[@backingInt(M.k)] }, .{ .module = try Name.full(a, prefix, "self_attn.v_proj"), .n = c.kvDim(), .w = w[@backingInt(M.v)] } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.o_proj"), .n = c.hidden, .w = w[@backingInt(M.o)] }}, c.heads * c.head_dim),
                .q_norm = try b.tensor(try Name.full(a, prefix, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim}),
                .k_norm = try b.tensor(try Name.full(a, prefix, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim}),
            } },
        };
        const shared: ?[3]projection.Triple = if (!c.moe()) null else .{
            try b.quant(try Name.full(a, prefix, widths.leaf(c, .gate)), c.intermediate, c.hidden, w[@backingInt(M.gate)]),
            try b.quant(try Name.full(a, prefix, widths.leaf(c, .up)), c.intermediate, c.hidden, w[@backingInt(M.up)]),
            try b.quant(try Name.full(a, prefix, widths.leaf(c, .down)), c.hidden, c.intermediate, w[@backingInt(M.down)]),
        };
        const moe: ?Moe = if (!c.moe()) null else .{
            .shared_gate = weightsOf(shared.?[0], c.intermediate, c.hidden),
            .shared_up = weightsOf(shared.?[1], c.intermediate, c.hidden),
            .shared_down = weightsOf(shared.?[2], c.hidden, c.intermediate),
            .router = try b.router(prefix, c, w[@backingInt(M.router)], w[@backingInt(M.shared_gate)]),
            .gate = try b.experts(try Name.full(a, prefix, widths.leaf(c, .egate)), c.experts, c.moe_intermediate, c.hidden, w[@backingInt(M.egate)], c.top_k + 1),
            .up = try b.experts(try Name.full(a, prefix, widths.leaf(c, .eup)), c.experts, c.moe_intermediate, c.hidden, w[@backingInt(M.eup)], c.top_k + 1),
            .down = try b.experts(try Name.full(a, prefix, widths.leaf(c, .edown)), c.experts, c.hidden, c.moe_intermediate, w[@backingInt(M.edown)], 1),
        };
        // with experts the shared expert's lane layouts come from its resident tensors, else the dense MLP's are staged
        const gu = if (shared) |t| try b.stack(&.{ t[0], t[1] }) else try b.linear(&.{ .{ .module = try Name.full(a, prefix, widths.leaf(c, .gate)), .n = c.intermediate, .w = w[@backingInt(M.gate)] }, .{ .module = try Name.full(a, prefix, widths.leaf(c, .up)), .n = c.intermediate, .w = w[@backingInt(M.up)] } }, c.hidden);
        const down = if (shared) |t| try b.stack(&.{t[2]}) else try b.linear(&.{.{ .module = try Name.full(a, prefix, widths.leaf(c, .down)), .n = c.hidden, .w = w[@backingInt(M.down)] }}, c.intermediate);
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = gu, .down = down, .moe = moe, .mixer = mixer };
    }
};

const Part = struct { module: []const u8, n: usize, w: Width };

fn checked(t: Tensor, dtype: cp.DType, shape: []const usize) !Tensor {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

/// The graph from `checkpoint`, whose `repacked` tensors stay in their files: each Linear is staged, repacked, dropped.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, c: Config, ws: *const widths.Widths) !Weights {
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
    for (layers, 0..) |*l, i| l.* = try b.layer(c, i);
    const embed = try b.quant("model.embed_tokens", c.vocab, c.hidden, ws.embed);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const head = try b.linear(&.{.{ .module = "lm_head", .n = c.vocab, .w = ws.head }}, c.hidden);
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

/// A borrowed quantized tensor as the row kernels' matrix.
fn weightsOf(t: projection.Triple, n: usize, k: usize) row.Weights {
    return .{ .w = t.weight.buffer, .w_off = t.weight.offset, .scales = t.scales.buffer, .s_off = t.scales.offset, .biases = t.biases.buffer, .b_off = t.biases.offset, .n = n, .k = k, .group = t.group, .bits = t.bits };
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

//! Borrows native scalar tensors and owns only byte-preserving affine layouts; rows and columns follow config.Shard.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const cfg = @import("config.zig");
const projection = @import("projection.zig");
const lane = @import("core").lane_projection;
const row_gpu = @import("core").row_projection;
const widths = @import("widths.zig");
const tfq3_mod = @import("tfq3.zig");
const repack = @import("gpu_repacked.zig");
const Width = widths.Width;
const Ref = projection.Ref;
const Linear = projection.Linear;
pub const Tensor = cp.Tensor;

pub const Gdn = struct { qkv: Linear, zba: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, kv: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
/// The routed block: the router (experts, shared gate, zero rows), the stacked experts, the shared expert's matrices.
pub const Moe = struct { router: row_gpu.Weights, gate: row_gpu.Experts, up: row_gpu.Experts, down: row_gpu.Experts, shared_gate: row_gpu.Weights, shared_up: row_gpu.Weights, shared_down: row_gpu.Weights };
/// With experts, `gu` and `down` are the shared expert's.
pub const Layer = struct { norm: Tensor, post_norm: Tensor, gu: Linear, down: Linear, moe: ?Moe = null, mixer: union(@import("config.zig").Kind) { linear: Gdn, attention: Attention } };

/// The 27B's MTP draft head (`mtp-4bit.safetensors` beside the checkpoint); its norms hold 1 + w, as the 35B's do.
pub const Mtp = struct { norm_embed: Tensor, norm_hidden: Tensor, fc: Linear, layer: Layer, norm: Tensor };

/// The draft head's file, beside the checkpoint's shards (as CUDA's 35B loader names it).
pub const mtp_file = "mtp-4bit.safetensors";

pub const Weights = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    physical_layers: []Layer = &.{},
    embed: projection.Triple,
    embed_bf16: ?Tensor = null,
    norm: Tensor,
    head: Linear,
    /// The draft head, once loadMtp read it; embed and head stay the target's.
    mtp: ?Mtp = null,
    /// Optional zero-copy TFQ3 weight store (mmap + no-copy GPU views), freed after owned buffers.
    tfq3: ?tfq3_mod.Store = null,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    runs: std.ArrayList([]projection.Run) = .empty,
    sn_runs: std.ArrayList([]@import("loop.zig").SnRun) = .empty,
    mhc: []@import("loop.zig").MhcLayer = &.{},

    pub fn deinit(w: *Weights) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.allocator);
        if (w.tfq3) |*t| t.deinit();
        for (w.runs.items) |r| w.allocator.free(r);
        w.runs.deinit(w.allocator);
        for (w.sn_runs.items) |r| w.allocator.free(r);
        w.sn_runs.deinit(w.allocator);
        if (w.mhc.len > 0) w.allocator.free(w.mhc);
        if (w.physical_layers.len > 0) w.allocator.free(w.physical_layers);
        w.allocator.free(w.layers);
    }

    /// Whether every MLP projection is 4-bit g64 in one matrix (the Neural Engine share's format).
    pub fn uniformMlp(w: *const Weights) bool {
        for (w.layers) |l| for ([_]Linear{ l.gu, l.down }) |p| if (!uniform(p)) return false;
        return true;
    }
};

fn uniform(p: Linear) bool {
    return p.sn == null and p.tfq3 == null and p.bf16 == null and p.bits == 4 and p.group == 64 and p.runs.len == 0;
}

/// Whether `name` is a weight, scales or biases tensor of a repacked module or the LM head.
pub fn repacked(name: []const u8, count: usize) bool {
    return repack.repacked(name, count);
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
    tfq3: ?*const tfq3_mod.Store = null,

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

    /// `experts` [n, k] matrices stacked in `module` at width `w`, borrowed where the files packed them.
    fn experts(b: Builder, module: []const u8, count: usize, n: usize, k: usize, w: Width, repeat: usize) !row_gpu.Experts {
        const t = try b.quantAt(module, &.{count}, n, k, w);
        const out: row_gpu.Experts = .{ .one = weightsOf(t, n, k), .experts = count, .repeat = repeat };
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
    fn router(b: Builder, prefix: []const u8, c: cfg.Config, w: Width, shared: Width) !row_gpu.Weights {
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
        const out: row_gpu.Weights = .{ .w = buffer, .w_off = parts[0][1], .scales = buffer, .s_off = parts[1][1], .biases = buffer, .b_off = parts[2][1], .n = n, .k = k, .group = w.group, .bits = w.bits };
        try out.validate();
        return out;
    }

    /// Parts at their widths and K windows, all staged before any is viewed: the stage is one reused buffer.
    fn linear(b: Builder, parts: []const Part, k: usize) !Linear {
        for (parts) |p| if (widths.isTfq3(p.w) or widths.isBf16(p.w)) return b.linearMixed(parts, k);
        return b.linearAffine(parts, k);
    }

    fn linearAffine(b: Builder, parts: []const Part, k: usize) !Linear {
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

    fn tfq3Found(b: Builder, p: Part, k: usize) !tfq3_mod.Found {
        if (p.span.len != 0 and (p.span.from != 0 or p.span.len != p.n)) return error.UnsupportedQuantization;
        const store = b.tfq3 orelse return error.UnsupportedQuantization;
        const f = store.find(p.module, p.n, k) orelse return error.BadNativeTensor;
        if (f.bits != p.w.bits) return error.BadNativeTensor;
        return f;
    }

    fn bf16Linear(b: Builder, parts: []const Part, k: usize) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const names = try a.alloc([]const u8, parts.len);
        for (parts, 0..) |p, i| names[i] = try std.mem.concat(a, u8, &.{ b.prefix, p.module, ".weight" });
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        var slices: [3][]const u8 = undefined;
        var rows_total: usize = 0;
        var gn_total: usize = 0;
        for (parts, 0..) |p, i| {
            const v = try rowView(try checked(try b.stage.read(b.checkpoint, names[i]), .bf16, &.{ p.n, k }), p.span);
            slices[i] = v.host(u8)[0..v.bytes];
            rows_total += v.shape[0];
            gn_total += p.n;
        }
        return tfq3_mod.packBf16Linear(b.allocator, b.device, b.owned, slices[0..parts.len], rows_total, gn_total, k);
    }

    /// Mixed TFQ3/BF16/affine stacks: fuses contiguous TFQ3 pairs (gate+up, k+v) and adjacent BF16/affine runs.
    fn linearMixed(b: Builder, parts: []const Part, k: usize) !Linear {
        if (parts.len == 0 or parts.len > 3) return error.BadProjectionShape;
        if (parts.len == 2 and widths.isTfq3(parts[0].w) and widths.isTfq3(parts[1].w)) {
            const f0 = try b.tfq3Found(parts[0], k);
            const f1 = try b.tfq3Found(parts[1], k);
            if (tfq3_mod.check.canFusePair(f0.part, f1.part)) return f0.linear(parts[0].n + parts[1].n, k, @intCast(parts[0].n));
        }
        var built: [3]projection.Run = undefined;
        var count: usize = 0;
        var first: u32 = 0;
        var idx: usize = 0;
        while (idx < parts.len) {
            const w0 = parts[idx].w;
            var j = idx + 1;
            const lin = if (widths.isTfq3(w0))
                (try b.tfq3Found(parts[idx], k)).linear(parts[idx].n, k, 0)
            else if (widths.isBf16(w0)) blk: {
                while (j < parts.len and widths.isBf16(parts[j].w)) : (j += 1) {}
                break :blk try b.bf16Linear(parts[idx..j], k);
            } else blk: {
                while (j < parts.len and std.meta.eql(parts[j].w, w0)) : (j += 1) {}
                break :blk try b.linearAffine(parts[idx..j], k);
            };
            built[count] = .{ .linear = lin, .first = first };
            first += lin.n;
            count += 1;
            idx = j;
        }
        if (count == 1) return built[0].linear;
        return b.wrapRuns(built[0..count], first);
    }

    fn wrapRuns(b: Builder, items: []const projection.Run, first: u32) !Linear {
        const runs = try b.allocator.dupe(projection.Run, items);
        b.runs.append(b.allocator, runs) catch |err| {
            b.allocator.free(runs);
            return err;
        };
        var fused = runs[0].linear;
        fused.n = first;
        fused.gn = 0;
        for (runs) |r| fused.gn += r.linear.gn;
        fused.raw = null;
        fused.tfq3 = null;
        fused.bf16 = null;
        fused.runs = runs;
        return fused;
    }

    /// One tensor a rank takes in blocks (GDN q | k | v); `n` is the split's key, `rows` each block's checkpoint rows.
    fn blocks(b: Builder, module: []const u8, n: usize, k: usize, w: Width, spans: []const cfg.Span, rows: []const usize) !Linear {
        if (widths.isTfq3(w)) {
            var total: usize = 0;
            for (spans) |span| {
                if (span.from != total) return error.UnsupportedQuantization;
                total = span.end();
            }
            if (total != n) return error.UnsupportedQuantization;
            return (try b.tfq3Found(.{ .module = module, .n = n, .w = w }, k)).linear(n, k, 0);
        }
        if (widths.isBf16(w)) return b.bf16Linear(&.{.{ .module = module, .n = n, .w = w }}, k);
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
        var built: [3]projection.Run = undefined;
        var first: u32 = 0;
        for (list, built[0..list.len]) |r, *run| {
            run.* = .{ .linear = try projection.prepare(b.device, b.allocator, b.owned, parts[r[0]..][0..r[1]], 32), .first = first };
            first += run.linear.n;
        }
        return b.wrapRuns(built[0..list.len], first);
    }

    fn layer(b: Builder, s: *const cfg.Shard, i: usize) !Layer {
        var name: [32]u8 = undefined;
        return b.block(s, try std.fmt.bufPrint(&name, "model.layers.{d}.", .{i}), s.local.kind(i), b.widths.layers[i]);
    }

    /// One decoder layer under `prefix` (a target layer's, or the draft head's `mtp.layers.0.`) at widths `w`.
    fn block(b: Builder, s: *const cfg.Shard, prefix: []const u8, kind: cfg.Kind, w: [widths.members]widths.Width) !Layer {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const Name = struct {
            fn full(allocator: std.mem.Allocator, p: []const u8, leaf: []const u8) ![]const u8 {
                return std.mem.concat(allocator, u8, &.{ p, leaf });
            }
        };
        const c = s.local;
        const g = s.global;
        const M = widths.Member;
        const qkv = s.qkvSpans();
        const qkvRows = [3]usize{ g.k_heads * g.dk, g.k_heads * g.dk, g.gdnValueDim() };
        const convRows = s.convSpans();
        const mixer: @FieldType(Layer, "mixer") = switch (kind) {
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
        const gu = if (shared) |t| try b.stack(&.{ t[0], t[1] }) else try b.linear(&.{
            .{ .module = try Name.full(a, prefix, "mlp.gate_proj"), .n = g.intermediate, .w = w[@backingInt(M.gate)], .span = s.intermediateRows() },
            .{ .module = try Name.full(a, prefix, "mlp.up_proj"), .n = g.intermediate, .w = w[@backingInt(M.up)], .span = s.intermediateRows() },
        }, g.hidden);
        const down = if (shared) |t| try b.stack(&.{t[2]}) else try b.linear(&.{.{ .module = try Name.full(a, prefix, "mlp.down_proj"), .n = g.hidden, .w = w[@backingInt(M.down)], .span = s.outputRows() }}, g.intermediate);
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = gu, .down = down, .moe = moe, .mixer = mixer };
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
    return loadWithTfq3(allocator, device, checkpoint, s, ws, null);
}

/// `load` taking ownership of an optional validated TFQ3 store (`tfq3_store.*` is cleared once transferred).
pub fn loadWithTfq3(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, s: *const cfg.Shard, ws: *const widths.Widths, tfq3_store: ?*tfq3_mod.Store) !Weights {
    const c = s.local;
    if (c.looped) return @import("loop.zig").loadWeights(allocator, device, checkpoint, c, ws);
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
    const tfq3_ptr: ?*const tfq3_mod.Store = if (tfq3_store) |ptr| ptr else null;
    const b = Builder{ .allocator = allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &owned, .stage = &stage, .runs = &runs, .widths = ws, .tfq3 = tfq3_ptr };
    for (layers, 0..) |*l, i| l.* = try b.layer(s, i);
    const embed: projection.Triple = if (widths.isBf16(ws.embed)) blk: {
        const w = try b.tensor("model.embed_tokens.weight", .bf16, &.{ g.vocab, g.hidden });
        break :blk .{ .weight = w, .scales = w, .biases = w, .bits = widths.bf16.bits, .group = widths.bf16.group };
    } else try b.quant("model.embed_tokens", g.vocab, g.hidden, ws.embed);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const stored_head = try std.mem.concat(allocator, u8, &.{ prefix, "lm_head.weight" });
    defer allocator.free(stored_head);
    const has_head = checkpoint.has(stored_head) or (tfq3_ptr != null and tfq3_ptr.?.lookupBits("lm_head", g.vocab, g.hidden) != null);
    // a tied head repacks the embedding's bytes into its own lane buffers; the embedding keeps its file layout
    const head_module = try @import("tied_head.zig").module(g.tied_head, has_head);
    const head = try b.linear(&.{.{ .module = head_module, .n = g.vocab, .w = ws.head, .span = s.vocabRows() }}, g.hidden);
    if (device.tensorUnits()) try laneOrder(allocator, device, layers, head);
    return .{ .allocator = allocator, .layers = layers, .embed = embed, .norm = norm, .head = head, .tfq3 = if (tfq3_store) |ptr| ptr.* else null, .owned = owned, .runs = runs };
}

/// The draft head from `checkpoint`, whose `mtp.*` tensors were added with `repacked` deferral; one rank, 4-bit g64.
pub fn loadMtp(w: *Weights, device: mtl.Device, checkpoint: *const cp.Checkpoint, s: *const cfg.Shard) !void {
    if (s.ranks != 1) return error.MtpNeedsOneRank;
    if (s.local.moe()) return error.MtpNeedsDenseModel;
    if (s.global.tied_head) return error.MtpNeedsUntiedHead;
    const c = s.local;
    const prefix: []const u8 = if (checkpoint.has("language_model.mtp.norm.weight")) "language_model." else if (checkpoint.has("mtp.norm.weight")) "" else return error.MissingMtpHead;
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    const ws: widths.Widths = .{ .allocator = w.allocator, .layers = &.{}, .embed = widths.uniform, .head = widths.uniform };
    const b = Builder{ .allocator = w.allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &w.owned, .stage = &stage, .runs = &w.runs, .widths = &ws };
    const four: [widths.members]widths.Width = @splat(widths.uniform);
    const head: Mtp = .{
        .norm_embed = try b.tensor("mtp.pre_fc_norm_embedding.weight", .bf16, &.{c.hidden}),
        .norm_hidden = try b.tensor("mtp.pre_fc_norm_hidden.weight", .bf16, &.{c.hidden}),
        // the input is [norm_embed(embed(token)), norm_hidden(hidden)]: embedding first, as the file's fc was trained
        .fc = try b.linear(&.{.{ .module = "mtp.fc", .n = c.hidden, .w = widths.uniform }}, 2 * c.hidden),
        .layer = try b.block(s, "mtp.layers.0.", .attention, four),
        .norm = try b.tensor("mtp.norm.weight", .bf16, &.{c.hidden}),
    };
    if (device.tensorUnits()) {
        var jobs: std.ArrayList(lane.Order) = .empty;
        defer jobs.deinit(w.allocator);
        const t = head.layer.mixer.attention;
        for ([_]Linear{ head.fc, t.q, t.kv, t.out, head.layer.gu, head.layer.down }) |p| try regJobs(w.allocator, &jobs, p);
        try lane.regOrder(device, jobs.items);
    }
    w.mtp = head;
}

/// Tensor-unit projections leave load in lane_projection.reg_layout, the order quant_gpu's reg kernel reads.
pub fn laneOrder(a: std.mem.Allocator, device: mtl.Device, layers: []const Layer, head: Linear) !void {
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

/// 4-, 5-, and 6-bit g64 matrices take the reg kernel; other widths keep lane_projection.weight_layout.
fn regJobs(a: std.mem.Allocator, jobs: *std.ArrayList(lane.Order), p: Linear) std.mem.Allocator.Error!void {
    for (p.runs) |r| try regJobs(a, jobs, r.linear);
    if (p.tfq3 != null or p.bf16 != null) return;
    const reg_bits = p.bits == 4 or p.bits == 5 or p.bits == 6;
    if (p.runs.len == 0 and reg_bits and p.group == 64 and p.n % 32 == 0) try jobs.append(a, .{ .w = .{ .buf = p.words.buffer, .off = p.words.offset }, .n = p.n, .k = p.k, .bits = p.bits });
}

/// A borrowed quantized tensor as the row kernels' matrix.
fn weightsOf(t: projection.Triple, n: usize, k: usize) row_gpu.Weights {
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

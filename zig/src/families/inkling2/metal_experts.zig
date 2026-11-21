//! Inkling 2 Mini routed W4A4 and W8A8. Borrow the original checkpoint bytes, no expanded or repacked weights.
//! Scratch is reused by the serial encoder; weights and all borrowed arguments live until command completion.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const Precision = @import("plan.zig").Precision;
pub const View = @import("../../core/exl3_projection.zig").View;

pub const threads = 256;
const simd_width = 32;
const max_width = 2304;
const max_experts = 128;
const max_topk = 8;
const dot_rows = 4 * (threads / simd_width);

const Kernel = enum { quant_nvfp4, quant_mxfp8, gate_up_nvfp4, gate_up_mxfp8, down_nvfp4, down_mxfp8, combine };
pub const symbols = blk: {
    const names = @typeInfo(Kernel).@"enum".field_names;
    var result: [names.len][:0]const u8 = undefined;
    for (names, 0..) |name, i| result[i] = "inkling2_exp_" ++ name;
    break :blk result;
};

const QuantArgs = extern struct { rows: u32, k: u32 };
const DotArgs = extern struct {
    pairs: u32,
    src_div: u32,
    k: u32,
    n: u32,
    experts: u32,
    out_width: u32,
    gate_cap: f32,
    up_cap: f32,
};
const CombineArgs = extern struct { rows: u32, topk: u32, hidden: u32, experts: u32 };

comptime {
    std.debug.assert(@sizeOf(QuantArgs) == 8 and @sizeOf(DotArgs) == 32 and @sizeOf(CombineArgs) == 16);
}

pub const LoadOptions = struct {
    /// Contract controls compile substitutions through the same Store and dispatches.
    source: []const u8 = sources.inkling2_experts,
    /// Device probes preserve intermediates for inspection; inference reuses them.
    shared_scratch: bool = false,
};

const Matrix = struct {
    weight: View,
    scales: View,
    amax: View,
    denominator: View,

    fn load(cp: *const ck.Checkpoint, i: usize, comptime name: []const u8, prec: Precision, e: u32, n: u32, k: u32) !Matrix {
        const nv = prec == .nvfp4;
        const weight = try checked(try tensor(cp, i, name ++ "_weight"), if (nv) .u8 else .f8_e4m3,
            &.{ e, n, if (nv) k / 2 else k });
        const scales = try checked(try tensor(cp, i, name ++ "_weight_scale"), if (nv) .f8_e4m3 else .u8,
            &.{ e, n, k / if (nv) @as(u32, 16) else 32 });
        // MXFP8 kernels ignore these arguments; bind a valid existing view without allocating dummy buffers.
        const amax = if (nv) try checked(try tensor(cp, i, name ++ "_weight_amax"), .f32, &.{e}) else weight;
        const denom_t = if (nv) try tensor(cp, i, name ++ "_weight_outer_scale_denominator") else null;
        const denom = if (denom_t) |t| try checked(t, .f32, &.{1}) else weight;
        if (denom_t) |t| {
            const d = t.host(f32)[0];
            if (!std.math.isFinite(d) or d <= 0 or !std.math.isFinite(d * 1536.0)) return error.ExpertScale;
        }
        return .{ .weight = weight, .scales = scales, .amax = amax, .denominator = denom };
    }
};

const Layer = struct { precision: Precision, gate_up: Matrix, down: Matrix };

const QuantRegion = struct {
    values: usize,
    scales: usize,
    amax: usize,

    fn reserve(at: *usize, elements: usize, rows: usize, has_nv: bool, has_mx: bool) QuantRegion {
        return .{
            .values = Scratch.reserve(at, if (has_mx) elements else elements / 2),
            .scales = Scratch.reserve(at, elements / (if (has_nv) @as(usize, 16) else 32)),
            .amax = Scratch.reserve(at, rows * @sizeOf(f32)),
        };
    }
};

const Scratch = struct {
    buffer: mtl.Buffer,
    quants: [2]QuantRegion,
    stage: usize,
    down: usize,

    fn reserve(cursor: *usize, bytes: usize) usize {
        const off = cursor.*;
        cursor.* = std.mem.alignForward(usize, off + bytes, ck.alignment);
        return off;
    }

    fn init(device: mtl.Device, rows: u32, pairs: u32, hidden: usize, inter: usize, has_nv: bool, has_mx: bool, shared: bool) !Scratch {
        var bytes: usize = 0;
        const quant = QuantRegion.reserve(&bytes, @max(rows * hidden, pairs * inter), pairs, has_nv, has_mx);
        const quants = if (!shared) [2]QuantRegion{ quant, quant } else .{
            quant, QuantRegion.reserve(&bytes, pairs * inter, pairs, has_nv, has_mx),
        };
        const stage = reserve(&bytes, pairs * @max(hidden, inter) * @sizeOf(u16));
        const down = if (shared) reserve(&bytes, pairs * hidden * @sizeOf(u16)) else stage;
        if (bytes > device.maxBufferLength()) return error.ExpertScratchTooLarge;
        return .{
            .buffer = try device.buffer(bytes, if (shared) mtl.ResourceOptions.shared else mtl.ResourceOptions.private),
            .quants = quants,
            .stage = stage,
            .down = down,
        };
    }

    fn at(s: Scratch, offset: usize) View {
        return .{ .buffer = s.buffer, .offset = offset };
    }
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    layers: []const Layer,
    pipes: [symbols.len]mtl.Pipeline,
    scratch: Scratch,
    hidden: u32,
    inter: u32,
    experts: u32,
    topk: u32,
    max_rows: u32,
    gate_cap: f32,
    up_cap: f32,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, cp: *const ck.Checkpoint, c: *const config.Config, max_rows: u32) !Store {
        return loadWith(gpa, device, cp, c, max_rows, .{});
    }

    pub fn loadWith(gpa: std.mem.Allocator, d: mtl.Device, cp: *const ck.Checkpoint, c: *const config.Config, max_rows: u32, options: LoadOptions) !Store {
        const t = &c.text;
        if (t.layers == 0 or t.layers > config.max_layers or max_rows == 0 or
            t.hidden == 0 or t.hidden > max_width or t.hidden % 32 != 0 or
            t.expert_inter == 0 or t.expert_inter > max_width or t.expert_inter % 32 != 0 or
            t.experts == 0 or t.experts > max_experts or t.topk == 0 or t.topk > @min(t.experts, max_topk) or
            !std.math.isFinite(t.gate_cap) or t.gate_cap < 0 or !std.math.isFinite(t.up_cap) or t.up_cap < 0)
            return error.ExpertGeometry;
        const pairs = try std.math.mul(u32, max_rows, t.topk);
        const layers = try gpa.alloc(Layer, t.layers);
        errdefer gpa.free(layers);
        var has_nv = false;
        var has_mx = false;
        for (layers, 0..) |*layer, i| {
            const precision: Precision = switch ((try tensor(cp, i, "w13_weight")).dtype) {
                .u8 => .nvfp4,
                .f8_e4m3 => .mxfp8,
                else => return error.ExpertFormat,
            };
            has_nv = has_nv or precision == .nvfp4;
            has_mx = has_mx or precision == .mxfp8;
            layer.* = .{
                .precision = precision,
                .gate_up = try Matrix.load(cp, i, "w13", precision, t.experts, 2 * t.expert_inter, t.hidden),
                .down = try Matrix.load(cp, i, "w2", precision, t.experts, t.hidden, t.expert_inter),
            };
        }
        var compile = mtl.CompileOptions.mlx();
        compile.math_functions = .precise;
        const lib = try mtl.Library.fromSource(d, options.source, compile);
        defer lib.deinit();
        var pipes: [symbols.len]mtl.Pipeline = undefined;
        var loaded: usize = 0;
        errdefer for (pipes[0..loaded]) |p| p.deinit();
        for (symbols, &pipes) |symbol, *p| {
            p.* = try mtl.Pipeline.init(d, lib, symbol, false);
            loaded += 1;
            if (p.maxThreads() < threads or p.simdWidth() != simd_width) return error.ExpertLaunchLimit;
        }
        const scratch = try Scratch.init(d, max_rows, pairs, t.hidden, t.expert_inter, has_nv, has_mx, options.shared_scratch);
        return .{
            .allocator = gpa,
            .layers = layers,
            .pipes = pipes,
            .scratch = scratch,
            .hidden = t.hidden,
            .inter = t.expert_inter,
            .experts = t.experts,
            .topk = t.topk,
            .max_rows = max_rows,
            .gate_cap = t.gate_cap,
            .up_cap = t.up_cap,
        };
    }

    pub fn scratchBytes(s: *const Store) u64 {
        return s.scratch.buffer.length();
    }

    /// The checkpoint owner manages its shard residency. Only scratch belongs to the Store.
    pub fn setResident(s: *Store, hook: ?mtl.ResidencySet.Follow) void {
        if (s.follow) |f| f.leave(f.ctx, s.scratch.buffer);
        s.follow = hook;
        if (hook) |f| f.join(f.ctx, s.scratch.buffer);
    }

    pub fn deinit(s: *Store) void {
        s.setResident(null);
        s.scratch.buffer.deinit();
        for (s.pipes) |p| p.deinit();
        s.allocator.free(s.layers);
    }

    fn pipe(s: *const Store, k: Kernel) mtl.Pipeline {
        return s.pipes[@intFromEnum(k)];
    }

    fn quant(s: *const Store, e: mtl.ComputeEncoder, prec: Precision, stage: usize, src: View, rows: u32, k: u32) void {
        const b = s.scratch;
        const q = b.quants[stage];
        e.setPipeline(s.pipe(if (prec == .nvfp4) .quant_nvfp4 else .quant_mxfp8));
        bind(e, &.{ src, b.at(q.values), b.at(q.scales), b.at(q.amax) });
        e.setValue(QuantArgs{ .rows = rows, .k = k }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(threads, 1, 1));
    }

    fn dot(s: *const Store, e: mtl.ComputeEncoder, layer: *const Layer, gate_up: bool, ids: View, weights: View, pairs: u32) void {
        const nv = layer.precision == .nvfp4;
        const m = if (gate_up) &layer.gate_up else &layer.down;
        const b = s.scratch;
        const q = b.quants[if (gate_up) 0 else 1];
        e.setPipeline(s.pipe(if (gate_up) (if (nv) .gate_up_nvfp4 else .gate_up_mxfp8) else (if (nv) .down_nvfp4 else .down_mxfp8)));
        bind(e, &.{ m.weight, m.scales, m.amax, m.denominator, b.at(q.values), b.at(q.scales), b.at(q.amax), ids, weights, b.at(if (gate_up) b.stage else b.down) });
        const out_width = if (gate_up) s.inter else s.hidden;
        e.setValue(DotArgs{
            .pairs = pairs,
            .src_div = if (gate_up) s.topk else 1,
            .k = if (gate_up) s.hidden else s.inter,
            .n = if (gate_up) 2 * s.inter else s.hidden,
            .experts = s.experts,
            .out_width = out_width,
            .gate_cap = s.gate_cap,
            .up_cap = s.up_cap,
        }, 10);
        e.dispatchGroups(mtl.Size.of(std.math.divCeil(u32, out_width, if (gate_up) dot_rows / 2 else dot_rows) catch unreachable, pairs, 1), mtl.Size.of(threads, 1, 1));
    }

    /// BF16 x/out [rows, hidden], u32 ids and FP32 weights [rows, topk]. IDs >= experts are inactive slots.
    /// The serial caller orders producers before this operation and consumers after it; no submission or host sync.
    pub fn encode(s: *const Store, e: mtl.ComputeEncoder, physical_layer: u32, x: View, ids: View, weights: View, out: View, rows: u32) !void {
        if (rows == 0 or rows > s.max_rows or physical_layer >= s.layers.len) return error.ExpertDispatch;
        const pairs = rows * s.topk;
        const row_bytes = @as(usize, rows) * s.hidden * @sizeOf(u16);
        const route_bytes = @as(usize, pairs) * @sizeOf(u32);
        try checkView(x, row_bytes, 2);
        try checkView(out, row_bytes, 2);
        try checkView(ids, route_bytes, 4);
        try checkView(weights, route_bytes, 4);
        if (overlap(out, row_bytes, x, row_bytes) or overlap(out, row_bytes, ids, route_bytes) or
            overlap(out, row_bytes, weights, route_bytes)) return error.ExpertAliasing;
        const layer = &s.layers[physical_layer];
        s.quant(e, layer.precision, 0, x, rows, s.hidden);
        e.barrier();
        s.dot(e, layer, true, ids, weights, pairs);
        e.barrier();
        s.quant(e, layer.precision, 1, s.scratch.at(s.scratch.stage), pairs, s.inter);
        e.barrier();
        s.dot(e, layer, false, ids, weights, pairs);
        e.barrier();
        e.setPipeline(s.pipe(.combine));
        bind(e, &.{ s.scratch.at(s.scratch.down), out });
        e.setValue(CombineArgs{ .rows = rows, .topk = s.topk, .hidden = s.hidden, .experts = s.experts }, 2);
        e.dispatchThreads(mtl.Size.of(s.hidden, rows, 1), mtl.Size.of(threads, 1, 1));
    }

    /// Synchronous only for isolated device tests. Inference encodes into the existing serial encoder.
    pub fn run(s: *const Store, queue: mtl.Queue, layer: u32, x: View, ids: View, weights: View, out: View, rows: u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        s.encode(e, layer, x, ids, weights, out, rows) catch |err| {
            e.end();
            return err;
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("inkling2 expert store: {s}", .{msg});
            return error.GpuFailed;
        }
    }
};

fn tensor(cp: *const ck.Checkpoint, i: usize, comptime suffix: []const u8) !ck.Tensor {
    var name: [128]u8 = undefined;
    return cp.get(try std.fmt.bufPrint(&name, "model.llm.layers.{d}.mlp.experts." ++ suffix, .{i}));
}

fn checked(t: ck.Tensor, dtype: ck.DType, shape: []const usize) !View {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, shape, t.shape[0..t.rank]) or
        t.bytes != dtype.size() * product(shape)) return error.ExpertFormat;
    const v: View = .{ .buffer = t.buffer, .offset = t.offset };
    try checkView(v, t.bytes, 4);
    return v;
}

fn product(shape: []const usize) usize {
    var p: usize = 1;
    for (shape) |d| p *= d;
    return p;
}

fn checkView(v: View, n: usize, alignment: usize) !void {
    const bytes = v.buffer.length();
    if (v.offset % alignment != 0 or v.offset > bytes or n > bytes - v.offset) return error.ExpertBuffer;
}

fn overlap(a: View, an: usize, b: View, bn: usize) bool {
    return a.buffer.id == b.buffer.id and a.offset < b.offset + bn and b.offset < a.offset + an;
}

fn bind(e: mtl.ComputeEncoder, views: []const View) void {
    for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
}
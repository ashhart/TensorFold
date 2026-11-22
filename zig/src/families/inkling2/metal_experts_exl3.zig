//! Decode resident EXL3 expert packs through the shared trellis projections; own only descriptors and BF16 scratch.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const exl3_load = @import("../../core/exl3_load.zig");
const config = @import("config.zig");
pub const View = exl3.View;

const combine_threads = 256;
const max_width = 2304;
const max_experts = 128;
const max_topk = 8;
const proj_names = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

const CombineArgs = extern struct { rows: u32, topk: u32, hidden: u32, experts: u32 };
const SwigluArgs = extern struct { pairs: u32, inter: u32, gate_cap: f32, up_cap: f32 };

comptime {
    std.debug.assert(@sizeOf(CombineArgs) == 16 and @sizeOf(SwigluArgs) == 16);
}

pub const LoadOptions = struct {
    /// Contract controls compile substitutions through the same Store and dispatches.
    source: []const u8 = sources.inkling2_experts,
    /// Device probes read the BF16 intermediates; inference keeps them device-private.
    shared_scratch: bool = false,
};

/// One admitted projection's parts: the fp16 scales, or the packed sign words; the bias is optional.
const Scales = enum { f16, words };

/// A group's tensors checked against the config's K,N: trellis [K/16,N/16,8*halves] i16, scales, marker, bias.
fn matrix(cp: *const ck.Checkpoint, a: std.mem.Allocator, layer: usize, expert: usize, proj: []const u8, k: usize, n: usize) !exl3.Matrix {
    var buf: [192]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, proj });
    var tb: [208]u8 = undefined;
    const trellis = try cp.get(try std.fmt.bufPrint(&tb, "{s}.trellis", .{prefix}));
    if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.shape[0] * 16 != k or
        trellis.shape[1] * 16 != n or trellis.shape[2] % 8 != 0) return error.ExpertFormat;
    const halves = trellis.shape[2] / 8;
    if (halves < 2 or halves > 16) return error.ExpertFormat;
    const su = try scale(cp, prefix, "suh", "su", k);
    const sv = try scale(cp, prefix, "svh", "sv", n);
    if (su == null or sv == null or su.?.kind != sv.?.kind) return error.ExpertFormat; // one fp16, one packed: a half-written group
    var bias: ?View = null;
    var bb: [208]u8 = undefined;
    const bias_name = try std.fmt.bufPrint(&bb, "{s}.bias", .{prefix});
    if (cp.has(bias_name)) {
        const t = try cp.get(bias_name);
        if (t.dtype != .f16 or t.rank != 1 or t.shape[0] != n or t.bytes != n * 2) return error.ExpertFormat;
        bias = .{ .buffer = t.buffer, .offset = t.offset };
    }
    return .{
        .trellis = .{ .buffer = trellis.buffer, .offset = trellis.offset },
        .suh = su.?.tensor,
        .svh = sv.?.tensor,
        .bias = bias,
        .k = k,
        .n = n,
        .width = .{ .halves = @intCast(halves) },
        .codebook = try codebook(cp, a, prefix),
        .suh_format = if (su.?.kind == .f16) .f16 else .packed_signs,
        .svh_format = if (sv.?.kind == .f16) .f16 else .packed_signs,
    };
}

const Part = struct { tensor: View, kind: Scales };

/// The fp16 scale name, or the packed sign words; a missing pair is refused.
fn scale(cp: *const ck.Checkpoint, prefix: []const u8, f16_name: []const u8, words_name: []const u8, len: usize) !?Part {
    var buf: [224]u8 = undefined;
    const full = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, f16_name });
    if (!cp.has(full)) {
        const words = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, words_name });
        if (!cp.has(words)) return null;
        const t = try cp.get(words);
        if (t.dtype != .i16 or t.rank != 1 or t.shape[0] != len / 16 or t.bytes != len / 8) return error.ExpertFormat;
        return .{ .tensor = .{ .buffer = t.buffer, .offset = t.offset }, .kind = .words };
    }
    const t = try cp.get(full);
    if (t.dtype != .f16 or t.rank != 1 or t.shape[0] != len or t.bytes != len * 2) return error.ExpertFormat;
    return .{ .tensor = .{ .buffer = t.buffer, .offset = t.offset }, .kind = .f16 };
}

/// format.py's markers: a scalar int32 per codebook; absent both, the inst3 default applies.
fn codebook(cp: *const ck.Checkpoint, a: std.mem.Allocator, prefix: []const u8) !exl3.format.Codebook {
    var found: ?exl3.format.Codebook = null;
    for (exl3_load.markers) |m| {
        const name = try std.mem.concat(a, u8, &.{ prefix, ".", m.name });
        if (!cp.has(name)) continue;
        if (found != null) return error.ExpertFormat; // both markers: not a loadable group
        const t = try cp.get(name);
        if (t.dtype != .i32 or t.bytes != 4) return error.ExpertFormat;
        if (std.mem.readInt(u32, t.host(u8)[0..4], .little) != m.value) return error.ExpertFormat;
        found = switch (m.codebook) {
            .inst3 => .inst3,
            .mcg => .mcg,
            .mul1 => .mul1,
        };
    }
    return found orelse .inst3;
}

fn collect(bufs: *std.ArrayList(mtl.Buffer), b: mtl.Buffer, gpa: std.mem.Allocator) !void {
    for (bufs.items) |seen| if (seen.id == b.id) return;
    try bufs.append(gpa, b);
}

fn reserve(cursor: *usize, bytes: usize) usize {
    const off = cursor.*;
    cursor.* = std.mem.alignForward(usize, off + bytes, ck.alignment);
    return off;
}

fn plane(s: *const Store, at: usize) View {
    return .{ .buffer = s.scratch, .offset = at };
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

pub const Store = struct {
    allocator: std.mem.Allocator,
    descriptors: mtl.Buffer,
    proj: exl3.Pipelines,
    combine: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    scratch: mtl.Buffer,
    /// Byte offsets of the gate, up, activation and down BF16 planes, then the projection scratch.
    planes: [5]usize,
    /// Every shard buffer the descriptors can name; encode marks them for indirect access.
    weight_bufs: []mtl.Buffer,
    hidden: u32,
    inter: u32,
    experts: u32,
    topk: u32,
    max_rows: u32,
    layers: usize,
    gate_cap: f32,
    up_cap: f32,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, cp: *const ck.Checkpoint, c: *const config.Config, max_rows: u32) !Store {
        return loadWith(gpa, device, cp, c, max_rows, .{});
    }

    pub fn loadWith(gpa: std.mem.Allocator, d: mtl.Device, cp: *const ck.Checkpoint, c: *const config.Config, max_rows: u32, options: LoadOptions) !Store {
        const t = &c.text;
        if (t.layers == 0 or t.layers > config.max_layers or max_rows == 0 or
            t.hidden == 0 or t.hidden > max_width or t.hidden % 128 != 0 or
            t.expert_inter == 0 or t.expert_inter > max_width or t.expert_inter % 128 != 0 or
            t.experts == 0 or t.experts > max_experts or t.topk == 0 or t.topk > @min(t.experts, max_topk) or
            !std.math.isFinite(t.gate_cap) or t.gate_cap < 0 or !std.math.isFinite(t.up_cap) or t.up_cap < 0)
            return error.ExpertGeometry;
        const pairs = try std.math.mul(u32, max_rows, t.topk);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const count = t.layers * 3 * t.experts;
        const descriptors = try d.buffer(count * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared);
        errdefer descriptors.deinit();
        var bufs: std.ArrayList(mtl.Buffer) = .empty;
        defer bufs.deinit(gpa);
        const table = descriptors.slice(exl3.ExpertDescriptor, count);
        var written: usize = 0;
        for (0..t.layers) |i| {
            for (proj_names, 0..) |pname, p| {
                const k: usize = if (p == 2) t.expert_inter else t.hidden;
                const n: usize = if (p == 2) t.hidden else t.expert_inter;
                for (0..t.experts) |e| {
                    const m = try matrix(cp, a, i, e, pname, k, n);
                    table[written] = try exl3.descriptor(m);
                    written += 1;
                    try collect(&bufs, m.trellis.buffer, gpa);
                    try collect(&bufs, m.suh.buffer, gpa);
                    try collect(&bufs, m.svh.buffer, gpa);
                    if (m.bias) |b| try collect(&bufs, b.buffer, gpa);
                }
            }
        }
        std.debug.assert(written == count);
        var proj = try exl3.Pipelines.load(d);
        errdefer proj.deinit();
        var compile = mtl.CompileOptions.mlx();
        compile.math_functions = .precise;
        const lib = try mtl.Library.fromSource(d, options.source, compile);
        defer lib.deinit();
        const combine = try mtl.Pipeline.init(d, lib, "inkling2_exp_combine", false);
        errdefer combine.deinit();
        const swiglu = try mtl.Pipeline.init(d, lib, "inkling2_exp_swiglu", false);
        errdefer swiglu.deinit();
        if (combine.maxThreads() < combine_threads or combine.simdWidth() != 32 or swiglu.maxThreads() < combine_threads)
            return error.ExpertLaunchLimit;
        const inter: usize = t.expert_inter;
        const hidden: usize = t.hidden;
        var cursor: usize = 0;
        const gate_y = reserve(&cursor, @as(usize, pairs) * inter * 2);
        const up_y = reserve(&cursor, @as(usize, pairs) * inter * 2);
        const act = reserve(&cursor, @as(usize, pairs) * inter * 2);
        const down_y = reserve(&cursor, @as(usize, pairs) * hidden * 2);
        const exl3_need = @max(try exl3.scratchBytes(hidden, inter, pairs), try exl3.scratchBytes(inter, hidden, pairs));
        const exl3_off = reserve(&cursor, exl3_need);
        if (cursor > d.maxBufferLength()) return error.ExpertScratchTooLarge;
        const scratch = try d.buffer(cursor, if (options.shared_scratch) mtl.ResourceOptions.shared else mtl.ResourceOptions.private);
        errdefer scratch.deinit();
        const weight_bufs = try bufs.toOwnedSlice(gpa);
        errdefer gpa.free(weight_bufs);
        return .{
            .allocator = gpa,
            .descriptors = descriptors,
            .proj = proj,
            .combine = combine,
            .swiglu = swiglu,
            .scratch = scratch,
            .planes = .{ gate_y, up_y, act, down_y, exl3_off },
            .weight_bufs = weight_bufs,
            .hidden = t.hidden,
            .inter = t.expert_inter,
            .experts = t.experts,
            .topk = t.topk,
            .max_rows = max_rows,
            .layers = t.layers,
            .gate_cap = t.gate_cap,
            .up_cap = t.up_cap,
        };
    }

    fn expertsAt(s: *const Store, layer: usize, proj: usize) exl3.Experts {
        const index = (layer * 3 + proj) * @as(usize, s.experts);
        return .{
            .descriptors = .{ .buffer = s.descriptors, .offset = index * @sizeOf(exl3.ExpertDescriptor) },
            .count = @intCast(s.experts),
            .k = @intCast(if (proj == 2) s.inter else s.hidden),
            .n = @intCast(if (proj == 2) s.hidden else s.inter),
        };
    }

    pub fn scratchBytes(s: *const Store) u64 {
        return s.scratch.length();
    }

    /// The checkpoint owner manages shard residency; the Store owns descriptors and scratch.
    pub fn setResident(s: *Store, hook: ?mtl.ResidencySet.Follow) void {
        if (s.follow) |f| {
            f.leave(f.ctx, s.scratch);
            f.leave(f.ctx, s.descriptors);
        }
        s.follow = hook;
        if (hook) |f| {
            f.join(f.ctx, s.scratch);
            f.join(f.ctx, s.descriptors);
        }
    }

    pub fn deinit(s: *Store) void {
        s.setResident(null);
        s.scratch.deinit();
        s.descriptors.deinit();
        s.swiglu.deinit();
        s.combine.deinit();
        s.proj.deinit();
        s.allocator.free(s.weight_bufs);
    }

    /// Encode BF16 projections for active IDs, weighted by FP32 route scores, without submitting.
    pub fn encode(s: *const Store, e: mtl.ComputeEncoder, physical_layer: u32, x: View, ids: View, weights: View, out: View, rows: u32) !void {
        if (rows == 0 or rows > s.max_rows or @as(usize, physical_layer) >= s.layers) return error.ExpertDispatch;
        const pairs = rows * s.topk;
        const row_bytes = @as(usize, rows) * s.hidden * @sizeOf(u16);
        const route_bytes = @as(usize, pairs) * @sizeOf(u32);
        try checkView(x, row_bytes, 2);
        try checkView(out, row_bytes, 2);
        try checkView(ids, route_bytes, 4);
        try checkView(weights, route_bytes, 4);
        if (overlap(out, row_bytes, x, row_bytes) or overlap(out, row_bytes, ids, route_bytes) or
            overlap(out, row_bytes, weights, route_bytes)) return error.ExpertAliasing;
        // The descriptors reach their weights through GPU addresses: mark every shard they can name.
        for (s.weight_bufs) |b| e.useResource(b, mtl.ResourceUsage.read);
        const layer = @as(usize, physical_layer);
        const gy = plane(s, s.planes[0]);
        const uy = plane(s, s.planes[1]);
        const ay = plane(s, s.planes[2]);
        const dy = plane(s, s.planes[3]);
        const sc = plane(s, s.planes[4]);
        const gate_rows: exl3.Rows = .{
            .count = @intCast(pairs),
            .x_stride = @intCast(s.hidden),
            .y_stride = @intCast(s.inter),
            .output = .bf16,
            .routes = .{ .ids = ids, .repeat = @intCast(s.topk) },
        };
        try exl3.encodeIndexed(&s.proj, e, s.expertsAt(layer, 0), x, gy, sc, gate_rows);
        e.barrier();
        try exl3.encodeIndexed(&s.proj, e, s.expertsAt(layer, 1), x, uy, sc, gate_rows);
        e.barrier();
        e.setPipeline(s.swiglu);
        bind(e, &.{ gy, uy, ay });
        e.setValue(SwigluArgs{ .pairs = pairs, .inter = s.inter, .gate_cap = s.gate_cap, .up_cap = s.up_cap }, 3);
        e.dispatchThreads(mtl.Size.of(std.math.mul(u32, pairs, s.inter) catch return error.ExpertDispatch, 1, 1), mtl.Size.of(combine_threads, 1, 1));
        e.barrier();
        try exl3.encodeIndexed(&s.proj, e, s.expertsAt(layer, 2), ay, dy, sc, .{
            .count = @intCast(pairs),
            .x_stride = @intCast(s.inter),
            .y_stride = @intCast(s.hidden),
            .output = .bf16,
            .routes = .{ .ids = ids, .repeat = 1 },
        });
        e.barrier();
        e.setPipeline(s.combine);
        bind(e, &.{ dy, ids, weights, out });
        e.setValue(CombineArgs{ .rows = rows, .topk = s.topk, .hidden = s.hidden, .experts = s.experts }, 4);
        e.dispatchThreads(mtl.Size.of(s.hidden, rows, 1), mtl.Size.of(combine_threads, 1, 1));
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
            std.log.err("inkling2 exl3 expert store: {s}", .{msg});
            return error.GpuFailed;
        }
    }
};

test "inkling2 exl3 store geometry mirrors the packed store's limits" {
    try std.testing.expect(@sizeOf(CombineArgs) == 16);
    try std.testing.expect(@sizeOf(SwigluArgs) == 16);
    try std.testing.expectEqual(@as(usize, 3), proj_names.len);
}

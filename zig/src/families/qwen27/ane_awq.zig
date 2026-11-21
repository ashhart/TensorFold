//! Hybrid ANE+GPU MLP builder and AWQ g128 remainder kernels for SenseNova-Loop checkpoints.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const ane = mtl.ane;
const cp = core.checkpoint_metal;
const sn_awq = core.sensenova_loop.awq;
const linear_gpu = core.sensenova_loop.linear_gpu;
const Config = @import("config.zig").Config;
const wts = @import("gpu_weights.zig");
const projection = @import("projection.zig");
const loop = @import("loop.zig");
const Ref = projection.Ref;
const width: u32 = 128;

pub const add_ane_bf16_source =
    \\// Adds ANE fp16 plane outputs onto optional BF16 GPU remainder projections.
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\#pragma METAL fp contract(off)
    \\#ifndef Q27_ANE_CTL
    \\#define Q27_ANE_CTL 0
    \\#endif
    \\kernel void q27_add_ane_bf16(const device bfloat* gpu_out [[buffer(0)]],
    \\    const device half* planes [[buffer(1)]], device bfloat* y [[buffer(2)]],
    \\    constant uint4& args [[buffer(3)]], uint2 pos [[thread_position_in_grid]]) {
    \\  const uint hidden = args.x, rows = args.y, base = args.z, has_gpu = args.w;
    \\  if (pos.x >= hidden || pos.y >= rows) return;
    \\#if Q27_ANE_CTL == 2
    \\  const uint r = base + pos.y + 1u;
    \\#else
    \\  const uint r = base + pos.y;
    \\#endif
    \\  const float a = float(planes[pos.x * 128u + pos.y]);
    \\  const float g = has_gpu ? float(gpu_out[r * hidden + pos.x]) : 0.0f;
    \\#if Q27_ANE_CTL == 1
    \\  const float sum = (isfinite(a) && isfinite(g)) ? (g + a) : 0.0f;
    \\#else
    \\  const float sum = g + a;
    \\#endif
    \\  y[r * hidden + pos.x] = bfloat(sum);
    \\}
;

/// True when `w` stores SenseNova-Loop AWQ g128 physical MLP projections compatible with a `columns`-wide ANE split.
pub fn supports(w: *const wts.Weights, c: Config, columns: u32) bool {
    if (!c.looped or columns == 0 or columns > c.intermediate or columns % 128 != 0) return false;
    if (c.hidden % 128 != 0 or c.intermediate % 128 != 0 or w.physical_layers.len != c.physicalLayers()) return false;
    for (w.physical_layers) |l| {
        if (l.gu.sn == null or l.down.sn == null) return false;
        if (l.gu.bits != 4 or l.gu.group != 128 or l.down.bits != 4 or l.down.group != 128) return false;
    }
    return true;
}

/// GPU bytes required for physical AWQ remainder weights `[columns .. intermediate)`.
pub fn tailBytes(c: Config, columns: u32, tensor_units: bool) u64 {
    if (!c.looped or columns >= c.intermediate) return 0;
    const rem = c.intermediate - columns;
    const layout: sn_awq.Layout = if (tensor_units) .m5_tile else .m3_row;
    const gu_w = sn_awq.weightWords(2 * rem, c.hidden, layout) * 4 + sn_awq.metaCount(2 * rem, c.hidden) * 8;
    const dn_w = sn_awq.weightWords(c.hidden, rem, layout) * 4 + sn_awq.metaCount(c.hidden, rem) * 8;
    return @as(u64, c.physicalLayers()) * @as(u64, gu_w + dn_w);
}

pub const SnTail = struct {
    a: std.mem.Allocator,
    gu: []projection.Linear,
    down: []projection.Linear,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    sn_runs: std.ArrayList([]loop.SnRun) = .empty,
    add_lib: mtl.Library,
    add_pipe: mtl.Pipeline,
    rem: u32,
    int8_active: bool,

    pub fn deinit(t: *SnTail) void {
        t.add_pipe.deinit();
        t.add_lib.deinit();
        for (t.sn_runs.items) |r| t.a.free(r);
        t.sn_runs.deinit(t.a);
        for (t.owned.items) |b| b.deinit();
        t.owned.deinit(t.a);
        t.a.free(t.gu);
        t.a.free(t.down);
        t.a.destroy(t);
    }

    pub fn addAneBf16(t: *const SnTail, e: mtl.ComputeEncoder, gpu_out: Ref, planes: Ref, y: Ref, hidden: u32, rows: u32, base: u32, has_gpu: bool) void {
        if (rows == 0) return;
        e.setPipeline(t.add_pipe);
        e.setBuffer(gpu_out.buffer, gpu_out.offset, 0);
        e.setBuffer(planes.buffer, planes.offset, 1);
        e.setBuffer(y.buffer, y.offset, 2);
        e.setValue([4]u32{ hidden, rows, base, @intFromBool(has_gpu) }, 3);
        e.dispatchThreads(mtl.Size.of(hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn build(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, c: Config, columns: u32, slices: u32, want_int8: bool, inputs: []const ane.Surface, outputs: []const ane.Surface, programs: []ane.Program, requests: []ane.Request) !*SnTail {
        const phys_n = c.physicalLayers();
        if (c.hidden % 128 != 0 or c.intermediate % 128 != 0 or columns == 0 or columns > c.intermediate or columns % 128 != 0 or inputs.len != slices or outputs.len != slices or programs.len != phys_n or requests.len != phys_n * slices) return error.BadAneShare;
        const add_lib = try mtl.Library.fromSource(device, add_ane_bf16_source, mtl.CompileOptions.mlx());
        errdefer add_lib.deinit();
        const add_pipe = try mtl.Pipeline.init(device, add_lib, "q27_add_ane_bf16", false);
        errdefer add_pipe.deinit();
        const t = try a.create(SnTail);
        errdefer a.destroy(t);
        const rem: u32 = @intCast(c.intermediate - columns);
        t.* = .{ .a = a, .gu = try a.alloc(projection.Linear, phys_n), .down = try a.alloc(projection.Linear, phys_n), .add_lib = add_lib, .add_pipe = add_pipe, .rem = rem, .int8_active = want_int8 };
        errdefer {
            for (t.sn_runs.items) |r| a.free(r);
            t.sn_runs.deinit(a);
            for (t.owned.items) |b| b.deinit();
            t.owned.deinit(a);
            a.free(t.gu);
            a.free(t.down);
        }
        const prefix = textNamespace(ck);
        var sn_gpu = linear_gpu.LinearGpu.init(a, device);
        defer sn_gpu.deinit();
        var stage = cp.Stage{ .device = device };
        defer stage.deinit();
        var made: usize = 0;
        errdefer for (0..made) |layer| {
            for (requests[layer * slices ..][0..slices]) |req| req.deinit();
            programs[layer].deinit();
        };
        const f16_blob = try ane.Blob.init(a, &.{ 2 * columns * c.hidden, c.hidden * columns });
        defer f16_blob.deinit(a);
        const f16_mil = try ane.mlpSlice(a, c.hidden, columns, width, f16_blob.offsets[0], f16_blob.offsets[1]);
        defer a.free(f16_mil);
        var i8_blob: ?ane.Blob = if (want_int8) try ane.Blob.initInt8(a, 2 * columns * c.hidden, 2 * columns, c.hidden * columns, c.hidden) else null;
        defer if (i8_blob) |b| b.deinit(a);
        const i8_mil: ?[]u8 = if (i8_blob) |b| try ane.mlpSliceInt8(a, c.hidden, columns, width, b.offsets[0], b.offsets[1], b.offsets[2], b.offsets[3]) else null;
        defer if (i8_mil) |m| a.free(m);
        for (0..phys_n) |phys| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const ta = arena.allocator();
            var names: [12][]const u8 = undefined;
            var n_names: usize = 0;
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| try appendAwqNames(ta, ck, prefix, phys, proj, &names, &n_names);
            try stage.reserve(ck, names[0..n_names]);
            defer stage.reset();
            const g_raw = try loadAwqRaw(ta, ck, &stage, prefix, phys, "gate_proj", c.intermediate, c.hidden);
            const u_raw = try loadAwqRaw(ta, ck, &stage, prefix, phys, "up_proj", c.intermediate, c.hidden);
            const d_raw = try loadAwqRaw(ta, ck, &stage, prefix, phys, "down_proj", c.hidden, c.intermediate);
            const f1 = f16_blob.values(0, 2 * columns * c.hidden);
            try dequantPrefix(g_raw, columns, c.hidden, f1[0 .. columns * c.hidden]);
            try dequantPrefix(u_raw, columns, c.hidden, f1[columns * c.hidden ..]);
            try dequantPrefix(d_raw, c.hidden, columns, f16_blob.values(1, c.hidden * columns));
            var prog: ?ane.Program = null;
            if (t.int8_active) {
                quantRows(f1, 2 * columns, c.hidden, i8_blob.?.int8Values(0, 2 * columns * c.hidden), i8_blob.?.values(1, 2 * columns));
                quantRows(f16_blob.values(1, c.hidden * columns), c.hidden, columns, i8_blob.?.int8Values(2, c.hidden * columns), i8_blob.?.values(3, c.hidden));
                prog = ane.Program.init(i8_mil.?, i8_blob.?.bytes) catch blk: {
                    std.log.warn("qwen27: ANE INT8 MIL compile failed on layer {d}; falling back to FP16 ANE slices", .{phys});
                    t.int8_active = false;
                    break :blk null;
                };
            }
            programs[phys] = prog orelse try ane.Program.init(f16_mil, f16_blob.bytes);
            const own = requests[phys * slices ..][0..slices];
            for (own, inputs, outputs, 0..) |*r, inp, outp, req_made| r.* = programs[phys].request(inp, outp) catch |err| {
                for (own[0..req_made]) |x| x.deinit();
                programs[phys].deinit();
                return err;
            };
            if (rem > 0) {
                const gt = try sliceOut(ta, g_raw, columns, rem);
                const ut = try sliceOut(ta, u_raw, columns, rem);
                const dt = sliceIn(d_raw, columns, rem);
                t.gu[phys] = try packSnLinear(a, &sn_gpu, &t.owned, &t.sn_runs, &.{ gt, ut });
                t.down[phys] = try packSnLinear(a, &sn_gpu, &t.owned, &t.sn_runs, &.{dt});
            }
            made += 1;
        }
        std.log.info("qwen27: ANE AWQ share built layers={d} columns={d} rem={d} mode={s}", .{ phys_n, columns, rem, if (t.int8_active) "int8" else "fp16" });
        return t;
    }
};

pub fn textNamespace(ck: *const cp.Checkpoint) []const u8 {
    for ([_][]const u8{ "model.language_model.", "language_model.model.", "model." }) |p| {
        var buf: [96]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{p}) catch continue;
        if (ck.has(name)) return p;
    }
    return "model.";
}

fn appendAwqNames(ta: std.mem.Allocator, ck: *const cp.Checkpoint, prefix: []const u8, phys: usize, proj: []const u8, out: *[12][]const u8, idx: *usize) !void {
    const base = try std.fmt.allocPrint(ta, "{s}layers.{d}.mlp.{s}", .{ prefix, phys, proj });
    for ([_][]const u8{ ".qweight", ".qzeros", ".scales" }) |suf| {
        out[idx.*] = try std.mem.concat(ta, u8, &.{ base, suf });
        idx.* += 1;
    }
    const is_n = try std.mem.concat(ta, u8, &.{ base, ".input_scale" });
    if (ck.has(is_n)) {
        out[idx.*] = is_n;
        idx.* += 1;
    }
}

fn loadAwqRaw(ta: std.mem.Allocator, ck: *const cp.Checkpoint, stage: *cp.Stage, prefix: []const u8, phys: usize, proj: []const u8, n: usize, k: usize) !sn_awq.Part {
    const base = try std.fmt.allocPrint(ta, "{s}layers.{d}.mlp.{s}", .{ prefix, phys, proj });
    const qw_n = try std.mem.concat(ta, u8, &.{ base, ".qweight" });
    const qz_n = try std.mem.concat(ta, u8, &.{ base, ".qzeros" });
    const sc_n = try std.mem.concat(ta, u8, &.{ base, ".scales" });
    const is_n = try std.mem.concat(ta, u8, &.{ base, ".input_scale" });
    const has_is = ck.has(is_n);
    const qw = try stage.read(ck, qw_n);
    const qz = try stage.read(ck, qz_n);
    const sc = try stage.read(ck, sc_n);
    const sdt: sn_awq.ScaleDType = switch (sc.dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        else => return error.BadAneShare,
    };
    const is: ?[]const f32 = if (has_is) (try stage.read(ck, is_n)).host(f32)[0..k] else null;
    return .{ .n = n, .k = k, .qweight = qw.host(u32), .qzeros = qz.host(u32), .scales = sc.buffer.contents()[sc.offset..][0..sc.bytes], .scales_dtype = sdt, .input_scale = is };
}

fn dequantPrefix(p: sn_awq.Part, rows: usize, cols: usize, out: []f16) !void {
    if (rows > p.n or cols > p.k or out.len != rows * cols) return error.BadAneShare;
    const pn8 = p.n / 8;
    for (0..rows) |nl| for (0..cols) |ki| {
        const g = ki / sn_awq.group_size;
        const q = sn_awq.unpackNibble(p.qweight[ki * pn8 + nl / 8], nl % 8);
        const z = sn_awq.unpackNibble(p.qzeros[g * pn8 + nl / 8], nl % 8);
        const sc = sn_awq.loadScale(p.scales_dtype, p.scales, g * p.n + nl);
        var val: f64 = (@as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt(z))) * @as(f64, sc);
        if (p.input_scale) |is| val /= @as(f64, is[ki]);
        out[nl * cols + ki] = @floatCast(val);
    };
}

pub fn quantRows(src: []const f16, rows: usize, cols: usize, out_q: []i8, out_s: []f16) void {
    for (0..rows) |r| {
        var amax: f32 = 0;
        for (src[r * cols ..][0..cols]) |v| amax = @max(amax, @abs(@as(f32, @floatCast(v))));
        const sh: f16 = if (amax == 0) 0 else @floatCast(amax / 127.0);
        const sf: f32 = @floatCast(sh);
        out_s[r] = sh;
        for (src[r * cols ..][0..cols], out_q[r * cols ..][0..cols]) |v, *q| {
            q.* = if (sf == 0) 0 else @intFromFloat(std.math.clamp(@round(@as(f32, @floatCast(v)) / sf), -127.0, 127.0));
        }
    }
}

fn sliceOut(ta: std.mem.Allocator, p: sn_awq.Part, col0: usize, rem: usize) !sn_awq.Part {
    const pn8 = p.n / 8;
    const tn8 = rem / 8;
    const c8 = col0 / 8;
    const kg = p.k / sn_awq.group_size;
    const eb = p.scales_dtype.byteSize();
    const qw = try ta.alloc(u32, p.k * tn8);
    const qz = try ta.alloc(u32, kg * tn8);
    const sc = try ta.alloc(u8, kg * rem * eb);
    for (0..p.k) |ki| @memcpy(qw[ki * tn8 ..][0..tn8], p.qweight[ki * pn8 + c8 ..][0..tn8]);
    for (0..kg) |g| {
        @memcpy(qz[g * tn8 ..][0..tn8], p.qzeros[g * pn8 + c8 ..][0..tn8]);
        @memcpy(sc[g * rem * eb ..][0 .. rem * eb], p.scales[(g * p.n + col0) * eb ..][0 .. rem * eb]);
    }
    return .{ .n = rem, .k = p.k, .qweight = qw, .qzeros = qz, .scales = sc, .scales_dtype = p.scales_dtype, .input_scale = p.input_scale };
}

fn sliceIn(p: sn_awq.Part, col0: usize, rem: usize) sn_awq.Part {
    const pn8 = p.n / 8;
    const g0 = col0 / sn_awq.group_size;
    const eb = p.scales_dtype.byteSize();
    return .{
        .n = p.n,
        .k = rem,
        .qweight = p.qweight[col0 * pn8 ..],
        .qzeros = p.qzeros[g0 * pn8 ..],
        .scales = p.scales[g0 * p.n * eb ..],
        .scales_dtype = p.scales_dtype,
        .input_scale = if (p.input_scale) |is| is[col0..][0..rem] else null,
    };
}

fn packSnLinear(a: std.mem.Allocator, sn_gpu: *const linear_gpu.LinearGpu, owned: *std.ArrayList(mtl.Buffer), sn_runs: *std.ArrayList([]loop.SnRun), parts: []const sn_awq.Part) !projection.Linear {
    const mats = try sn_gpu.prepareAwq(parts);
    defer sn_gpu.allocator.free(mats);
    const runs = try a.alloc(loop.SnRun, mats.len);
    errdefer a.free(runs);
    var first: u32 = 0;
    for (mats, runs) |mat, *run| {
        var copy = mat;
        copy.owns = false;
        owned.appendSlice(a, &.{ copy.w, copy.sb }) catch |err| {
            var tmp = mat;
            tmp.deinit();
            return err;
        };
        if (copy.input_scale) |ib| try owned.append(a, ib);
        run.* = .{ .mat = .{ .awq = copy }, .first = first };
        first += copy.n;
    }
    try sn_runs.append(a, runs);
    const dummy: Ref = .{ .buffer = runs[0].mat.awq.w };
    return .{ .words = dummy, .pairs = dummy, .n = first, .k = @intCast(parts[0].k), .tile = 32, .slices = 1, .bits = 4, .group = 128, .sn = .{ .runs = runs, .n = first, .k = @intCast(parts[0].k) } };
}

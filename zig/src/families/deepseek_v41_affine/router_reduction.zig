//! Measure the full, exact router path on installed DeepSeek weights.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;
const Buffer = mtl.Buffer;
const rows_max = 1 + c.draft_block;
const RouteArgs = extern struct {
    experts: i32 = c.experts,
    topk: i32 = c.topk,
    scale: f32 = c.route_scale,
    trace_layer: i32 = -1,
};

pub const Layout = struct { cols: usize, groups: usize };
pub const layouts = [_]Layout{
    .{ .cols = 1, .groups = 8 }, // original
    .{ .cols = 1, .groups = 4 },
    .{ .cols = 1, .groups = 8 },
    .{ .cols = 2, .groups = 4 },
    .{ .cols = 2, .groups = 8 },
    .{ .cols = 4, .groups = 2 },
    .{ .cols = 4, .groups = 4 },
};
const selector_names = [_][:0]const u8{ "tf_route_reduce", "tf_route_merge" };

pub fn projection(device: mtl.Device, lib: mtl.Library, rows: usize, cols: usize) !mtl.Pipeline {
    var name: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "tf_router_project_{d}_{d}\x00", .{ rows, cols });
    return mtl.Pipeline.init(device, lib, label[0 .. label.len - 1 :0], false);
}

pub fn encodeProjection(e: mtl.ComputeEncoder, p: mtl.Pipeline, w: wts.Tensor, input: Buffer, out: Buffer, rows: usize, layout: usize) void {
    const g = layouts[layout];
    e.setPipeline(p);
    e.setBuffer(input, 0, 0);
    e.setBuffer(w.buffer, w.offset, 1);
    e.setBuffer(out, 0, 2);
    if (layout == 0) e.setValue(fwd.GemvArgs{ .rows = @intCast(rows), .n = c.experts, .k = c.hidden, .group_n = c.experts, .x_stride = c.hidden, .y_stride = c.experts }, 3);
    e.dispatchGroups(mtl.Size.of(c.experts / (g.cols * g.groups), 1, 1), mtl.Size.of(g.groups * 32, 1, 1));
}

const Output = struct {
    logits: Buffer,
    pick: Buffer,
    weights: Buffer,

    fn init(s: *st.Storage) !Output {
        return .{ .logits = try s.alloc(rows_max * c.experts * 4), .pick = try s.alloc(rows_max * c.topk * 4), .weights = try s.alloc(rows_max * c.topk * 4) };
    }

    fn verify(a: Output, b: Output, rows: usize, logits: bool) !void {
        if (!std.mem.eql(u32, a.pick.slice(u32, rows * c.topk), b.pick.slice(u32, rows * c.topk)))
            return error.RouterVariantPickDiffers;
        if (!std.mem.eql(u32, a.weights.slice(u32, rows * c.topk), b.weights.slice(u32, rows * c.topk)))
            return error.RouterVariantWeightDiffers;
        if (logits and !std.mem.eql(u32, a.logits.slice(u32, rows * c.experts), b.logits.slice(u32, rows * c.experts)))
            return error.RouterVariantLogitDiffers;
    }
};

const Trial = struct {
    m: *Model,
    w: *const wts.Moe,
    rows: usize,
    input: Buffer,
    images: Buffer,
    unused_trace: Buffer,
    projections: [layouts.len]mtl.Pipeline,
    selectors: [3]mtl.Pipeline,

    fn time(t: Trial, l: usize, select: usize, out: Output, reps: usize, synthetic: ?[2]Buffer) !f64 {
        const cb = t.m.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        for (0..reps) |r| {
            if (r != 0) e.barrier();
            if (synthetic == null) {
                encodeProjection(e, t.projections[l], t.w.router, t.input, out.logits, t.rows, l);
                e.barrier();
            }
            e.setPipeline(t.selectors[select]);
            const logits = if (synthetic) |s| s[0] else out.logits;
            e.setBuffer(logits, 0, 0);
            if (synthetic) |s| e.setBuffer(s[1], 0, 1) else e.setBuffer(t.w.bias.buffer, t.w.bias.offset, 1);
            e.setBuffer(out.pick, 0, 2);
            e.setBuffer(out.weights, 0, 3);
            e.setValue(RouteArgs{}, 4);
            e.setBuffer(t.images, 0, 5);
            if (synthetic) |s| e.setBuffer(s[1], 0, 6) else e.setBuffer(t.w.bias_vl.buffer, t.w.bias_vl.offset, 6);
            e.setBuffer(t.unused_trace, 0, 7);
            e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(c.experts, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.log.err("router reduction probe Metal error: {s}", .{why});
            return error.RouterVariantGpuFailed;
        }
        return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
    }
};

pub fn probe(m: *Model) !void {
    const lib = try mtl.Library.fromSource(m.device, @embedFile("router_reduction.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    var selectors: [3]mtl.Pipeline = undefined;
    selectors[0] = m.kernels.get("ds_route");
    var loaded: usize = 1;
    defer for (selectors[1..loaded]) |p| p.deinit();
    for (selector_names, 1..) |name, i| {
        selectors[i] = try mtl.Pipeline.init(m.device, lib, name, false);
        loaded += 1;
    }
    var memory: st.Storage = .{ .gpa = m.gpa, .device = m.device };
    defer memory.deinit();
    const input = try memory.alloc(rows_max * c.hidden * 2);
    const images = try memory.alloc(rows_max * 4);
    const synthetic_logits = try memory.alloc(rows_max * c.experts * 4);
    const synthetic_bias = try memory.alloc(c.experts * 4);
    const trace = try memory.alloc(4);
    const reference = try Output.init(&memory);
    const trial = try Output.init(&memory);
    for (images.slice(i32, rows_max), 0..) |*cell, i| cell.* = @intCast(i % 2);
    @memset(synthetic_bias.slice(u32, c.experts), 0);
    for (1..rows_max + 1) |rows| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var p: [layouts.len]mtl.Pipeline = undefined;
        p[0] = m.kernels.get("ds_gemv_bf_f32");
        var count: usize = 1;
        defer for (p[1..count]) |item| item.deinit();
        for (layouts[1..], 1..) |g, i| {
            p[i] = try projection(m.device, lib, rows, g.cols);
            count += 1;
        }
        for ([_]usize{ 0, 10, 24, 39 }) |layer| {
            const w = &m.weights.layers[layer].moe;
            if (w.router.offset % 8 != 0) return error.RouterVariantUnalignedWeights;
            for (input.slice(u16, rows * c.hidden), 0..) |*cell, i| {
                const f: f32 = @floatFromInt(@as(i32, @intCast((i * 127 + layer * 89 + 71) % 997)) - 498);
                cell.* = @truncate(@as(u32, @bitCast(f / 256.0)) >> 16);
            }
            const t = Trial{ .m = m, .w = w, .rows = rows, .input = input, .images = images, .unused_trace = trace, .projections = p, .selectors = selectors };
            _ = try t.time(0, 0, reference, 1, null);
            const arms = layouts.len + 2;
            for (1..arms) |i| {
                const l = if (i < layouts.len) i else 0;
                const s = if (i < layouts.len) 0 else i - layouts.len + 1;
                _ = try t.time(l, s, trial, 1, null);
                reference.verify(trial, rows, true) catch |err| {
                    std.log.err("router variant differs layer={d} rows={d} layout={d} selector={d}", .{ layer, rows, l, s });
                    return err;
                };
                std.log.info("router exact layer={d} rows={d} arm={d} f32={d} ids={d}", .{ layer, rows, i, rows * (c.experts + c.topk), rows * c.topk });
            }
            if (layer == 0) for (0..2) |special| {
                @memset(synthetic_logits.slice(u32, rows * c.experts), 0);
                if (special != 0) {
                    for (synthetic_logits.slice(f32, rows * c.experts), 0..) |*cell, i|
                        cell.* = @as(f32, @floatFromInt(i % 17)) * 0.25;
                    synthetic_logits.slice(u32, rows * c.experts)[17] = 0x7fc00000;
                }
                const synthetic = [2]Buffer{ synthetic_logits, synthetic_bias };
                _ = try t.time(0, 0, reference, 1, synthetic);
                for (1..selectors.len) |s| {
                    _ = try t.time(0, s, trial, 1, synthetic);
                    try reference.verify(trial, rows, false);
                    std.log.info("router special rows={d} fixture={d} selector={d} f32={d} ids={d}", .{ rows, special, s, rows * c.topk, rows * c.topk });
                }
            };
            for (0..7) |run| for (0..arms) |j| {
                const i = if (run % 2 == 0) j else arms - 1 - j;
                const l = if (i < layouts.len) i else 0;
                const s = if (i < layouts.len) 0 else i - layouts.len + 1;
                const us = try t.time(l, s, trial, 10, null);
                std.log.info("router time layer={d} rows={d} arm={d} run={d} us={d:.4}", .{ layer, rows, i, run, us });
            };
        }
    }
}

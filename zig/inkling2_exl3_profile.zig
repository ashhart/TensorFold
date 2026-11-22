//! Decode expert-store GPU timing on a real Inkling-2 pack: whole encode() and, for EXL3, per-projection stages.
//! Usage: tf-inkling2-exl3-profile <pack-dir> [--rows N] [--iters N] [--official] [--stages]
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("src/core/exl3_projection.zig");
const ck = @import("src/core/checkpoint_metal.zig");
const exl3s = @import("src/families/inkling2/metal_experts_exl3.zig");
const ofs = @import("src/families/inkling2/metal_experts.zig");
const mapped_mod = @import("src/families/inkling2/metal_mapped.zig");
const config = @import("src/families/inkling2/config.zig");

const SwigluArgs = extern struct { pairs: u32, inter: u32, gate_cap: f32, up_cap: f32 };
const CombineArgs = extern struct { rows: u32, topk: u32, hidden: u32, experts: u32 };
const glue_threads = 256;

const Stage = enum(u4) { whole, gate, up, swiglu, down, combine, empty, rot_in, matmul, rot_out };

const Ctx = struct {
    exl3_store: ?*const exl3s.Store = null,
    ofs_store: ?*const ofs.Store = null,
    layer: u32 = 0,
    rows: u32 = 1,
    x: exl3.View = undefined,
    ids: exl3.View = undefined,
    weights: exl3.View = undefined,
    out: exl3.View = undefined,
    stage: Stage = .whole,
    proj: usize = 0,
};
var g: Ctx = .{};

fn bindV(e: mtl.ComputeEncoder, v: exl3.View, index: usize) void {
    e.setBuffer(v.buffer, v.offset, index);
}

fn expertsAt(s: *const exl3s.Store, layer: usize, proj: usize) exl3.Experts {
    const index = (layer * 3 + proj) * @as(usize, s.experts);
    return .{
        .descriptors = .{ .buffer = s.descriptors, .offset = index * @sizeOf(exl3.ExpertDescriptor) },
        .count = @intCast(s.experts),
        .k = @intCast(if (proj == 2) s.inter else s.hidden),
        .n = @intCast(if (proj == 2) s.hidden else s.inter),
    };
}

/// exl3_projection.zig's TfExl3Args, same extern layout (96 bytes).
const Exl3Args = extern struct {
    k: u32,
    n: u32,
    rows: u32,
    halves: u32,
    codebook: u32,
    su_signs: u32,
    sv_signs: u32,
    has_bias: u32,
    output_f32: u32,
    input_ids: u32,
    output_ids: u32,
    routed: u32,
    expert_id: u32,
    repeat: u32,
    indexed: u32 = 0,
    experts: u32 = 0,
    x_stride: u64,
    y_stride: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
};

comptime {
    std.debug.assert(@sizeOf(Exl3Args) == 96);
}

/// One sub-dispatch of exl3_projection.stages() for the indexed expert path, same buffers/args/grid.
fn exl3Sub(s: *const exl3s.Store, e: mtl.ComputeEncoder, layer: u32, proj: usize, x: exl3.View, y: exl3.View, ids: exl3.View, rows: u32, which: Stage) !void {
    if (rows == 0 or rows > s.max_rows or @as(usize, layer) >= s.layers) return error.ExpertDispatch;
    const pairs = rows * s.topk;
    const k: u32 = if (proj == 2) s.inter else s.hidden;
    const n: u32 = if (proj == 2) s.hidden else s.inter;
    const repeat: u32 = if (proj == 2) 1 else s.topk;
    const sc: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[4] };
    const sums_off = sc.offset + ((@as(usize, pairs) * k * 2 + 15) & ~@as(usize, 15));
    const sums: exl3.View = .{ .buffer = s.scratch, .offset = sums_off };
    const a: Exl3Args = .{
        .k = k,
        .n = n,
        .rows = pairs,
        .halves = 0,
        .codebook = 0,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 0,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 1,
        .expert_id = 0,
        .repeat = repeat,
        .indexed = 1,
        .experts = s.experts,
        .x_stride = if (proj == 2) @intCast(s.inter) else @intCast(s.hidden),
        .y_stride = if (proj == 2) @intCast(s.hidden) else @intCast(s.inter),
        .tile_k_stride = 0,
        .tile_n_stride = 0,
    };
    const l: usize = layer;
    const experts = expertsAt(s, l, proj);
    const routes: exl3.View = ids;
    switch (which) {
        .rot_in => {
            e.setPipeline(s.proj.rotate_in);
            bindV(e, x, 0);
            bindV(e, sc, 1);
            bindV(e, sc, 2);
            e.setValue(a, 3);
            bindV(e, sc, 4);
            bindV(e, routes, 5);
            bindV(e, experts.descriptors, 6);
            e.dispatchGroups(.{ .width = k / 128, .height = pairs }, .{ .width = 32 });
        },
        .matmul => {
            e.setPipeline(s.proj.matmul);
            bindV(e, sc, 0);
            bindV(e, sc, 1);
            bindV(e, sums, 2);
            e.setValue(a, 3);
            bindV(e, routes, 4);
            bindV(e, experts.descriptors, 5);
            e.dispatchGroups(.{ .width = n / 16, .height = pairs }, .{ .width = 32 });
        },
        .rot_out => {
            e.setPipeline(s.proj.rotate_out);
            bindV(e, sums, 0);
            bindV(e, sc, 1);
            bindV(e, sc, 2);
            bindV(e, y, 3);
            e.setValue(a, 4);
            bindV(e, sc, 5);
            bindV(e, routes, 6);
            bindV(e, experts.descriptors, 7);
            e.dispatchGroups(.{ .width = n / 128, .height = pairs }, .{ .width = 32 });
        },
        else => return error.BadStage,
    }
}

/// One stage of the EXL3 encode() sequence, byte-for-byte the same dispatches as Store.encode.
fn exl3Stage(s: *const exl3s.Store, e: mtl.ComputeEncoder, layer: u32, x: exl3.View, ids: exl3.View, weights: exl3.View, out: exl3.View, rows: u32, which: Stage) !void {
    if (rows == 0 or rows > s.max_rows or @as(usize, layer) >= s.layers) return error.ExpertDispatch;
    const pairs = rows * s.topk;
    const gy: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[0] };
    const uy: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[1] };
    const ay: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[2] };
    const dy: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[3] };
    const sc: exl3.View = .{ .buffer = s.scratch, .offset = s.planes[4] };
    const l: usize = layer;
    const gate_rows: exl3.Rows = .{
        .count = @as(usize, pairs),
        .x_stride = @as(usize, s.hidden),
        .y_stride = @as(usize, s.inter),
        .output = .bf16,
        .routes = .{ .ids = ids, .repeat = @as(usize, s.topk) },
    };
    switch (which) {
        .gate => try exl3.encodeIndexed(&s.proj, e, expertsAt(s, l, 0), x, gy, sc, gate_rows),
        .up => try exl3.encodeIndexed(&s.proj, e, expertsAt(s, l, 1), x, uy, sc, gate_rows),
        .swiglu => {
            e.setPipeline(s.swiglu);
            bindV(e, gy, 0);
            bindV(e, uy, 1);
            bindV(e, ay, 2);
            e.setValue(SwigluArgs{ .pairs = pairs, .inter = s.inter, .gate_cap = s.gate_cap, .up_cap = s.up_cap }, 3);
            e.dispatchThreads(mtl.Size.of(std.math.mul(u32, pairs, s.inter) catch return error.ExpertDispatch, 1, 1), mtl.Size.of(glue_threads, 1, 1));
        },
        .down => try exl3.encodeIndexed(&s.proj, e, expertsAt(s, l, 2), ay, dy, sc, .{
            .count = @as(usize, pairs),
            .x_stride = @as(usize, s.inter),
            .y_stride = @as(usize, s.hidden),
            .output = .bf16,
            .routes = .{ .ids = ids, .repeat = 1 },
        }),
        .combine => {
            e.setPipeline(s.combine);
            bindV(e, dy, 0);
            bindV(e, ids, 1);
            bindV(e, weights, 2);
            bindV(e, out, 3);
            e.setValue(CombineArgs{ .rows = rows, .topk = s.topk, .hidden = s.hidden, .experts = s.experts }, 4);
            e.dispatchThreads(mtl.Size.of(s.hidden, rows, 1), mtl.Size.of(glue_threads, 1, 1));
        },
        else => return error.BadStage,
    }
}

fn dispatch(e: mtl.ComputeEncoder) !void {
    if (g.exl3_store) |s| {
        switch (g.stage) {
            .whole => try s.encode(e, g.layer, g.x, g.ids, g.weights, g.out, g.rows),
            .empty => {},
            .rot_in, .matmul, .rot_out => {
                const plane_at: usize = if (g.proj == 2) s.planes[3] else if (g.proj == 1) s.planes[1] else s.planes[0];
                const yplane: exl3.View = .{ .buffer = s.scratch, .offset = plane_at };
                try exl3Sub(s, e, g.layer, g.proj, g.x, yplane, g.ids, g.rows, g.stage);
            },
            else => try exl3Stage(s, e, g.layer, g.x, g.ids, g.weights, g.out, g.rows, g.stage),
        }
    } else if (g.ofs_store) |s| {
        try s.encode(e, g.layer, g.x, g.ids, g.weights, g.out, g.rows);
    } else return error.NoStore;
}

fn lt(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn stats(name: []const u8, times: []const f64) void {
    const sorted = std.heap.page_allocator.dupe(f64, times) catch @panic("oom");
    defer std.heap.page_allocator.free(sorted);
    std.mem.sort(f64, sorted, {}, lt);
    const n = sorted.len;
    var sum: f64 = 0;
    for (sorted) |t| sum += t;
    std.debug.print("{s:24} n={d:3} mean={d:8.3} med={d:8.3} min={d:8.3} max={d:8.3} ms\n", .{ name, n, sum / @as(f64, @floatFromInt(n)), sorted[n / 2], sorted[0], sorted[n - 1] });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-inkling2-exl3-profile <pack-dir> [--rows N] [--iters N] [--official] [--stages]\n", .{});
        return error.Usage;
    }
    const dir = args[1];
    var rows: u32 = 1;
    var iters: usize = 30;
    var official = false;
    var stages = false;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--official")) {
            official = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--stages")) {
            stages = true;
            continue;
        }
        if (i + 1 >= args.len) return error.OptionValueRequired;
        const v = try std.fmt.parseInt(u32, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--rows")) rows = v else if (std.mem.eql(u8, args[i], "--iters")) iters = @intCast(v) else return error.UnknownOption;
        i += 1;
    }

    var why: config.Why = .{};
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(json);
    const cfg = try config.parse(gpa, json, &why);
    const t = &cfg.text;
    std.debug.print("config: hidden={d} inter={d} experts={d} topk={d} layers={d} mtp={}\n", .{ t.hidden, t.expert_inter, t.experts, t.topk, t.layers, cfg.mtp != null });

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();

    var mapped = try mapped_mod.Set.open(gpa, io, dir);
    defer mapped.deinit();
    var cp = ck.Checkpoint.init(gpa);
    defer cp.deinit();
    try mapped.adopt(&cp, device);
    mapped.warm(8);

    var exl3_seen: usize = 0;
    for (0..t.layers) |layer| {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.0.gate_proj.trellis", .{layer});
        if (cp.has(name)) exl3_seen += 1;
    }
    const is_exl3 = !official and exl3_seen == @as(usize, t.layers);
    std.debug.print("format: {s} (exl3 layers {d}/{d})\n", .{ if (is_exl3) "exl3" else "official", exl3_seen, t.layers });

    const max_rows: u32 = 512;
    const pairs = rows * t.topk;

    const x_buf = try device.buffer(@as(usize, rows) * @as(usize, t.hidden) * 2, mtl.ResourceOptions.shared);
    defer x_buf.deinit();
    {
        const xs = x_buf.slice(u16, @as(usize, rows) * @as(usize, t.hidden));
        for (xs, 0..) |*v, j| v.* = if (j % 7 == 0) 0 else 0x3F80;
    }
    const ids_buf = try device.buffer(@as(usize, pairs) * 4, mtl.ResourceOptions.shared);
    defer ids_buf.deinit();
    for (ids_buf.slice(u32, pairs), 0..) |*v, j| v.* = @intCast(j % t.experts);
    const w_buf = try device.buffer(@as(usize, pairs) * 4, mtl.ResourceOptions.shared);
    defer w_buf.deinit();
    for (w_buf.slice(f32, pairs)) |*v| v.* = 1.0;
    const out_buf = try device.buffer(@as(usize, rows) * @as(usize, t.hidden) * 2, mtl.ResourceOptions.shared);
    defer out_buf.deinit();

    g.x = .{ .buffer = x_buf };
    g.ids = .{ .buffer = ids_buf };
    g.weights = .{ .buffer = w_buf };
    g.out = .{ .buffer = out_buf };
    g.rows = rows;

    const whole_times = try gpa.alloc(f64, iters);
    defer gpa.free(whole_times);

    if (is_exl3) {
        var store = try exl3s.Store.loadWith(gpa, device, &cp, &cfg, max_rows, .{});
        defer store.deinit();
        g.exl3_store = &store;
        std.debug.print("exl3 store: scratch {d:.1} MiB, layers {d}\n", .{ @as(f64, @floatFromInt(store.scratchBytes())) / (1 << 20), store.layers });
        for (0..3) |_| {
            g.stage = .whole;
            g.layer = 0;
            _ = try timeOnce(queue);
        }
        for (whole_times, 0..) |*slot, it| {
            g.stage = .whole;
            g.layer = @intCast(it % store.layers);
            slot.* = try timeOnce(queue);
        }
        stats("exl3 encode() whole", whole_times);

        if (stages) {
            const stage_names = [_][]const u8{ "gate_idx", "up_idx", "swiglu", "down_idx", "combine", "empty_cb", "rot_in_g", "matmul_g", "rot_out_g", "rot_in_d", "matmul_d", "rot_out_d" };
            const stage_vals = [_]Stage{ .gate, .up, .swiglu, .down, .combine, .empty, .rot_in, .matmul, .rot_out, .rot_in, .matmul, .rot_out };
            var acc: [12][]f64 = undefined;
            for (&acc) |*a| a.* = try gpa.alloc(f64, iters);
            defer {
                for (&acc) |a| gpa.free(a);
            }
            for (0..3) |_| {
                for (stage_vals, 0..) |sv, s| {
                    g.stage = sv;
                    g.proj = if (s >= 9) 2 else 0;
                    g.layer = 0;
                    _ = try timeOnce(queue);
                }
            }
            for (0..iters) |it| {
                for (stage_vals, 0..) |sv, s| {
                    g.stage = sv;
                    g.proj = if (s >= 9) 2 else 0;
                    g.layer = @intCast(it % store.layers);
                    acc[s][it] = try timeOnce(queue);
                }
            }
            for (stage_names, 0..) |nm, s| stats(nm, acc[s]);
        }
    } else {
        var store = try ofs.Store.loadWith(gpa, device, &cp, &cfg, max_rows, .{});
        defer store.deinit();
        g.ofs_store = &store;
        for (0..3) |_| {
            g.layer = 0;
            _ = try timeOnce(queue);
        }
        for (whole_times, 0..) |*slot, it| {
            g.layer = @intCast(it % store.layers.len);
            slot.* = try timeOnce(queue);
        }
        stats("official encode() whole", whole_times);
    }
}

fn timeOnce(queue: mtl.Queue) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    dispatch(e) catch |err| {
        e.end();
        return err;
    };
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("gpu failure: {s}", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e3;
}

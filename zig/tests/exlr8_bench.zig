//! EXLR8 Metal projections on a real MoE layer: FP64 bounds and row windows on its experts, then weight bytes a second.

const std = @import("std");
const mtl = @import("metal");
const ex = @import("exlr8");
const op = @import("exlr8_metal").projection;
const ref = @import("exlr8_ref");
const layer_ = @import("exlr8_layer");
const L = ex.experts;
const a = std.heap.page_allocator;
const slots = layer_.slots;

fn buffer(device: mtl.Device, items: anytype) !mtl.Buffer {
    const raw = std.mem.sliceAsBytes(items);
    const b = try device.buffer(@max(raw.len, 16), mtl.ResourceOptions.shared);
    @memcpy(b.contents()[0..raw.len], raw);
    return b;
}
fn zeros(device: mtl.Device, len: usize) !mtl.Buffer {
    const b = try device.buffer(len, mtl.ResourceOptions.shared);
    @memset(b.contents()[0..len], 0);
    return b;
}

/// A set's jobs and slot tables on the device.
const Set = struct { host: layer_.Set, up: mtl.Buffer, down: mtl.Buffer, sv: mtl.Buffer, suh: mtl.Buffer };

/// Every buffer a round reads and writes, inputs and outputs for 16 rows.
const Gpu = struct {
    weights: mtl.Buffer,
    bases: mtl.Buffer,
    halves: mtl.Buffer,
    signs: mtl.Buffer,
    x: mtl.Buffer,
    act: mtl.Buffer,
    w: mtl.Buffer,
    r: mtl.Buffer,
    gu: mtl.Buffer,
    ar: mtl.Buffer,
    z: mtl.Buffer,
    y: mtl.Buffer,
    sets: []Set,
    /// Sets a job for every projection (1: one; 2 or 4 split a job's blocks).
    sets_a_job: usize = 1,

    fn tables(g: Gpu, jobs: mtl.Buffer, count: usize) op.Tables {
        return .{ .weights = g.weights, .bases = g.bases, .halves = g.halves, .signs = g.signs, .jobs = jobs, .count = count, .sets = g.sets_a_job };
    }
};

const Stage = enum { rotate_in, gate_up, rotate_down, down, combine };
const all = [_]Stage{ .rotate_in, .gate_up, .rotate_down, .down, .combine };

fn round(queue: mtl.Queue, p: *const op.Pipelines, g: Gpu, s: Set, rows: u32, first: usize, stages: []const Stage) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (stages, 0..) |st, i| {
        if (i > 0) e.barrier();
        switch (st) {
            .rotate_in => try op.encodeRotate(p, e, .{ .buf = g.x, .off = first * L.hidden * 2 }, g.signs, g.halves, .{ .buf = g.r }, layer_.rotateIn(rows), 48, 1, null),
            .gate_up => try op.encodeProject(p, e, .{ .buf = g.r }, g.tables(s.up, s.host.up.len), .{ .buf = g.gu }, rows, L.hidden, slots * 4096, true, null),
            .rotate_down => try op.encodeRotate(p, e, .{ .buf = g.act, .off = first * slots * 2048 * 2 }, g.signs, g.halves, .{ .buf = g.ar }, layer_.rotateDown(rows), 16, slots, .{ .buf = s.suh }),
            .down => try op.encodeProject(p, e, .{ .buf = g.ar }, g.tables(s.down, s.host.down.len), .{ .buf = g.z }, rows, slots * 2048, slots * L.hidden, true, null),
            .combine => try op.encodeCombine(p, e, .{ .buf = g.z }, .{ .buf = g.w, .off = first * slots * 4 }, .{ .buf = s.sv }, g.halves, g.signs, .{ .buf = g.y }, layer_.combine(rows), 48),
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU failure: {s}\n", .{m});
        return error.GpuFailure;
    }
    return cb.gpuSeconds();
}

/// Each row alone, then every window 1-16: the gate/up sums and the combined down output, byte for byte.
fn windows(queue: mtl.Queue, p: *const op.Pipelines, g: Gpu) !usize {
    const gu_row = slots * 4096 * 4;
    const y_row = L.hidden * 4;
    const alone = try a.alloc(u8, 16 * (gu_row + y_row));
    defer a.free(alone);
    for (0..16) |r| {
        _ = try round(queue, p, g, g.sets[0], 1, r, &all);
        @memcpy(alone[r * gu_row ..][0..gu_row], g.gu.contents()[0..gu_row]);
        @memcpy(alone[16 * gu_row + r * y_row ..][0..y_row], g.y.contents()[0..y_row]);
    }
    var pairs: usize = 0;
    for (1..17) |m| {
        _ = try round(queue, p, g, g.sets[0], @intCast(m), 0, &all);
        for (0..m) |r| {
            if (!std.mem.eql(u8, g.gu.contents()[r * gu_row ..][0..gu_row], alone[r * gu_row ..][0..gu_row]) or
                !std.mem.eql(u8, g.y.contents()[r * y_row ..][0..y_row], alone[16 * gu_row + r * y_row ..][0..y_row])) return error.RowInvariantMismatch;
            pairs += 1;
        }
    }
    return pairs;
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        std.debug.print("tf-exlr8-bench DIR [MANIFEST] [LAYER] [SETS]: home, layer 40 and 4 sets by default\n", .{});
        return;
    }
    const manifest_name = if (argv.len > 2) argv[2] else "home";
    const number = if (argv.len > 3) try std.fmt.parseInt(u16, argv[3], 10) else 40;
    const count = if (argv.len > 4) try std.fmt.parseInt(usize, argv[4], 10) else 4;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try op.Pipelines.load(device);
    defer p.deinit();
    var layer = try L.Layer.open(a, init.io, argv[1], number);
    defer layer.close(init.io);
    const mpath = try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-{s}.json", .{ argv[1], manifest_name });
    var m = try ex.manifest.Manifest.load(a, init.io, mpath);
    defer m.deinit();
    const widths = m.layers[number] orelse return error.LayerHasNoExperts;
    const host = try layer_.Host.build(&layer, widths, count);
    const in = try layer_.Inputs.init();
    const sets = try a.alloc(Set, count);
    for (sets, host.sets) |*s, h| s.* = .{ .host = h, .up = try buffer(device, h.up), .down = try buffer(device, h.down), .sv = try buffer(device, &h.sv), .suh = try buffer(device, &h.suh) };
    const g: Gpu = .{
        .weights = try buffer(device, host.weights.items),
        .bases = try buffer(device, host.bases.items),
        .halves = try buffer(device, host.halves.items),
        .signs = try buffer(device, &host.signs),
        .x = try buffer(device, in.x),
        .act = try buffer(device, in.act),
        .w = try buffer(device, in.w),
        .r = try zeros(device, 16 * L.hidden * 4),
        .gu = try zeros(device, 16 * slots * 4096 * 4),
        .ar = try zeros(device, 16 * slots * 2048 * 4),
        .z = try zeros(device, 16 * slots * L.hidden * 4),
        .y = try zeros(device, 16 * L.hidden * 4),
        .sets = sets,
    };
    std.debug.print("EXLR8 layer {d} ({s}) on {s}: {d} sets of 8 routed experts + the shared one, {d:.1} MB of weights a set\n", .{ number, manifest_name, device.name(), count, @as(f64, @floatFromInt(host.sets[0].up_bytes + host.sets[0].down_bytes)) / 1e6 });
    var check: ref.Check = .{};
    _ = try round(queue, &p, g, sets[0], 16, 0, &all);
    try layer_.fp64(&layer, widths, host.sets[0], in, g.gu.slice(f32, 16 * slots * 4096), g.y.slice(f32, 16 * L.hidden), &check);
    std.debug.print("PASS FP64: {d} values (gate/up halves and the combined down), max error {e}, max {d:.4} of the bound\n", .{ check.values, check.max_error, check.max_fraction });
    std.debug.print("PASS rows: {d} row/window pairs byte-equal to rows alone (gate/up and combined down)\n", .{try windows(queue, &p, g)});
    for (0..30) |i| _ = try round(queue, &p, g, sets[i % count], 16, 0, &all);
    var t: [24]f64 = undefined;
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| {
        var us: [all.len]f64 = undefined;
        for (all, 0..) |st, si| {
            for (&t, 0..) |*v, i| v.* = try round(queue, &p, g, sets[i % count], rows, 0, &.{st});
            us[si] = median(&t) * 1e6;
        }
        for (&t, 0..) |*v, i| v.* = try round(queue, &p, g, sets[i % count], rows, 0, &all);
        const whole = median(&t) * 1e6;
        const up: f64 = @floatFromInt(host.sets[0].up_bytes);
        const down: f64 = @floatFromInt(host.sets[0].down_bytes);
        std.debug.print("rows {d:2}: layer {d:7.1} us | rotate x {d:5.1} us, gate/up {d:7.1} us ({d:5.1} GB/s), rotate down inputs {d:5.1} us, down {d:7.1} us ({d:5.1} GB/s), combine {d:5.1} us | rotations {d:4.1}% of the stages\n", .{
            rows, whole, us[0], us[1], up / us[1] / 1e3, us[2], us[3], down / us[3] / 1e3, us[4], 100 * (us[0] + us[2]) / (us[0] + us[1] + us[2] + us[3] + us[4]),
        });
        if (rows > 4) continue;
        // The gate/up jobs again with their blocks split over 2 and 4 sets a job (one set where no kernel fits).
        var split: [2]f64 = undefined;
        for ([_]usize{ 2, 4 }, &split) |n, *out| {
            var gs = g;
            gs.sets_a_job = n;
            for (&t, 0..) |*v, i| v.* = try round(queue, &p, gs, sets[i % count], rows, 0, &.{.gate_up});
            out.* = median(&t) * 1e6;
        }
        std.debug.print("rows {d:2}: gate/up {d:7.1} us at 2 sets a job, {d:7.1} us at 4\n", .{ rows, split[0], split[1] });
    }
}

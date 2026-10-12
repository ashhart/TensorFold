//! EXLR8 Metal projections on a real MoE layer: FP64 bounds and row windows on its experts, then weight bytes a second.

const std = @import("std");
const mtl = @import("metal");
const ex = @import("exlr8");
const op = @import("exlr8_projection");
const ref = @import("exlr8_ref.zig");
const codec = ex.codec;
const L = ex.experts;
const a = std.heap.page_allocator;
const opts = mtl.ResourceOptions.shared;
const slots = 9;

/// Every buffer one set of nine experts (eight routed and the shared one) needs, with its jobs.
const Set = struct {
    experts: [slots]u16,
    jobs_up: mtl.Buffer,
    jobs_down: mtl.Buffer,
    up: usize,
    down: usize,
    up_bytes: usize,
    down_bytes: usize,
    sv: mtl.Buffer,
    suh: mtl.Buffer,
};

const Layer = struct {
    weights: mtl.Buffer,
    bases: mtl.Buffer,
    halves: mtl.Buffer,
    signs: mtl.Buffer,
    sets: []Set,

    fn tables(l: Layer, jobs: mtl.Buffer, count: usize) op.Tables {
        return .{ .weights = l.weights, .bases = l.bases, .halves = l.halves, .signs = l.signs, .jobs = jobs, .count = count };
    }
};

fn buffer(device: mtl.Device, bytes: usize) !mtl.Buffer {
    const b = try device.buffer(@max(bytes, 16), opts);
    @memset(b.contents()[0..b.length()], 0);
    return b;
}

const Builder = struct {
    w: std.ArrayList(u8) = .empty,
    bases: std.ArrayList(u64) = .empty,
    halves: std.ArrayList(f16) = .empty,

    fn scales(b: *Builder, v: []align(1) const f16) !u32 {
        const at: u32 = @intCast(b.halves.items.len);
        for (v) |x| try b.halves.append(a, x);
        return at;
    }
    fn half(b: *Builder, h: L.Half) !usize {
        const at = b.w.items.len;
        try b.w.appendSlice(a, h.trellis);
        return at;
    }
};

fn build(device: mtl.Device, layer: *const L.Layer, widths: *const ex.manifest.LayerWidths, sets: usize) !Layer {
    var b: Builder = .{};
    defer b.w.deinit(a);
    defer b.bases.deinit(a);
    defer b.halves.deinit(a);
    const out = try a.alloc(Set, sets);
    for (out, 0..) |*s, si| {
        var up: std.ArrayList(op.Job) = .empty;
        defer up.deinit(a);
        var down: std.ArrayList(op.Job) = .empty;
        defer down.deinit(a);
        var sv: [slots]u32 = undefined;
        var suh: [slots]u32 = undefined;
        s.up_bytes = 0;
        s.down_bytes = 0;
        for (0..slots) |slot| {
            const e: u16 = if (slot == slots - 1) L.shared else @intCast((si * 8 + slot * 31 + 7) % 256);
            s.experts[slot] = e;
            for ([_]L.Proj{ .gate, .up }, 0..) |p, pi| for (0..L.halves) |h| {
                const k = if (e == L.shared) (if (p == .gate) widths.shared_gate[h] else widths.shared_up[h]) else (if (p == .gate) widths.gate[e][h] else widths.up[e][h]);
                const half = try layer.half(e, p, h, k);
                const at = try b.half(half);
                s.up_bytes += half.trellis.len;
                const base: u32 = @intCast(b.bases.items.len);
                for (0..48) |ib| try b.bases.append(a, at + ib * 64 * codec.tileBytes(k));
                try up.append(a, .{ .k = k, .blocks = 48, .base = base, .in_scale = try b.scales(half.block), .out_scale = try b.scales(half.chan), .y_col = @intCast(slot * 4096 + pi * 2048 + h * 128) });
            };
            const kd = if (e == L.shared) widths.shared_down else widths.down[e];
            var at: [L.halves]usize = undefined;
            var chan: [2048]f16 = undefined;
            for (0..L.halves) |h| {
                const half = try layer.half(e, .down, h, kd);
                at[h] = try b.half(half);
                s.down_bytes += half.trellis.len;
                for (half.chan, 0..) |c, i| chan[h * 128 + i] = c;
                if (h == 0) sv[slot] = try b.scales(half.block);
            }
            suh[slot] = try b.scales(&chan);
            for (0..48) |ob| {
                const base: u32 = @intCast(b.bases.items.len);
                for (0..L.halves) |h| try b.bases.append(a, at[h] + ob * 64 * codec.tileBytes(kd));
                try down.append(a, .{ .k = kd, .blocks = L.halves, .base = base, .x_col = @intCast(slot * 2048), .y_col = @intCast(slot * L.hidden + ob * 128), .raw = 1 });
            }
        }
        s.up = up.items.len;
        s.down = down.items.len;
        s.jobs_up = try buffer(device, up.items.len * @sizeOf(op.Job));
        @memcpy(s.jobs_up.slice(op.Job, up.items.len), up.items);
        s.jobs_down = try buffer(device, down.items.len * @sizeOf(op.Job));
        @memcpy(s.jobs_down.slice(op.Job, down.items.len), down.items);
        s.sv = try buffer(device, slots * 4);
        @memcpy(s.sv.slice(u32, slots), &sv);
        s.suh = try buffer(device, slots * 4);
        @memcpy(s.suh.slice(u32, slots), &suh);
    }
    const l: Layer = .{ .weights = try buffer(device, b.w.items.len), .bases = try buffer(device, b.bases.items.len * 8), .halves = try buffer(device, b.halves.items.len * 2), .signs = try buffer(device, 2 * L.hidden), .sets = out };
    @memcpy(l.weights.contents()[0..b.w.items.len], b.w.items);
    @memcpy(l.bases.slice(u64, b.bases.items.len), b.bases.items);
    @memcpy(l.halves.slice(f16, b.halves.items.len), b.halves.items);
    @memcpy(l.signs.slice(i8, L.hidden), layer.signs_in);
    @memcpy(l.signs.slice(i8, 2 * L.hidden)[L.hidden..], layer.signs_out);
    return l;
}

/// The round's buffers for up to 16 rows: inputs, rotations, outputs and routing weights.
const Io = struct {
    x: mtl.Buffer,
    act: mtl.Buffer,
    r: mtl.Buffer,
    gu: mtl.Buffer,
    ar: mtl.Buffer,
    z: mtl.Buffer,
    y: mtl.Buffer,
    w: mtl.Buffer,

    fn init(device: mtl.Device) !Io {
        var prng = std.Random.DefaultPrng.init(0xE8B);
        const rnd = prng.random();
        const io: Io = .{
            .x = try buffer(device, 16 * L.hidden * 2),
            .act = try buffer(device, 16 * slots * 2048 * 2),
            .r = try buffer(device, 16 * L.hidden * 4),
            .gu = try buffer(device, 16 * slots * 4096 * 4),
            .ar = try buffer(device, 16 * slots * 2048 * 4),
            .z = try buffer(device, 16 * slots * L.hidden * 4),
            .y = try buffer(device, 16 * L.hidden * 4),
            .w = try buffer(device, 16 * slots * 4),
        };
        for (io.x.slice(u16, 16 * L.hidden)) |*v| v.* = ref.bf16Of(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -1000, 1000))) * 0.0021);
        for (io.act.slice(u16, 16 * slots * 2048)) |*v| v.* = ref.bf16Of(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -1000, 1000))) * 0.0013);
        for (io.w.slice(f32, 16 * slots)) |*v| v.* = @as(f32, @floatFromInt(rnd.intRangeLessThan(i32, 1, 400))) / 512.0;
        return io;
    }
};

const Stage = enum { rotate_in, gate_up, rotate_down, down, combine };

fn encode(p: *const op.Pipelines, e: mtl.ComputeEncoder, l: Layer, s: Set, io: Io, rows: u32, first: usize, stages: []const Stage) !void {
    for (stages, 0..) |st, i| {
        if (i > 0) e.barrier();
        switch (st) {
            .rotate_in => try op.encodeRotate(p, e, .{ .buf = io.x, .off = first * L.hidden * 2 }, l.signs, l.halves, .{ .buf = io.r }, .{ .rows = rows, .x_stride = L.hidden, .r_stride = L.hidden, .sign = 0 }, 48, 1, null),
            .gate_up => try op.encodeProject(p, e, .{ .buf = io.r }, l.tables(s.jobs_up, s.up), .{ .buf = io.gu }, rows, L.hidden, slots * 4096, true),
            .rotate_down => try op.encodeRotate(p, e, .{ .buf = io.act, .off = first * slots * 2048 * 2 }, l.signs, l.halves, .{ .buf = io.ar }, .{ .rows = rows, .x_stride = slots * 2048, .r_stride = slots * 2048, .slot_cols = 2048, .scales = 1 }, 16, slots, .{ .buf = s.suh }),
            .down => try op.encodeProject(p, e, .{ .buf = io.ar }, l.tables(s.jobs_down, s.down), .{ .buf = io.z }, rows, slots * 2048, slots * L.hidden, true),
            .combine => try op.encodeCombine(p, e, .{ .buf = io.z }, .{ .buf = io.w, .off = first * slots * 4 }, .{ .buf = s.sv }, l.halves, l.signs, .{ .buf = io.y }, .{ .rows = rows, .slots = slots, .z_stride = slots * L.hidden, .z_slot = L.hidden, .y_stride = L.hidden, .y_f32 = 1, .sign = L.hidden }, 48),
        }
    }
}

const all = [_]Stage{ .rotate_in, .gate_up, .rotate_down, .down, .combine };

fn round(queue: mtl.Queue, p: *const op.Pipelines, l: Layer, s: Set, io: Io, rows: u32, first: usize, stages: []const Stage) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try encode(p, e, l, s, io, rows, first, stages);
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
fn windows(queue: mtl.Queue, p: *const op.Pipelines, l: Layer, io: Io) !usize {
    const s = l.sets[0];
    const gu_row = slots * 4096 * 4;
    const y_row = L.hidden * 4;
    const alone_gu = try a.alloc(u8, 16 * gu_row);
    defer a.free(alone_gu);
    const alone_y = try a.alloc(u8, 16 * y_row);
    defer a.free(alone_y);
    for (0..16) |r| {
        _ = try round(queue, p, l, s, io, 1, r, &all);
        @memcpy(alone_gu[r * gu_row ..][0..gu_row], io.gu.contents()[0..gu_row]);
        @memcpy(alone_y[r * y_row ..][0..y_row], io.y.contents()[0..y_row]);
    }
    var pairs: usize = 0;
    for (1..17) |m| {
        _ = try round(queue, p, l, s, io, @intCast(m), 0, &all);
        for (0..m) |r| {
            if (!std.mem.eql(u8, io.gu.contents()[r * gu_row ..][0..gu_row], alone_gu[r * gu_row ..][0..gu_row]) or
                !std.mem.eql(u8, io.y.contents()[r * y_row ..][0..y_row], alone_y[r * y_row ..][0..y_row])) return error.RowInvariantMismatch;
            pairs += 1;
        }
    }
    return pairs;
}

/// FP64 of every gate/up output and every combined down output of 16 rows, against the exact W of each job.
fn fp64(layer: *const L.Layer, widths: *const ex.manifest.LayerWidths, l: Layer, io: Io, check: *ref.Check) !void {
    const book = try a.create(codec.Codebook);
    defer a.destroy(book);
    book.init();
    const s = l.sets[0];
    const codes = try a.alloc(u8, L.hidden * 128);
    defer a.free(codes);
    var suh: [L.hidden]f32 = undefined;
    var svh: [128]f32 = undefined;
    var in: [L.hidden]f64 = undefined;
    var in_abs: [L.hidden]f64 = undefined;
    var block: [48]f32 = undefined;
    const gu = io.gu.slice(f32, 16 * slots * 4096);
    const want_y = try a.alloc(f64, 16 * L.hidden);
    defer a.free(want_y);
    const mag_y = try a.alloc(f64, 16 * L.hidden);
    defer a.free(mag_y);
    @memset(want_y, 0);
    @memset(mag_y, 0);
    for (s.experts, 0..) |e, slot| {
        for ([_]L.Proj{ .gate, .up }, 0..) |p, pi| for (0..L.halves) |h| {
            const k = if (e == L.shared) (if (p == .gate) widths.shared_gate[h] else widths.shared_up[h]) else (if (p == .gate) widths.gate[e][h] else widths.up[e][h]);
            const half = try layer.half(e, p, h, k);
            try codec.decodeRun(book, k, .out_major, half.trellis, L.hidden, 128, codes);
            half.suh(&suh);
            half.svh(&svh);
            for (&block, half.block) |*v, c| v.* = c;
            const exact = try ref.Exact.init(a, codes, 48, &suh, &svh);
            defer exact.deinit(a);
            for (0..16) |r| {
                for (&in, &in_abs, 0..) |*v, *va, i| {
                    v.* = ref.bf16Value(io.x.slice(u16, 16 * L.hidden)[r * L.hidden + i]);
                    va.* = @abs(v.*);
                }
                var y: [128]f64 = undefined;
                const m = exact.row(&in, &in_abs, &block, &y);
                for (0..128) |c| try check.add(gu[r * slots * 4096 + slot * 4096 + pi * 2048 + h * 128 + c], y[c], @abs(svh[c]) / 128 * m);
            }
        };
        const kd = if (e == L.shared) widths.shared_down else widths.down[e];
        const dcodes = try a.alloc(u8, 2048 * 128);
        defer a.free(dcodes);
        var dsuh: [2048]f32 = undefined;
        for (0..48) |ob| {
            var sb: f32 = 0;
            for (0..L.halves) |h| {
                const half = try layer.half(e, .down, h, kd);
                const tb = 64 * codec.tileBytes(kd);
                try codec.decodeRun(book, kd, .out_major, half.trellis[ob * tb ..][0..tb], 128, 128, dcodes[h * 128 * 128 ..][0 .. 128 * 128]);
                for (half.chan, 0..) |c, i| dsuh[h * 128 + i] = c;
                sb = half.block[ob];
            }
            for (&svh, 0..) |*v, c| v.* = @as(f32, @floatFromInt(layer.signs_out[ob * 128 + c])) * sb;
            const exact = try ref.Exact.init(a, dcodes, 16, &dsuh, &svh);
            defer exact.deinit(a);
            const ones: [16]f32 = @splat(1);
            for (0..16) |r| {
                for (in[0..2048], in_abs[0..2048], 0..) |*v, *va, i| {
                    v.* = ref.bf16Value(io.act.slice(u16, 16 * slots * 2048)[(r * slots + slot) * 2048 + i]);
                    va.* = @abs(v.* * dsuh[i]);
                }
                var y: [128]f64 = undefined;
                const m = exact.row(in[0..2048], in_abs[0..2048], &ones, &y);
                const w: f64 = io.w.slice(f32, 16 * slots)[r * slots + slot];
                for (0..128) |c| {
                    want_y[r * L.hidden + ob * 128 + c] += w * y[c];
                    mag_y[r * L.hidden + ob * 128 + c] += @abs(w * svh[c]) / 128 * m;
                }
            }
        }
    }
    for (io.y.slice(f32, 16 * L.hidden), want_y, mag_y) |got, want, mag| try check.add(got, want, mag);
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        std.debug.print("tf-exlr8-bench DIR [MANIFEST] [LAYER] [SETS]: layer 40, home and 4 sets by default\n", .{});
        return;
    }
    const manifest_name = if (argv.len > 2) argv[2] else "home";
    const number = if (argv.len > 3) try std.fmt.parseInt(u16, argv[3], 10) else 40;
    const sets = if (argv.len > 4) try std.fmt.parseInt(usize, argv[4], 10) else 4;
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
    const l = try build(device, &layer, widths, sets);
    const io = try Io.init(device);
    std.debug.print("EXLR8 layer {d} ({s}) on {s}: {d} sets of 8 routed experts + the shared one, {d:.1} MB of weights a set\n", .{ number, manifest_name, device.name(), sets, @as(f64, @floatFromInt(l.sets[0].up_bytes + l.sets[0].down_bytes)) / 1e6 });
    var check: ref.Check = .{};
    _ = try round(queue, &p, l, l.sets[0], io, 16, 0, &all);
    try fp64(&layer, widths, l, io, &check);
    std.debug.print("PASS FP64: {d} values (gate/up halves and the combined down), max error {e}, max {d:.4} of the bound\n", .{ check.values, check.max_error, check.max_fraction });
    std.debug.print("PASS rows: {d} row/window pairs byte-equal to rows alone (gate/up and combined down)\n", .{try windows(queue, &p, l, io)});
    for (0..30) |i| _ = try round(queue, &p, l, l.sets[i % sets], io, 16, 0, &all);
    const iters = 24;
    var t: [iters]f64 = undefined;
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| {
        var stage_us: [all.len]f64 = undefined;
        for (all, 0..) |st, si| {
            for (&t, 0..) |*v, i| v.* = try round(queue, &p, l, l.sets[i % sets], io, rows, 0, &.{st});
            stage_us[si] = median(&t) * 1e6;
        }
        for (&t, 0..) |*v, i| v.* = try round(queue, &p, l, l.sets[i % sets], io, rows, 0, &all);
        const whole = median(&t) * 1e6;
        const gu_bytes = l.sets[0].up_bytes;
        const down_bytes = l.sets[0].down_bytes;
        std.debug.print("rows {d:2}: layer {d:7.1} us | rotate x {d:5.1} us, gate/up {d:7.1} us ({d:5.1} GB/s), rotate down inputs {d:5.1} us, down {d:7.1} us ({d:5.1} GB/s), combine {d:5.1} us | rotations {d:4.1}% of the stages\n", .{
            rows,                                                  whole,                                                                                                     stage_us[0], stage_us[1],
            @as(f64, @floatFromInt(gu_bytes)) / stage_us[1] / 1e3, stage_us[2],                                                                                               stage_us[3], @as(f64, @floatFromInt(down_bytes)) / stage_us[3] / 1e3,
            stage_us[4],                                           100 * (stage_us[0] + stage_us[2]) / (stage_us[0] + stage_us[1] + stage_us[2] + stage_us[3] + stage_us[4]),
        });
    }
}

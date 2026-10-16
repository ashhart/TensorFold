//! EXLR8 lane projections on CUDA: the Metal check's cases and frozen digests, then a real MoE layer's FP64 and GB/s.
const std = @import("std");
const cuda = @import("cuda");
const ex = @import("exlr8");
const ref = @import("exlr8_ref");
const cases = @import("exlr8_cases");
const layer_ = @import("exlr8_layer");
const frozen = @import("exlr8_frozen");
const check = @import("check.zig");
const Gpu = check.Gpu;
const L = ex.experts;
const slots = layer_.slots;
const rows_max = cases.rows_max;

/// Device buffers for one test, freed together.
const Rig = struct {
    gpu: Gpu,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn put(r: *Rig, items: anytype) !u64 {
        const raw = std.mem.sliceAsBytes(items);
        const b = if (raw.len == 0) try cuda.DeviceBuffer.alloc(r.gpu.d, 16) else try cuda.DeviceBuffer.fromHost(r.gpu.d, raw);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }
    fn zeros(r: *Rig, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(len, 16));
        try b.fill8(0, null);
        // the memset runs on the legacy stream, which a non-blocking stream's kernels do not wait for
        try r.gpu.d.check(r.gpu.d.api.cuStreamSynchronize(null), "cuStreamSynchronize");
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }
    fn get(r: *Rig, ptr: u64, out: []u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = out.len };
        try b.download(0, out);
    }
    fn deinit(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
    }
};

fn rowsArgs(rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) ex.job.Rows {
    return .{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) };
}

fn load(gpu: Gpu) !struct { mod: cuda.Module, k: cuda.exlr8.Kernels } {
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.exlr8);
    errdefer mod.unload();
    return .{ .mod = mod, .k = try cuda.exlr8.Kernels.resolve(mod) };
}

/// `tf-cuda-test exlr8`: every state's weight, then each case's windows, FP64 bound and frozen Metal digests.
pub fn run(gpu: Gpu) !void {
    var l = try load(gpu);
    defer l.mod.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const book = try gpu.gpa.create(ex.codec.Codebook);
    defer gpu.gpa.destroy(book);
    book.init();
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const out = try rig.zeros(65536 * 4);
    try l.k.codebook(s, out);
    try s.synchronize();
    var got: [65536 * 4]u8 = undefined;
    try rig.get(out, &got);
    for (0..65536) |i| try check.expect(std.mem.readInt(u32, got[4 * i ..][0..4], .little) == @as(u32, @bitCast(ex.codec.e4m3Value(book.codes[i]))), "exlr8 weight of state {d}", .{i});
    check.pass("65536 states: the GPU's weights equal the reference codebook's E4M3 values bit for bit", .{});
    var bound: ref.Check = .{};
    var pairs: usize = 0;
    for (cases.cases, 0..) |c, i| {
        const d = try cases.Data.init(c, 0xE8 + i);
        defer d.deinit();
        // One, two and four sets a job: the split kernels must give the one-set kernel's bytes.
        for ([_]usize{ 1, 2, 4 }) |sets| {
            const o = try windows(gpu, &l.k, s, d, sets);
            defer o.deinit();
            pairs += try cases.rowsEqual(o);
            if (sets == 1) try cases.fp64(book, d, o, &bound);
            try frozen.compare(c.name, o.windows, o.width * o.elem);
        }
        check.pass("{s}: windows 1-16 byte-equal to rows alone and to the Metal kernels' frozen windows, at 1, 2 and 4 sets a job", .{c.name});
    }
    check.pass("{d} cases, {d} row/window pairs, {d} FP64 values; max error {e}, max {d:.4} of the bound", .{ cases.cases.len, pairs, bound.values, bound.max_error, bound.max_fraction });
}

fn windows(gpu: Gpu, k: *const cuda.exlr8.Kernels, s: cuda.Stream, d: cases.Data, sets: usize) !cases.Outputs {
    const c = d.c;
    const plan = try cases.Plan.init(d);
    defer plan.deinit();
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const weights = try rig.put(d.tiles);
    const bases = try rig.put(plan.bases);
    const halves = try rig.put(d.scales);
    const signs = try rig.put(d.signs);
    const jobs = try rig.put(plan.jobs);
    const x = try rig.put(d.x);
    const wts = try rig.put(d.weights);
    const suh_of = try rig.put(plan.suh_of);
    const sv = try rig.put(plan.sv_of);
    const rot = try rig.zeros(rows_max * d.xWidth() * 4);
    const z = try rig.zeros(rows_max * c.slots * d.nOut() * 4);
    const y = try rig.zeros(rows_max * d.yWidth() * 4);
    const o = try cases.Outputs.init(d);
    var at: usize = 0;
    for (0..rows_max + 1) |m| for (0..if (m == 0) rows_max else 1) |pass| {
        const rows: u32 = if (m == 0) 1 else @intCast(m);
        const first = if (m == 0) pass else 0;
        try k.rotate(s, x + first * d.xWidth() * 2, signs, halves, rot, if (c.down) suh_of else 0, cases.Plan.rotate(d, rows), c.blocks, c.slots);
        if (c.down) {
            try k.project(s, rot, weights, bases, halves, signs, jobs, plan.jobs.len, rowsArgs(rows, d.xWidth(), c.slots * d.nOut(), true), z, 0, sets);
            try k.combine(s, z, wts + first * c.slots * 4, sv, halves, signs, cases.Plan.combine(d, rows), c.outs, y);
        } else try k.project(s, rot, weights, bases, halves, signs, jobs, plan.jobs.len, rowsArgs(rows, d.xWidth(), d.yWidth(), c.y_f32), y, 0, sets);
        try s.synchronize();
        const bytes = rows * o.width * o.elem;
        if (m == 0) {
            try rig.get(y, o.alone[pass * o.width * o.elem ..][0..bytes]);
        } else {
            try rig.get(y, o.windows[at..][0..bytes]);
            at += bytes;
        }
    };
    return o;
}

/// A real layer's buffers on the device: every set's jobs and slot tables, inputs and outputs for 16 rows.
const Layer = struct {
    rig: Rig,
    weights: u64,
    bases: u64,
    halves: u64,
    signs: u64,
    x: u64,
    act: u64,
    w: u64,
    r: u64,
    gu: u64,
    ar: u64,
    z: u64,
    y: u64,
    up: []u64,
    down: []u64,
    sv: []u64,
    suh: []u64,
};

const Stage = enum { rotate_in, gate_up, rotate_down, down, combine };
const all = [_]Stage{ .rotate_in, .gate_up, .rotate_down, .down, .combine };

fn stages(k: *const cuda.exlr8.Kernels, s: cuda.Stream, g: *const Layer, host: layer_.Host, set: usize, rows: u32, first: usize, which: []const Stage) !void {
    return stagesSplit(k, s, g, host, set, rows, first, which, 1);
}

fn stagesSplit(k: *const cuda.exlr8.Kernels, s: cuda.Stream, g: *const Layer, host: layer_.Host, set: usize, rows: u32, first: usize, which: []const Stage, sets: usize) !void {
    const h = host.sets[set];
    for (which) |st| switch (st) {
        .rotate_in => try k.rotate(s, g.x + first * L.hidden * 2, g.signs, g.halves, g.r, 0, layer_.rotateIn(rows), 48, 1),
        .gate_up => try k.project(s, g.r, g.weights, g.bases, g.halves, g.signs, g.up[set], h.up.len, rowsArgs(rows, L.hidden, slots * 4096, true), g.gu, 0, sets),
        .rotate_down => try k.rotate(s, g.act + first * slots * 2048 * 2, g.signs, g.halves, g.ar, g.suh[set], layer_.rotateDown(rows), 16, slots),
        .down => try k.project(s, g.ar, g.weights, g.bases, g.halves, g.signs, g.down[set], h.down.len, rowsArgs(rows, slots * 2048, slots * L.hidden, true), g.z, 0, 1),
        .combine => try k.combine(s, g.z, g.w + first * slots * 4, g.sv[set], g.halves, g.signs, layer_.combine(rows), 48, g.y),
    };
}

/// `tf-cuda-test exlr8-bench DIR [MANIFEST] [LAYER] [SETS]`: a real layer's FP64 and windows, then GB/s by stage.
pub fn bench(gpu: Gpu, dir: []const u8, manifest_name: []const u8, number: u16, count: usize) !void {
    var l = try load(gpu);
    defer l.mod.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const a = gpu.gpa;
    var layer = try L.Layer.open(a, gpu.io, dir, number);
    defer layer.close(gpu.io);
    const mpath = try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-{s}.json", .{ dir, manifest_name });
    defer a.free(mpath);
    var m = try ex.manifest.Manifest.load(a, gpu.io, mpath);
    defer m.deinit();
    const widths = m.layers[number] orelse return error.LayerHasNoExperts;
    const host = try layer_.Host.build(&layer, widths, count);
    const in = try layer_.Inputs.init();
    var g: Layer = undefined;
    g.rig = .{ .gpu = gpu };
    defer g.rig.deinit();
    g.weights = try g.rig.put(host.weights.items);
    g.bases = try g.rig.put(host.bases.items);
    g.halves = try g.rig.put(host.halves.items);
    g.signs = try g.rig.put(&host.signs);
    g.x = try g.rig.put(in.x);
    g.act = try g.rig.put(in.act);
    g.w = try g.rig.put(in.w);
    g.r = try g.rig.zeros(16 * L.hidden * 4);
    g.gu = try g.rig.zeros(16 * slots * 4096 * 4);
    g.ar = try g.rig.zeros(16 * slots * 2048 * 4);
    g.z = try g.rig.zeros(16 * slots * L.hidden * 4);
    g.y = try g.rig.zeros(16 * L.hidden * 4);
    for ([_]*[]u64{ &g.up, &g.down, &g.sv, &g.suh }) |t| t.* = try a.alloc(u64, count);
    for (host.sets, 0..) |h, i| {
        g.up[i] = try g.rig.put(h.up);
        g.down[i] = try g.rig.put(h.down);
        g.sv[i] = try g.rig.put(&h.sv);
        g.suh[i] = try g.rig.put(&h.suh);
    }
    var name_buf: [256]u8 = undefined;
    std.debug.print("EXLR8 layer {d} ({s}) on {s}: {d} sets of 8 routed experts + the shared one, {d:.1} MB of weights a set\n", .{ number, manifest_name, try gpu.ctx.name(&name_buf), count, @as(f64, @floatFromInt(host.sets[0].up_bytes + host.sets[0].down_bytes)) / 1e6 });
    try stages(&l.k, s, &g, host, 0, 16, 0, &all);
    try s.synchronize();
    const gu = try a.alloc(f32, 16 * slots * 4096);
    defer a.free(gu);
    const y = try a.alloc(f32, 16 * L.hidden);
    defer a.free(y);
    try g.rig.get(g.gu, std.mem.sliceAsBytes(gu));
    try g.rig.get(g.y, std.mem.sliceAsBytes(y));
    var bound: ref.Check = .{};
    try layer_.fp64(&layer, widths, host.sets[0], in, gu, y, &bound);
    check.pass("FP64: {d} values (gate/up halves and the combined down), max error {e}, max {d:.4} of the bound", .{ bound.values, bound.max_error, bound.max_fraction });
    const gu_row = slots * 4096 * 4;
    const y_row = L.hidden * 4;
    const alone = try a.alloc(u8, 16 * (gu_row + y_row));
    defer a.free(alone);
    for (0..16) |r| {
        try stages(&l.k, s, &g, host, 0, 1, r, &all);
        try s.synchronize();
        try g.rig.get(g.gu, alone[r * gu_row ..][0..gu_row]);
        try g.rig.get(g.y, alone[16 * gu_row + r * y_row ..][0..y_row]);
    }
    var pairs: usize = 0;
    for (1..17) |w| {
        try stages(&l.k, s, &g, host, 0, @intCast(w), 0, &all);
        try s.synchronize();
        try g.rig.get(g.gu, std.mem.sliceAsBytes(gu));
        try g.rig.get(g.y, std.mem.sliceAsBytes(y));
        for (0..w) |r| {
            try check.sameBytes("exlr8 gate/up row", std.mem.sliceAsBytes(gu)[r * gu_row ..][0..gu_row], alone[r * gu_row ..][0..gu_row]);
            try check.sameBytes("exlr8 down row", std.mem.sliceAsBytes(y)[r * y_row ..][0..y_row], alone[16 * gu_row + r * y_row ..][0..y_row]);
            pairs += 1;
        }
    }
    check.pass("rows: {d} row/window pairs byte-equal to rows alone (gate/up and combined down)", .{pairs});
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for (0..30) |i| try stages(&l.k, s, &g, host, i % count, 16, 0, &all);
    var t: [24]f64 = undefined;
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| {
        var us: [all.len + 1]f64 = undefined;
        for (0..all.len + 1) |si| {
            for (&t, 0..) |*v, i| {
                try start.record(s);
                try stages(&l.k, s, &g, host, i % count, rows, 0, if (si == all.len) &all else all[si..][0..1]);
                try end.record(s);
                try end.synchronize();
                v.* = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1e3;
            }
            us[si] = check.median(&t);
        }
        const up: f64 = @floatFromInt(host.sets[0].up_bytes);
        const down: f64 = @floatFromInt(host.sets[0].down_bytes);
        std.debug.print("RESULT rows {d:2}: layer {d:7.1} us | rotate x {d:5.1} us, gate/up {d:7.1} us ({d:5.1} GB/s), rotate down inputs {d:5.1} us, down {d:7.1} us ({d:5.1} GB/s), combine {d:5.1} us | rotations {d:4.1}% of the stages\n", .{
            rows, us[5], us[0], us[1], up / us[1] / 1e3, us[2], us[3], down / us[3] / 1e3, us[4], 100 * (us[0] + us[2]) / (us[0] + us[1] + us[2] + us[3] + us[4]),
        });
        if (rows > 4) continue;
        // The gate/up jobs again with their blocks split over 2 and 4 sets a job (one set where no kernel fits).
        var split: [2]f64 = undefined;
        for ([_]usize{ 2, 4 }, &split) |sets, *out| {
            for (&t, 0..) |*v, i| {
                try start.record(s);
                try stagesSplit(&l.k, s, &g, host, i % count, rows, 0, &.{.gate_up}, sets);
                try end.record(s);
                try end.synchronize();
                v.* = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1e3;
            }
            out.* = check.median(&t);
        }
        std.debug.print("RESULT rows {d:2}: gate/up {d:7.1} us at 2 sets a job, {d:7.1} us at 4\n", .{ rows, split[0], split[1] });
    }
}

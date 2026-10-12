//! EXLR8 Metal projections on synthetic weights: every state's weight, FP64 bounds, rows equal alone and in windows.

const std = @import("std");
const mtl = @import("metal");
const ex = @import("exlr8");
const op = @import("exlr8_metal").projection;
const ref = @import("exlr8_ref");
const cases = @import("exlr8_cases");
const frozen = @import("fixtures/exlr8/frozen.zig");
const codec = ex.codec;
const a = std.heap.page_allocator;
const rows_max = cases.rows_max;

fn buffer(device: mtl.Device, bytes: anytype) !mtl.Buffer {
    const raw = std.mem.sliceAsBytes(bytes);
    const b = try device.buffer(@max(raw.len, 16), mtl.ResourceOptions.shared);
    @memcpy(b.contents()[0..raw.len], raw);
    return b;
}
fn zeros(device: mtl.Device, len: usize) !mtl.Buffer {
    const b = try device.buffer(@max(len, 16), mtl.ResourceOptions.shared);
    @memset(b.contents()[0..b.length()], 0);
    return b;
}
fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.debug.print("GPU command failed: {s}\n", .{message});
        return error.GpuFailure;
    }
}

fn checkWeights(device: mtl.Device, queue: mtl.Queue, p: *const op.Pipelines, book: *const codec.Codebook) !void {
    const out = try zeros(device, 65536 * 4);
    defer out.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(p.weights);
    e.setBuffer(out, 0, 0);
    e.dispatchThreads(.{ .width = 65536 }, .{ .width = 256 });
    e.end();
    try finish(cb);
    for (out.slice(f32, 65536), 0..) |got, s| {
        if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(codec.e4m3Value(book.codes[s])))) return error.Exlr8WeightMismatch;
    }
    std.debug.print("PASS 65536 states: the GPU's weights equal the reference codebook's E4M3 values bit for bit\n", .{});
}

/// Each row alone, then rows 0..m together for m = 1..16, through the rotation, projection and (down) fan-in.
fn run(device: mtl.Device, queue: mtl.Queue, p: *const op.Pipelines, d: cases.Data, sets: usize) !cases.Outputs {
    const c = d.c;
    const plan = try cases.Plan.init(d);
    defer plan.deinit();
    const bufs = [_]mtl.Buffer{
        try buffer(device, d.tiles),                  try buffer(device, plan.bases),                       try buffer(device, d.scales),
        try buffer(device, d.signs),                  try buffer(device, plan.jobs),                        try buffer(device, d.x),
        try buffer(device, d.weights),                try buffer(device, plan.suh_of),                      try buffer(device, plan.sv_of),
        try zeros(device, rows_max * d.xWidth() * 4), try zeros(device, rows_max * c.slots * d.nOut() * 4), try zeros(device, rows_max * d.yWidth() * 4),
    };
    defer for (bufs) |b| b.deinit();
    const weights, const bases, const halves, const signs, const jobs, const x, const wts, const suh_of, const sv, const rot, const z, const y = bufs;
    const t: op.Tables = .{ .weights = weights, .bases = bases, .halves = halves, .signs = signs, .jobs = jobs, .count = plan.jobs.len, .sets = sets };
    const out = try cases.Outputs.init(d);
    var at: usize = 0;
    for (0..rows_max + 1) |m| for (0..if (m == 0) rows_max else 1) |pass| {
        const rows: u32 = if (m == 0) 1 else @intCast(m);
        const first = if (m == 0) pass else 0;
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try op.encodeRotate(p, e, .{ .buf = x, .off = first * d.xWidth() * 2 }, signs, halves, .{ .buf = rot }, cases.Plan.rotate(d, rows), c.blocks, c.slots, if (c.down) op.Ref{ .buf = suh_of } else null);
        e.barrier();
        if (c.down) {
            try op.encodeProject(p, e, .{ .buf = rot }, t, .{ .buf = z }, rows, d.xWidth(), c.slots * d.nOut(), true, null);
            e.barrier();
            try op.encodeCombine(p, e, .{ .buf = z }, .{ .buf = wts, .off = first * c.slots * 4 }, .{ .buf = sv }, halves, signs, .{ .buf = y }, cases.Plan.combine(d, rows), c.outs);
        } else try op.encodeProject(p, e, .{ .buf = rot }, t, .{ .buf = y }, rows, d.xWidth(), d.yWidth(), c.y_f32, null);
        e.end();
        try finish(cb);
        const bytes = rows * out.width * out.elem;
        if (m == 0) {
            @memcpy(out.alone[pass * out.width * out.elem ..][0..bytes], y.contents()[0..bytes]);
        } else {
            @memcpy(out.windows[at..][0..bytes], y.contents()[0..bytes]);
            at += bytes;
        }
    };
    return out;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const freeze = argv.len == 2 and std.mem.eql(u8, argv[1], "--freeze");
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch |err| {
        if (err != error.NoDevice) return err;
        std.debug.print("SKIP EXLR8 projections: no Metal device\n", .{});
        return;
    };
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try op.Pipelines.load(device);
    defer p.deinit();
    const book = try a.create(codec.Codebook);
    defer a.destroy(book);
    book.init();
    std.debug.print("EXLR8 Metal projections on {s}; synthetic weights, no model files\n", .{device.name()});
    try checkWeights(device, queue, &p, book);
    var check: ref.Check = .{};
    var pairs: usize = 0;
    for (cases.cases, 0..) |c, i| {
        const d = try cases.Data.init(c, 0xE8 + i);
        defer d.deinit();
        // One, two and four sets a job: the split kernels must give the one-set kernel's bytes.
        for ([_]usize{ 1, 2, 4 }) |sets| {
            const o = try run(device, queue, &p, d, sets);
            defer o.deinit();
            pairs += try cases.rowsEqual(o);
            if (sets == 1) try cases.fp64(book, d, o, &check);
            if (freeze and sets == 1) {
                frozen.print(c.name, o.windows, o.width * o.elem);
            } else try frozen.compare(c.name, o.windows, o.width * o.elem);
        }
        std.debug.print("PASS {s}: windows 1-16 byte-equal to rows alone{s}, at 1, 2 and 4 sets a job\n", .{ c.name, if (freeze) "" else ", frozen windows byte-equal" });
    }
    std.debug.print("PASS {d} cases, {d} row/window pairs, {d} FP64 values; max error {e}, max {d:.4} of the bound\n", .{ cases.cases.len, pairs, check.values, check.max_error, check.max_fraction });
}

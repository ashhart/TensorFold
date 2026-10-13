//! EXLR8 dense formats on Metal: weights equal the reference decode bit for bit, FP64 bounds, rows alone == windows.

const std = @import("std");
const mtl = @import("metal");
const ex = @import("exlr8");
const dg = @import("exlr8_metal").dense_gpu;
const op = @import("exlr8_metal").projection;
const ref = @import("exlr8_ref");
const cases = @import("exlr8_dense_cases");
const real = @import("exlr8_dense_real");
const frozen = @import("fixtures/exlr8/frozen_dense.zig");
const st = ex.safetensors;
const dense = ex.dense;
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

const Gpu = struct { device: mtl.Device, queue: mtl.Queue, dp: dg.Pipelines, tp: op.Pipelines };

/// Rows [out0, out0 + n) of the matrix decoded on the GPU as fp32 (n x in).
fn decode(g: *const Gpu, w: mtl.Buffer, j: ex.job.DenseJob, out: mtl.Buffer) !void {
    const jobs = try buffer(g.device, &[_]ex.job.DenseJob{j});
    defer jobs.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    try dg.encodeDecode(&g.dp, e, w, .{ .buf = jobs }, j, .{ .buf = out });
    e.end();
    try finish(cb);
}

/// The job's rows through the projection: each row alone, then windows of 1-16 rows; returns both.
fn project(g: *const Gpu, w: mtl.Buffer, j: ex.job.DenseJob, x: []const f32, y_f32: bool) !struct { alone: []u8, windows: []u8 } {
    const elem: usize = if (y_f32) 4 else 2;
    const row = @as(usize, j.outs) * elem;
    const groups = try ex.job.denseGroups(a, &.{j});
    defer a.free(groups);
    const bufs = [_]mtl.Buffer{ try buffer(g.device, &[_]ex.job.DenseJob{j}), try buffer(g.device, groups), try buffer(g.device, x), try zeros(g.device, rows_max * row) };
    defer for (bufs) |b| b.deinit();
    const t: dg.Tables = .{ .weights = w, .jobs = bufs[0], .groups = bufs[1], .count = groups.len };
    const alone = try a.alloc(u8, rows_max * row);
    const windows = try a.alloc(u8, 136 * row);
    var at: usize = 0;
    for (0..rows_max + 1) |m| for (0..if (m == 0) rows_max else 1) |pass| {
        const rows: usize = if (m == 0) 1 else m;
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        try dg.encodeProject(&g.dp, e, .{ .buf = bufs[2], .off = (if (m == 0) pass else 0) * j.ins * 4 }, t, .{ .buf = bufs[3] }, rows, j.ins, j.outs, y_f32);
        e.end();
        try finish(cb);
        if (m == 0) {
            @memcpy(alone[pass * row ..][0..row], bufs[3].contents()[0..row]);
        } else {
            @memcpy(windows[at..][0 .. m * row], bufs[3].contents()[0 .. m * row]);
            at += m * row;
        }
    };
    return .{ .alone = alone, .windows = windows };
}

fn synthetic(g: *const Gpu, freeze: bool) !void {
    var check: cases.Check = .{};
    var pairs: usize = 0;
    for (cases.cases, 0..) |c, i| {
        const p = try cases.synthetic(c, 0xD3 + i);
        defer p.deinit();
        const w = try buffer(g.device, p.bytes);
        defer w.deinit();
        const j = try p.job(0, c.out);
        const out = try zeros(g.device, c.out * c.in * 4);
        defer out.deinit();
        try decode(g, w, j, out);
        try cases.decodeEqual(p.m, 0, c.out, out.slice(f32, c.out * c.in));
        const x = try cases.inputs(c.in, 0x5EED + i);
        defer a.free(x);
        const r = try project(g, w, j, x, c.y_f32);
        defer a.free(r.alone);
        defer a.free(r.windows);
        const row = c.out * @as(usize, if (c.y_f32) 4 else 2);
        pairs += try cases.rowsEqual(r.windows, r.alone, row);
        try check.rows(p.m, x, 0, c.out, r.alone, c.y_f32);
        const d = cases.digest(r.windows);
        if (freeze) {
            std.debug.print("    .{{ .name = \"{s}\", .windows = \"{s}\" }},\n", .{ c.name, d });
        } else try frozen.compare(c.name, &d);
        std.debug.print("PASS {s}: {d} x {d} decode bit-equal, windows 1-16 equal rows alone{s}\n", .{ c.name, c.out, c.in, if (freeze) "" else ", frozen bytes equal" });
    }
    std.debug.print("PASS {d} synthetic formats, {d} row/window pairs, {d} FP64 values; max error {e}, max {d:.4} of the bound\n", .{ cases.cases.len, pairs, check.values, check.max_error, check.max_fraction });
}

/// A real matrix stored by row: its whole decode hashes to the reference's digest; 256 rows' projection checked.
fn rowMatrix(g: *const Gpu, m: dense.Matrix, want: real.Expected, check: *cases.Check) !usize {
    const p = try cases.pack(m);
    defer p.deinit();
    const w = try buffer(g.device, p.bytes);
    defer w.deinit();
    const r = m.record;
    const chunk = @min(r.out, @max(32, (256 << 20) / (r.in * 4)) / 32 * 32);
    const out = try zeros(g.device, chunk * r.in * 4);
    defer out.deinit();
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var o0: usize = 0;
    while (o0 < r.out) : (o0 += chunk) {
        const n = @min(chunk, r.out - o0);
        try decode(g, w, try p.job(o0, n), out);
        h.update(out.contents()[0 .. n * r.in * 4]);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    if (!std.mem.eql(u8, &real.hex(d), want.w orelse return error.MissingDigest)) return error.Exlr8DecodeDigestMismatch;
    const outs: usize = @min(r.out, 256);
    const x = try real.inputs(rows_max, r.in, 0xF00D + r.out);
    defer a.free(x);
    const res = try project(g, w, try p.job(0, outs), x, true);
    defer a.free(res.alone);
    defer a.free(res.windows);
    try check.rows(m, x, 0, outs, res.alone, true);
    return try cases.rowsEqual(res.windows, res.alone, outs * 4);
}

/// A real trellis matrix: its codes T on the GPU hash to the reference's digest; two output blocks against FP64.
fn trellisMatrix(g: *const Gpu, book: *const ex.codec.Codebook, m: dense.Matrix, want: real.Expected, check: *ref.Check) !usize {
    const t = try real.Trellis.init(m);
    defer t.deinit();
    const r = m.record;
    const w = try buffer(g.device, m.data[0].bytes);
    defer w.deinit();
    const runs = try buffer(g.device, t.runs);
    defer runs.deinit();
    const blocks = try zeros(g.device, t.runs.len * 16384);
    defer blocks.deinit();
    var cb = g.queue.commandBuffer();
    var e = cb.compute(.serial);
    try op.encodeTileCodes(&g.tp, e, w, .{ .buf = runs }, r.k, t.runs.len, .{ .buf = blocks });
    e.end();
    try finish(cb);
    const codes = try a.alloc(u8, r.in * r.out);
    defer a.free(codes);
    t.assemble(blocks.contents()[0 .. t.runs.len * 16384], codes);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(codes, &d, .{});
    if (!std.mem.eql(u8, &real.hex(d), want.tiles orelse return error.MissingDigest)) return error.Exlr8TileDigestMismatch;
    const x = try real.inputs(rows_max, r.in, 0xBEEF + r.in);
    defer a.free(x);
    const bufs = [_]mtl.Buffer{ try buffer(g.device, t.bases), try buffer(g.device, t.halves), try buffer(g.device, t.signs), try buffer(g.device, t.jobs), try buffer(g.device, x), try zeros(g.device, rows_max * r.in * 4), try zeros(g.device, rows_max * 256 * 4) };
    defer for (bufs) |b| b.deinit();
    const tables: op.Tables = .{ .weights = w, .bases = bufs[0], .halves = bufs[1], .signs = bufs[2], .jobs = bufs[3], .count = t.jobs.len };
    const alone = try a.alloc(u8, rows_max * 256 * 4);
    defer a.free(alone);
    const windows = try a.alloc(u8, 136 * 256 * 4);
    defer a.free(windows);
    var at: usize = 0;
    for (0..rows_max + 1) |mw| for (0..if (mw == 0) rows_max else 1) |pass| {
        const rows: u32 = if (mw == 0) 1 else @intCast(mw);
        cb = g.queue.commandBuffer();
        e = cb.compute(.serial);
        try op.encodeRotate(&g.tp, e, .{ .buf = bufs[4], .off = (if (mw == 0) pass else 0) * r.in * 4 }, bufs[2], bufs[1], .{ .buf = bufs[5] }, t.rotate(rows), r.in / 128, 1, null);
        e.barrier();
        try op.encodeProject(&g.tp, e, .{ .buf = bufs[5] }, tables, .{ .buf = bufs[6] }, rows, r.in, 256, true, null);
        e.end();
        try finish(cb);
        const bytes = rows * 256 * 4;
        if (mw == 0) {
            @memcpy(alone[pass * 1024 ..][0..1024], bufs[6].contents()[0..1024]);
        } else {
            @memcpy(windows[at..][0..bytes], bufs[6].contents()[0..bytes]);
            at += bytes;
        }
    };
    try t.fp64(book, x, rows_max, std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(alone))), check);
    return try cases.rowsEqual(windows, alone, 1024);
}

fn realFiles(g: *const Gpu, io: std.Io, root: []const u8) !void {
    const fixture = try real.fixtureJson(a, io);
    defer fixture.deinit();
    const book = try a.create(ex.codec.Codebook);
    defer a.destroy(book);
    book.init();
    var rows_check: cases.Check = .{};
    var t_check: ref.Check = .{};
    var pairs: usize = 0;
    var row_n: usize = 0;
    var trellis_n: usize = 0;
    for (real.files) |name| {
        const path = try std.fmt.allocPrint(a, "{s}/dense/{s}.safetensors", .{ root, name });
        defer a.free(path);
        var f = try st.File.open(a, io, path);
        defer f.close(io);
        const names = try dense.names(a, &f);
        defer a.free(names);
        for (names) |n| {
            const m = try dense.Matrix.get(&f, n);
            const want = try real.expected(fixture.value, name, n);
            if (m.record.format == .trellis) {
                pairs += try trellisMatrix(g, book, m, want, &t_check);
                trellis_n += 1;
            } else {
                pairs += try rowMatrix(g, m, want, &rows_check);
                row_n += 1;
            }
            std.debug.print("PASS {s} {s}: format {d}, {d} x {d}, the GPU's decode equals the reference's digest\n", .{ name, n, @backingInt(m.record.format), m.record.out, m.record.in });
        }
    }
    std.debug.print("PASS {d} row-stored and {d} trellis matrices: decode digests equal, {d} row/window pairs; FP64 {d} + {d} values, max {d:.4} / {d:.4} of the bound\n", .{ row_n, trellis_n, pairs, rows_check.values, t_check.values, rows_check.max_fraction, t_check.max_fraction });
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const freeze = argv.len == 2 and std.mem.eql(u8, argv[1], "--freeze");
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch |err| {
        if (err != error.NoDevice) return err;
        std.debug.print("SKIP EXLR8 dense formats: no Metal device\n", .{});
        return;
    };
    defer device.deinit();
    var g: Gpu = .{ .device = device, .queue = try device.queue(), .dp = try dg.Pipelines.load(device), .tp = try op.Pipelines.load(device) };
    defer g.queue.deinit();
    defer g.dp.deinit();
    defer g.tp.deinit();
    std.debug.print("EXLR8 dense formats on {s}\n", .{device.name()});
    try synthetic(&g, freeze);
    const root = init.environ_map.get("TF_EXLR8_DIR") orelse {
        std.debug.print("SKIP real matrices: set TF_EXLR8_DIR to a folder with dense/exlr8-dense-{{000,003,040,078}} and the vocab file\n", .{});
        return;
    };
    try realFiles(&g, init.io, root);
}

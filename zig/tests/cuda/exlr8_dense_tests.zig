//! EXLR8 dense formats on CUDA: decode bit-equal to the reference, FP64, rows alone == windows, Metal's frozen bytes.
const std = @import("std");
const cuda = @import("cuda");
const ex = @import("exlr8");
const ref = @import("exlr8_ref");
const cases = @import("exlr8_dense_cases");
const real = @import("exlr8_dense_real");
const frozen = @import("exlr8_frozen_dense");
const check = @import("check.zig");
const Gpu = check.Gpu;
const st = ex.safetensors;
const dense = ex.dense;
const rows_max = cases.rows_max;
const a = std.heap.page_allocator;

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

const K = struct { dense: cuda.exlr8.Dense, trellis: cuda.exlr8.Kernels, s: cuda.Stream };

fn rowsArgs(rows: usize, x_stride: usize, y_stride: usize, y_f32: bool) ex.job.Rows {
    return .{ .rows = @intCast(rows), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .y_f32 = @intFromBool(y_f32) };
}

/// The job's rows decoded as fp32 (outs x ins) on the device.
fn decode(rig: *Rig, k: *const K, w: u64, j: ex.job.DenseJob, out: []u8) !void {
    const jobs = try rig.put(&[_]ex.job.DenseJob{j});
    const o = try rig.zeros(out.len);
    try k.dense.decode(k.s, w, jobs, j.outs, j.ins, o);
    try k.s.synchronize();
    try rig.get(o, out);
}

/// Each row alone, then windows of 1-16 rows, through the job's matrix.
fn project(rig: *Rig, k: *const K, w: u64, j: ex.job.DenseJob, x: []const f32, y_f32: bool, alone: []u8, windows: []u8) !void {
    const row = @as(usize, j.outs) * @as(usize, if (y_f32) 4 else 2);
    const groups = try ex.job.denseGroups(a, &.{j});
    defer a.free(groups);
    const jobs = try rig.put(&[_]ex.job.DenseJob{j});
    const g = try rig.put(groups);
    const xd = try rig.put(x);
    const y = try rig.zeros(rows_max * row);
    var at: usize = 0;
    for (0..rows_max + 1) |m| for (0..if (m == 0) rows_max else 1) |pass| {
        const rows: usize = if (m == 0) 1 else m;
        try k.dense.project(k.s, xd + (if (m == 0) pass else 0) * j.ins * 4, w, jobs, g, groups.len, rowsArgs(rows, j.ins, j.outs, y_f32), y);
        try k.s.synchronize();
        if (m == 0) {
            try rig.get(y, alone[pass * row ..][0..row]);
        } else {
            try rig.get(y, windows[at..][0 .. m * row]);
            at += m * row;
        }
    };
}

fn synthetic(gpu: Gpu, k: *const K) !void {
    var bound: cases.Check = .{};
    var pairs: usize = 0;
    for (cases.cases, 0..) |c, i| {
        var rig: Rig = .{ .gpu = gpu };
        defer rig.deinit();
        const p = try cases.synthetic(c, 0xD3 + i);
        defer p.deinit();
        const w = try rig.put(p.bytes);
        const j = try p.job(0, c.out);
        const got = try a.alloc(u8, c.out * c.in * 4);
        defer a.free(got);
        try decode(&rig, k, w, j, got);
        try cases.decodeEqual(p.m, 0, c.out, std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(got))));
        const x = try cases.inputs(c.in, 0x5EED + i);
        defer a.free(x);
        const row = c.out * @as(usize, if (c.y_f32) 4 else 2);
        const alone = try a.alloc(u8, rows_max * row);
        defer a.free(alone);
        const windows = try a.alloc(u8, 136 * row);
        defer a.free(windows);
        try project(&rig, k, w, j, x, c.y_f32, alone, windows);
        pairs += try cases.rowsEqual(windows, alone, row);
        try bound.rows(p.m, x, 0, c.out, alone, c.y_f32);
        try frozen.compare(c.name, &cases.digest(windows));
        check.pass("{s}: decode bit-equal, windows 1-16 equal rows alone and the Metal kernels' frozen windows", .{c.name});
    }
    check.pass("{d} synthetic formats, {d} row/window pairs, {d} FP64 values; max {d:.4} of the bound", .{ cases.cases.len, pairs, bound.values, bound.max_fraction });
}

fn rowMatrix(gpu: Gpu, k: *const K, m: dense.Matrix, want: real.Expected, bound: *cases.Check) !usize {
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const p = try cases.pack(m);
    defer p.deinit();
    const w = try rig.put(p.bytes);
    const r = m.record;
    const chunk = @min(r.out, @max(32, (256 << 20) / (r.in * 4)) / 32 * 32);
    const out = try a.alloc(u8, chunk * r.in * 4);
    defer a.free(out);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var o0: usize = 0;
    while (o0 < r.out) : (o0 += chunk) {
        const n = @min(chunk, r.out - o0);
        try decode(&rig, k, w, try p.job(o0, n), out[0 .. n * r.in * 4]);
        h.update(out[0 .. n * r.in * 4]);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    try check.expect(std.mem.eql(u8, &real.hex(d), want.w orelse ""), "decode digest", .{});
    const outs: usize = @min(r.out, 256);
    const x = try real.inputs(rows_max, r.in, 0xF00D + r.out);
    defer a.free(x);
    const alone = try a.alloc(u8, rows_max * outs * 4);
    defer a.free(alone);
    const windows = try a.alloc(u8, 136 * outs * 4);
    defer a.free(windows);
    try project(&rig, k, w, try p.job(0, outs), x, true, alone, windows);
    try bound.rows(m, x, 0, outs, alone, true);
    return cases.rowsEqual(windows, alone, outs * 4);
}

fn trellisMatrix(gpu: Gpu, k: *const K, book: *const ex.codec.Codebook, m: dense.Matrix, want: real.Expected, bound: *ref.Check) !usize {
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const t = try real.Trellis.init(m);
    defer t.deinit();
    const r = m.record;
    const w = try rig.put(m.data[0].bytes);
    const blocks = try rig.zeros(t.runs.len * 16384);
    try k.trellis.tileCodes(k.s, w, try rig.put(t.runs), r.k, t.runs.len, blocks);
    try k.s.synchronize();
    const got = try a.alloc(u8, t.runs.len * 16384);
    defer a.free(got);
    try rig.get(blocks, got);
    const codes = try a.alloc(u8, r.in * r.out);
    defer a.free(codes);
    t.assemble(got, codes);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(codes, &d, .{});
    try check.expect(std.mem.eql(u8, &real.hex(d), want.tiles orelse ""), "tile codes digest", .{});
    const x = try real.inputs(rows_max, r.in, 0xBEEF + r.in);
    defer a.free(x);
    const bases = try rig.put(t.bases);
    const halves = try rig.put(t.halves);
    const signs = try rig.put(t.signs);
    const jobs = try rig.put(t.jobs);
    const xd = try rig.put(x);
    const rot = try rig.zeros(rows_max * r.in * 4);
    const y = try rig.zeros(rows_max * 256 * 4);
    const alone = try a.alloc(u8, rows_max * 1024);
    defer a.free(alone);
    const windows = try a.alloc(u8, 136 * 1024);
    defer a.free(windows);
    var at: usize = 0;
    for (0..rows_max + 1) |mw| for (0..if (mw == 0) rows_max else 1) |pass| {
        const rows: u32 = if (mw == 0) 1 else @intCast(mw);
        try k.trellis.rotate(k.s, xd + (if (mw == 0) pass else 0) * r.in * 4, signs, halves, rot, 0, t.rotate(rows), r.in / 128, 1);
        try k.trellis.project(k.s, rot, w, bases, halves, signs, jobs, t.jobs.len, rowsArgs(rows, r.in, 256, true), y, 0, 1);
        try k.s.synchronize();
        if (mw == 0) {
            try rig.get(y, alone[pass * 1024 ..][0..1024]);
        } else {
            try rig.get(y, windows[at..][0 .. rows * 1024]);
            at += rows * 1024;
        }
    };
    try t.fp64(book, x, rows_max, std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(alone))), bound);
    return cases.rowsEqual(windows, alone, 1024);
}

/// `tf-cuda-test exlr8-dense [DIR]`: the synthetic formats, then with DIR every real dense matrix the fixture names.
pub fn run(gpu: Gpu, dir: ?[]const u8) !void {
    var dm = try cuda.Module.load(gpu.d, cuda.kernels.exlr8_dense);
    defer dm.unload();
    var tm = try cuda.Module.load(gpu.d, cuda.kernels.exlr8);
    defer tm.unload();
    var k: K = .{ .dense = try cuda.exlr8.Dense.resolve(dm), .trellis = try cuda.exlr8.Kernels.resolve(tm), .s = try cuda.Stream.init(gpu.d, true) };
    defer k.s.deinit();
    try synthetic(gpu, &k);
    const root = dir orelse return;
    const fixture = try real.fixtureJson(a, gpu.io);
    defer fixture.deinit();
    const book = try a.create(ex.codec.Codebook);
    defer a.destroy(book);
    book.init();
    var rb: cases.Check = .{};
    var tb: ref.Check = .{};
    var pairs: usize = 0;
    var n: usize = 0;
    for (real.files) |name| {
        const path = try std.fmt.allocPrint(a, "{s}/dense/{s}.safetensors", .{ root, name });
        defer a.free(path);
        var f = try st.File.open(a, gpu.io, path);
        defer f.close(gpu.io);
        const names = try dense.names(a, &f);
        defer a.free(names);
        for (names) |mn| {
            const m = try dense.Matrix.get(&f, mn);
            const want = try real.expected(fixture.value, name, mn);
            pairs += if (m.record.format == .trellis) try trellisMatrix(gpu, &k, book, m, want, &tb) else try rowMatrix(gpu, &k, m, want, &rb);
            n += 1;
        }
    }
    check.pass("{d} real dense matrices: decode digests equal the reference's, {d} row/window pairs; FP64 max {d:.4} / {d:.4} of the bound", .{ n, pairs, rb.max_fraction, tb.max_fraction });
}

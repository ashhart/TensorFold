//! TF_Q35_DFLASH_ABSORB_BATCH: absorb projections over all streams' rows give each row its own launch's bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const Rig = quant_ref.RigT;
const hidden = 5120;
const taps = 5;
const kv = 1024; // DFlash2's k and v rows: kv heads x head dim

fn sums(r: *Rig, x: []const u16, rows: usize, k: usize) ![]f32 {
    const out = try r.a.alloc(f32, rows * k / 64);
    for (0..rows) |row| for (0..k / 64) |g| {
        var s: f32 = 0;
        for (0..64) |j| s += @floatCast(quant_ref.helpers.bfOf(x[row * k + g * 64 + j]));
        out[row * (k / 64) + g] = s;
    };
    return out;
}

fn weight(r: *Rig, a4: *const qlinear.Affine4Kernels, n: usize, k: usize) !qlinear.Affine4 {
    const words = try r.a.alloc(u32, n * k / 8);
    for (words) |*w| w.* = r.rng.int(u32);
    const scales = try r.a.alloc(u16, n * k / 64);
    const biases = try r.a.alloc(u16, n * k / 64);
    for (scales, biases) |*sv, *bv| {
        sv.* = quant_ref.helpers.bf16Of((r.rng.float(f32) * 0.9 + 0.1) * 0.0025);
        bv.* = quant_ref.helpers.bf16Of(-(r.rng.float(f32) * 0.4 + 0.8) * 0.02);
    }
    const lay = qlinear.Affine4.layout(n, k);
    const q: qlinear.Affine4 = .{ .w = try r.zeros(lay.words), .s = try r.zeros(lay.scales), .b = try r.zeros(lay.scales), .n = n, .k = k, .npad = lay.npad };
    try a4.pack(r.s, try r.dev(std.mem.sliceAsBytes(words)), try r.dev(std.mem.sliceAsBytes(scales)), try r.dev(std.mem.sliceAsBytes(biases)), q);
    return q;
}

/// precision.affine's live rows: verify in tiles of verifyRows().
fn live(a4: *const qlinear.Affine4Kernels, s: cuda.Stream, x: u64, xs: u64, q: qlinear.Affine4, out: u64, rows: usize) !void {
    const tile = a4.verifyRows();
    var at: usize = 0;
    while (at < rows) : (at += tile) try a4.verify(s, x + at * q.k * 2, xs + at * (q.k / 64) * 4, q, out + at * q.n * 2, @min(tile, rows - at));
}

/// One pass's outputs: fc over the taps, then k and v (one call, as projManySums takes them) over hidden rows.
const Outs = struct { fc: u64, k: u64, v: u64 };

fn project(r: *Rig, a4: *const qlinear.Affine4Kernels, fc: qlinear.Affine4, k: qlinear.Affine4, v: qlinear.Affine4, xt: []const u16, xh: []const u16, rows: usize) !Outs {
    const o: Outs = .{ .fc = try r.zeros(rows * hidden * 2), .k = try r.zeros(rows * kv * 2), .v = try r.zeros(rows * kv * 2) };
    if (rows == 0) return o;
    try live(a4, r.s, try r.dev(std.mem.sliceAsBytes(xt)), try r.dev(std.mem.sliceAsBytes(try sums(r, xt, rows, taps * hidden))), fc, o.fc, rows);
    const dh = try r.dev(std.mem.sliceAsBytes(xh));
    const dhs = try r.dev(std.mem.sliceAsBytes(try sums(r, xh, rows, hidden)));
    if (a4.pdl and rows > 1) {
        try a4.verifyManyRows(r.s, dh, dhs, &.{ .{ .q = k, .out = o.k }, .{ .q = v, .out = o.v } }, rows);
    } else {
        try live(a4, r.s, dh, dhs, k, o.k, rows);
        try live(a4, r.s, dh, dhs, v, o.v, rows);
    }
    return o;
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(1049);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk = cuda.kernels;
    const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*md| md.unload();
    for (images, &mods) |img, *md| {
        md.* = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const gb10 = (try gpu.ctx.capability()) == 121;
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, gb10);
    defer a4.deinit();
    const fc = try weight(&r, &a4, hidden, taps * hidden);
    const k = try weight(&r, &a4, kv, hidden);
    const v = try weight(&r, &a4, kv, hidden);
    var cells: usize = 0;
    for ([_]usize{ 1, 4, 8, 16 }) |streams| {
        // uneven kept rows, 0 to 16 a stream, at most one pass's rows in all
        var n: [16]usize = undefined;
        var total: usize = 0;
        for (n[0..streams]) |*c| {
            c.* = @min(r.rng.uintAtMost(usize, 16), a4.verifyRows() - total);
            total += c.*;
        }
        if (total == 0) {
            n[0] = 1;
            total = 1;
        }
        const xt = try r.bfs(total * taps * hidden, 1.0);
        const xh = try r.bfs(total * hidden, 1.0);
        for ([_]u16{ 0x7FC0, 0x7F80, 0xFF80 }, 0..) |special, j| if (total > j) { // NaN and infinities planted
            xt[j * taps * hidden + 3] = special;
            xh[j * hidden + 7] = special;
        };
        const batched = try project(&r, &a4, fc, k, v, xt, xh, total);
        var row0: usize = 0;
        for (n[0..streams]) |c| {
            const own = try project(&r, &a4, fc, k, v, xt[row0 * taps * hidden ..][0 .. c * taps * hidden], xh[row0 * hidden ..][0 .. c * hidden], c);
            inline for (.{ "fc", "k", "v" }, .{ hidden, kv, kv }) |name, w| {
                const want = try r.back(@field(own, name), c * w * 2);
                const got = try r.back(@field(batched, name) + row0 * w * 2, c * w * 2);
                try check.sameBytes("absorb batch: " ++ name ++ " rows of one stream", got, want);
            }
            row0 += c;
            cells += 1;
        }
        // planted control: one row's input changed moves that row's k and nothing else
        const last = total - 1;
        xh[last * hidden] ^= 0x0040;
        const moved = try project(&r, &a4, fc, k, v, xt, xh, total);
        const before = try r.back(batched.k, total * kv * 2);
        const after = try r.back(moved.k, total * kv * 2);
        try check.expect(!std.mem.eql(u8, before[last * kv * 2 ..], after[last * kv * 2 ..]), "absorb batch: the planted row's k moves ({d} streams)", .{streams});
        try check.sameBytes("absorb batch: the other rows' k stay", after[0 .. last * kv * 2], before[0 .. last * kv * 2]);
    }
    check.pass("absorb-batch: {d} stream cells at 1/4/8/16 streams, fc, k and v rows byte-equal batched and alone, NaN/Inf planted, control caught", .{cells});
}

//! TF_NEMOTRON_WIDE_ROUNDS: Lightning's dense calls at 17-32 rows give each row its bytes from 16-row calls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const Rig = quant_ref.RigT;

/// Lightning's dense projections (n, k): in_proj, out_proj (and o), qkv, MTP eh_proj, draft head, lm_head.
const shapes = [_][2]usize{ .{ 10304, 2688 }, .{ 2688, 4096 }, .{ 4608, 2688 }, .{ 2688, 5376 }, .{ 49152, 2688 }, .{ 131072, 2688 } };
/// A wide round's rows: the narrow rounds it replaces are 16 rows, then the rest.
const row_counts = [_]usize{ 17, 25, 32 };

fn sums(r: *Rig, x: []const u16, rows: usize, k: usize) ![]f32 {
    const out = try r.a.alloc(f32, rows * k / 64);
    for (0..rows) |row| for (0..k / 64) |g| {
        var s: f32 = 0;
        for (0..64) |j| s += @floatCast(quant_ref.helpers.bfOf(x[row * k + g * 64 + j]));
        out[row * (k / 64) + g] = s;
    };
    return out;
}

/// A weight in each format a wide round takes: MLX affine-4 (packed as served), NVFP4, and ModelOpt's FP8 and bf16.
const W = union(enum) { affine4: qlinear.Affine4, nvfp4: cuda.qmmf.Weight, fp8: cuda.qmmf.Weight, bf16: cuda.qmmf.Weight };

fn affine(r: *Rig, a4: *const qlinear.Affine4Kernels, n: usize, k: usize) !W {
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
    return .{ .affine4 = q };
}

fn fp4(r: *Rig, n: usize, k: usize) !W {
    const npad = (n + 127) / 128 * 128;
    const codes = try r.a.alloc(u8, npad * k / 2);
    r.rng.bytes(codes);
    const scales = try r.a.alloc(u8, npad * k / 16);
    for (scales) |*v| v.* = 0x30 + r.rng.uintLessThan(u8, 16);
    return .{ .nvfp4 = .{ .mode = .fp4, .codes = try r.dev(codes), .scales = try r.dev(scales), .scale = 1, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) } };
}

/// ModelOpt FP8: finite e4m3 codes on the lane's padded rows, one scale.
fn fp8(r: *Rig, n: usize, k: usize) !W {
    const codes = try r.a.alloc(u8, cuda.modelopt.padded(n) * k);
    for (codes) |*c| c.* = r.rng.uintLessThan(u8, 0x70) | (r.rng.int(u8) & 0x80);
    return .{ .fp8 = cuda.modelopt.fp8Weight(try r.dev(codes), 0.75, n, k) };
}

/// ModelOpt bf16: small finite values on the lane's padded rows.
fn bf16(r: *Rig, n: usize, k: usize) !W {
    const vals = try r.a.alloc(u16, cuda.modelopt.padded(n) * k);
    for (vals) |*v| v.* = quant_ref.helpers.bf16Of((r.rng.float(f32) - 0.5) * 0.05);
    return .{ .bf16 = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(vals)), n, k) };
}

/// One call of `rows` rows from row `at`, as the engine's dense projection makes it.
fn call(l: qlinear.Linear, r: *Rig, w: W, x: u64, xs: u64, part: u64, out: u64, at: usize, rows: usize) !void {
    const k: usize = switch (w) {
        inline else => |q| q.k,
    };
    const n: usize = switch (w) {
        inline else => |q| q.n,
    };
    const weight: qlinear.Weight = switch (w) {
        .affine4 => |q| .{ .affine4 = q },
        .nvfp4 => |q| .{ .nvfp4 = q },
        .fp8 => |q| .{ .fp8 = q },
        .bf16 => |q| .{ .bf16 = q },
    };
    const ax = x + at * k * 2;
    const axs = xs + at * (k / 64) * 4;
    // nemotron Ops.dense with the switch on: affine-4 rows past 16 take verify's tiles
    if (weight == .affine4 and rows > qlinear.Affine4Kernels.max_rows) return l.affine4.?.verify(r.s, ax, axs, weight.affine4, out + at * n * 2, rows);
    try l.decode(r.s, weight, .{ .x = ax, .sums = axs, .part = part }, out + at * n * 2, rows);
}

pub fn run(gpu: Gpu) !void {
    if ((try gpu.ctx.capability()) != 121) {
        check.pass("wide-rounds: N/A off GB10 (the switch stays off there: affine-4 has no 32-row verify)", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(1050);
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
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer a4.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, 12);
    defer lane.unload();
    const l: qlinear.Linear = .{ .affine4 = &a4, .lane = &lane };
    const part = try r.zeros(64 << 20); // slice partials where the lane reduces them
    var cells: usize = 0;
    for (shapes) |sh| for ([_]std.meta.Tag(W){ .affine4, .nvfp4, .fp8, .bf16 }) |format| {
        const mark = r.bufs.items.len;
        defer {
            for (r.bufs.items[mark..]) |*b| b.free();
            r.bufs.shrinkRetainingCapacity(mark);
        }
        const n = sh[0];
        const k = sh[1];
        const w = switch (format) {
            .affine4 => try affine(&r, &a4, n, k),
            .nvfp4 => try fp4(&r, n, k),
            .fp8 => try fp8(&r, n, k),
            .bf16 => try bf16(&r, n, k),
        };
        const x = try r.bfs(32 * k, 1.0);
        for ([_]u16{ 0x7FC0, 0x7F80, 0xFF80 }, [_]usize{ 2, 19, 30 }) |special, row| x[row * k + 5] = special; // NaN, +-Inf
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const dxs = try r.dev(std.mem.sliceAsBytes(try sums(&r, x, 32, k)));
        for (row_counts) |rows| {
            const wide = try r.zeros(rows * n * 2);
            const narrow = try r.zeros(rows * n * 2);
            try call(l, &r, w, dx, dxs, part, wide, 0, rows);
            try call(l, &r, w, dx, dxs, part, narrow, 0, 16);
            try call(l, &r, w, dx, dxs, part, narrow, 16, rows - 16);
            try check.sameBytes("wide rounds: each format's rows equal their 16-row calls", try r.back(wide, rows * n * 2), try r.back(narrow, rows * n * 2));
            cells += 1;
        }
        // planted control: row 20's input moved changes row 20 of the wide call and no other row
        const before = try r.zeros(32 * n * 2);
        try call(l, &r, w, dx, dxs, part, before, 0, 32);
        x[20 * k] ^= 0x0040;
        const cx = try r.dev(std.mem.sliceAsBytes(x));
        const cxs = try r.dev(std.mem.sliceAsBytes(try sums(&r, x, 32, k)));
        const after = try r.zeros(32 * n * 2);
        try call(l, &r, w, cx, cxs, part, after, 0, 32);
        const b = try r.back(before, 32 * n * 2);
        const a = try r.back(after, 32 * n * 2);
        try check.expect(!std.mem.eql(u8, b[20 * n * 2 ..][0 .. n * 2], a[20 * n * 2 ..][0 .. n * 2]), "wide rounds: the planted row moves ([{d}, {d}])", .{ n, k });
        try check.sameBytes("wide rounds: rows beside the planted one stay", a[0 .. 20 * n * 2], b[0 .. 20 * n * 2]);
        try check.sameBytes("wide rounds: rows after the planted one stay", a[21 * n * 2 ..], b[21 * n * 2 ..]);
        x[20 * k] ^= 0x0040;
    };
    check.pass("wide-rounds: {d} cells, Lightning's dense shapes in affine-4, NVFP4, FP8 and bf16 at 17/25/32 rows byte-equal to 16 + the rest, NaN/Inf planted, control caught", .{cells});
}

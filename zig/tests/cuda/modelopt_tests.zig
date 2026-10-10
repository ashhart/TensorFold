//! ModelOpt formats on the GPU: repacks against host twins; FP8, BF16 lane rows and relu^2 NVFP4 experts against fp64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const modelopt = cuda.modelopt;
const qmmf = cuda.qmmf;
const fx4 = cuda.experts.Nvfp4Experts;

fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * std.math.pow(f64, 2, -6) else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

fn bf(v: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
}

fn tobf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// Random bytes; e4m3 codes skip NaN (0x7F, 0xFF) and, as scales, stay positive in [2^-4, 2^3).
fn fill(rng: std.Random, out: []u8, what: enum { any, e4m3, scale }) void {
    for (out) |*b| b.* = switch (what) {
        .any => rng.int(u8),
        .e4m3 => blk: {
            const v = rng.int(u8);
            break :blk if (v & 0x7F == 0x7F) v ^ 1 else v;
        },
        .scale => 0x18 + rng.uintLessThan(u8, 0x30),
    };
}

fn bf16s(gpa: std.mem.Allocator, rng: std.Random, n: usize, spread: f32) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = tobf((rng.float(f32) - 0.5) * spread);
    return out;
}

fn dev(gpu: Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

fn same(gpu: Gpu, what: []const u8, b: cuda.DeviceBuffer, want: []const u8) !void {
    const got = try check.download(gpu, b);
    defer gpu.gpa.free(got);
    try check.sameBytes(what, got[0..want.len], want);
}

pub fn run(gpu: Gpu) !void {
    var prng = std.Random.DefaultPrng.init(0x5eed_f4f8);
    const rng = prng.random();
    var pack = try modelopt.Packer.load(gpu.d);
    defer pack.unload();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    try packers(gpu, rng, &pack, stream);
    var lane = try qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var gemm = try cuda.nvfp4.Prompt.load(gpu.d);
    defer gemm.unload();
    const lin: cuda.qlinear.Linear = .{ .lane = &lane, .gemm = &gemm };
    for ([_]qmmf.Mode{ .fp8, .bf16 }) |mode| try lanes(gpu, rng, &pack, lin, stream, mode);
    try experts(gpu, rng, &pack, stream);
}

/// Every GPU repack byte-equal to its host twin.
fn packers(gpu: Gpu, rng: std.Random, p: *const modelopt.Packer, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const n = 130;
    const k = 128;
    const np = modelopt.padded(n);
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    fill(rng, codes, .e4m3);
    const want8 = try a.alloc(u8, np * k);
    defer a.free(want8);
    cuda.fp8.packCodes(want8, codes, n, k);
    var src = try dev(gpu, codes);
    defer src.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, np * k * 2);
    defer out.free();
    try p.fp8Lane(s, src.ptr, out.ptr, n, k);
    try s.synchronize();
    try same(gpu, "fp8 lane codes", out, want8);
    const w16 = try bf16s(a, rng, n * k, 2);
    defer a.free(w16);
    const want16 = try a.alloc(u16, np * k);
    defer a.free(want16);
    qmmf.packBf16(want16, w16, n, k);
    var src16 = try dev(gpu, std.mem.sliceAsBytes(w16));
    defer src16.free();
    try p.bf16Lane(s, src16.ptr, out.ptr, n, k);
    try s.synchronize();
    try same(gpu, "bf16 lane", out, std.mem.sliceAsBytes(want16));
    const c4 = codes[0 .. n * k / 2];
    const s4 = try a.alloc(u8, n * k / 16);
    defer a.free(s4);
    fill(rng, s4, .scale);
    const words = try a.alloc(u32, np / 64 * (k / 64) * 512);
    defer a.free(words);
    cuda.nvfp4.packWords(words, c4, n, k);
    const tiles = try a.alloc(u8, np * (k / 16));
    defer a.free(tiles);
    cuda.nvfp4.packScales(tiles, s4, n, k);
    var dc = try dev(gpu, c4);
    defer dc.free();
    var ds = try dev(gpu, s4);
    defer ds.free();
    var dt = try cuda.DeviceBuffer.alloc(gpu.d, tiles.len);
    defer dt.free();
    try p.fp4Lane(s, dc.ptr, ds.ptr, out.ptr, dt.ptr, n, k);
    try s.synchronize();
    try same(gpu, "fp4 lane words", out, std.mem.sliceAsBytes(words));
    try same(gpu, "fp4 lane scale tiles", dt, tiles);
    // experts: 3 of [64, 96], then one of [64, 192] packed as two column halves
    const e = 3;
    const en = 64;
    const ek = 96;
    const blk = en / fx4.cols * (ek / 32) * fx4.words;
    const ec = try a.alloc(u8, e * en * ek); // codes [e][en][ek/2] in the first half, room for [en][2 ek/2]
    defer a.free(ec);
    fill(rng, ec, .any);
    const es = try a.alloc(u8, e * en * ek / 8);
    defer a.free(es);
    fill(rng, es, .scale);
    const want = try a.alloc(u32, e * blk);
    defer a.free(want);
    for (0..e) |x| fx4.packOne(want[x * blk ..][0..blk], ec[x * en * ek / 2 ..][0 .. en * ek / 2], es[x * en * ek / 16 ..][0 .. en * ek / 16], en, ek);
    var dec = try dev(gpu, ec);
    defer dec.free();
    var des = try dev(gpu, es);
    defer des.free();
    try p.fp4Experts(s, dec.ptr, des.ptr, out.ptr, e, en, ek, 0, ek, 1, 0);
    try s.synchronize();
    try same(gpu, "nvfp4 expert blocks", out, std.mem.sliceAsBytes(want));
    for (0..2) |h| { // the shared down projection's halves: one [en, 2 ek] matrix, inputs h * ek ..
        const half_c = try a.alloc(u8, en * ek / 2);
        defer a.free(half_c);
        const half_s = try a.alloc(u8, en * ek / 16);
        defer a.free(half_s);
        for (0..en) |r| {
            @memcpy(half_c[r * ek / 2 ..][0 .. ek / 2], ec[r * ek + h * ek / 2 ..][0 .. ek / 2]);
            @memcpy(half_s[r * ek / 16 ..][0 .. ek / 16], es[r * ek / 8 + h * ek / 16 ..][0 .. ek / 16]);
        }
        fx4.packOne(want[0..blk], half_c, half_s, en, ek);
        try p.fp4Experts(s, dec.ptr, des.ptr, out.ptr, 1, en, 2 * ek, h * ek, ek, 1, 0);
        try s.synchronize();
        try same(gpu, "nvfp4 expert half", out, std.mem.sliceAsBytes(want[0..blk]));
    }
    check.pass("modelopt repack: fp8, bf16 and fp4 lane tiles and nvfp4 expert blocks (and column halves) equal the host packers", .{});
}

/// The lane matmul (decode rows) and prompt rows for one mode against fp64; rows equal at every row count and chunking.
fn lanes(gpu: Gpu, rng: std.Random, p: *const modelopt.Packer, lin: cuda.qlinear.Linear, s: cuda.Stream, mode: qmmf.Mode) !void {
    const a = gpu.gpa;
    const n = 320;
    const k = 2048;
    const fp8 = mode == .fp8;
    const scale: f32 = if (fp8) 0.0123 else 1.0;
    const raw = try a.alloc(u8, n * k * 2);
    defer a.free(raw);
    const w = try a.alloc(f64, n * k);
    defer a.free(w);
    if (fp8) {
        fill(rng, raw[0 .. n * k], .e4m3);
        for (w, raw[0 .. n * k]) |*v, b| v.* = e4m3(b) * scale;
    } else {
        const w16 = try bf16s(a, rng, n * k, 0.25);
        defer a.free(w16);
        @memcpy(raw, std.mem.sliceAsBytes(w16));
        for (w, w16) |*v, b| v.* = bf(b);
    }
    var src = try dev(gpu, raw[0 .. n * k * (if (fp8) @as(usize, 1) else 2)]);
    defer src.free();
    var packed_w = try cuda.DeviceBuffer.alloc(gpu.d, modelopt.padded(n) * k * 2);
    defer packed_w.free();
    if (fp8) try p.fp8Lane(s, src.ptr, packed_w.ptr, n, k) else try p.bf16Lane(s, src.ptr, packed_w.ptr, n, k);
    const q: cuda.qlinear.Weight = if (fp8) .{ .fp8 = modelopt.fp8Weight(packed_w.ptr, scale, n, k) } else .{ .bf16 = modelopt.bf16Weight(packed_w.ptr, n, k) };
    const m = 300;
    const x = try bf16s(a, rng, m * k, 4);
    defer a.free(x);
    var dx = try dev(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
    defer y.free();
    var y2 = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
    defer y2.free();
    try lin.decode(s, q, .{ .x = dx.ptr }, y.ptr, 16);
    try s.synchronize();
    const wide = try check.download(gpu, y);
    defer a.free(wide);
    for (1..16) |rows| {
        try y2.fill8(0xff, s.handle);
        try lin.decode(s, q, .{ .x = dx.ptr }, y2.ptr, rows);
        try s.synchronize();
        try same(gpu, "decode rows at a narrower width", y2, wide[0 .. rows * n * 2]);
    }
    var worst = [2]f64{ 0, 0 };
    try lin.prompt(s, q, .{ .x = dx.ptr }, y.ptr, m);
    try s.synchronize();
    const prompt = try check.download(gpu, y);
    defer a.free(prompt);
    try lin.prompt(s, q, .{ .x = dx.ptr }, y2.ptr, 128);
    try s.synchronize();
    try same(gpu, "prompt rows in a chunk of 128", y2, prompt[0 .. 128 * n * 2]);
    for (0..m) |r| for (0..n) |c| {
        var ref: f64 = 0;
        for (0..k) |i| ref += bf(x[r * k + i]) * w[c * k + i];
        const tol = 1e-2 * @max(@abs(ref), 1.0);
        for ([2][]const u8{ wide, prompt }, &worst) |out, *err| {
            if (r >= 16 and out.ptr == wide.ptr) continue;
            const got = bf(std.mem.readInt(u16, out[(r * n + c) * 2 ..][0..2], .little));
            err.* = @max(err.*, @abs(got - ref) / tol);
        }
    };
    try check.expect(worst[0] <= 1 and worst[1] <= 1, "{t} rows within 1% of fp64: decode {d:.3}, prompt {d:.3} of the bound", .{ mode, worst[0], worst[1] });
    check.pass("{t} lane: rows 1-16 byte-equal, 300 prompt rows chunk-invariant, worst |err| / (1% of max(|ref|, 1)): decode {d:.3}, prompt {d:.3}", .{ mode, worst[0], worst[1] });
}

/// relu^2 NVFP4 experts (one up projection, Nemotron's): decode and prompt forms against fp64, then each other.
fn experts(gpu: Gpu, rng: std.Random, p: *const modelopt.Packer, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const e = 4;
    const width = 64;
    const dims = 128;
    const rows = 5;
    const slots = 2;
    const pairs = rows * slots;
    var codes: [2][]u8 = undefined;
    var scales: [2][]u8 = undefined;
    var dev_w: [2]cuda.DeviceBuffer = undefined;
    const gs = [2][e]f32{ .{ 0.031, 0.017, 0.022, 0.05 }, .{ 0.012, 0.02, 0.009, 0.015 } };
    for (0..2) |i| {
        codes[i] = try a.alloc(u8, e * width * dims / 2);
        scales[i] = try a.alloc(u8, e * width * dims / 16);
        fill(rng, codes[i], .any);
        fill(rng, scales[i], .scale);
    }
    defer for (0..2) |i| {
        a.free(codes[i]);
        a.free(scales[i]);
    };
    for (0..2) |i| {
        const n: usize = if (i == 0) width else dims;
        const k: usize = if (i == 0) dims else width;
        var dc = try dev(gpu, codes[i]);
        defer dc.free();
        var ds = try dev(gpu, scales[i]);
        defer ds.free();
        dev_w[i] = try cuda.DeviceBuffer.alloc(gpu.d, fx4.bytes(e, n, k, 1));
        try p.fp4Experts(s, dc.ptr, ds.ptr, dev_w[i].ptr, e, n, k, 0, k, 1, 0);
        try s.synchronize();
    }
    defer for (&dev_w) |*b| b.free();
    var dgs = try dev(gpu, std.mem.sliceAsBytes(&gs));
    defer dgs.free();
    const l: cuda.experts.Layer = .{ .format = .nvfp4, .up = dev_w[0].ptr, .down = dev_w[1].ptr, .up_scale = dgs.ptr, .down_scale = dgs.ptr + e * 4, .width = width, .dims = dims, .experts = e, .relu2 = true };
    const picks = [pairs]i32{ 0, 3, 1, 3, 2, 2, 0, 1, 3, 0 };
    const x = try bf16s(a, rng, rows * dims, 4);
    defer a.free(x);
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try fx4.load(gpu.d, sms);
    defer ex.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    const g: cuda.experts.Grouped = .{ .nvfp4 = &ex };
    var bufs: [5]cuda.DeviceBuffer = undefined;
    for (&bufs, [5]usize{ pairs * 4, cuda.grouped.maxItems(pairs, e, 16) * 12, 8, 4, 4 }) |*b, n| b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer for (&bufs) |*b| b.free();
    const plan: cuda.grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    var dp = try dev(gpu, std.mem.sliceAsBytes(&picks));
    defer dp.free();
    var dx = try dev(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * width * 2);
    defer act.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * dims * 4);
    defer y.free();
    var outs: [2][2][]u8 = undefined; // [decode, prompt][act, y]
    for (0..2) |form| {
        const tile: usize = if (form == 0) 16 else cuda.experts.promptTile(.nvfp4);
        try router.route(s, dp.ptr, pairs, e, tile, plan);
        const items = cuda.grouped.maxItems(pairs, e, tile);
        try act.fill8(0xff, s.handle);
        try y.fill8(0xff, s.handle);
        const up_in: cuda.experts.Rows = .{ .x = dx.ptr, .stride = dims, .slots = slots };
        const down_in: cuda.experts.Rows = .{ .x = act.ptr, .stride = width };
        if (form == 0) {
            try g.decode(s, .up, up_in, l, plan, act.ptr, items, -1);
            try g.decode(s, .down_f32, down_in, l, plan, y.ptr, items, -1);
        } else {
            try g.prompt(s, .up, up_in, l, plan, act.ptr, items, -1);
            try g.prompt(s, .down_bf16, down_in, l, plan, y.ptr, items, -1);
        }
        try s.synchronize();
        outs[form] = .{ try check.download(gpu, act), try check.download(gpu, y) };
    }
    defer for (outs) |o| for (o) |b| a.free(b);
    try check.sameBytes("relu^2 up: prompt form == decode form", outs[1][0], outs[0][0]);
    var worst: f64 = 0;
    for (0..pairs) |pr| {
        const ex_i: usize = @intCast(picks[pr]);
        const row = pr / slots;
        for (0..width) |j| { // act = bf16(relu(bf16(x . w_up * g))^2)
            var acc: f64 = 0;
            for (0..dims) |i| acc += bf(x[row * dims + i]) * weight(codes[0], scales[0], ex_i, width, dims, j, i);
            const pre = bf(tobf(@floatCast(acc * gs[0][ex_i])));
            const want = @max(pre, 0) * @max(pre, 0);
            const got = bf(std.mem.readInt(u16, outs[0][0][(pr * width + j) * 2 ..][0..2], .little));
            worst = @max(worst, @abs(got - want) / (2e-2 * @max(@abs(want), 1.0)));
        }
        for (0..dims) |c| { // y = act . w_down * g, from the kernel's own act
            var acc: f64 = 0;
            for (0..width) |j| acc += bf(std.mem.readInt(u16, outs[0][0][(pr * width + j) * 2 ..][0..2], .little)) * weight(codes[1], scales[1], ex_i, dims, width, c, j);
            const want = acc * gs[1][ex_i];
            const got: f64 = @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, outs[0][1][(pr * dims + c) * 4 ..][0..4], .little))));
            worst = @max(worst, @abs(got - want) / (1e-3 * @max(@abs(want), 1.0)));
            const sum16 = std.mem.readInt(u16, outs[1][1][(pr * dims + c) * 2 ..][0..2], .little);
            try check.expect(sum16 == tobf(@floatCast(got)), "prompt-form bf16 sum is the decode form's fp32 sum rounded", .{});
        }
    }
    try check.expect(worst <= 1, "relu^2 experts within bounds of fp64 ({d:.3} of the bound)", .{worst});
    check.pass("nvfp4 relu^2 experts: decode and prompt forms agree byte for byte, worst error {d:.3} of the bound (2% up, 0.1% down)", .{worst});
}

/// Weight [r, c] of expert `x` of [n, k] NVFP4 matrices: e2m1 code times its e4m3 block scale.
fn weight(codes: []const u8, scales: []const u8, x: usize, n: usize, k: usize, r: usize, c: usize) f64 {
    const byte = codes[(x * n + r) * (k / 2) + c / 2];
    const code: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
    return e2m1(code) * e4m3(scales[(x * n + r) * (k / 16) + c / 16]);
}

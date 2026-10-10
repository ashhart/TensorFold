//! The quantized products against float64 references on synthetic weights, and every row's bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const fp8 = cuda.fp8;
const nvfp4 = cuda.nvfp4;
const qmmf = cuda.qmmf;
const qlinear = cuda.qlinear;
const grouped = cuda.grouped;
const experts = cuda.experts;

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0, // the largest difference as a fraction of its tolerance, over every check

    pub fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    pub fn zeros(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(bytes, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle); // a launch that writes nothing fails its check
        return b.ptr;
    }

    pub fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
        const out = try r.a.alloc(u8, len);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, out);
        return out;
    }

    /// bf16 inputs in [-scale, scale).
    pub fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = bf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    /// `got` (bf16 or fp32 at `ptr`) within `tol` of `want` everywhere; tol is per element.
    pub fn close(r: *Rig, name: []const u8, ptr: u64, f32_out: bool, want: []const f64, tol: []const f64) !void {
        const raw = try r.back(ptr, want.len * @as(usize, if (f32_out) 4 else 2));
        var most: f64 = 0;
        var at: usize = 0;
        for (want, tol, 0..) |w, t, i| {
            const g: f64 = if (f32_out) std.mem.bytesToValue(f32, raw[4 * i ..][0..4]) else bf(std.mem.bytesToValue(u16, raw[2 * i ..][0..2]));
            const q = @abs(g - w) / t;
            if (!(q <= most)) {
                most = q;
                at = i;
            }
        }
        r.worst = @max(r.worst, most);
        std.debug.print("RESULT {s}: largest difference {d:.3} of its tolerance (element {d})\n", .{ name, most, at });
        try check.expect(most <= 1, "{s} within its float64 tolerance", .{name});
    }
};

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// An e4m3 byte (no NaN codes in use).
fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    if (e == 0) return sign * m / 8.0 * std.math.pow(f64, 2, -6);
    return sign * (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// An e2m1 nibble.
fn e2m1(n: u8) f64 {
    const vals = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * vals[n & 7];
}

fn fp8Code(r: *Rig) u8 {
    while (true) {
        const b = r.rng.int(u8);
        if (b & 0x7f != 0x7f) return b; // e4m3 NaN codes out: a checkpoint holds none
    }
}

/// y = x w^T by `value(row, col)`, and its bound: a bf16 ulp (2^-7 of |y|) and 2^-12 of sum |x w|.
fn reference(r: *Rig, x: []const u16, m: usize, k: usize, n: usize, ctx: anytype, comptime value: fn (@TypeOf(ctx), usize, usize) f64) !struct { want: []f64, tol: []f64 } {
    const want = try r.a.alloc(f64, m * n);
    const tol = try r.a.alloc(f64, m * n);
    const wd = try r.a.alloc(f64, n * k);
    for (0..n) |j| for (0..k) |i| {
        wd[j * k + i] = value(ctx, j, i);
    };
    for (0..m) |row| for (0..n) |j| {
        var sum: f64 = 0;
        var abs: f64 = 0;
        for (0..k) |i| {
            const t = bf(x[row * k + i]) * wd[j * k + i];
            sum += t;
            abs += @abs(t);
        }
        want[row * n + j] = sum;
        tol[row * n + j] = @abs(sum) * std.math.pow(f64, 2, -7) + abs * std.math.pow(f64, 2, -12) + 1e-30;
    };
    return .{ .want = want, .tol = tol };
}

/// A wide call's first rows must equal every narrower call's rows byte for byte.
fn sameRows(r: *Rig, name: []const u8, wide: u64, narrow: u64, bytes: usize) !void {
    const a = try r.back(wide, bytes);
    const b = try r.back(narrow, bytes);
    try check.sameBytes(name, b, a);
}

const Fp8Ctx = struct { codes: []const u8, inv: []const f32, k: usize };
fn fp8Value(c: Fp8Ctx, row: usize, col: usize) f64 {
    return e4m3(c.codes[row * c.k + col]) * c.inv[(row / 128) * (c.k / 128) + col / 128];
}

const Fp4Ctx = struct { codes: []const u8, scales: []const u8, global: f32, k: usize };
fn fp4Value(c: Fp4Ctx, row: usize, col: usize) f64 {
    const byte = c.codes[row * (c.k / 2) + col / 2];
    const nib: u8 = if (col & 1 == 0) byte & 0xF else byte >> 4;
    return e2m1(nib) * e4m3(c.scales[row * (c.k / 16) + col / 16]) * c.global;
}

const rows = [_]usize{ 1, 3, 17, 40, 300 };

fn lanes(r: *Rig, lin: qlinear.Linear, n: usize, k: usize) !void {
    const x = try r.bfs(rows[rows.len - 1] * k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    // block-FP8: e4m3 codes and fp32 scale_inv a 128 x 128 block
    const codes = try r.a.alloc(u8, n * k);
    for (codes) |*c| c.* = fp8Code(r);
    const inv = try r.a.alloc(f32, ((n + 127) / 128) * (k / 128));
    for (inv) |*v| v.* = r.rng.float(f32) * 0.01 + 0.001;
    const w8 = try r.a.alloc(u8, fp8.padded(n) * k);
    fp8.packCodes(w8, codes, n, k);
    const bs = try r.a.alloc(f32, fp8.padded(n) * (k / fp8.group));
    fp8.packScales(bs, inv, n, k);
    const w = fp8.weight(try r.dev(w8), try r.dev(std.mem.sliceAsBytes(bs)), n, k);
    // NVFP4: e2m1 codes, an e4m3 scale a 16 inputs, a global scale
    const c4 = try r.a.alloc(u8, n * k / 2);
    r.rng.bytes(c4);
    const s4 = try r.a.alloc(u8, n * k / 16);
    for (s4) |*v| v.* = 0x20 + r.rng.uintLessThan(u8, 0x28); // positive e4m3, no NaN
    const words = try r.a.alloc(u32, nvfp4.padded(n) / 64 * (k / nvfp4.group) * 512);
    nvfp4.packWords(words, c4, n, k);
    const bs4 = try r.a.alloc(u8, nvfp4.padded(n) * (k / 16));
    nvfp4.packScales(bs4, s4, n, k);
    const global: f32 = 0.0123;
    const w4 = nvfp4.weight(try r.dev(std.mem.sliceAsBytes(words)), try r.dev(bs4), global, n, k);
    const wide = rows[rows.len - 1];
    const part = try r.zeros(8 * wide * n * 4);
    var name: [96]u8 = undefined;
    inline for (.{ .{ "fp8g", w, Fp8Ctx{ .codes = codes, .inv = inv, .k = k }, fp8Value }, .{ "nvfp4", w4, Fp4Ctx{ .codes = c4, .scales = s4, .global = global, .k = k }, fp4Value } }) |f| {
        const view: qlinear.Weight = if (comptime std.mem.eql(u8, f[0], "fp8g")) .{ .fp8g = f[1] } else .{ .nvfp4 = f[1] };
        const ref = try reference(r, x, wide, k, n, f[2], f[3]);
        for ([_]bool{ false, true }) |prompt| {
            const outs = try r.a.alloc(u64, rows.len);
            for (rows, outs) |m, *o| {
                o.* = try r.zeros(m * n * 2);
                const in: qlinear.Rows = .{ .x = dx, .ldx = k, .part = part };
                if (prompt) try lin.prompt(r.s, view, in, o.*, m) else try lin.decode(r.s, view, in, o.*, m);
            }
            const path = if (prompt) "prompt" else "decode";
            try r.close(try std.fmt.bufPrint(&name, "{s} {s} {d} rows x [{d}, {d}]", .{ f[0], path, wide, n, k }), outs[rows.len - 1], false, ref.want, ref.tol);
            for (rows[0 .. rows.len - 1], outs[0 .. rows.len - 1]) |m, o| try sameRows(r, try std.fmt.bufPrint(&name, "{s} {s}: {d} rows equal the {d}-row call's", .{ f[0], path, m, wide }), outs[rows.len - 1], o, m * n * 2);
        }
    }
    check.pass("qlinear fp8g and nvfp4, [{d}, {d}]: decode and prompt within float64 tolerance, every row count's bits equal", .{ n, k });
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const cap = try gpu.ctx.capability();
    var lane = try qmmf.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    var pg = try nvfp4.Prompt.load(gpu.d);
    defer pg.unload();
    const lin: qlinear.Linear = .{ .lane = &lane, .nvfp4_prompt = &pg };
    try lanes(&r, lin, 320, 2560);
    try lanes(&r, lin, 2560, 512);
    try @import("quant_ref_experts.zig").run(&r);
    check.pass("quant-ref: every product within {d:.3} of its float64 tolerance", .{r.worst});
}

pub const RigT = Rig;
pub const helpers = struct {
    pub const bfOf = bf;
    pub const bf16Of = bf16;
    pub const e4m3Of = e4m3;
    pub const e2m1Of = e2m1;
    pub const fp8CodeOf = fp8Code;
};

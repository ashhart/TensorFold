//! MLX 4-bit affine (group 64) matvec for m = 1..16 rows on the matrix engine (qmv4d); a row's bits do not depend on m.
const std = @import("std");
const rt = @import("xpu").rt;

const spv = @import("xpu").kernels.qwen_mlx4;

/// Row-major words / scales / biases to the kernel's block-interleaved layout (16-row blocks); rows % 16, in % 256.
pub fn repack(in: u32, rows: u32, w: []const u8, s: []const u8, b: []const u8, wo: []u8, so: []u8, bo: []u8) void {
    const nb = rows / 16;
    const nt: usize = 8;
    var th: [nt]?std.Thread = undefined;
    for (&th, 0..) |*t, i| t.* = std.Thread.spawn(.{}, repackBlocks, .{ in, w, s, b, wo, so, bo, nb * i / nt, nb * (i + 1) / nt }) catch blk: {
        repackBlocks(in, w, s, b, wo, so, bo, nb * i / nt, nb * (i + 1) / nt);
        break :blk null;
    };
    for (th) |t| if (t) |x| x.join();
}

fn repackBlocks(in: u32, w: []const u8, s: []const u8, b: []const u8, wo: []u8, so: []u8, bo: []u8, rb0: usize, rb1: usize) void {
    const groups = in / 64;
    const words = in / 8;
    const ws = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    const wd = std.mem.bytesAsSlice(u32, @as([]align(1) u8, wo));
    const ss = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, s));
    const sd = std.mem.bytesAsSlice(u16, @as([]align(1) u8, so));
    const bs = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, b));
    const bd = std.mem.bytesAsSlice(u16, @as([]align(1) u8, bo));
    for (rb0..rb1) |rb| for (0..groups) |g| for (0..16) |lane| {
        const row = rb * 16 + lane;
        const dst = (rb * groups + g) * 16 + lane;
        @memcpy(wd[dst * 8 ..][0..8], ws[row * words + g * 8 ..][0..8]);
        sd[dst] = ss[row * groups + g];
        bd[dst] = bs[row * groups + g];
    };
}

/// K splits: about 850 sub-groups, each a multiple of 4 groups of 64 inputs (shape only: one reduction for every m).
pub fn splitsFor(in: u32, rows: u32) u32 {
    const groups = in / 64;
    var best: u32 = 1;
    var score: f64 = 1e9;
    const target: f64 = if (std.c.getenv("MLX4_TARGET")) |v| @floatFromInt(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 850) else 850.0;
    var s: u32 = 1;
    while (s <= 16) : (s += 1) {
        if (groups % s != 0 or (groups / s) % 4 != 0) continue;
        const sc = @abs(@log(@as(f64, @floatFromInt(rows / 16 * s)) / target));
        if (sc < score) {
            score = sc;
            best = s;
        }
    }
    return best;
}

pub const Block = struct {
    r: *rt.Runtime,
    m: rt.Module,
    kn1: [4]rt.Kernel, // the NB = 1 forms for m <= 8 (rows not a multiple of 32)
    kd: [5]rt.Kernel, // qmv4d_M_G_NB: 1_1, 2_1, 4_1, 8_1, 8_2 with NB blocks (the NB = 2 forms for m <= 8)
    xprep: rt.Kernel,
    finish: rt.Kernel,
    finish_gu: rt.Kernel, // merge of the gate and up partials + SwiGLU
    nb: u32, // 16-row blocks a sub-group for m <= 8
    xt: rt.Buffer, // [16 rows][max_in] bf16
    z: rt.Buffer, // [S <= 16][16][max_rows] fp32

    pub fn init(r: *rt.Runtime, max_in: u32, max_rows: u32) !Block {
        var m = try r.module(spv);
        const nb: u32 = if (std.c.getenv("MLX4_NB")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2) else 2;
        var kd: [5]rt.Kernel = undefined;
        var nm: [32]u8 = undefined;
        const ms = [_]u32{ 1, 2, 4, 8 };
        for (ms, 0..) |mw, i| kd[i] = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4d_{d}_1_{d}", .{ mw, nb }, 0), .{ 16, 1, 1 });
        kd[4] = try m.kernel("qmv4d_8_2_1", .{ 16, 1, 1 });
        var kn1: [4]rt.Kernel = undefined;
        for (ms, 0..) |mw, i| kn1[i] = try m.kernel(try std.fmt.bufPrintSentinel(&nm, "qmv4d_{d}_1_1", .{mw}, 0), .{ 16, 1, 1 });
        return .{
            .r = r,
            .m = m,
            .kd = kd,
            .kn1 = kn1,
            .nb = nb,
            .xprep = try m.kernel("qmv4_xprep", .{ 16, 1, 1 }),
            .finish = try m.kernel("qmv4_finish", .{ 16, 1, 1 }),
            .finish_gu = try m.kernel("qmv4_finish_gu", .{ 16, 1, 1 }),
            .xt = try r.alloc(@as(usize, 16) * max_in * 2),
            .z = try r.alloc(@as(usize, 2) * 16 * 16 * max_rows * 4),
        };
    }

    pub fn deinit(self: *Block) void {
        self.xt.free();
        self.z.free();
    }

    /// y[r * rows + y_off + row] (bf16, or fp32 when f32out) for r < m <= 16 from the repacked weights (in % 256).
    pub fn matvec(self: *Block, w: rt.Buffer, s: rt.Buffer, b: rt.Buffer, x: rt.Buffer, y: rt.Buffer, in: u32, rows: u32, m: u32, y_off: u32, f32out: bool) !void {
        return self.matvecP(w, s, b, x, y, in, rows, m, y_off, f32out, true);
    }

    /// `matvec`; prep = false reuses the activation prepass (xt) of the previous call (same x, in, m).
    pub fn matvecP(self: *Block, w: rt.Buffer, s: rt.Buffer, b: rt.Buffer, x: rt.Buffer, y: rt.Buffer, in: u32, rows: u32, m: u32, y_off: u32, f32out: bool, prep: bool) !void {
        _ = try self.launchK(w, s, b, x, y, self.z, in, rows, m, y_off, f32out, prep, true);
    }

    /// act [m][rows] = bf16(silu(gate x) * up x) in three launches; false (nothing launched) when the split count is 1.
    pub fn gateUp(self: *Block, wg: rt.Buffer, sg: rt.Buffer, bg: rt.Buffer, wu: rt.Buffer, su: rt.Buffer, bu: rt.Buffer, x: rt.Buffer, act: rt.Buffer, in: u32, rows: u32, m: u32, prep: bool) !bool {
        if (splitsFor(in, rows / self.nb) == 1) return false;
        const half: usize = self.z.len / 2;
        const zu: rt.Buffer = .{ .rt = self.z.rt, .ptr = @ptrFromInt(@intFromPtr(self.z.ptr.?) + half), .len = half };
        const zg: rt.Buffer = .{ .rt = self.z.rt, .ptr = self.z.ptr, .len = half };
        const sp = try self.launchK(wg, sg, bg, x, act, zg, in, rows, m, 0, false, prep, false);
        _ = try self.launchK(wu, su, bu, x, act, zu, in, rows, m, 0, false, false, false);
        var kf = &self.finish_gu;
        try kf.setBuffer(0, zg);
        try kf.setBuffer(1, zu);
        try kf.setBuffer(2, act);
        try kf.setU32(3, rows);
        try kf.setU32(4, sp);
        try kf.launch(.{ (rows + 15) / 16, m, 1 });
        return true;
    }

    fn launchK(self: *Block, w: rt.Buffer, s: rt.Buffer, b: rt.Buffer, x: rt.Buffer, y: rt.Buffer, zb: rt.Buffer, in: u32, rows: u32, m: u32, y_off: u32, f32out: bool, prep: bool, fin: bool) !u32 {
        if (m == 0 or m > 16 or rows % 16 != 0 or in % 256 != 0) return error.Invalid;
        const idx: usize = if (m == 1) 0 else if (m == 2) 1 else if (m <= 4) 2 else if (m <= 8) 3 else 4;
        const mw: u32 = if (idx == 0) 1 else if (idx == 1) 2 else if (idx == 2) 4 else 8;
        var nbk: u32 = if (idx == 4) 1 else self.nb;
        const one = rows % (16 * nbk) != 0;
        if (one) nbk = 1;
        const sp = splitsFor(in, rows / self.nb); // one split count for every m (the sums' grouping is part of a row's bits)
        if (sp > 1 and @as(usize, rows) * 16 * 4 * sp > zb.len) return error.Invalid;
        if (prep) {
            var kx = &self.xprep;
            try kx.setBuffer(0, x);
            try kx.setBuffer(1, self.xt);
            try kx.setU32(2, in);
            try kx.setU32(3, m);
            try kx.setU32(4, mw);
            try kx.launch(.{ in / 16, (m + mw - 1) / mw * mw, 1 });
        }
        var kd = if (one) &self.kn1[idx] else &self.kd[idx];
        try kd.setBuffer(0, w);
        try kd.setBuffer(1, s);
        try kd.setBuffer(2, b);
        try kd.setBuffer(3, self.xt);
        try kd.setBuffer(4, y);
        try kd.setBuffer(5, zb);
        try kd.setU32(6, in);
        try kd.setU32(7, y_off);
        try kd.setU32(8, rows);
        try kd.setU32(9, rows);
        try kd.setU32(10, m);
        try kd.setU32(11, @intFromBool(f32out));
        try kd.setU32(12, sp);
        try kd.setU32(13, 0x3F803F80);
        try kd.launch(.{ rows / (16 * nbk), sp, 1 });
        if (sp > 1 and fin) {
            var kf = &self.finish;
            try kf.setBuffer(0, zb);
            try kf.setBuffer(1, y);
            try kf.setU32(2, rows);
            try kf.setU32(3, sp);
            try kf.setU32(4, y_off);
            try kf.setU32(5, rows);
            try kf.setU32(6, @intFromBool(f32out));
            try kf.launch(.{ (rows + 15) / 16, m, 1 });
        }
        return sp;
    }
};

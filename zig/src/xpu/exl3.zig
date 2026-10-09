//! EXL3 (trellis) quantized linear layers on the Intel GPU (kernels/xpu/exl3.cl); a row's bits depend on that row only.

const std = @import("std");
const rt = @import("rt.zig");

/// Tile columns a sub-group of exl3_gemm computes (GEMM_GN in exl3.cl).
pub const gemm_gn = 2;

/// Fewest rows for which forwardPrefill takes the 2D-block-load GEMM (EXL3_PF2D_MIN overrides).
fn pf2dMin() u32 {
    return if (std.c.getenv("EXL3_PF2D_MIN")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 256) else 256;
}

pub const Dtype = enum(u32) { f16 = 0, bf16 = 1, f32 = 2 };
pub const Codebook = enum(u32) { inst3 = 0, mcg = 1, mul1 = 2 };

const gpa = std.heap.page_allocator;
/// Profile of forwardPrefill2d (exl3_test): every stage synced and timed; ns of rotation, decode, GEMM, finish.
pub var prof_on = false;
pub var prof_ns: [4]u64 = .{ 0, 0, 0, 0 };
fn profNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
fn profStage(r: *rt.Runtime, i: usize, t0: *u64) void {
    if (!prof_on) return;
    r.sync() catch {};
    const t = profNow();
    prof_ns[i] += t - t0.*;
    t0.* = t;
}
const spv_pf2d = @import("kernels.zig").exl3_pf2d;

fn setLocal(k: *rt.Kernel, index: u32, bytes: usize) !void {
    try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, index, bytes, null), "setLocal");
}

pub fn at(b: rt.Buffer, off: usize) rt.Buffer {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

fn setU64(k: *rt.Kernel, index: u32, v: u64) !void {
    var x = v;
    try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, index, 8, @ptrCast(&x)), "setU64");
}

/// K splits for a K x N layer: sub-groups per 128-column block, each summing a contiguous k range; depends on (K, N).
pub fn splitFor(k: u32, n: u32) u32 {
    const kt = k / 16;
    const nb: f64 = @floatFromInt(n / 128);
    var best: u32 = 1;
    var best_score: f64 = 1e9;
    var d: u32 = 1;
    while (d <= 64) : (d += 1) {
        if (kt % d != 0 or kt / d < 4) continue;
        const score = @abs(@log(nb * @as(f64, @floatFromInt(d)) / 650.0));
        if (score < best_score) {
            best_score = score;
            best = d;
        }
    }
    return best;
}

/// Floats of split-K partials per row (splits * N): Scratch needs max_rows times the largest of these.
pub fn zn(k: u32, n: u32) u64 {
    return @as(u64, splitFor(k, n)) * n;
}

/// Kernels shared by every layer: input rotation, split-K finish, and per-(width, codebook) decode and matmul kernels.
pub const Engine = struct {
    r: *rt.Runtime,
    m: *rt.Module,
    rot: rt.Kernel,
    fin: rt.Kernel,
    gemm: rt.Kernel,
    pfgemm: rt.Kernel,
    rot_k: rt.Kernel,
    pf2d_module: rt.Module,
    pf2d: rt.Kernel,
    sets: [17][3]?Set,

    pub const Set = struct { decw: ?rt.Kernel = null, mx1: rt.Kernel, mx2: rt.Kernel, dec: rt.Kernel, decv: rt.Kernel, decr: rt.Kernel };

    pub fn init(r: *rt.Runtime, m: *rt.Module) !Engine {
        var e: Engine = .{ .r = r, .m = m, .rot = try m.kernel("exl3_rot_in_t", .{ 16, 1, 1 }), .fin = try m.kernel("exl3_finish", .{ 16, 1, 1 }), .gemm = try m.kernel("exl3_gemm", .{ 16, 1, 1 }), .pfgemm = try m.kernel("exl3_pfgemm", .{ 16, 1, 1 }), .rot_k = try m.kernel("exl3_rot_in_k", .{ 512, 1, 1 }), .pf2d_module = try r.moduleWith(spv_pf2d, "-cl-intel-256-GRF-per-thread"), .pf2d = undefined, .sets = undefined };
        e.pf2d = try e.pf2d_module.kernel("exl3_pfgemm2d", .{ 16, 1, 1 }); // 128 accumulator registers a lane: the large-GRF build
        for (&e.sets) |*row| for (row) |*s| {
            s.* = null;
        };
        return e;
    }

    pub fn deinit(self: *Engine) void {
        self.rot.deinit();
        self.fin.deinit();
        self.gemm.deinit();
        self.pfgemm.deinit();
        self.rot_k.deinit();
        self.pf2d.deinit();
        self.pf2d_module.deinit();
        for (&self.sets) |*row| for (row) |*sl| if (sl.*) |*ks| {
            if (ks.decw) |*g| g.deinit();
            ks.mx1.deinit();
            ks.mx2.deinit();
            ks.dec.deinit();
            ks.decv.deinit();
            ks.decr.deinit();
        };
    }

    pub fn set(self: *Engine, k2: u32, cb: Codebook) !*Set {
        if (k2 > 16 or k2 < 2) return error.Invalid;
        const slot = &self.sets[k2][@intFromEnum(cb)];
        if (slot.* == null) {
            var nm: [48]u8 = undefined;
            const c = @intFromEnum(cb);
            slot.* = .{
                .decw = if (k2 == 6 and cb == .mul1) (self.m.kernel("exl3_decw_6_2", .{ 128, 1, 1 }) catch null) else null,
                .mx1 = try self.m.kernel(try std.fmt.bufPrintSentinel(&nm, "exl3_mx_{d}_{d}_1", .{ k2, c }, 0), .{ 16, 1, 1 }),
                .mx2 = try self.m.kernel(try std.fmt.bufPrintSentinel(&nm, "exl3_mx_{d}_{d}_2", .{ k2, c }, 0), .{ 16, 1, 1 }),
                .dec = try self.m.kernel(try std.fmt.bufPrintSentinel(&nm, "exl3_dec_{d}_{d}", .{ k2, c }, 0), .{ 16, 1, 1 }),
                .decv = try self.m.kernel(try std.fmt.bufPrintSentinel(&nm, "exl3_decv_{d}_{d}", .{ k2, c }, 0), .{ 16, 1, 1 }),
                .decr = try self.m.kernel(try std.fmt.bufPrintSentinel(&nm, "exl3_decr_{d}_{d}", .{ k2, c }, 0), .{ 16, 1, 1 }),
            };
        }
        return &slot.*.?;
    }
};

/// Work buffers for the forward pass (one call at a time): the rotated input and the split-K partial sums.
pub const Scratch = struct {
    xt: rt.Buffer,
    z: rt.Buffer,
    max_rows: u32,

    pub fn init(r: *rt.Runtime, max_rows: u32, max_k: u32, max_zn: u64) !Scratch {
        return .{ .xt = try r.alloc(@as(usize, (max_rows + 7) / 8 * 8) * max_k * 2), .z = try r.alloc(@max(@as(usize, max_rows) * max_zn * 4, 16)), .max_rows = max_rows };
    }

    pub fn deinit(self: *Scratch) void {
        self.xt.free();
        self.z.free();
    }
};

/// Work buffers for prefill-sized row counts: the rotated input, a chunk of decoded fp16 weights and the fp32 product.
pub const Prefill = struct {
    xt: rt.Buffer,
    w: rt.Buffer,
    c: rt.Buffer,
    max_rows: u32,
    chunk: u32, // 16-column tiles decoded at a time (a multiple of 8)

    pub fn init(r: *rt.Runtime, max_rows: u32, max_k: u32, max_n: u32, chunk: u32) !Prefill {
        return .{ .xt = try r.alloc(@as(usize, (max_rows + 63) / 64 * 64) * max_k * 2), .w = try r.alloc(@as(usize, max_k) * chunk * 32), .c = try r.alloc(@as(usize, max_rows) * max_n * 4), .max_rows = max_rows, .chunk = chunk };
    }

    pub fn deinit(self: *Prefill) void {
        self.xt.free();
        self.w.free();
        self.c.free();
    }
};

pub const Desc = struct {
    k: u32,
    n: u32,
    k2: u32, // 2 * bits per weight: 2, 3 (1.5 bits, mul1 only), 4, 5, 6, 7, 8, 10, 12, 14, 16
    cb: Codebook,
    trellis: []const u8, // the checkpoint's int16 [K/16, N/16, 16 * bits] tensor, as stored
    suh: []const u8, // fp16 [K]
    svh: []const u8, // fp16 [N]
    bias: ?[]const u8 = null, // fp16 [N]
};

pub const Layer = struct {
    k: u32,
    n: u32,
    k2: u32,
    cb: Codebook,
    sp: u32, // K splits: sub-groups a column block, each summing a contiguous k range into its own partial
    cols: rt.Buffer, // trellis words, nt-major: [N/128, 8, K/16, 4 * K2]
    suh: rt.Buffer,
    svh: rt.Buffer,
    bias: rt.Buffer,
    has_bias: bool,

    pub fn init(eng: *Engine, d: Desc) !Layer {
        const r = eng.r;
        if (d.k % 128 != 0 or d.n % 128 != 0) return error.Invalid; // the prefill decode takes 4 k tiles a step: K a multiple of 64 (128 holds)
        const tw: usize = 16 * d.k2; // bytes a tile
        if (d.trellis.len != @as(usize, d.k / 16) * (d.n / 16) * tw or d.suh.len != d.k * 2 or d.svh.len != d.n * 2) return error.Invalid;
        // the kernels read 16-byte vectors up to 3 words past the last tile: the device copy has 64 bytes of padding
        const packed_ = try gpa.alloc(u8, d.trellis.len);
        defer gpa.free(packed_);
        const kt = d.k / 16;
        for (0..d.n / 16) |c| for (0..kt) |t| {
            const s = (t * (d.n / 16) + c) * tw;
            @memcpy(packed_[(c * kt + t) * tw ..][0..tw], d.trellis[s..][0..tw]);
        };
        var l: Layer = .{ .k = d.k, .n = d.n, .k2 = d.k2, .cb = d.cb, .sp = splitFor(d.k, d.n), .cols = try r.alloc(packed_.len + 64), .suh = try r.alloc(d.suh.len), .svh = try r.alloc(d.svh.len), .bias = undefined, .has_bias = d.bias != null };
        l.bias = if (d.bias) |b| try r.alloc(b.len) else l.svh;
        try r.upload(l.cols, packed_);
        try r.upload(l.suh, d.suh);
        try r.upload(l.svh, d.svh);
        if (d.bias) |b| try r.upload(l.bias, b);
        try r.sync(); // the copies read the caller's and our temporary memory until they complete
        _ = try eng.set(d.k2, d.cb);
        return l;
    }

    pub fn deinit(self: *Layer) void {
        self.cols.free();
        self.suh.free();
        self.svh.free();
        if (self.has_bias) self.bias.free();
    }

    pub fn bytes(self: Layer) usize {
        return self.cols.len - 64;
    }

    /// y [rows, N] = x [rows, K] @ W + bias; any rows <= scratch.max_rows (the kernel does up to 16 rows a pass).
    pub fn forward(self: *Layer, eng: *Engine, s: *Scratch, x: rt.Buffer, xdt: Dtype, rows: u32, y: rt.Buffer, ydt: Dtype) !void {
        try self.rotIn(eng, s, x, xdt, rows);
        try self.prepare(eng, s, rows, y, ydt);
        try self.launch(eng, rows);
    }

    pub fn rotIn(self: *Layer, eng: *Engine, s: *Scratch, x: rt.Buffer, xdt: Dtype, rows: u32) !void {
        if (rows == 0 or rows > s.max_rows) return error.Invalid;
        try self.rotInto(eng, s.xt, x, xdt, rows);
    }

    fn rotInto(self: *Layer, eng: *Engine, xt: rt.Buffer, x: rt.Buffer, xdt: Dtype, rows: u32) !void {
        return self.rotIntoPad(eng, xt, x, xdt, rows, 8);
    }

    /// The rotated input with the row count rounded up to `pad` (zero rows).
    fn rotIntoPad(self: *Layer, eng: *Engine, xt: rt.Buffer, x: rt.Buffer, xdt: Dtype, rows: u32, pad: u32) !void {
        var k = &eng.rot;
        try k.setBuffer(0, x);
        try k.setU32(1, @intFromEnum(xdt));
        try k.setBuffer(2, self.suh);
        try k.setBuffer(3, xt);
        try k.setU32(4, self.k);
        try k.setU32(5, rows);
        try k.launch(.{ self.k / 128, (rows + pad - 1) / pad * pad, 1 });
    }

    /// The matmul kernel for `rows` rows (8 rows a DPAS group; passes of 8 or 16 rows each decode W once).
    pub fn mxKernel(self: *Layer, eng: *Engine, rows: u32) !*rt.Kernel {
        const set = try eng.set(self.k2, self.cb);
        return if (rows > 8) &set.mx2 else &set.mx1;
    }

    /// Sets the matmul and finish arguments (kept until the next layer's prepare on the same kernels).
    pub fn prepare(self: *Layer, eng: *Engine, s: *Scratch, rows: u32, y: rt.Buffer, ydt: Dtype) !void {
        const sp = self.sp;
        if (@as(usize, sp) * rows * self.n * 4 > s.z.len) return error.Invalid;
        var kk = try self.mxKernel(eng, rows);
        const kt: u64 = self.k / 16;
        const tw: u64 = 4 * self.k2;
        try kk.setBuffer(0, s.xt);
        try kk.setBuffer(1, self.cols);
        try setU64(kk, 2, tw);
        try setU64(kk, 3, kt * tw);
        try setU64(kk, 4, 8 * kt * tw);
        try kk.setBuffer(5, s.z);
        try kk.setU32(6, rows);
        try kk.setU32(7, self.k);
        try kk.setU32(8, self.n);
        try kk.setU32(9, sp);
        {
            var f = &eng.fin;
            try f.setBuffer(0, s.z);
            try f.setBuffer(1, self.svh);
            try f.setBuffer(2, self.bias);
            try f.setU32(3, @intFromBool(self.has_bias));
            try f.setBuffer(4, y);
            try f.setU32(5, @intFromEnum(ydt));
            try f.setU32(6, rows);
            try f.setU32(7, self.n);
            try f.setU32(8, sp);
        }
    }

    pub fn launch(self: *Layer, eng: *Engine, rows: u32) !void {
        var kk = try self.mxKernel(eng, rows);
        try kk.launch(.{ self.n / 128, self.sp, 2 });
        try eng.fin.launch(.{ self.n / 128, rows, 1 });
    }

    /// W_q [K, N] row-major (the rotated-domain weight) as fp16 or bf16, for prefill-style GEMMs or checks.
    pub fn decode(self: *Layer, eng: *Engine, w: rt.Buffer, odt: Dtype) !void {
        var kk = &(try eng.set(self.k2, self.cb)).dec;
        const kt: u64 = self.k / 16;
        const tw: u64 = 4 * self.k2;
        try kk.setBuffer(0, self.cols);
        try kk.setBuffer(1, w);
        try kk.setU32(2, self.n);
        try setU64(kk, 3, tw);
        try setU64(kk, 4, kt * tw);
        try setU64(kk, 5, 8 * kt * tw);
        try kk.setU32(6, @intFromBool(odt == .bf16));
        try kk.launch(.{ self.n / 16, self.k / 16, 1 });
    }

    /// Prompt chunks: decode W_q once per column chunk, one GEMM; bits differ from forward() (other summation order).
    pub fn forwardPrefill(self: *Layer, eng: *Engine, p: *Prefill, x: rt.Buffer, xdt: Dtype, rows: u32, y: rt.Buffer, ydt: Dtype) !void {
        if (rows == 0 or rows > p.max_rows) return error.Invalid;
        if (self.n % 64 == 0 and rows >= pf2dMin() and std.c.getenv("EXL3_OLDGEMM") == null and std.c.getenv("EXL3_FRAGGEMM") == null) return self.forwardPrefill2d(eng, p, x, xdt, rows, y, ydt);
        const wide = self.n % 64 == 0 and std.c.getenv("EXL3_OLDGEMM") == null; // 16 x 64 tiles, row panels fastest (ntc is a multiple of 4: chunks are multiples of 8 tiles)
        try self.rotIntoPad(eng, p.xt, x, xdt, rows, if (wide) 64 else 8);
        const nt_all = self.n / 16;
        var nt_first: u32 = 0;
        while (nt_first < nt_all) {
            const ntc = @min(p.chunk, nt_all - nt_first);
            try self.decodeFragments(eng, p.w, nt_first, ntc);
            var g = if (wide) &eng.pfgemm else &eng.gemm;
            try g.setBuffer(0, p.xt);
            try g.setBuffer(1, p.w);
            try g.setBuffer(2, p.c);
            try g.setU32(3, rows);
            try g.setU32(4, self.k);
            try g.setU32(5, self.n);
            try g.setU32(6, ntc);
            try g.setU32(7, nt_first);
            if (wide) try g.launch(.{ (rows + 15) / 16, ntc / 4, 1 }) else try g.launch(.{ ntc / gemm_gn, (rows + 15) / 16, 1 });
            nt_first += ntc;
        }
        var f = &eng.fin;
        try f.setBuffer(0, p.c);
        try f.setBuffer(1, self.svh);
        try f.setBuffer(2, self.bias);
        try f.setU32(3, @intFromBool(self.has_bias));
        try f.setBuffer(4, y);
        try f.setU32(5, @intFromEnum(ydt));
        try f.setU32(6, rows);
        try f.setU32(7, self.n);
        try f.setU32(8, 1);
        try f.launch(.{ self.n / 128, rows, 1 });
    }

    /// forwardPrefill on 2D block loads (32 columns x 64 tokens a sub-group, ascending-k DPAS chain); n % 64 == 0.
    fn forwardPrefill2d(self: *Layer, eng: *Engine, p: *Prefill, x: rt.Buffer, xdt: Dtype, rows: u32, y: rt.Buffer, ydt: Dtype) !void {
        const rp = (rows + 63) / 64 * 64;
        var pt: u64 = 0;
        if (prof_on) {
            try eng.r.sync();
            pt = profNow();
        }
        var rk = &eng.rot_k;
        try rk.setBuffer(0, x);
        try rk.setU32(1, @intFromEnum(xdt));
        try rk.setBuffer(2, self.suh);
        try rk.setBuffer(3, p.xt);
        try rk.setU32(4, self.k);
        try rk.setU32(5, rows);
        try rk.setU32(6, rp);
        try rk.launch(.{ self.k / 128, rp / 32, 1 });
        profStage(eng.r, 0, &pt);
        const nt_all = self.n / 16;
        var nt_first: u32 = 0;
        while (nt_first < nt_all) {
            const ntc = @min(p.chunk, nt_all - nt_first);
            var d = &(try eng.set(self.k2, self.cb)).decr;
            const kt: u64 = self.k / 16;
            const tw: u64 = 4 * self.k2;
            try d.setBuffer(0, self.cols);
            try d.setBuffer(1, p.w);
            try setU64(d, 2, tw);
            try setU64(d, 3, kt * tw);
            try setU64(d, 4, 8 * kt * tw);
            try d.setU32(5, ntc);
            try d.setU32(6, nt_first);
            try d.setU32(7, self.k);
            if (self.k % 512 == 0 and (try eng.set(self.k2, self.cb)).decw != null) {
                d = &((try eng.set(self.k2, self.cb)).decw.?);
                try d.setBuffer(0, self.cols);
                try d.setBuffer(1, p.w);
                try setU64(d, 2, tw);
                try setU64(d, 3, kt * tw);
                try setU64(d, 4, 8 * kt * tw);
                try d.setU32(5, ntc);
                try d.setU32(6, nt_first);
                try d.setU32(7, self.k);
                try d.launch(.{ ntc, self.k / 512, 1 });
            } else try d.launch(.{ ntc, self.k / 64, 1 });
            profStage(eng.r, 1, &pt);
            var g = &eng.pf2d;
            try g.setBuffer(0, p.w);
            try g.setBuffer(1, p.xt);
            try g.setBuffer(2, p.c);
            try g.setU32(3, rows);
            try g.setU32(4, rp);
            try g.setU32(5, self.k);
            try g.setU32(6, ntc * 16);
            try g.setU32(7, self.n);
            try g.setU32(8, nt_first * 16);
            try g.launch(.{ rp / 64, ntc / 2, 1 });
            profStage(eng.r, 2, &pt);
            nt_first += ntc;
        }
        var f = &eng.fin;
        try f.setBuffer(0, p.c);
        try f.setBuffer(1, self.svh);
        try f.setBuffer(2, self.bias);
        try f.setU32(3, @intFromBool(self.has_bias));
        try f.setBuffer(4, y);
        try f.setU32(5, @intFromEnum(ydt));
        try f.setU32(6, rows);
        try f.setU32(7, self.n);
        try f.setU32(8, 1);
        try f.launch(.{ self.n / 128, rows, 1 });
        profStage(eng.r, 3, &pt);
    }

    /// W_q of tile columns nt_first .. nt_first + ntc as fp16 DPAS B fragments: wv [K/16][ntc][8][16] ints.
    pub fn decodeFragments(self: *Layer, eng: *Engine, wv: rt.Buffer, nt_first: u32, ntc: u32) !void {
        var d = &(try eng.set(self.k2, self.cb)).decv;
        const kt: u64 = self.k / 16;
        const tw: u64 = 4 * self.k2;
        try d.setBuffer(0, self.cols);
        try d.setBuffer(1, wv);
        try setU64(d, 2, tw);
        try setU64(d, 3, kt * tw);
        try setU64(d, 4, 8 * kt * tw);
        try d.setU32(5, ntc);
        try d.setU32(6, nt_first);
        try d.launch(.{ ntc, self.k / 64, 1 });
    }
};

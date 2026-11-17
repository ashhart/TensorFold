//! ZONOS3's windowed sums on Metal: the codec's five, resampler, teacher's conv (contract_zonos3_windows.zig's twin).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");
const conv = @import("metal_zonos3_conv.zig");

const c = contract.spec;
const g = contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

fn kDeep(s: c.Shape) usize {
    return s.k;
}

pub const Depthwise = enum { codec, teacher };

/// Rows x channels x taps: the codec's (7 taps, a bias, history in place), the teacher's (31, zeros before row 0).
pub fn depthwiseSpec(which: Depthwise, fault: bool) c.Spec {
    return switch (which) {
        .codec => .{ .name = if (fault) "control: metal zonos3 zc_dwconv launched one row short" else "metal zonos3 zc_dwconv (history in place)", .accum = .f32, .depth = kDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 16, .n = 768, .k = 8 }, .{ .rows = 1, .n = 768, .k = 8 }, .{ .rows = 300, .n = 768, .k = 8 } } },
        .teacher => .{ .name = if (fault) "control: metal zonos3 zt_dwconv launched one tap short" else "metal zonos3 zt_dwconv", .accum = .f32, .depth = kDeep, .out = .f32, .served = &.{ .{ .rows = 64, .n = 1024, .k = 31 }, .{ .rows = 1, .n = 1024, .k = 31 }, .{ .rows = 999, .n = 1024, .k = 31 } } },
    };
}

/// A channel's causal convolution over the rows before and at each row, an fma chain from the oldest tap.
pub const DwConv = struct {
    pipe: mtl.Pipeline,
    which: Depthwise,
    /// control: a row (codec) or a tap (teacher) fewer than the reference
    fault: bool = false,
    T: usize = 0,
    C: usize = 0,
    /// the rows before row 0 ([taps - 1][C]: the codec's history, the teacher's zeros), then the rows
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    dev: [3]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub const sums = {};

    fn taps(self: *const DwConv) usize {
        return if (self.which == .codec) 7 else 31;
    }

    pub fn spec(self: *const DwConv) c.Spec {
        return depthwiseSpec(self.which, self.fault);
    }

    pub fn control(self: *const DwConv) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *DwConv, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        a.free(self.b);
        self.x = &.{};
        self.w = &.{};
        self.b = &.{};
    }

    pub fn prepare(self: *DwConv, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const K = self.taps();
        if (shape.rows == 0 or shape.n == 0 or shape.k != (if (self.which == .codec) K + 1 else K)) return false;
        self.free(a);
        const T = shape.rows;
        const C = shape.n;
        self.T = T;
        self.C = C;
        const past = K - 1;
        self.x = try ref.values(a, rng, (past + T) * C, .random, 2);
        self.w = try ref.values(a, rng, C * K, .random, 1);
        self.b = try ref.values(a, rng, C, .random, 1);
        if (self.which == .teacher) @memset(self.x[0 .. past * C], 0);
        if (fill == .stress) {
            @memset(self.w, 1);
            @memset(self.b, 0);
            const from: usize = if (self.which == .codec) 0 else past;
            for (0..C) |ch| ref.stressAxis(self.x, from * C + ch, C, past + T - from, rng);
        }
        if (fill == .nonfinite) ref.plant(self.x[past * C ..], T, C);
        self.dev = .{ try ref.input(rig, self.x[past * C ..]), try ref.input(rig, self.w), try ref.input(rig, self.b) };
        self.out[0] = try rig.output(T * C * 4);
        if (self.which == .codec) self.out[1] = try rig.output(C * 6 * 4);
        return true;
    }

    pub fn run(self: *DwConv, rig: *ref.Rig) !void {
        const T = self.T;
        const C = self.C;
        const less = ref.one(self.fault);
        if (self.which == .codec) {
            // hist [C][6]: a channel's six inputs before row 0, oldest first
            const hist = try rig.gpa().alloc(f32, C * 6);
            defer rig.gpa().free(hist);
            for (hist, 0..) |*v, i| v.* = self.x[(i % 6) * C + i / 6];
            try ref.restore(rig, self.out[1], std.mem.sliceAsBytes(hist));
            const A = extern struct { T: i32, C: i32 };
            try rig.go(self.pipe, .{ (C + 255) / 256, 1, 1 }, 256, &.{ self.dev[0], self.dev[1], self.dev[2], self.out[1], self.out[0] }, A{ .T = ref.int(T - less), .C = ref.int(C) }, 0);
        } else {
            const A = extern struct { K: i32, C: i32 };
            try rig.go(self.pipe, .{ (C + 255) / 256, T, 1 }, 256, &.{ self.dev[0], self.dev[1], self.out[0] }, A{ .K = ref.int(31 - less), .C = ref.int(C) }, 0);
        }
    }

    pub fn count(self: *const DwConv) usize {
        return self.T * self.C + (if (self.which == .codec) self.C * 6 else 0);
    }

    pub fn solve(self: *const DwConv, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const T = self.T;
        const C = self.C;
        const K = self.taps();
        var scratch: [32]f64 = undefined;
        var terms: [32]Iv = undefined;
        for (0..C) |ch| {
            for (0..T) |t| {
                var n: usize = 0;
                if (self.which == .codec) {
                    terms[0] = Iv.of(self.b[ch]);
                    n = 1;
                }
                for (0..K) |j| {
                    // the teacher's kernel skips the rows before row 0
                    if (self.which == .teacher and t + j < K - 1) continue;
                    terms[n] = Iv.of(self.w[ch * K + j]).mul(Iv.of(self.x[(t + j) * C + ch]));
                    n += 1;
                }
                out[t * C + ch] = s.sum(terms[0..n], n, &scratch);
            }
            if (self.which == .codec) for (0..6) |j| {
                out[T * C + ch * 6 + j] = Iv.of(self.x[(T + j) * C + ch]);
            };
        }
    }
};

const book_rows = 4096;
const used_rows = 16;
const residuals = 9;

/// Frames x the latent's width x its ten terms.
pub fn latentSpec(fault: bool) c.Spec {
    return .{ .name = if (fault) "control: metal zonos3 zc_latent launched one row short" else "metal zonos3 zc_latent (in place)", .accum = .f32, .depth = kDeep, .out = .f32, .in_place = true, .served = &.{ .{ .rows = 16, .n = 1024, .k = 10 }, .{ .rows = 1, .n = 1024, .k = 10 }, .{ .rows = 300, .n = 1024, .k = 10 } } };
}

/// z plus the nine residual codebooks' rows of a frame's codes, in codebook order.
pub const Latent = struct {
    pipe: mtl.Pipeline,
    /// control: the launch covers rows - 1 rows
    fault: bool = false,
    shape: c.Shape = undefined,
    z: []f32 = &.{},
    /// [9][4096][n] as the kernel strides it; the codes stay below `used_rows`
    tables: []f32 = &.{},
    ids: []i32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Latent) c.Spec {
        return latentSpec(self.fault);
    }

    pub fn control(self: *const Latent) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Latent, a: std.mem.Allocator) void {
        a.free(self.z);
        a.free(self.tables);
        a.free(self.ids);
        self.z = &.{};
        self.tables = &.{};
        self.ids = &.{};
    }

    fn row(self: *const Latent, book: usize, code: usize) []f32 {
        return self.tables[(book * book_rows + code) * self.shape.n ..][0..self.shape.n];
    }

    pub fn prepare(self: *Latent, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != residuals + 1) return false;
        self.free(a);
        self.shape = shape;
        const T = shape.rows;
        const n = shape.n;
        self.z = try ref.values(a, rng, T * n, .random, 2);
        self.tables = try a.alloc(f32, ((residuals - 1) * book_rows + used_rows) * n);
        @memset(self.tables, 0);
        for (0..residuals) |book| for (0..used_rows) |code| for (self.row(book, code)) |*v| {
            v.* = rng.float(f32) * 2 - 1;
        };
        self.ids = try a.alloc(i32, T * 10);
        for (self.ids, 0..) |*v, i| v.* = if (fill == .stress) @intCast(i % 10) else @intCast(rng.uintLessThan(u32, used_rows));
        if (fill == .stress) for (0..n) |d| {
            var line: [residuals + 1]f32 = undefined;
            ref.stress(&line, rng);
            for (0..T) |t| self.z[t * n + d] = line[0];
            for (0..residuals) |book| self.row(book, book + 1)[d] = line[1 + book];
        };
        if (fill == .nonfinite) ref.plant(self.z, T, n);
        self.dev = .{ try ref.input(rig, self.ids), try ref.input(rig, self.tables) };
        self.out = try rig.output(T * n * 4);
        return true;
    }

    pub fn run(self: *Latent, rig: *ref.Rig) !void {
        const n = self.shape.n;
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.z));
        const A = extern struct { D: i32 };
        try rig.go(self.pipe, .{ (n + 255) / 256, self.shape.rows - ref.one(self.fault), 1 }, 256, &.{ self.dev[0], self.dev[1], self.out }, A{ .D = ref.int(n) }, 0);
    }

    pub fn count(self: *const Latent) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Latent, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        var scratch: [residuals + 1]f64 = undefined;
        for (out, 0..) |*o, at| {
            var terms: [residuals + 1]Iv = undefined;
            terms[0] = Iv.of(self.z[at]);
            for (0..residuals) |book| terms[1 + book] = Iv.of(self.row(book, @intCast(self.ids[at / n * 10 + 1 + book]))[at % n]);
            o.* = s.sum(&terms, residuals + 1, &scratch);
        }
    }
};

const hop = 184;

/// Head frames x the branch's FFT size x the frames a sample overlaps (736 / 184 and 2,208 / 184).
pub fn olaSpec(fault: bool) c.Spec {
    return .{ .name = if (fault) "control: metal zonos3 zc_ola with its hop one sample long" else "metal zonos3 zc_ola", .accum = .f32, .depth = kDeep, .out = .f32, .served = &.{ .{ .rows = 16, .n = 736, .k = 4 }, .{ .rows = 16, .n = 2208, .k = 12 }, .{ .rows = 2, .n = 736, .k = 4 }, .{ .rows = 64, .n = 2208, .k = 12 } } };
}

/// A sample's overlapping windowed frames summed in frame order, over the same frames' squared windows.
pub const Ola = struct {
    pipe: mtl.Pipeline,
    /// control: the device's hop is 185
    fault: bool = false,
    F: usize = 0,
    N: usize = 0,
    n0: usize = 0,
    samples: usize = 0,
    frames: []f32 = &.{},
    win2: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Ola) c.Spec {
        return olaSpec(self.fault);
    }

    pub fn control(self: *const Ola) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Ola, a: std.mem.Allocator) void {
        a.free(self.frames);
        a.free(self.win2);
        self.frames = &.{};
        self.win2 = &.{};
    }

    /// The frames whose window covers centered sample `i` of the launch: first and one past the last.
    fn span(self: *const Ola, i: usize) [2]usize {
        const p = self.n0 + i + self.N / 2;
        return .{ if (p < self.N) 0 else (p - self.N) / hop + 1, @min(p / hop, self.F - 1) + 1 };
    }

    pub fn prepare(self: *Ola, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n < 2 * hop or shape.n % hop != 0 or shape.k != shape.n / hop) return false;
        self.free(a);
        self.F = shape.rows;
        self.N = shape.n;
        const all = if (self.F > 1) (self.F - 1) * hop else self.N / 2;
        self.n0 = if (fill == .stress) 0 else rng.uintLessThan(usize, all / 2);
        self.samples = all - self.n0;
        self.frames = try ref.values(a, rng, self.F * self.N, .random, 2);
        self.win2 = try a.alloc(f32, self.N);
        for (self.win2) |*v| v.* = if (fill == .stress) 1 else 0.1 + rng.float(f32);
        if (fill == .stress) for (0..self.samples) |i| {
            const fs = self.span(i);
            var line: [16]f32 = undefined;
            ref.stress(line[0 .. fs[1] - fs[0]], rng);
            for (fs[0]..fs[1]) |fr| self.frames[fr * self.N + (i + self.N / 2 - fr * hop)] = line[fr - fs[0]];
        };
        if (fill == .nonfinite) ref.plant(self.frames, self.F, self.N);
        self.dev = .{ try ref.input(rig, self.frames), try ref.input(rig, self.win2) };
        self.out = try rig.output(self.samples * 4);
        return true;
    }

    pub fn run(self: *Ola, rig: *ref.Rig) !void {
        const A = extern struct { F: i32, N: i32, hop: i32, pad: i32 = 0, n0: i64, count: i64 };
        const args: A = .{ .F = ref.int(self.F), .N = ref.int(self.N), .hop = ref.int(hop + ref.one(self.fault)), .n0 = ref.long(self.n0), .count = ref.long(self.samples) };
        try rig.go(self.pipe, .{ (self.samples + 255) / 256, 1, 1 }, 256, &.{ self.dev[0], self.dev[1], self.out }, args, 0);
    }

    pub fn count(self: *const Ola) usize {
        return self.samples;
    }

    pub fn solve(self: *const Ola, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var scratch: [16]f64 = undefined;
        for (out, 0..) |*o, i| {
            const fs = self.span(i);
            const p = self.n0 + i + self.N / 2;
            var top: [16]Iv = undefined;
            var env: [16]Iv = undefined;
            for (fs[0]..fs[1]) |fr| {
                top[fr - fs[0]] = Iv.of(self.frames[fr * self.N + p - fr * hop]);
                env[fr - fs[0]] = Iv.of(self.win2[p - fr * hop]);
            }
            const n = fs[1] - fs[0];
            o.* = s.sum(top[0..n], n, &scratch).div(s.sum(env[0..n], n, &scratch));
        }
    }
};

pub const Taps = enum { mix, resample, resample_in };

const frame = 2944;

/// Frames x a frame's samples x taps: the mixer's 2 x 127, the resamplers' 770 (736 to 735) and 769 (735 to 736).
pub fn tapsSpec(which: Taps, fault: bool) c.Spec {
    return switch (which) {
        .mix => .{ .name = if (fault) "control: metal zonos3 zc_mix reading its branches one sample early" else "metal zonos3 zc_mix", .accum = .f32, .depth = kDeep, .out = .f32, .served = &.{ .{ .rows = 4, .n = frame, .k = 254 }, .{ .rows = 1, .n = frame, .k = 254 }, .{ .rows = 16, .n = frame, .k = 254 } } },
        .resample => .{ .name = if (fault) "control: metal zonos3 zc_resample reading its input one sample early" else "metal zonos3 zc_resample", .accum = .f32, .depth = kDeep, .out = .f32, .served = &.{ .{ .rows = 4, .n = 2940, .k = 770 }, .{ .rows = 1, .n = 2940, .k = 770 }, .{ .rows = 16, .n = 2940, .k = 770 } } },
        .resample_in => .{ .name = if (fault) "control: metal zonos3 ze_resample_in reading its input one sample early" else "metal zonos3 ze_resample_in", .accum = .f32, .depth = kDeep, .out = .f32, .served = &.{ .{ .rows = 4, .n = frame, .k = 769 }, .{ .rows = 1, .n = frame, .k = 769 }, .{ .rows = 16, .n = frame, .k = 769 } } },
    };
}

/// An fma chain over a sample's taps in tap order, the input zero outside its length.
pub const Fir = struct {
    pipe: mtl.Pipeline,
    which: Taps,
    /// control: the device's input starts one sample early
    fault: bool = false,
    first: usize = 0,
    samples: usize = 0,
    total: usize = 0,
    /// the mixer's two branches are `total` apart
    x: []f32 = &.{},
    w: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Fir) c.Spec {
        return tapsSpec(self.which, self.fault);
    }

    pub fn control(self: *const Fir) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Fir, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    /// Phases x taps of the kernel, and the input and output steps of a phase cycle.
    fn geometry(self: *const Fir) [4]usize {
        return switch (self.which) {
            .mix => .{ 2, 127, 1, 1 },
            .resample => .{ 735, 770, 736, 735 },
            .resample_in => .{ 736, 769, 735, 736 },
        };
    }

    pub fn prepare(self: *Fir, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        const geo = self.geometry();
        self.samples = shape.rows * shape.n;
        self.first = if (fill == .stress or self.which == .resample_in) 0 else rng.uintLessThan(usize, 2000);
        // the input ends inside the last outputs' taps
        self.total = (self.first + self.samples) * geo[2] / geo[3] + rng.uintLessThan(usize, 60);
        const branches: usize = if (self.which == .mix) 2 else 1;
        self.x = try ref.values(a, rng, branches * self.total, .random, 1);
        self.w = try ref.values(a, rng, geo[0] * geo[1], .random, 1);
        if (fill == .stress) {
            @memset(self.w, 1);
            for (self.x) |*v| v.* = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
            ref.stress(self.x[0..@min(64, self.total)], rng);
        }
        if (fill == .nonfinite) {
            self.x[5] = ref.nan;
            self.x[self.total / 2] = ref.inf;
            self.x[self.x.len - 3] = -ref.inf;
        }
        // one sample ahead of the input, read only by the control's launch: a Metal buffer cannot start before its base
        const padded = try a.alloc(f32, self.x.len + 1);
        defer a.free(padded);
        padded[0] = 7;
        @memcpy(padded[1..], self.x);
        self.dev = .{ try ref.input(rig, padded), try ref.input(rig, self.w) };
        self.out = try rig.output(self.samples * 4);
        return true;
    }

    pub fn run(self: *Fir, rig: *ref.Rig) !void {
        const x = self.dev[0] + 4 * (1 - ref.one(self.fault));
        const groups = [3]usize{ (self.samples + 255) / 256, 1, 1 };
        const total = ref.long(self.total);
        const first = ref.long(self.first);
        const n = ref.long(self.samples);
        switch (self.which) {
            .mix => {
                const A = extern struct { stride: i64, total: i64, n0: i64, count: i64 };
                try rig.go(self.pipe, groups, 256, &.{ x, self.dev[1], self.out }, A{ .stride = total, .total = total, .n0 = first, .count = n }, 0);
            },
            .resample => {
                const A = extern struct { total: i64, m0: i64, count: i64 };
                try rig.go(self.pipe, groups, 256, &.{ x, self.dev[1], self.out }, A{ .total = total, .m0 = first, .count = n }, 0);
            },
            .resample_in => {
                const A = extern struct { total: i64, count: i64 };
                try rig.go(self.pipe, groups, 256, &.{ x, self.dev[1], self.out }, A{ .total = total, .count = n }, 0);
            },
        }
    }

    pub fn count(self: *const Fir) usize {
        return self.samples;
    }

    pub fn solve(self: *const Fir, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const geo = self.geometry();
        const terms = try a.alloc(Iv, 800);
        defer a.free(terms);
        var scratch: [800]f64 = undefined;
        const total: i64 = @intCast(self.total);
        for (out, 0..) |*o, i| {
            const m = self.first + i;
            var n: usize = 0;
            if (self.which == .mix) {
                for (0..2) |b| for (0..127) |j| {
                    const p = @as(i64, @intCast(m + j)) - 63;
                    if (p < 0 or p >= total) continue;
                    terms[n] = Iv.of(self.w[b * 127 + j]).mul(Iv.of(self.x[b * self.total + @as(usize, @intCast(p))]));
                    n += 1;
                };
            } else {
                const phase = m % geo[0];
                for (0..geo[1]) |j| {
                    const p = @as(i64, @intCast(geo[2] * (m / geo[0]) + j)) - 17;
                    if (p < 0 or p >= total) continue;
                    terms[n] = Iv.of(self.w[phase * geo[1] + j]).mul(Iv.of(self.x[@intCast(p)]));
                    n += 1;
                }
            }
            o.* = s.sum(terms[0..n], @max(n, 1), &scratch);
        }
    }
};

/// The group's registrations, each followed by its control, then the encoder's convolution's.
pub const Set = struct {
    dwconv: [2][2]ref.Gated(DwConv),
    latent: [2]ref.Gated(Latent),
    ola: [2]ref.Gated(Ola),
    fir: [3][2]ref.Gated(Fir),
    conv: conv.Set,

    pub fn init(set: *Set, libs: *ref.Libs, rig: ref.Rig, list: ref.List) !void {
        for (0..2) |j| {
            const fault = j == 1;
            set.dwconv[0][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_dwconv"), .which = .codec, .fault = fault } };
            set.dwconv[1][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.teacher, "zt_dwconv"), .which = .teacher, .fault = fault } };
            set.latent[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_latent"), .fault = fault } };
            set.ola[j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_ola"), .fault = fault } };
            set.fir[0][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_mix"), .which = .mix, .fault = fault } };
            set.fir[1][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.codec, "zc_resample"), .which = .resample, .fault = fault } };
            set.fir[2][j] = .{ .rig = rig, .k = .{ .pipe = try libs.pipe(.encoder, "ze_resample_in"), .which = .resample_in, .fault = fault } };
        }
        for (&set.dwconv) |*pair| for (pair) |*x| try list.add(x.kernel());
        for (&set.latent) |*x| try list.add(x.kernel());
        for (&set.ola) |*x| try list.add(x.kernel());
        for (&set.fir) |*pair| for (pair) |*x| try list.add(x.kernel());
        try set.conv.init(libs, rig, list);
    }

    pub fn deinit(set: *Set) void {
        set.conv.deinit();
        for (&set.fir) |*pair| for (pair) |*x| x.deinit();
        for (&set.ola) |*x| x.deinit();
        for (&set.latent) |*x| x.deinit();
        for (&set.dwconv) |*pair| for (pair) |*x| x.deinit();
    }
};

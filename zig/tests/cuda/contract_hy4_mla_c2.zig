//! Contract cases for `hy4_mla_c2`: same-engine exact byte comparisons across all absorb/expand variants + fp64 bounds.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const C2 = hy4.c2.C2;

const lat: usize = hy4.latent;
const row_w: usize = hy4.row;
const nope_dim: usize = 192;
const v_dim: usize = 256;
const kvb_rows: usize = nope_dim + v_dim;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn c2ExpandDepth(_: c.Shape) usize {
    return lat + 16;
}

const c2_absorb_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4 * lat, .k = nope_dim },
    .{ .rows = 9, .n = 4 * lat, .k = nope_dim },
    .{ .rows = 17, .n = 4 * lat, .k = nope_dim },
};

pub const C2Absorb = struct {
    rig: rig_.Rig,
    c2: *const C2,
    shape: c.Shape = undefined,
    heads: usize = 0,
    qb: []u16 = &.{},
    kvb: []u16 = &.{},
    wukt: []u16 = &.{},
    in: [2]u64 = @splat(0),
    out_dev: u64 = 0,

    pub fn deinit(self: *C2Absorb) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *C2Absorb) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.qb, &self.kvb, &self.wukt }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    pub fn kernel(self: *C2Absorb) g.Kernel {
        const s: c.Spec = .{
            .name = "cuda hy4_mla_c2 hy4c2_absorb (d1_x, d2_x, p_x exact 2-plane)",
            .accum = .none,
            .depth = zeroDepth,
            .out = .bf16,
            .nan = .{ .stated = "2-plane bf16 sum matches float64 W_UK_T projection, variant bytes match, preserves NaN" },
            .served = &c2_absorb_shapes,
        };
        return .{ .spec = s, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(C2Absorb);

    pub fn count(self: *const C2Absorb) usize {
        return 2 * self.shape.rows * self.heads * row_w;
    }

    pub fn reference(_: *C2Absorb, _: []f64, _: []f64) !void {}

    pub fn prepare(self: *C2Absorb, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != nope_dim or shape.n % lat != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / lat;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ldq = self.heads * 256;
        self.qb = try a.alloc(u16, shape.rows * ldq);
        self.kvb = try a.alloc(u16, self.heads * kvb_rows * lat);
        for (self.qb) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        for (self.kvb) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        if (fill == .nonfinite) {
            self.qb[0] = c.toBf16(std.math.nan(f32));
        }
        const rep = try hy4.c2.repack(a, std.mem.sliceAsBytes(self.kvb), self.heads);
        a.free(rep.uv);
        self.wukt = rep.uk;
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.qb));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.wukt));
        self.out_dev = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *C2Absorb) !void {
        const out_b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out_dev, .len = self.count() * 2 };
        try out_b.fill8(0xff, self.rig.s.handle);
        const qlo = self.shape.rows * self.heads * row_w;
        try hy4.c2.absorb(self.c2, self.rig.s, self.in[0], self.heads * 256, self.shape.rows, self.heads, self.in[1], self.out_dev, qlo);
        try self.rig.s.synchronize();
    }

    pub fn check(self: *C2Absorb, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const raw = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(raw);
        const words: []align(1) const u16 = std.mem.bytesAsSlice(u16, raw);
        const qlo = self.shape.rows * self.heads * row_w;
        const ldq = self.heads * 256;
        // Same-engine byte comparison: all 3 absorb variants (d1_x, d2_x, p_x) must emit identical hi/lo bytes.
        var cmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, raw.len);
        defer cmp.free();
        const cmp_host = try a.alloc(u8, raw.len);
        defer a.free(cmp_host);
        for (0..3) |vi| {
            try cmp.fill8(0xff, self.rig.s.handle);
            try hy4.c2.absorbVariant(self.c2, vi, self.rig.s, self.in[0], ldq, self.shape.rows, self.heads, self.in[1], cmp.ptr, qlo);
            try self.rig.s.synchronize();
            try cmp.download(0, cmp_host);
            if (!std.mem.eql(u8, raw, cmp_host)) return "hy4c2_absorb variant bytes differ across d1_x/d2_x/p_x";
        }
        // Row-invariance byte comparison against R=1 d1_x for multi-row shapes.
        if (self.shape.rows > 1) {
            const row_elems = self.heads * row_w;
            var solo = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, 2 * row_elems * 2);
            defer solo.free();
            const solo_h = try a.alloc(u8, 2 * row_elems * 2);
            defer a.free(solo_h);
            for ([_]usize{ 0, self.shape.rows / 2, self.shape.rows - 1 }) |ri| {
                try solo.fill8(0xff, self.rig.s.handle);
                try hy4.c2.absorbVariant(self.c2, 0, self.rig.s, self.in[0] + ri * ldq * 2, ldq, 1, self.heads, self.in[1], solo.ptr, row_elems);
                try self.rig.s.synchronize();
                try solo.download(0, solo_h);
                const hi_off = ri * row_elems * 2;
                const lo_off = (qlo + ri * row_elems) * 2;
                if (!std.mem.eql(u8, raw[hi_off..][0 .. row_elems * 2], solo_h[0 .. row_elems * 2]) or
                    !std.mem.eql(u8, raw[lo_off..][0 .. row_elems * 2], solo_h[row_elems * 2 ..][0 .. row_elems * 2]))
                    return "hy4c2_absorb multi-row bytes differ from R=1 d1_x";
            }
        }
        var fin: usize = 0;
        var nonfin: usize = 0;
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const o_base = (r * self.heads + h) * row_w;
            for (lat..row_w) |d| {
                if (words[o_base + d] != 0 or words[qlo + o_base + d] != 0) return "absorb wrote past latent columns";
            }
            for (0..lat) |col| {
                var acc: f64 = 0;
                var mag: f64 = 0;
                for (0..nope_dim) |k| {
                    const x = c.fromBf16(self.qb[r * ldq + h * 256 + k]) * c.fromBf16(self.kvb[(h * kvb_rows + k) * lat + col]);
                    acc += x;
                    mag += @abs(x);
                }
                const hi = c.fromBf16(words[o_base + col]);
                const lo = c.fromBf16(words[qlo + o_base + col]);
                const got = if (std.math.isFinite(hi)) hi + lo else hi;
                if (gate == .nonfinite) {
                    if (c.classOf(acc) == .finite) fin += 1 else nonfin += 1;
                    if (c.classOf(got) != c.classOf(acc)) return "absorb non-finite class mismatch";
                } else {
                    const lim = c.gamma(nope_dim, 0x1p-24) * mag + 0x1p-14 * @abs(acc) + 0x1p-20;
                    if (!(std.math.isFinite(got) and @abs(got - acc) <= lim)) return "absorb 2-plane sum exceeded float64 bound";
                }
            }
        };
        if (gate == .nonfinite and (fin == 0 or nonfin == 0)) return "absorb non-finite test lacked mixed finite/non-finite outputs";
        return null;
    }
};

const c2_expand_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4 * v_dim, .k = lat },
    .{ .rows = 9, .n = 4 * v_dim, .k = lat },
    .{ .rows = 17, .n = 4 * v_dim, .k = lat },
    .{ .rows = 33, .n = 4 * v_dim, .k = lat },
};

pub const C2Expand = struct {
    rig: rig_.Rig,
    c2: *const C2,
    shape: c.Shape = undefined,
    heads: usize = 0,
    lat_in: []f32 = &.{},
    gate_in: []u16 = &.{},
    kvb: []u16 = &.{},
    wuv: []u16 = &.{},
    in: [3]u64 = @splat(0),
    out_dev: u64 = 0,

    pub fn deinit(self: *C2Expand) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *C2Expand) void {
        const a = self.rig.gpu.gpa;
        a.free(self.lat_in);
        self.lat_in = &.{};
        inline for (.{ &self.gate_in, &self.kvb, &self.wuv }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    pub fn kernel(self: *C2Expand) g.Kernel {
        const s: c.Spec = .{
            .name = "cuda hy4_mla_c2 hy4c2_expand_f32lat (d1, d2, d4k64s2, p + sigmoid gate)",
            .accum = .none,
            .depth = zeroDepth,
            .ops = 6,
            .out = .bf16,
            .served = &c2_expand_shapes,
        };
        return .{ .spec = s, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(C2Expand);

    pub fn count(self: *const C2Expand) usize {
        return self.shape.rows * self.heads * v_dim;
    }

    pub fn prepare(self: *C2Expand, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != lat or shape.n % v_dim != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / v_dim;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ldg = self.heads * v_dim;
        self.lat_in = try a.alloc(f32, shape.rows * self.heads * lat);
        self.gate_in = try a.alloc(u16, shape.rows * ldg);
        self.kvb = try a.alloc(u16, self.heads * kvb_rows * lat);
        for (self.lat_in) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
        for (self.gate_in) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2.0);
        for (self.kvb) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        if (fill == .nonfinite) {
            self.lat_in[0] = std.math.nan(f32);
            self.gate_in[v_dim] = c.toBf16(std.math.nan(f32));
        }
        const rep = try hy4.c2.repack(a, std.mem.sliceAsBytes(self.kvb), self.heads);
        a.free(rep.uk);
        self.wuv = rep.uv;
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.lat_in));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.wuv));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.gate_in));
        self.out_dev = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *C2Expand) !void {
        try hy4.c2.expand(self.c2, self.rig.s, self.in[0], self.shape.rows, self.heads, self.in[1], self.in[2], self.heads * v_dim, self.out_dev);
        try self.rig.s.synchronize();
    }

    pub fn check(self: *C2Expand, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return null;
        const a = self.rig.gpu.gpa;
        const raw = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(raw);
        const ldg = self.heads * v_dim;
        // Same-engine byte comparison: all 4 expand variants (d1, d2, d4k64s2, p) must emit identical bf16 bytes.
        var cmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, raw.len);
        defer cmp.free();
        const cmp_host = try a.alloc(u8, raw.len);
        defer a.free(cmp_host);
        for (0..4) |vi| {
            try cmp.fill8(0xff, self.rig.s.handle);
            try hy4.c2.expandVariant(self.c2, vi, self.rig.s, self.in[0], self.shape.rows, self.heads, self.in[1], self.in[2], ldg, cmp.ptr);
            try self.rig.s.synchronize();
            try cmp.download(0, cmp_host);
            if (!std.mem.eql(u8, raw, cmp_host)) return "hy4c2_expand_f32lat variant bytes differ across d1/d2/d4k64s2/p";
        }
        // Row-invariance byte comparison against R=1 d1 for multi-row shapes.
        if (self.shape.rows > 1) {
            const row_bytes = ldg * 2;
            var solo = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, row_bytes);
            defer solo.free();
            const solo_h = try a.alloc(u8, row_bytes);
            defer a.free(solo_h);
            for ([_]usize{ 0, self.shape.rows / 2, self.shape.rows - 1 }) |ri| {
                try solo.fill8(0xff, self.rig.s.handle);
                try hy4.c2.expandVariant(self.c2, 0, self.rig.s, self.in[0] + ri * self.heads * lat * 4, 1, self.heads, self.in[1], self.in[2] + ri * row_bytes, ldg, solo.ptr);
                try self.rig.s.synchronize();
                try solo.download(0, solo_h);
                if (!std.mem.eql(u8, raw[ri * row_bytes ..][0..row_bytes], solo_h)) return "hy4c2_expand_f32lat multi-row bytes differ from R=1 d1";
            }
        }
        const n = self.count();
        const want = try a.alloc(f64, n);
        defer a.free(want);
        const mag = try a.alloc(f64, n);
        defer a.free(mag);
        try self.reference(want, mag);
        if (gate == .nonfinite) {
            var planted: usize = 0;
            for (want, 0..) |w, i| {
                if (c.classOf(w) != .finite) planted += 1;
                if (c.classOf(c.read(.bf16, raw, i)) != c.classOf(w)) return "hy4c2_expand_f32lat non-finite class mismatch";
            }
            if (planted == 0 or planted == n) return "hy4c2_expand_f32lat non-finite test lacked mixed finite/non-finite outputs";
        } else {
            const allow = try a.alloc(f64, n);
            defer a.free(allow);
            try self.bounds(want, mag, allow);
            for (want, allow, 0..) |w, limit, i| {
                const got = c.read(.bf16, raw, i);
                const q = if (got == w) 0 else @abs(got - w) / limit;
                if (!(q <= 1)) return "hy4c2_expand_f32lat output exceeded float64 bound";
            }
        }
        return null;
    }

    pub fn reference(self: *C2Expand, want: []f64, mag: []f64) !void {
        const ldg = self.heads * v_dim;
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const l_base = (r * self.heads + h) * lat;
            for (0..v_dim) |v| {
                var acc: f64 = 0;
                var amag: f64 = 0;
                for (0..lat) |col| {
                    const x = @as(f64, self.lat_in[l_base + col]) * c.fromBf16(self.kvb[(h * kvb_rows + nope_dim + v) * lat + col]);
                    acc += x;
                    amag += @abs(x);
                }
                const gval = c.fromBf16(self.gate_in[r * ldg + h * v_dim + v]);
                const sig = 1.0 / (1.0 + @exp(-gval));
                const idx = (r * self.heads + h) * v_dim + v;
                want[idx] = acc * sig;
                mag[idx] = amag * @abs(sig);
            }
        };
    }

    pub fn bounds(self: *const C2Expand, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec: c.Spec = .{ .name = "", .accum = .f32, .depth = c2ExpandDepth, .ops = 8, .out = .bf16, .served = &c2_expand_shapes };
        for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m);
    }
};

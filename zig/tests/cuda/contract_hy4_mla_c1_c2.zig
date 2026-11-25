//! Contract cases for `hy4_mla_attn_paged` (C1 exact 2-plane + paged + wrong-page control) and `hy4_mla_c2`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const C1 = hy4.c1.C1;
const C2 = hy4.c2.C2;

const lat: usize = hy4.latent;
const row_w: usize = hy4.row;
const nope_dim: usize = 192;
const v_dim: usize = 256;
const kvb_rows: usize = nope_dim + v_dim;
const page_rows: usize = 256;
const num_streams: usize = 2;
const poison_bf16: u16 = 0xffff;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn c1Depth(s: c.Shape) usize {
    return row_w + s.k + 16;
}

fn c2ExpandDepth(_: c.Shape) usize {
    return lat + 16;
}

fn buildPageTable(a: std.mem.Allocator, pages_per_stream: usize, pt_stride: usize) !struct { pt: []u32, phys_pages: usize } {
    const total_mapped = num_streams * pages_per_stream;
    const phys_pages = total_mapped + 2;
    const perm = try a.alloc(u32, phys_pages);
    defer a.free(perm);
    for (0..phys_pages) |i| perm[i] = @intCast((i * 5 + 3) % phys_pages);
    if (phys_pages % 5 == 0) {
        for (0..phys_pages) |i| perm[i] = @intCast((phys_pages - 1 - i + 2) % phys_pages);
    }
    const pt = try a.alloc(u32, num_streams * pt_stride);
    @memset(pt, perm[total_mapped]);
    for (0..num_streams) |s| for (0..pages_per_stream) |p| {
        pt[s * pt_stride + p] = perm[s * pages_per_stream + p];
    };
    return .{ .pt = pt, .phys_pages = phys_pages };
}

fn hasMixedFinite(out_ty: c.Elem, bytes: []const u8, n: usize) bool {
    var fin: usize = 0;
    var nonfin: usize = 0;
    for (0..n) |i| {
        if (c.classOf(c.read(out_ty, bytes, i)) == .finite) fin += 1 else nonfin += 1;
    }
    return fin > 0 and nonfin > 0;
}

pub const C1Mode = enum { x16, x16_rows, x16_qreg, x16_qreg_rows, x16_paged, x16_qreg_paged, x16_paged_wrong };

const c1_split_shapes = [_]c.Shape{
    .{ .rows = 2, .n = 16 * lat, .k = 257 },
    .{ .rows = 1, .n = 32 * lat, .k = 513 },
    .{ .rows = 3, .n = 16 * lat, .k = 255 },
};
const c1_qreg_shapes = [_]c.Shape{
    .{ .rows = 4, .n = 16 * lat, .k = 257 },
    .{ .rows = 5, .n = 16 * lat, .k = 256 },
};
const c1_paged_shapes = [_]c.Shape{
    .{ .rows = 2, .n = 16 * lat, .k = 257 },
    .{ .rows = 2, .n = 16 * lat, .k = 255 },
    .{ .rows = 3, .n = 16 * lat, .k = 256 },
    .{ .rows = 2, .n = 16 * lat, .k = 513 },
};
const c1_qreg_paged_shapes = [_]c.Shape{
    .{ .rows = 4, .n = 16 * lat, .k = 257 },
    .{ .rows = 4, .n = 16 * lat, .k = 255 },
    .{ .rows = 4, .n = 16 * lat, .k = 256 },
    .{ .rows = 4, .n = 16 * lat, .k = 513 },
};

pub const C1Case = struct {
    rig: rig_.Rig,
    c1: *const C1,
    mode: C1Mode,
    shape: c.Shape = undefined,
    heads: usize = 0,
    ctx: usize = 0,
    pt_stride: usize = 0,
    qa: []u16 = &.{},
    cache_contig: []u16 = &.{},
    cache_paged: []u16 = &.{},
    pos: []i32 = &.{},
    rs: []i32 = &.{},
    sinks: []f32 = &.{},
    pt_dev: []u32 = &.{},
    row_ptrs: []u64 = &.{},
    in: [7]u64 = @splat(0),
    out_dev: u64 = 0,
    base_dev: u64 = 0,

    pub fn deinit(self: *C1Case) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *C1Case) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.qa, &self.cache_contig, &self.cache_paged }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.pos, &self.rs }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        a.free(self.sinks);
        a.free(self.pt_dev);
        a.free(self.row_ptrs);
        self.sinks = &.{};
        self.pt_dev = &.{};
        self.row_ptrs = &.{};
    }

    fn isPaged(self: *const C1Case) bool {
        return self.mode == .x16_paged or self.mode == .x16_qreg_paged or self.mode == .x16_paged_wrong;
    }

    fn isPerRow(self: *const C1Case) bool {
        return self.mode == .x16_rows or self.mode == .x16_qreg_rows;
    }

    fn singleSplit(self: *const C1Case) bool {
        return self.mode == .x16_qreg or self.mode == .x16_qreg_rows or self.mode == .x16_qreg_paged;
    }

    pub fn kernel(self: *C1Case) g.Kernel {
        const name = switch (self.mode) {
            .x16 => "cuda hy4_mla_attn_paged hy4_mla_dec_x16 + hy4_mla_fold (exact 2-plane split+fold)",
            .x16_rows => "cuda hy4_mla_attn_paged hy4_mla_dec_x16_rows + hy4_mla_fold (per-row caches)",
            .x16_qreg => "cuda hy4_mla_attn_paged hy4_mla_dec_x16_qreg (register-resident single-split)",
            .x16_qreg_rows => "cuda hy4_mla_attn_paged hy4_mla_dec_x16_qreg_rows (qreg per-row caches)",
            .x16_paged => "cuda hy4_mla_attn_paged hy4_mla_dec_x16_paged + hy4_mla_fold (non-identity 256-row pages)",
            .x16_qreg_paged => "cuda hy4_mla_attn_paged hy4_mla_dec_x16_qreg_paged (qreg non-identity pages)",
            .x16_paged_wrong => "control: hy4_mla_attn_paged hy4_mla_dec_x16_paged wrong page table",
        };
        const rep: ?[]const u8 = if (self.isPaged() or self.isPerRow()) "contiguous hy4_mla_dec_x16" else null;
        const s: c.Spec = .{
            .name = name,
            .accum = .none,
            .depth = zeroDepth,
            .out = .f32,
            .nan = if (rep != null) .{ .stated = "bytes match equivalent contiguous C1 launch" } else .propagate,
            .replaces = rep,
            .served = switch (self.mode) {
                .x16, .x16_rows => &c1_split_shapes,
                .x16_qreg, .x16_qreg_rows => &c1_qreg_shapes,
                .x16_paged, .x16_paged_wrong => &c1_paged_shapes,
                .x16_qreg_paged => &c1_qreg_paged_shapes,
            },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.mode == .x16_paged_wrong) .shapes else null };
    }

    const vt = g.hooks(C1Case);

    pub fn count(self: *const C1Case) usize {
        return self.shape.rows * self.heads * lat;
    }

    pub fn prepare(self: *C1Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n % lat != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / lat;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const nkeys = shape.k;
        self.ctx = nkeys + 16;
        const qlo = rows * self.heads * row_w;
        self.qa = try a.alloc(u16, 2 * qlo);
        for (0..qlo) |i| {
            const x = (rng.float(f32) * 2 - 1) * 0.5;
            const hi = c.toBf16(x);
            const lo = c.toBf16(x - @as(f32, @floatCast(c.fromBf16(hi))));
            self.qa[i] = hi;
            self.qa[qlo + i] = lo;
        }
        self.pos = try a.alloc(i32, rows);
        self.rs = try a.alloc(i32, rows);
        self.sinks = try a.alloc(f32, self.heads);
        for (self.pos, 0..) |*p, r| p.* = @intCast(nkeys - 1 - @min(r, 2));
        for (self.rs, 0..) |*s, r| s.* = @intCast(if (self.isPaged() or self.isPerRow()) (r & 1) else 0);
        for (self.sinks) |*s| s.* = rng.float(f32) * 2 - 1;
        const n_streams: usize = if (self.isPaged() or self.isPerRow()) num_streams else 1;
        self.cache_contig = try a.alloc(u16, n_streams * self.ctx * row_w);
        @memset(self.cache_contig, poison_bf16);
        const pages_per_stream = @max(@as(usize, 2), (nkeys + page_rows - 1) / page_rows);
        self.pt_stride = pages_per_stream + 1;
        const tbl = try buildPageTable(a, pages_per_stream, self.pt_stride);
        defer a.free(tbl.pt);
        self.pt_dev = try a.dupe(u32, tbl.pt);
        if (self.mode == .x16_paged_wrong) {
            std.mem.swap(u32, &self.pt_dev[0], &self.pt_dev[1]);
        }
        self.cache_paged = try a.alloc(u16, tbl.phys_pages * page_rows * row_w);
        @memset(self.cache_paged, poison_bf16);
        for (0..n_streams) |st| for (0..pages_per_stream) |pg| {
            const phys_base = @as(usize, tbl.pt[st * self.pt_stride + pg]) * page_rows * row_w;
            for (self.cache_paged[phys_base .. phys_base + page_rows * row_w]) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
            for (0..page_rows) |slot| {
                const t = pg * page_rows + slot;
                if (t < nkeys) {
                    const src = self.cache_paged[phys_base + slot * row_w ..][0..row_w];
                    const dst = self.cache_contig[(st * self.ctx + t) * row_w ..][0..row_w];
                    @memcpy(dst, src);
                }
            }
        };
        if (fill == .nonfinite) {
            self.qa[0] = c.toBf16(std.math.nan(f32));
            self.qa[qlo] = 0;
            if (self.heads > 1) {
                self.sinks[1] = std.math.nan(f32);
            }
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.qa));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.cache_contig));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.cache_paged));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.in[4] = try self.rig.input(std.mem.sliceAsBytes(self.rs));
        self.in[5] = try self.rig.input(std.mem.sliceAsBytes(self.sinks));
        self.in[6] = try self.rig.input(std.mem.sliceAsBytes(self.pt_dev));
        if (self.isPerRow()) {
            self.row_ptrs = try a.alloc(u64, rows);
            const stream_bytes: u64 = @intCast(self.ctx * row_w * 2);
            for (self.row_ptrs, 0..) |*ptr, r| ptr.* = self.in[1] + @as(u64, @intCast(self.rs[r])) * stream_bytes;
            self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.row_ptrs));
        }
        self.out_dev = try self.rig.output(self.count() * 4);
        self.base_dev = try self.rig.scratch(self.count() * 4);
        return true;
    }

    pub fn run(self: *C1Case) !void {
        const qlo = self.shape.rows * self.heads * row_w;
        if (self.isPerRow()) {
            try hy4.c1.attnRows(self.c1, self.rig.s, self.in[0], qlo, self.heads, self.in[2], .{
                .kind = 1,
                .ring = self.ctx,
                .single_split = self.singleSplit(),
            }, self.in[3], self.in[5], self.out_dev, self.shape.rows);
        } else {
            try hy4.c1.attn(self.c1, self.rig.s, self.in[0], qlo, self.heads, if (self.isPaged()) self.in[2] else self.in[1], .{
                .kind = 1,
                .ring = self.ctx,
                .single_split = self.singleSplit(),
                .page_table = if (self.isPaged()) self.in[6] else 0,
                .pt_stride = if (self.isPaged()) self.pt_stride else 0,
                .row_stream = self.in[4],
            }, self.in[3], self.in[5], self.out_dev, self.shape.rows);
        }
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *C1Case) ![]const u8 {
        const qlo = self.shape.rows * self.heads * row_w;
        try hy4.c1.attn(self.c1, self.rig.s, self.in[0], qlo, self.heads, self.in[1], .{
            .kind = 1,
            .ring = self.ctx,
            .single_split = self.singleSplit(),
            .row_stream = self.in[4],
        }, self.in[3], self.in[5], self.base_dev, self.shape.rows);
        return self.rig.bytes(self.base_dev, self.count() * 4);
    }

    pub fn check(self: *C1Case, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite or !(self.isPaged() or self.isPerRow())) return null;
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        if (!hasMixedFinite(.f32, want, self.count())) return "non-finite input did not produce mixed finite/non-finite outputs";
        if (!std.mem.eql(u8, got, want)) return "paged/per-row non-finite output bytes differ from contiguous baseline";
        return null;
    }

    pub fn reference(self: *C1Case, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const qlo = self.shape.rows * self.heads * row_w;
        const scores = try a.alloc(f64, self.shape.k);
        defer a.free(scores);
        for (0..self.shape.rows) |r| {
            const st: usize = @intCast(self.rs[r]);
            const n: usize = @as(usize, @intCast(self.pos[r])) + 1;
            for (0..self.heads) |h| {
                const q_base = (r * self.heads + h) * row_w;
                var top: f64 = self.sinks[h];
                for (0..n) |t| {
                    const k_base = (st * self.ctx + t) * row_w;
                    var dot: f64 = 0;
                    for (0..row_w) |d| {
                        const qv = c.fromBf16(self.qa[q_base + d]) + c.fromBf16(self.qa[qlo + q_base + d]);
                        dot += qv * c.fromBf16(self.cache_contig[k_base + d]);
                    }
                    scores[t] = dot * 0.0625;
                    top = if (std.math.isNan(top) or std.math.isNan(scores[t])) std.math.nan(f64) else @max(top, scores[t]);
                }
                var den: f64 = @exp(self.sinks[h] - top);
                for (0..n) |t| den += @exp(scores[t] - top);
                const o_base = (r * self.heads + h) * lat;
                for (0..lat) |d| {
                    var acc: f64 = 0;
                    var amag: f64 = 0;
                    for (0..n) |t| {
                        const w = @exp(scores[t] - top) / den;
                        const kv = c.fromBf16(self.cache_contig[(st * self.ctx + t) * row_w + d]);
                        acc += w * kv;
                        amag += @abs(w * kv);
                    }
                    want[o_base + d] = acc;
                    mag[o_base + d] = @max(amag, 1.0);
                }
            }
        }
    }

    pub fn bounds(self: *const C1Case, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec: c.Spec = .{ .name = "", .accum = .f32, .depth = c1Depth, .ops = 16, .out = .f32, .served = &c1_split_shapes };
        for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m) + 0x1p-14 * m;
    }
};

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
            .nan = .{ .stated = "2-plane bf16 sum matches float64 W_UK_T projection and preserves NaN" },
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
        try self.rig.zero(self.out_dev, self.count() * 2);
        const qlo = self.shape.rows * self.heads * row_w;
        try hy4.c2.absorb(self.c2, self.rig.s, self.in[0], self.heads * 256, self.shape.rows, self.heads, self.in[1], self.out_dev, qlo);
        try self.rig.s.synchronize();
    }

    pub fn check(self: *C2Absorb, gate: g.Gate) !?[]const u8 {
        const raw = try self.rig.outputBytes();
        const words = std.mem.bytesAsSlice(u16, raw);
        const qlo = self.shape.rows * self.heads * row_w;
        const ldq = self.heads * 256;
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

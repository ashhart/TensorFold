//! Paged pool and score contract cases: non-identity pages, stream/group offsets, tails, and wrong-page controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_index_paged");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_index_paged.zig");
const pool_k = @import("contract_hy4_index_paged_pool.zig");
const score_k = @import("contract_hy4_index_paged_score.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const page_groups: usize = 64;
const num_streams: usize = 2;
const pool_g_base: usize = 59;
const poison_bf16: u16 = 0xffff;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

/// Deterministic non-identity physical page table with two unmapped poison pages per pool.
fn buildPageTable(a: std.mem.Allocator, pages_per_stream: usize, pt_stride: usize) !struct { pt: []u32, unmapped: [2]u32, phys_pages: usize } {
    const total_mapped = num_streams * pages_per_stream;
    const phys_pages = total_mapped + 2;
    const perm = try a.alloc(u32, phys_pages);
    defer a.free(perm);
    for (0..phys_pages) |i| perm[i] = @intCast((i * 5 + 3) % phys_pages);
    // Guarantee bijection when gcd(5, phys_pages) != 1 by reversing/rotating non-coprime counts.
    if (phys_pages % 5 == 0) {
        for (0..phys_pages) |i| perm[i] = @intCast((phys_pages - 1 - i + 2) % phys_pages);
    }
    const pt = try a.alloc(u32, num_streams * pt_stride);
    @memset(pt, perm[total_mapped]);
    for (0..num_streams) |s| for (0..pages_per_stream) |p| {
        pt[s * pt_stride + p] = perm[s * pages_per_stream + p];
    };
    return .{ .pt = pt, .unmapped = .{ perm[total_mapped], perm[total_mapped + 1] }, .phys_pages = phys_pages };
}

fn hasMixedFinite(out_ty: c.Elem, bytes: []const u8, n: usize) bool {
    var fin: usize = 0;
    var nonfin: usize = 0;
    for (0..n) |i| {
        if (c.classOf(c.read(out_ty, bytes, i)) == .finite) fin += 1 else nonfin += 1;
    }
    return fin > 0 and nonfin > 0;
}

const pool_paged_served = [_]c.Shape{
    .{ .rows = 8, .n = hy4.dim, .k = hy4.dim },
    .{ .rows = 1, .n = hy4.dim, .k = hy4.dim },
    .{ .rows = 17, .n = hy4.dim, .k = hy4.dim },
    .{ .rows = 65, .n = hy4.dim, .k = hy4.dim },
    .{ .rows = 133, .n = hy4.dim, .k = hy4.dim },
};

pub const PoolPaged = struct {
    rig: rig_.Rig,
    variant: pool_k.Variant,
    f: cuda.Function,
    base_1cta: cuda.Function,
    persist_clusters: usize = 1,
    wrong_page: bool = false,
    shape: c.Shape = undefined,
    gpool: usize = 0,
    pt_stride: usize = 0,
    phys_pages: usize = 0,
    unmapped: [2]u32 = .{ 0, 0 },
    js: []i32 = &.{},
    jg: []i32 = &.{},
    pt_true: []u32 = &.{},
    pt_dev: []u32 = &.{},
    tk: []u16 = &.{},
    wv: []f32 = &.{},
    wg: []f32 = &.{},
    ape: []f32 = &.{},
    knw: []f32 = &.{},
    rt: []f32 = &.{},
    expected: []u16 = &.{},
    in: [9]u64 = @splat(0),
    out_dev: u64 = 0,
    base_dev: u64 = 0,

    pub fn deinit(self: *PoolPaged) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *PoolPaged) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.js, &self.jg }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.pt_true, &self.pt_dev }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.tk, &self.expected }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.wv, &self.wg, &self.ape, &self.knw, &self.rt }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    pub fn kernel(self: *PoolPaged) g.Kernel {
        const name = if (self.wrong_page)
            "control: hy4_index_paged tf_hy4_idx_pool_1cta_paged wrong page table"
        else switch (self.variant) {
            .cta1 => "cuda hy4_index_paged tf_hy4_idx_pool_1cta_paged (non-identity paged 4:1 pooling)",
            .small => "cuda hy4_index_paged tf_hy4_idx_pool_small_paged (4-CTA DSMEM paged pooling)",
            .persist => "cuda hy4_index_paged tf_hy4_idx_pool_persist_paged (4-CTA persistent paged pooling)",
        };
        const s: c.Spec = .{
            .name = name,
            .accum = .none,
            .depth = zeroDepth,
            .out = .bf16,
            .nan = .{ .stated = "gathered paged keys match contiguous tf_hy4_idx_pool_1cta bytes" },
            .replaces = "contiguous tf_hy4_idx_pool_1cta",
            .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
            .served = &pool_paged_served,
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.wrong_page) .shapes else null };
    }

    const vt = g.hooks(PoolPaged);

    pub fn count(self: *const PoolPaged) usize {
        return self.phys_pages * page_groups * hy4.dim;
    }

    pub fn reference(_: *PoolPaged, _: []f64, _: []f64) !void {}

    pub fn prepare(self: *PoolPaged, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hy4.dim or shape.k != hy4.dim) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const jobs = shape.rows;
        self.js = try a.alloc(i32, jobs);
        self.jg = try a.alloc(i32, jobs);
        var max_g: usize = 0;
        for (0..jobs) |jb| {
            const st: usize = jb & 1;
            const grp: usize = (if (st == 0) @as(usize, 2) else @as(usize, 5)) + (jb / 2);
            self.js[jb] = @intCast(st);
            self.jg[jb] = @intCast(grp);
            max_g = @max(max_g, grp);
        }
        self.gpool = max_g + 4;
        const ctx = self.gpool * hy4.ratio;
        const pages_per_stream = @max(@as(usize, 2), (pool_g_base + self.gpool + page_groups - 1) / page_groups);
        self.pt_stride = pages_per_stream + 1;
        const tbl = try buildPageTable(a, pages_per_stream, self.pt_stride);
        self.pt_true = tbl.pt;
        self.unmapped = tbl.unmapped;
        self.phys_pages = tbl.phys_pages;
        self.pt_dev = try a.dupe(u32, self.pt_true);
        if (self.wrong_page) {
            std.mem.swap(u32, &self.pt_dev[0], &self.pt_dev[self.pt_stride]);
        }
        self.tk = try a.alloc(u16, num_streams * ctx * hy4.dim);
        self.wv = try a.alloc(f32, hy4.dim * hy4.dim);
        self.wg = try a.alloc(f32, hy4.dim * hy4.dim);
        self.ape = try a.alloc(f32, hy4.ratio * hy4.dim);
        self.knw = try a.alloc(f32, hy4.dim);
        self.rt = try a.alloc(f32, num_streams * self.gpool * hy4.rope_pairs * 2);
        const phys_elems = self.phys_pages * page_groups * hy4.dim;
        self.expected = try a.alloc(u16, phys_elems);
        for (self.tk) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (self.wv) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
        for (self.wg) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
        for (self.ape) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
        for (self.knw) |*v| v.* = 0.5 + rng.float(f32);
        const freqs = cuda.mla.ropeFreqs(1e7);
        for (0..num_streams) |st| for (0..self.gpool) |gg| for (0..hy4.rope_pairs) |j| {
            const ang = @as(f64, @floatFromInt(4 * gg + st * 17)) * @as(f64, freqs[j]);
            const at = ((st * self.gpool + gg) * hy4.rope_pairs + j) * 2;
            self.rt[at] = @floatCast(@cos(ang));
            self.rt[at + 1] = @floatCast(@sin(ang));
        };
        if (fill == .nonfinite) {
            const s0: usize = @intCast(self.js[0]);
            const g0: usize = @intCast(self.jg[0]);
            self.tk[(s0 * ctx + 4 * g0) * hy4.dim] = c.toBf16(std.math.nan(f32));
            if (jobs > 1) {
                const s1: usize = @intCast(self.js[1]);
                const g1: usize = @intCast(self.jg[1]);
                self.tk[(s1 * ctx + 4 * g1) * hy4.dim + 5] = c.toBf16(std.math.inf(f32));
            }
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.tk));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.wv));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.wg));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.ape));
        self.in[4] = try self.rig.input(std.mem.sliceAsBytes(self.knw));
        self.in[5] = try self.rig.input(std.mem.sliceAsBytes(self.rt));
        self.in[6] = try self.rig.input(std.mem.sliceAsBytes(self.js));
        self.in[7] = try self.rig.input(std.mem.sliceAsBytes(self.jg));
        self.in[8] = try self.rig.input(std.mem.sliceAsBytes(self.pt_dev));
        self.out_dev = try self.rig.output(phys_elems * 2);
        self.base_dev = try self.rig.scratch(num_streams * self.gpool * hy4.dim * 2);
        return true;
    }

    fn launchPool(self: *const PoolPaged, f: cuda.Function, v: pool_k.Variant, out: u64, pt: u64, pt_stride: usize, g_base: usize) !void {
        const jobs = self.shape.rows;
        const cluster: ?u32 = if (v == .cta1) null else 4;
        const smem: u32 = switch (v) {
            .cta1 => hy4.pool_1cta_smem,
            .small => 0,
            .persist => hy4.pool_persist_smem,
        };
        const gx: u32 = switch (v) {
            .cta1 => @intCast((jobs + 15) / 16),
            .small => @intCast(4 * jobs),
            .persist => @intCast(4 * @min((jobs + 31) / 32, self.persist_clusters)),
        };
        const bx: u32 = if (v == .persist) 256 else 128;
        const w: hy4.PoolWeights = .{ .values = self.in[1], .gates = self.in[2], .ape = self.in[3], .key_norm = self.in[4], .eps = set.eps };
        try hy4.poolWith(f, self.rig.s, cluster, smem, gx, bx, self.in[6], self.in[7], 0, jobs, self.in[0], self.gpool * hy4.ratio, w, self.in[5], self.gpool, 1, out, pt, pt_stride, g_base);
    }

    pub fn run(self: *PoolPaged) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out_dev, .len = self.phys_pages * page_groups * hy4.dim * 2 };
        try b.fill8(0xff, self.rig.s.handle);
        try self.launchPool(self.f, self.variant, self.out_dev, self.in[8], self.pt_stride, pool_g_base);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *PoolPaged) ![]const u8 {
        try self.launchPool(self.base_1cta, .cta1, self.base_dev, 0, 0, 0);
        const raw = try self.rig.bytes(self.base_dev, num_streams * self.gpool * hy4.dim * 2);
        const words = std.mem.bytesAsSlice(u16, raw);
        @memset(self.expected, poison_bf16);
        for (0..self.shape.rows) |jb| {
            const st: usize = @intCast(self.js[jb]);
            const grp: usize = @intCast(self.jg[jb]);
            const ug = pool_g_base + grp;
            const phys_row = @as(usize, self.pt_true[st * self.pt_stride + (ug >> 6)]) * page_groups + (ug & 63);
            const src = words[(st * self.gpool + grp) * hy4.dim ..][0..hy4.dim];
            @memcpy(self.expected[phys_row * hy4.dim ..][0..hy4.dim], src);
        }
        return std.mem.sliceAsBytes(self.expected);
    }

    pub fn check(self: *PoolPaged, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite) return null;
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        var fin: usize = 0;
        var nonfin: usize = 0;
        for (0..self.shape.rows) |jb| {
            const st: usize = @intCast(self.js[jb]);
            const ug = pool_g_base + @as(usize, @intCast(self.jg[jb]));
            const phys_row = @as(usize, self.pt_true[st * self.pt_stride + (ug >> 6)]) * page_groups + (ug & 63);
            for (0..hy4.dim) |ch| {
                const w = c.read(.bf16, want, phys_row * hy4.dim + ch);
                if (c.classOf(w) == .finite) fin += 1 else nonfin += 1;
            }
        }
        if (fin == 0 or nonfin == 0) return "non-finite input did not produce mixed finite/non-finite logical keys";
        if (!std.mem.eql(u8, got, want)) return "paged non-finite output bytes differ from contiguous baseline";
        return null;
    }
};

const score_paged_served = [_]c.Shape{
    .{ .rows = 8, .n = 129, .k = hy4.dim },
    .{ .rows = 1, .n = 63, .k = hy4.dim },
    .{ .rows = 4, .n = 64, .k = hy4.dim },
    .{ .rows = 5, .n = 65, .k = hy4.dim },
    .{ .rows = 17, .n = 257, .k = hy4.dim },
};

pub const ScorePaged = struct {
    rig: rig_.Rig,
    kind: score_k.Kind,
    f: cuda.Function,
    base_contig: cuda.Function,
    wrong_page: bool = false,
    shape: c.Shape = undefined,
    gpool: usize = 0,
    pt_stride: usize = 0,
    iq: []u16 = &.{},
    hw: []f32 = &.{},
    pos: []i32 = &.{},
    rs: []i32 = &.{},
    gkc_contig: []u16 = &.{},
    gkc_paged: []u16 = &.{},
    pt_dev: []u32 = &.{},
    cs: []f32 = &.{},
    in: [8]u64 = @splat(0),
    out_dev: u64 = 0,
    base_dev: u64 = 0,

    pub fn deinit(self: *ScorePaged) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *ScorePaged) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.iq, &self.gkc_contig, &self.gkc_paged }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.hw, &self.cs }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        inline for (.{ &self.pos, &self.rs }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
        a.free(self.pt_dev);
        self.pt_dev = &.{};
    }

    pub fn kernel(self: *ScorePaged) g.Kernel {
        const name = if (self.wrong_page)
            "control: hy4_index_paged tf_hy4_idx_score_decode_paged wrong page table"
        else switch (self.kind) {
            .decode => "cuda hy4_index_paged tf_hy4_idx_score_decode_paged (non-identity paged decode score)",
            .decode_rope => "cuda hy4_index_paged tf_hy4_idx_score_decode_rope_paged (paged M-RoPE + score)",
            .prefill => "cuda hy4_index_paged tf_hy4_idx_score_prefill_paged (paged 8-row prefill score)",
        };
        const s: c.Spec = .{
            .name = name,
            .accum = .none,
            .depth = zeroDepth,
            .out = .f32,
            .nan = .{ .stated = "paged scores match equivalent contiguous score kernel bytes" },
            .replaces = "contiguous tf_hy4_idx_score",
            .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 64, .step = 1, .min = 1 }, .{} },
            .served = &score_paged_served,
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.wrong_page) .shapes else null };
    }

    const vt = g.hooks(ScorePaged);

    pub fn count(self: *const ScorePaged) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(_: *ScorePaged, _: []f64, _: []f64) !void {}

    pub fn prepare(self: *ScorePaged, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != hy4.dim) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const own = shape.n;
        self.gpool = own + 16;
        const pages_per_stream = @max(@as(usize, 2), (own + page_groups - 1) / page_groups);
        self.pt_stride = pages_per_stream + 1;
        const tbl = try buildPageTable(a, pages_per_stream, self.pt_stride);
        defer a.free(tbl.pt);
        self.pt_dev = try a.dupe(u32, tbl.pt);
        if (self.wrong_page) {
            std.mem.swap(u32, &self.pt_dev[0], &self.pt_dev[1]);
        }
        const max_pos = 4 * own + 4;
        self.iq = try a.alloc(u16, rows * hy4.heads * hy4.dim);
        self.hw = try a.alloc(f32, rows * hy4.heads);
        self.pos = try a.alloc(i32, rows);
        self.rs = try a.alloc(i32, rows);
        self.gkc_contig = try a.alloc(u16, num_streams * self.gpool * hy4.dim);
        self.gkc_paged = try a.alloc(u16, tbl.phys_pages * page_groups * hy4.dim);
        self.cs = try a.alloc(f32, max_pos * hy4.rope_pairs * 2);
        @memset(self.gkc_contig, poison_bf16);
        @memset(self.gkc_paged, poison_bf16);
        for (self.pos, 0..) |*p, r| p.* = @intCast(4 * own + (r % 4));
        // Mix grouped and alternating stream assignments to exercise stream switches inside a CTA.
        for (self.rs, 0..) |*s, r| s.* = @intCast(if (r < 2) 0 else if (r < 4) 1 else (r & 1));
        const freqs = cuda.mla.ropeFreqs(1e7);
        for (0..max_pos) |p| for (0..hy4.rope_pairs) |j| {
            const ang = @as(f64, @floatFromInt(p)) * @as(f64, freqs[j]);
            self.cs[(p * hy4.rope_pairs + j) * 2] = @floatCast(@cos(ang));
            self.cs[(p * hy4.rope_pairs + j) * 2 + 1] = @floatCast(@sin(ang));
        };
        for (self.iq) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (self.hw) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
        for (0..num_streams) |st| for (0..pages_per_stream) |pg| {
            const phys_base = @as(usize, tbl.pt[st * self.pt_stride + pg]) * page_groups * hy4.dim;
            for (self.gkc_paged[phys_base .. phys_base + page_groups * hy4.dim]) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (0..page_groups) |slot| {
                const gg = pg * page_groups + slot;
                if (gg < own) {
                    const src = self.gkc_paged[phys_base + slot * hy4.dim ..][0..hy4.dim];
                    const dst = self.gkc_contig[(st * self.gpool + gg) * hy4.dim ..][0..hy4.dim];
                    @memcpy(dst, src);
                }
            }
        };
        if (fill == .nonfinite) {
            self.iq[0 * hy4.heads * hy4.dim + 5] = c.toBf16(std.math.nan(f32));
            if (rows > 1) self.iq[1 * hy4.heads * hy4.dim + 0] = c.toBf16(std.math.inf(f32));
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.iq));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.hw));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.rs));
        self.in[4] = try self.rig.input(std.mem.sliceAsBytes(self.gkc_paged));
        self.in[5] = try self.rig.input(std.mem.sliceAsBytes(self.gkc_contig));
        self.in[6] = try self.rig.input(std.mem.sliceAsBytes(self.cs));
        self.in[7] = try self.rig.input(std.mem.sliceAsBytes(self.pt_dev));
        self.out_dev = try self.rig.output(self.count() * 4);
        self.base_dev = try self.rig.scratch(self.count() * 4);
        return true;
    }

    fn launchScore(self: *const ScorePaged, f: cuda.Function, gkc: u64, out: u64, pt: u64) !void {
        const rows = self.shape.rows;
        const own = self.shape.n;
        const rb: usize = if (self.kind == .prefill) 8 else 4;
        const nb = (rows + rb - 1) / rb;
        const split = hy4.scoreSplit(nb, own, 48, 3);
        const bx: u32 = if (self.kind == .prefill) 256 else 128;
        var a: cuda.Args = .{};
        inline for (.{ @as(u64, 0), @as(c_int, @intCast(rows)), self.in[0], self.in[1], self.in[3], self.in[2], gkc, @as(c_int, @intCast(self.gpool)), out, @as(c_int, @intCast(own)) }) |v| a.add(v);
        if (self.kind == .decode_rope) {
            inline for (.{ self.in[6], @as(u64, 0), @as(c_int, @intCast(rows)) }) |v| a.add(v);
        }
        if (pt != 0) {
            a.add(pt);
            a.add(@as(c_int, @intCast(self.pt_stride)));
        }
        try cuda.launch.launch(f, .{ .grid = .{ .x = split, .y = @intCast(nb) }, .block = .{ .x = bx } }, self.rig.s, &a);
    }

    pub fn run(self: *ScorePaged) !void {
        try self.launchScore(self.f, self.in[4], self.out_dev, self.in[7]);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *ScorePaged) ![]const u8 {
        try self.launchScore(self.base_contig, self.in[5], self.base_dev, 0);
        return self.rig.bytes(self.base_dev, self.count() * 4);
    }

    pub fn check(self: *ScorePaged, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite) return null;
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        if (!hasMixedFinite(.f32, want, self.count())) return "non-finite input did not produce mixed finite/non-finite scores";
        if (!std.mem.eql(u8, got, want)) return "paged non-finite score bytes differ from contiguous baseline";
        return null;
    }
};

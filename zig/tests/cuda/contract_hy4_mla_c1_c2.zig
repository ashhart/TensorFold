//! Contract cases for `hy4_mla_attn_paged` (C1 2-plane, paged, long indexed list, wrong-page control) + C2 re-exports.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const rig_ = @import("contract_rig.zig");
const c2_k = @import("contract_hy4_mla_c2.zig");

pub const C2Absorb = c2_k.C2Absorb;
pub const C2Expand = c2_k.C2Expand;

const c = core.contract.spec;
const g = core.contract.gates;
const C1 = hy4.c1.C1;

const lat: usize = hy4.latent;
const row_w: usize = hy4.row;
const page_rows: usize = 256;
const num_streams: usize = 2;
const dsa_keys: usize = 2048;
const list_w: usize = 512;
const poison_bf16: u16 = 0xffff;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn c1Depth(s: c.Shape) usize {
    return row_w + @min(s.k, dsa_keys) + 16;
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
    .{ .rows = 2, .n = 16 * lat, .k = 2120 },
};
const c1_qreg_shapes = [_]c.Shape{
    .{ .rows = 4, .n = 16 * lat, .k = 257 },
    .{ .rows = 5, .n = 16 * lat, .k = 256 },
    .{ .rows = 4, .n = 16 * lat, .k = 2120 },
};
const c1_paged_shapes = [_]c.Shape{
    .{ .rows = 2, .n = 16 * lat, .k = 257 },
    .{ .rows = 2, .n = 16 * lat, .k = 255 },
    .{ .rows = 3, .n = 16 * lat, .k = 256 },
    .{ .rows = 2, .n = 16 * lat, .k = 513 },
    .{ .rows = 2, .n = 16 * lat, .k = 2120 },
};
const c1_qreg_paged_shapes = [_]c.Shape{
    .{ .rows = 4, .n = 16 * lat, .k = 257 },
    .{ .rows = 4, .n = 16 * lat, .k = 255 },
    .{ .rows = 4, .n = 16 * lat, .k = 256 },
    .{ .rows = 4, .n = 16 * lat, .k = 513 },
    .{ .rows = 4, .n = 16 * lat, .k = 2120 },
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
    list_host: []i32 = &.{},
    sinks: []f32 = &.{},
    pt_dev: []u32 = &.{},
    row_ptrs: []u64 = &.{},
    in: [8]u64 = @splat(0),
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
        inline for (.{ &self.pos, &self.rs, &self.list_host }) |s| {
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
        self.list_host = try a.alloc(i32, rows * list_w);
        self.sinks = try a.alloc(f32, self.heads);
        for (self.pos, 0..) |*p, r| p.* = @intCast(nkeys - 1 - @min(r, 2));
        for (self.rs, 0..) |*s, r| s.* = @intCast(if (self.isPaged() or self.isPerRow()) (r & 1) else 0);
        // Non-monotonic group list spanning all pages plus -1 sentinels for long-context (pos >= 2048) rows.
        const max_grp: usize = @max(@as(usize, 1), nkeys / 4);
        for (0..rows) |r| for (0..list_w) |gi| {
            self.list_host[r * list_w + gi] = if ((gi % 37) == 36) -1 else @intCast((gi * 7 + r * 13) % max_grp);
        };
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
        self.in[7] = try self.rig.input(std.mem.sliceAsBytes(self.list_host));
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
                .list = self.in[7],
                .single_split = self.singleSplit(),
            }, self.in[3], self.in[5], self.out_dev, self.shape.rows);
        } else {
            try hy4.c1.attn(self.c1, self.rig.s, self.in[0], qlo, self.heads, if (self.isPaged()) self.in[2] else self.in[1], .{
                .kind = 1,
                .ring = self.ctx,
                .list = self.in[7],
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
            .list = self.in[7],
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
        const scores = try a.alloc(f64, dsa_keys);
        defer a.free(scores);
        const kpos = try a.alloc(i32, dsa_keys);
        defer a.free(kpos);
        for (0..self.shape.rows) |r| {
            const st: usize = @intCast(self.rs[r]);
            const prow = self.pos[r];
            const use_list = prow >= @as(i32, dsa_keys);
            const n: usize = if (use_list) dsa_keys else @as(usize, @intCast(prow)) + 1;
            for (0..n) |i| {
                if (!use_list) {
                    kpos[i] = @intCast(i);
                } else {
                    const gi = self.list_host[r * list_w + (i >> 2)];
                    const kp: i32 = if (gi < 0) -1 else 4 * gi + @as(i32, @intCast(i & 3));
                    kpos[i] = if (kp >= 0 and kp <= prow) kp else -1;
                }
            }
            for (0..self.heads) |h| {
                const q_base = (r * self.heads + h) * row_w;
                var top: f64 = self.sinks[h];
                for (0..n) |i| {
                    if (kpos[i] < 0) {
                        scores[i] = -std.math.inf(f64);
                        continue;
                    }
                    const k_base = (st * self.ctx + @as(usize, @intCast(kpos[i]))) * row_w;
                    var dot: f64 = 0;
                    for (0..row_w) |d| {
                        const qv = c.fromBf16(self.qa[q_base + d]) + c.fromBf16(self.qa[qlo + q_base + d]);
                        dot += qv * c.fromBf16(self.cache_contig[k_base + d]);
                    }
                    scores[i] = dot * 0.0625;
                    top = if (std.math.isNan(top) or std.math.isNan(scores[i])) std.math.nan(f64) else @max(top, scores[i]);
                }
                var den: f64 = @exp(self.sinks[h] - top);
                for (0..n) |i| {
                    if (kpos[i] >= 0) den += @exp(scores[i] - top);
                }
                const o_base = (r * self.heads + h) * lat;
                for (0..lat) |d| {
                    var acc: f64 = 0;
                    var amag: f64 = 0;
                    for (0..n) |i| {
                        if (kpos[i] < 0) continue;
                        const w = @exp(scores[i] - top) / den;
                        const kv = c.fromBf16(self.cache_contig[(st * self.ctx + @as(usize, @intCast(kpos[i]))) * row_w + d]);
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

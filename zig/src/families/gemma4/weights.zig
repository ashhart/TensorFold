//! Gemma 4 dense weights: each linear stack packed once for the core lane projection, norms read in place.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("../../core/lane_projection.zig");
const rowp = @import("../../core/row_projection.zig");
const ks = @import("kernel_sources");
const fused = @import("fused_lane.zig");
const cfg = @import("config.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A buffer and an offset into it.
pub const Ref = lane.Ref;

const fast: lane.Format = .{ .bits = cfg.widths.fast.bits, .group = cfg.group };

/// One packed linear (members of one width along N); with `y_stride`, it writes columns col.. of a stack's rows.
pub const Linear = struct {
    w: mtl.Buffer,
    sb: mtl.Buffer,
    n: usize,
    k: usize,
    format: lane.Format = fast,
    col: usize = 0,
    y_stride: ?usize = null,
    /// The row path's weights (a chip without tensor units): the checkpoint's arrays as stored.
    raw: ?rowp.Weights = null,

    /// The reg kernel's widths (4-, 5-, or 6-bit g64 on whole tiles), reading group sums from the input kernel.
    pub fn reg(x: Linear) bool {
        return x.format.group == cfg.group and x.n % 32 == 0 and (x.format.bits == 4 or x.format.bits == 5 or x.format.bits == 6);
    }

    /// `rows` of x [rows, k] through a general-unpack pipeline, which takes 16 rows an encode: y's rows y_stride apart.
    pub fn encodeRows(x: Linear, p: lane.Projection, e: mtl.ComputeEncoder, in: Ref, y: Ref, rows: usize) !void {
        return x.encodeRefsRows(p, e, in, .{ .buf = x.w }, .{ .buf = x.sb }, y, rows);
    }

    /// The same row encoder with explicit packed references, including guarded contract buffers.
    pub fn encodeRefsRows(x: Linear, p: lane.Projection, e: mtl.ComputeEncoder, in: Ref, w: Ref, sb: Ref, y: Ref, rows: usize) !void {
        const pitch = x.y_stride orelse x.n;
        var r: usize = 0;
        while (r < rows) : (r += 16) {
            const n = @min(16, rows - r);
            try p.encode(e, .{ .buf = in.buf, .off = in.off + r * x.k * 2 }, w, sb, .{ .buf = y.buf, .off = y.off + r * pitch * 2 }, @intCast(n));
        }
    }
};

/// A q|k|v or gate|up stack: one Linear per run of equal widths (gilby's runs), together `n` wide.
pub const Stack = struct {
    runs: [3]Linear = undefined,
    count: usize = 0,
    n: usize = 0,
    /// gate|up in one run, its rows interleaved in 16-row blocks (gemma_geglu's mode 1); split, gate then up.
    interleaved: bool = false,

    pub fn list(s: *const Stack) []const Linear {
        return s.runs[0..s.count];
    }
};

/// A linear as core/affine_mm reads it (prompt chunks): MLX's words, scales and biases, copied to aligned storage.
pub const Mlx = struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32, bits: u8 = 4 };

/// A bf16 linear read in place [n, k] (the E-series' per-layer linears, the converter's --dense-ple).
pub const Dense = struct { w: Ref, n: usize, k: usize };

/// The E-series' per-layer inputs: the 4/64 table, the projection of the embeddings and its norm.
pub const Ple = struct {
    table_w: Ref,
    table_s: Ref,
    table_b: Ref,
    model_projection: Dense,
    projection_norm: Ref,
};

pub const Layer = struct {
    /// q, k, v (k again on global layers, which have none), o, gate, up, down for prompt chunks (unset on E-series).
    mlx: [7]Mlx,
    qkv: Stack,
    o: Linear,
    gate_up: Stack,
    down: Linear,
    q_norm: Ref,
    k_norm: Ref,
    input_norm: Ref,
    post_attention: Ref,
    pre_feedforward: Ref,
    post_feedforward: Ref,
    scalar: Ref,
    /// The E-series' per-layer gate [width, hidden], projection [hidden, width] and the norm after it.
    ple_gate: ?Dense = null,
    ple_projection: ?Dense = null,
    ple_norm: ?Ref = null,
};

/// Python's lane_qmm.split_k: K slices fixed by the weight's shape, so every window width keeps one summation order.
pub fn referenceSplit(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The split each shape streams fastest at on one row (tf-gemma4-bench GEMMA_FUSED), not the reference's order.
pub fn splitK(geo: cfg.Geometry, n: usize, k: usize) usize {
    if (k == geo.hidden and n >= 2 * geo.intermediate) return 8; // the gate/up stack and the head: long chains at 1
    if (n == geo.hidden) return 4; // o and down: two waves of 256-thread groups at 8
    return referenceSplit(n, k);
}

/// 4/64: the reg kernel (codes in tensor-op registers, lane_qmm's words); other widths, the general unpack.
pub fn layout(geo: cfg.Geometry, x: Linear) lane.Layout {
    return .{ .n = x.n, .k = x.k, .format = x.format, .sk = splitK(geo, x.n, x.k), .precompute_sums = x.reg(), .reg = x.reg(), .y_stride = x.y_stride };
}

/// The same kernel with a layer's glue as its prologue (gemma4/fused.metal's TF_FUSE kinds) for one-row windows.
pub fn fusedLayout(geo: cfg.Geometry, x: Linear, fuse: u8) fused.Layout {
    const l = layout(geo, x);
    return .{ .projection = l, .source = ks.gemma4_fused, .defines = switch (fuse) {
        1 => "#define TF_FUSE 1\n",
        2 => "#define TF_FUSE 2\n",
        else => "#define TF_FUSE 3\n",
    }, .x_rows = if (fuse == 3) 0 else 1, .epilogue = fuse == 2, .threads = if (fuse == 3) 32 * l.sk else 256 };
}

pub const Weights = struct {
    /// The geometry's layers first; the rest unset.
    layers: [cfg.max_layers]Layer,
    head: Linear,
    /// The tied table's bits (gemma_embed's EB).
    embed_bits: u8,
    embed_w: Ref,
    embed_s: Ref,
    embed_b: Ref,
    final_norm: Ref,
    ple: ?Ple = null,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    /// Bytes the row path's aliases save: the per-member copies (`mlxCopy`) main also made without tensor units.
    aliased: u64 = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }
};

/// MLX affine words, scales and biases [x.n, x.k] at x's width in the lane layout (reg order at 4/5/6-bit g64).
pub fn packArrays(gpa: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, words: []const u32, scales: []const u16, biases: []const u16, x: Linear) !Linear {
    if (!device.tensorUnits()) return packRows(gpa, device, owned, words, scales, biases, x);
    const l = layout(geo, x);
    const wb = try device.buffer(l.weightWords() * 4, shared);
    errdefer wb.deinit();
    const sb = try device.buffer(l.metadataElements() * 2, shared);
    errdefer sb.deinit();
    try lane.pack(l, words, scales, biases, wb.slice(u32, l.weightWords()), sb.slice(u16, l.metadataElements()));
    if (l.reg) try lane.regOrder(device, &.{.{ .w = .{ .buf = wb }, .n = x.n, .k = x.k, .bits = x.format.bits }});
    try owned.ensureUnusedCapacity(gpa, 2);
    owned.appendAssumeCapacity(wb);
    owned.appendAssumeCapacity(sb);
    var out = x;
    out.w = wb;
    out.sb = sb;
    return out;
}

/// The row path's weights (a chip without tensor units): the checkpoint's arrays as stored, no lane layout.
pub fn packRows(gpa: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), words: []const u32, scales: []const u16, biases: []const u16, x: Linear) !Linear {
    const wb = try device.buffer(words.len * 4, shared);
    errdefer wb.deinit();
    const scb = try device.buffer(scales.len * 2, shared);
    errdefer scb.deinit();
    const bib = try device.buffer(biases.len * 2, shared);
    errdefer bib.deinit();
    @memcpy(wb.slice(u32, words.len), words);
    @memcpy(scb.slice(u16, scales.len), scales);
    @memcpy(bib.slice(u16, biases.len), biases);
    try owned.ensureUnusedCapacity(gpa, 3);
    owned.appendAssumeCapacity(wb);
    owned.appendAssumeCapacity(scb);
    owned.appendAssumeCapacity(bib);
    var out = x;
    out.w = wb;
    out.sb = scb;
    out.raw = rowp.Weights{ .w = wb, .scales = scb, .biases = bib, .n = x.n, .k = x.k, .group = x.format.group, .bits = x.format.bits, .sum = .f32 };
    return out;
}

/// The row path's projection (a chip without tensor units): the checkpoint's arrays as stored on the simdgroup tiles.
pub fn projectRow(ps: *Projections, device: mtl.Device, e: mtl.ComputeEncoder, raw: rowp.Weights, l: Linear, x: lane.Ref, y: mtl.Buffer, rows: usize) !void {
    return projectRowAt(ps, device, e, raw, l, x, y, 0, rows);
}

/// projectRow into y from byte offset `y_off` (the first row's place).
pub fn projectRowAt(ps: *Projections, device: mtl.Device, e: mtl.ComputeEncoder, raw: rowp.Weights, l: Linear, x: lane.Ref, y: mtl.Buffer, y_off: usize, rows: usize) !void {
    const p = try ps.simd(device);
    const mtx = rowp.simd_bits.Matrix{ .w = raw.w, .scales = raw.scales, .biases = raw.biases };
    const scalar = rows <= 2 and try ps.bitTwin(device, p, mtx, raw);
    const call = try rowp.simd_bits.call(p, raw.bits, raw.n, raw.k, rows, l.y_stride orelse l.n, scalar, 0);
    rowp.simd_bits.encode(e, call, mtx, x.buf, x.off, y, y_off + l.col * 2);
}

/// The lane pipelines, one per distinct (n, k, width, stride, prologue).
pub const Projections = struct {
    gpa: std.mem.Allocator,
    geo: cfg.Geometry,
    follow: ?mtl.ResidencySet.Follow = null,
    list: std.ArrayList(struct { x: Linear, p: lane.Projection }) = .empty,
    fused_list: std.ArrayList(struct { x: Linear, fuse: u8, p: fused.Projection }) = .empty,
    /// The row path's tiles (a chip without tensor units): the simdgroup matrices at any width.
    row_simd: ?rowp.simd_bits.Pipelines = null,
    /// Per [bits, n, k, whole]: whether the scalar twin is bit-equal to the tiles on this chip.
    bit_twins: std.AutoHashMapUnmanaged([4]usize, bool) = .empty,
    fn same(a: Linear, b: Linear) bool {
        return a.n == b.n and a.k == b.k and std.meta.eql(a.format, b.format) and a.y_stride == b.y_stride;
    }
    pub fn get(ps: *Projections, gpa: std.mem.Allocator, device: mtl.Device, x: Linear) !lane.Projection {
        for (ps.list.items) |v| if (same(v.x, x)) return v.p;
        const p = try lane.Projection.init(gpa, device, layout(ps.geo, x));
        errdefer p.deinit();
        try ps.list.append(gpa, .{ .x = x, .p = p });
        if (ps.follow) |f| mtl.resident.fields.apply(&p, f, true);
        return p;
    }
    pub fn getFused(ps: *Projections, gpa: std.mem.Allocator, device: mtl.Device, x: Linear, fuse: u8) !fused.Projection {
        for (ps.fused_list.items) |v| if (same(v.x, x) and v.fuse == fuse) return v.p;
        const p = try fused.Projection.init(gpa, device, fusedLayout(ps.geo, x, fuse));
        errdefer p.deinit();
        try ps.fused_list.append(gpa, .{ .x = x, .fuse = fuse, .p = p });
        return p;
    }
    /// The row path's tiles, built once (a chip without tensor units).
    pub fn simd(ps: *Projections, device: mtl.Device) !*rowp.simd_bits.Pipelines {
        if (ps.row_simd == null) ps.row_simd = try rowp.simd_bits.Pipelines.load(device);
        return &ps.row_simd.?;
    }

    /// Checks the scalar twin against the tiles once per width, shape and K split key on this chip.
    fn bitTwin(ps: *Projections, device: mtl.Device, p: *rowp.simd_bits.Pipelines, m: rowp.simd_bits.Matrix, raw: rowp.Weights) !bool {
        const key = [4]usize{ raw.bits, raw.n, raw.k, 0 };
        if (ps.bit_twins.get(key)) |ok| return ok;
        const ok = try rowp.simd_bits.twinExact(p, device, raw.bits, m, raw.n, raw.k, 0);
        if (!ok) std.log.info("gemma4: {d}-bit {d}x{d} rows take simdgroup tiles at every width (the scalar twin's bits differ on this chip)", .{ raw.bits, raw.n, raw.k });
        try ps.bit_twins.put(ps.gpa, key, ok);
        return ok;
    }

    pub fn deinit(ps: *Projections, gpa: std.mem.Allocator) void {
        ps.setResident(null);
        if (ps.row_simd) |*p| p.deinit();
        for (ps.list.items) |v| v.p.deinit();
        ps.list.deinit(gpa);
        for (ps.fused_list.items) |v| v.p.deinit();
        ps.fused_list.deinit(gpa);
        ps.bit_twins.deinit(ps.gpa);
    }

    pub fn appendBuffers(ps: *const Projections, gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        for (ps.list.items) |*v| try mtl.resident.fields.append(gpa, list, &v.p);
    }

    pub fn setResident(ps: *Projections, hook: ?mtl.ResidencySet.Follow) void {
        for (ps.list.items) |*v| {
            if (ps.follow) |f| mtl.resident.fields.apply(&v.p, f, false);
            if (hook) |f| mtl.resident.fields.apply(&v.p, f, true);
        }
        ps.follow = hook;
    }
};

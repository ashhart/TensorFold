//! Gemma 4 dense weights: each linear stack packed once for the core lane projection, norms read in place.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
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

    /// The reg kernel's width, which reads the group sums its input's kernel wrote; others sum their input themselves.
    pub fn reg(x: Linear) bool {
        return std.meta.eql(x.format, fast);
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

pub const Layer = struct {
    /// q, k, v (k again on global layers, which have none), o, gate, up, down for prompt chunks.
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
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }
};

/// A linear's three tensors copied into one aligned buffer: checkpoint tensors can sit at any byte offset in a shard.
fn mlxCopy(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), prefix: []const u8, member: []const u8, width: cfg.widths.Width) !Mlx {
    var name: [160]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, member }));
    const k = try storedK(c, prefix, member, width);
    const s = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, member }));
    const b = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, member }));
    const sb_at = std.mem.alignForward(usize, w.bytes, 256);
    const bb_at = sb_at + std.mem.alignForward(usize, s.bytes, 256);
    const buf = try device.buffer(bb_at + b.bytes, shared);
    errdefer buf.deinit();
    const dst = buf.slice(u8, bb_at + b.bytes);
    @memcpy(dst[0..w.bytes], w.host(u8)[0..w.bytes]);
    @memcpy(dst[sb_at..][0..s.bytes], s.host(u8)[0..s.bytes]);
    @memcpy(dst[bb_at..][0..b.bytes], b.host(u8)[0..b.bytes]);
    try owned.append(gpa, buf);
    return .{ .w = .{ .buf = buf }, .s = .{ .buf = buf, .off = sb_at }, .b = .{ .buf = buf, .off = bb_at }, .n = @intCast(w.shape[0]), .k = @intCast(k), .bits = width.bits };
}

/// A member's K at its config width (words [N, K * bits / 32], tables [N, K / group]); other storage refused.
fn storedK(c: *const ckpt.Checkpoint, prefix: []const u8, member: []const u8, width: cfg.widths.Width) !usize {
    var name: [160]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, member }));
    const s = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, member }));
    const b = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, member }));
    const ok = w.dtype == .u32 and w.rank == 2 and s.rank == 2 and b.rank == 2 and s.dtype == .bf16 and b.dtype == .bf16 and
        s.shape[0] == w.shape[0] and b.shape[0] == w.shape[0] and s.shape[1] == b.shape[1] and
        w.count() == w.shape[0] * w.shape[1] and s.count() == s.shape[0] * s.shape[1] and b.count() == s.count();
    const k = s.shape[1] * width.group;
    if (!ok or w.shape[1] * 32 != k * width.bits) {
        std.log.warn("gemma4: {s}{s} is not stored at its config width, {d}-bit in groups of {d}", .{ prefix, member, width.bits, width.group });
        return error.BadGemmaWeight;
    }
    return k;
}

fn ref(c: *const ckpt.Checkpoint, name: []const u8) !Ref {
    const t = try c.get(name);
    return .{ .buf = t.buffer, .off = t.offset };
}

/// Members of one width stacked along N into the lane layout; `interleave`: two members in alternate 16-row blocks.
fn pack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8, width: cfg.widths.Width, interleave: bool) !Linear {
    var name: [160]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (members) |m| {
        const mk = try storedK(c, prefix, m, width);
        if (k != 0 and k != mk) return error.BadGemmaWeight;
        n += (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }))).shape[0];
        k = mk;
    }
    const row_words = k * width.bits / 32;
    const groups = k / width.group;
    const words = try gpa.alloc(u32, n * row_words);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    defer gpa.free(biases);
    if (interleave and (members.len != 2 or n % 32 != 0)) return error.BadGemmaWeight;
    var row: usize = 0;
    for (members, 0..) |m, mi| {
        const w = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }))).host(u32);
        const s = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, m }))).host(u16);
        const b = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, m }))).host(u16);
        const rows = w.len / row_words;
        if (interleave and rows != n / 2) return error.BadGemmaWeight;
        for (0..rows) |r| {
            const dst = if (interleave) 32 * (r / 16) + 16 * mi + r % 16 else row + r;
            @memcpy(words[dst * row_words ..][0..row_words], w[r * row_words ..][0..row_words]);
            @memcpy(scales[dst * groups ..][0..groups], s[r * groups ..][0..groups]);
            @memcpy(biases[dst * groups ..][0..groups], b[r * groups ..][0..groups]);
        }
        row += rows;
    }
    return packArrays(gpa, device, owned, geo, words, scales, biases, .{ .w = undefined, .sb = undefined, .n = n, .k = k, .format = .{ .bits = width.bits, .group = width.group } });
}

/// A q|k|v or gate|up stack: each run of members of one width packed as one linear writing its columns of the rows.
fn stack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8, ws: []const cfg.widths.Width, glu: bool) !Stack {
    var s: Stack = .{};
    var first: usize = 0;
    while (first < members.len) {
        var end = first + 1;
        while (end < members.len and std.meta.eql(ws[end], ws[first])) end += 1;
        const whole = first == 0 and end == members.len;
        var x = try pack(gpa, device, c, owned, geo, prefix, members[first..end], ws[first], glu and whole);
        x.col = s.n;
        s.n += x.n;
        s.runs[s.count] = x;
        s.count += 1;
        first = end;
    }
    s.interleaved = glu and s.count == 1;
    if (s.count > 1) for (s.runs[0..s.count]) |*x| {
        x.y_stride = s.n; // each run writes its columns of the stack's rows (the lane words do not depend on it)
    };
    return s;
}

/// MLX affine words, scales and biases [x.n, x.k] at x's width in the lane layout (reg order at 4/64).
pub fn packArrays(gpa: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, words: []const u32, scales: []const u16, biases: []const u16, x: Linear) !Linear {
    if (!device.tensorUnits()) return packRows(gpa, device, owned, words, scales, biases, x);
    const l = layout(geo, x);
    const wb = try device.buffer(l.weightWords() * 4, shared);
    errdefer wb.deinit();
    const sb = try device.buffer(l.metadataElements() * 2, shared);
    errdefer sb.deinit();
    try lane.pack(l, words, scales, biases, wb.slice(u32, l.weightWords()), sb.slice(u16, l.metadataElements()));
    if (l.reg) try lane.regOrder(device, &.{.{ .w = .{ .buf = wb }, .n = x.n, .k = x.k }});
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

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, geo: cfg.Geometry, ws: *const cfg.widths.Widths) !Weights {
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    var p: [96]u8 = undefined;
    var name: [160]u8 = undefined;
    for (w.layers[0..geo.layers], 0..) |*l, i| {
        const pre = try std.fmt.bufPrint(&p, "language_model.model.layers.{d}.", .{i});
        const lw = ws.layers[i];
        const qkv: []const []const u8 = if (cfg.global(i)) &.{ "self_attn.q_proj", "self_attn.k_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
        l.qkv = try stack(gpa, device, c, &w.owned, geo, pre, qkv, lw[0..qkv.len], false);
        l.o = try pack(gpa, device, c, &w.owned, geo, pre, &.{"self_attn.o_proj"}, lw[3], false);
        l.gate_up = try stack(gpa, device, c, &w.owned, geo, pre, &.{ "mlp.gate_proj", "mlp.up_proj" }, lw[4..6], true);
        l.down = try pack(gpa, device, c, &w.owned, geo, pre, &.{"mlp.down_proj"}, lw[6], false);
        for (&l.mlx, cfg.widths.module_names, lw) |*x, member, width| {
            // global layers have no v_proj: k stands in, never read
            x.* = if (cfg.global(i) and std.mem.eql(u8, member, "self_attn.v_proj")) l.mlx[1] else try mlxCopy(gpa, device, c, &w.owned, pre, member, width);
        }
        inline for (.{ "q_norm", "k_norm", "input_norm", "post_attention", "pre_feedforward", "post_feedforward", "scalar" }, .{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" }) |field, t|
            @field(l, field) = try ref(c, try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t }));
    }
    w.head = try pack(gpa, device, c, &w.owned, geo, "language_model.model.", &.{"embed_tokens"}, ws.embed, false);
    w.embed_bits = ws.embed.bits;
    w.embed_w = try ref(c, "language_model.model.embed_tokens.weight");
    w.embed_s = try ref(c, "language_model.model.embed_tokens.scales");
    w.embed_b = try ref(c, "language_model.model.embed_tokens.biases");
    w.final_norm = try ref(c, "language_model.model.norm.weight");
    return w;
}

/// The row path's projection (a chip without tensor units): the checkpoint's arrays as stored on the simdgroup tiles.
pub fn projectRow(ps: *Projections, device: mtl.Device, e: mtl.ComputeEncoder, raw: rowp.Weights, l: Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize) !void {
    const p = try ps.simd(device);
    const mtx = rowp.simd_bits.Matrix{ .w = raw.w, .scales = raw.scales, .biases = raw.biases };
    const call = try rowp.simd_bits.call(p, raw.bits, raw.n, raw.k, rows, l.y_stride orelse l.n, false);
    rowp.simd_bits.encode(e, call, mtx, x, 0, y, l.col * 2);
}

/// The lane pipelines, one per distinct (n, k, width, stride, prologue).
pub const Projections = struct {
    geo: cfg.Geometry,
    follow: ?mtl.ResidencySet.Follow = null,
    list: std.ArrayList(struct { x: Linear, p: lane.Projection }) = .empty,
    fused_list: std.ArrayList(struct { x: Linear, fuse: u8, p: fused.Projection }) = .empty,
    /// The row path's tiles (a chip without tensor units): the simdgroup matrices at any width.
    row_simd: ?rowp.simd_bits.Pipelines = null,
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

    pub fn deinit(ps: *Projections, gpa: std.mem.Allocator) void {
        ps.setResident(null);
        if (ps.row_simd) |*p| p.deinit();
        for (ps.list.items) |v| v.p.deinit();
        ps.list.deinit(gpa);
        for (ps.fused_list.items) |v| v.p.deinit();
        ps.fused_list.deinit(gpa);
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

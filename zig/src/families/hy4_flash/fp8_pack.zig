//! W8A16 CUDA weight packing for Hy4-flash: qmmf BF16/FP8G projection shares and TP=1/2/4 FP8 routed experts.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const weights = @import("fp8_weights.zig");
const partition = @import("fp8_partition.zig");

const Tensor = core.checkpoint.Tensor;
const Share = partition.Share;
const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;
const fx8 = cuda.experts.Fp8Experts;

/// A projection share packed for `cuda.qmmf`: bf16 tiles, or e4m3 tiles with fp32 scale tiles.
pub const Packed = struct {
    mode: qmmf.Mode,
    codes: []u8,
    scales: []f32 = &.{},
    n: usize,
    k: usize,

    pub fn deinit(p: *Packed, gpa: std.mem.Allocator) void {
        gpa.free(p.codes);
        gpa.free(p.scales);
        p.* = undefined;
    }
};

/// Rows `rows` of a stored projection; multiple parts stack by rows into one packed `qmmf` projection.
pub const Part = struct { proj: weights.Proj, rows: Share };

fn inputs(p: weights.Proj) usize {
    return switch (p) {
        .bf16 => |t| t.dim(1),
        .fp8 => |f| f.weight.dim(1),
    };
}

fn checkProj(p: weights.Proj) !void {
    switch (p) {
        .bf16 => |t| {
            if (t.rank != 2 or t.dtype != .bf16) return error.UnexpectedTensor;
            const need = std.math.mul(usize, t.dim(0), t.dim(1)) catch return error.UnexpectedTensor;
            const bytes = std.math.mul(usize, need, 2) catch return error.UnexpectedTensor;
            if (t.bytes.len != bytes) return error.UnexpectedTensor;
        },
        .fp8 => |f| {
            const w = f.weight;
            const s = f.scale_inv;
            if (w.rank != 2 or w.dtype != .f8_e4m3 or s.rank != 2 or s.dtype != .bf16) return error.UnexpectedTensor;
            const w_bytes = std.math.mul(usize, w.dim(0), w.dim(1)) catch return error.UnexpectedTensor;
            const nb = core.quant.fp8.blocks(w.dim(0));
            const kb = core.quant.fp8.blocks(w.dim(1));
            const s_elems = std.math.mul(usize, nb, kb) catch return error.UnexpectedTensor;
            const s_bytes = std.math.mul(usize, s_elems, 2) catch return error.UnexpectedTensor;
            if (w.bytes.len != w_bytes or s.dim(0) != nb or s.dim(1) != kb or s.bytes.len != s_bytes) {
                return error.UnexpectedTensor;
            }
        },
    }
}

fn widened(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    errdefer gpa.free(out);
    try core.quant.fp8.widen(t, out, true);
    return out;
}

/// Pack `parts` stacked by rows and sliced to `cols` (whole 64-input groups) without requantizing.
pub fn pack(gpa: std.mem.Allocator, parts: []const Part, cols: Share) !Packed {
    if (parts.len == 0) return error.UnexpectedTensor;
    const mode: qmmf.Mode = if (parts[0].proj == .bf16) .bf16 else .fp8g;
    try checkProj(parts[0].proj);
    const width = inputs(parts[0].proj);
    var n: usize = 0;
    for (parts) |q| {
        try checkProj(q.proj);
        const same = (q.proj == .bf16) == (mode == .bf16);
        if (!same or inputs(q.proj) != width) return error.UnexpectedTensor;
        const end = std.math.add(usize, q.rows.lo, q.rows.len) catch return error.SplitOutOfRange;
        if (q.rows.len == 0 or end > q.proj.rows()) return error.SplitOutOfRange;
        if (mode == .fp8g and parts.len > 1 and q.rows.len % 64 != 0) return error.SplitNotOnBlocks;
        n = std.math.add(usize, n, q.rows.len) catch return error.SplitOutOfRange;
    }
    const col_end = std.math.add(usize, cols.lo, cols.len) catch return error.SplitOutOfRange;
    if (cols.len == 0 or col_end > width) return error.SplitOutOfRange;
    if (cols.lo % qmmf.group != 0 or cols.len % qmmf.group != 0) return error.SplitNotOnBlocks;

    const k = cols.len;
    const npad = fp8.padded(n);
    const elem: usize = if (mode == .bf16) 2 else 1;
    const rows = try gpa.alloc(u8, n * k * elem);
    defer gpa.free(rows);
    var at: usize = 0;
    for (parts) |q| {
        const bytes = switch (q.proj) {
            .bf16 => |t| t.bytes,
            .fp8 => |f| f.weight.bytes,
        };
        for (q.rows.lo..q.rows.lo + q.rows.len) |r| {
            @memcpy(rows[at * k * elem ..][0 .. k * elem], bytes[(r * width + cols.lo) * elem ..][0 .. k * elem]);
            at += 1;
        }
    }
    const codes = try gpa.alloc(u8, npad * k * elem);
    errdefer gpa.free(codes);
    if (mode == .bf16) {
        qmmf.packBf16(@alignCast(std.mem.bytesAsSlice(u16, codes)), @alignCast(std.mem.bytesAsSlice(u16, rows)), n, k);
        return .{ .mode = .bf16, .codes = codes, .n = n, .k = k };
    }
    fp8.packCodes(codes, rows, n, k);
    const kg = k / fp8.group;
    const scales = try gpa.alloc(f32, npad * kg);
    errdefer gpa.free(scales);
    @memset(scales, 1.0);
    var row: usize = 0;
    for (parts) |q| {
        const f = q.proj.fp8;
        const inv = try widened(gpa, f.scale_inv);
        defer gpa.free(inv);
        const one = if (parts.len == 1) scales else try gpa.alloc(f32, fp8.padded(q.rows.len) * kg);
        defer if (parts.len > 1) gpa.free(one);
        fp8.packScalesAt(one, inv, f.weight.dim(0), f.weight.dim(1), q.rows.lo, cols.lo, q.rows.len, k);
        if (parts.len > 1) {
            const tiles = q.rows.len / 64;
            @memcpy(scales[(row / 64) * kg * 64 ..][0 .. tiles * kg * 64], one[0 .. tiles * kg * 64]);
        }
        row += q.rows.len;
    }
    return .{ .mode = .fp8g, .codes = codes, .scales = scales, .n = n, .k = k };
}

/// Horizontally concatenate two BF16 tensors `a[rows, a_cols]` and `b[rows, b_cols]` and pack for `qmmf`.
pub fn packHConcat(gpa: std.mem.Allocator, a: Tensor, a_cols: Share, b: Tensor, b_cols: Share, rows: Share) !Packed {
    try checkProj(.{ .bf16 = a });
    try checkProj(.{ .bf16 = b });
    const n = rows.len;
    const k = std.math.add(usize, a_cols.len, b_cols.len) catch return error.SplitOutOfRange;
    if (n == 0 or a_cols.len == 0 or b_cols.len == 0) return error.SplitOutOfRange;
    const r_end = std.math.add(usize, rows.lo, n) catch return error.SplitOutOfRange;
    const a_end = std.math.add(usize, a_cols.lo, a_cols.len) catch return error.SplitOutOfRange;
    const b_end = std.math.add(usize, b_cols.lo, b_cols.len) catch return error.SplitOutOfRange;
    if (r_end > a.dim(0) or r_end > b.dim(0) or a_end > a.dim(1) or b_end > b.dim(1)) return error.SplitOutOfRange;
    if (k % qmmf.group != 0) return error.SplitNotOnBlocks;

    const aw = a.dim(1);
    const bw = b.dim(1);
    const npad = fp8.padded(n);
    const raw = try gpa.alloc(u16, n * k);
    defer gpa.free(raw);
    for (0..n) |i| {
        const r = rows.lo + i;
        @memcpy(std.mem.sliceAsBytes(raw[i * k ..][0..a_cols.len]), a.bytes[(r * aw + a_cols.lo) * 2 ..][0 .. a_cols.len * 2]);
        @memcpy(std.mem.sliceAsBytes(raw[i * k + a_cols.len ..][0..b_cols.len]), b.bytes[(r * bw + b_cols.lo) * 2 ..][0 .. b_cols.len * 2]);
    }
    const codes = try gpa.alloc(u8, npad * k * 2);
    errdefer gpa.free(codes);
    qmmf.packBf16(@alignCast(std.mem.bytesAsSlice(u16, codes)), raw, n, k);
    return .{ .mode = .bf16, .codes = codes, .n = n, .k = k };
}

/// One layer's routed FP8 experts sliced to a rank's intermediate share for `cuda.experts.Fp8Experts`.
pub const HostExperts = struct {
    up: []u32, // [E][w/32][L/32][2][32][2][4]: gate and up interleaved
    down: []u32, // [E][L/32][w/32][1][32][2][4]
    up_scale: []f32, // [E][2][w/128][L/128] (gate then up)
    down_scale: []f32, // [E][1][L/128][w/128]

    pub fn deinit(h: *HostExperts, gpa: std.mem.Allocator) void {
        gpa.free(h.up);
        gpa.free(h.down);
        gpa.free(h.up_scale);
        gpa.free(h.down_scale);
        h.* = undefined;
    }
};

/// Slice stacked `[E, 2w, L]` `gate_up` and `[E, L, w]` `down` FP8 experts by `share` and widen BF16 block scales.
pub fn hostExperts(gpa: std.mem.Allocator, m: weights.Moe, share: Share) !HostExperts {
    if (m.gate_up.rank != 3 or m.gate_up.dtype != .f8_e4m3 or m.gate_up_s.rank != 3 or m.gate_up_s.dtype != .bf16) {
        return error.UnexpectedTensor;
    }
    if (m.down.rank != 3 or m.down.dtype != .f8_e4m3 or m.down_s.rank != 3 or m.down_s.dtype != .bf16) {
        return error.UnexpectedTensor;
    }
    const e = m.gate_up.dim(0);
    const two = m.gate_up.dim(1);
    const l = m.gate_up.dim(2);
    const b = fp8.block;
    if (e == 0 or two == 0 or two % 2 != 0 or l == 0 or l % b != 0) return error.UnexpectedTensor;
    const full = two / 2;
    if (full % b != 0 or m.down.dim(0) != e or m.down.dim(1) != l or m.down.dim(2) != full) {
        return error.UnexpectedTensor;
    }
    const lb = l / b;
    const fb = full / b;
    if (m.gate_up_s.dim(0) != e or m.gate_up_s.dim(1) != 2 * fb or m.gate_up_s.dim(2) != lb) {
        return error.UnexpectedTensor;
    }
    if (m.down_s.dim(0) != e or m.down_s.dim(1) != lb or m.down_s.dim(2) != fb) return error.UnexpectedTensor;
    if (m.gate_up.bytes.len != e * two * l or m.down.bytes.len != e * l * full) return error.UnexpectedTensor;

    const w = share.len;
    const end = std.math.add(usize, share.lo, w) catch return error.SplitOutOfRange;
    if (w == 0 or end > full) return error.SplitOutOfRange;
    if (share.lo % b != 0 or w % b != 0) return error.SplitNotOnBlocks;

    const gu_s = try widened(gpa, m.gate_up_s);
    defer gpa.free(gu_s);
    const dn_s = try widened(gpa, m.down_s);
    defer gpa.free(dn_s);

    var h: HostExperts = .{ .up = &.{}, .down = &.{}, .up_scale = &.{}, .down_scale = &.{} };
    errdefer h.deinit(gpa);
    h.up = try gpa.alloc(u32, e * w * l / 2);
    h.down = try gpa.alloc(u32, e * w * l / 4);
    h.up_scale = try gpa.alloc(f32, e * 2 * (w / b) * lb);
    h.down_scale = try gpa.alloc(f32, e * lb * (w / b));

    const cols = try gpa.alloc(u8, l * w);
    defer gpa.free(cols);
    const wb = w / b;
    const sb = share.lo / b;
    for (0..e) |x| {
        const gu = m.gate_up.bytes[x * two * l ..][0 .. two * l];
        try fx8.packGateUp(gpa, h.up[x * w * l / 2 ..][0 .. w * l / 2], gu[share.lo * l ..][0 .. w * l], gu[(full + share.lo) * l ..][0 .. w * l], w, l);
        const dn = m.down.bytes[x * l * full ..][0 .. l * full];
        for (0..l) |r| @memcpy(cols[r * w ..][0..w], dn[r * full + share.lo ..][0..w]);
        fx8.packOne(h.down[x * w * l / 4 ..][0 .. w * l / 4], cols, l, w);
        const us = h.up_scale[x * 2 * wb * lb ..];
        for (0..2) |half| for (0..wb) |rb| for (0..lb) |cb| {
            us[(half * wb + rb) * lb + cb] = gu_s[(x * 2 * fb + half * fb + sb + rb) * lb + cb];
        };
        const ds = h.down_scale[x * lb * wb ..];
        for (0..lb) |rb| for (0..wb) |cb| {
            ds[rb * wb + cb] = dn_s[(x * lb + rb) * fb + sb + cb];
        };
    }
    return h;
}

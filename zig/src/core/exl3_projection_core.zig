//! The EXL3 projection's views, matrix descriptors and host argument helpers.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
pub const checkpoint = @import("exl3_load.zig");
pub const load = @import("exl3_metal_load.zig");
pub const format = @import("exl3_format.zig");
const decoder = @import("exl3_projection_decode.zig");
const knobs_mod = @import("exl3_knobs.zig");

pub const Knobs = knobs_mod.Knobs;
pub var fma_log_count: usize = 0;
/// The family host resolves this once at load. The defaults are the shipped bodies.
pub var knobs: Knobs = .{};

pub const View = struct {
    buffer: mtl.Buffer,
    offset: usize = 0,
};
pub const ScaleFormat = enum { f16, packed_signs };
pub const Output = enum { bf16, f32 };

/// A prompt-row int8 sidecar: the pre-quantized dense weight in the fp8 body's 16-k x 64-n block order.
pub const Int8 = struct {
    weight: View,
    scales: View,
    k: usize,
    n: usize,
    pub fn deinit(i: *const Int8) void {
        i.weight.buffer.deinit();
        i.scales.buffer.deinit();
    }
};

/// W[K,N] is stored as trellis[K/16,N/16,8*width.halves] int16, not a conventional [N,K] matrix.
pub const Matrix = struct {
    trellis: View,
    suh: View,
    svh: View,
    bias: ?View = null,
    k: usize,
    n: usize,
    width: format.Width,
    codebook: format.Codebook = .mul1,
    suh_format: ScaleFormat = .f16,
    svh_format: ScaleFormat = .f16,
    /// Byte strides between tiles; zero derives the checkpoint's stored K-major layout.
    tile_k_stride: usize = 0,
    tile_n_stride: usize = 0,
    /// The prompt-row int8 twin, when a sidecar is loaded.
    int8: ?Int8 = null,
    /// Whether that twin was packed pre-rotated (both Hadamards baked in).
    int8_direct: bool = false,

    pub fn validate(m: Matrix) !void {
        try shape(m.k, m.n);
        if (m.width.halves < 2 or m.width.halves > 16 or
            (m.width.isHalf() and (m.codebook != .mul1 or m.width.halves > 7))) return error.UnsupportedExl3Width;
        const row_bytes = try std.math.mul(usize, m.n / 16, m.nStride());
        const k_stride = if (m.tile_k_stride == 0) row_bytes else m.tile_k_stride;
        if (m.nStride() < m.tileBytes() or m.nStride() % 2 != 0 or
            k_stride < row_bytes or k_stride % 2 != 0)
            return error.InvalidExl3Stride;
    }

    pub fn tileBytes(m: Matrix) usize {
        return m.width.tileWords() * 2;
    }
    pub fn nStride(m: Matrix) usize {
        return if (m.tile_n_stride == 0) m.tileBytes() else m.tile_n_stride;
    }
    pub fn kStride(m: Matrix) !usize {
        return if (m.tile_k_stride == 0) std.math.mul(usize, m.n / 16, m.nStride()) else m.tile_k_stride;
    }
    pub fn storageBytes(m: Matrix) !usize {
        try m.validate();
        const last_k = try std.math.mul(usize, m.k / 16 - 1, try m.kStride());
        const last_n = try std.math.mul(usize, m.n / 16 - 1, m.nStride());
        return std.math.add(usize, try std.math.add(usize, last_k, last_n), m.tileBytes());
    }
    pub fn checkStorage(m: Matrix) !usize {
        const bytes = try m.storageBytes();
        try check(m.trellis, bytes, 2);
        try check(m.suh, m.k * 2 / (if (m.suh_format == .packed_signs) @as(usize, 16) else 1), 2);
        try check(m.svh, m.n * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1), 2);
        if (m.bias) |b| try check(b, m.n * 2, 2);
        return bytes;
    }
};

/// Individual expert filtering reads u32 IDs on GPU; repeat top-k slots share one input row.
pub const Routes = struct { ids: View, expert_id: u32 = 0, repeat: usize = 1 };

/// Strides are elements, X is BF16, IDs are u32[count]. Gather/scatter IDs must be in bounds and outputs unique.
pub const Rows = struct {
    count: usize,
    x_stride: usize,
    y_stride: usize,
    output: Output = .bf16,
    input_ids: ?View = null,
    output_ids: ?View = null,
    /// Required with a corresponding ID map; callers guarantee IDs below these capacities and unique output IDs.
    input_rows: usize = 0,
    output_rows: usize = 0,
    routes: ?Routes = null,
    /// Caller-prepared eight-row tiles, same expert per tile; maxInt(u32) padding must follow valid rows.
    grouped: bool = false,
    /// Row tile the routed groups were padded for (0: the 64-row default); the prompt tile follows it.
    tile_rows: u32 = 0,
    /// The caller wrote this dispatch's rotated input (row-major, k halfs a row) into the scratch base.
    prerotated: bool = false,
    /// Leave the projection's f32 sums in the scratch for a caller that runs the epilogue itself.
    no_epilogue: bool = false,
    /// The caller asks for the tensor-unit dense tile from EIGHT rows up (the shipped gate is 64).
    dense8: bool = false,
};

/// Metal 3 / tier-2 argument-buffer ABI; borrowed pointers remain resident until the encoded work completes.
pub const ExpertDescriptor = extern struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    bias: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
    halves: u32,
    codebook: u32,
    suh_signs: u32,
    svh_signs: u32,
    has_bias: u32,
    reserved: u32 = 0,
};

/// Populate a caller-owned shared descriptor table once; no weight copying, packing or dequantization.
pub fn descriptor(m: Matrix) !ExpertDescriptor {
    _ = try m.checkStorage();
    return .{
        .trellis = try address(m.trellis),
        .suh = try address(m.suh),
        .svh = try address(m.svh),
        .bias = if (m.bias) |v| try address(v) else 0,
        .tile_k_stride = try m.kStride(),
        .tile_n_stride = m.nStride(),
        .halves = m.width.halves,
        .codebook = @intCast(@backingInt(m.codebook)),
        .suh_signs = @intFromBool(m.suh_format == .packed_signs),
        .svh_signs = @intFromBool(m.svh_format == .packed_signs),
        .has_bias = @intFromBool(m.bias != null),
        // The entry's own output width.
        .reserved = @intCast(m.n),
    };
}

pub fn address(v: View) !u64 {
    return std.math.add(u64, v.buffer.gpuAddress(), v.offset);
}

/// Call once per encoder for indirectly addressed matrices, unless the caller declares them via residency sets.
pub fn useMatrix(e: mtl.ComputeEncoder, m: Matrix) void {
    e.useResource(m.trellis.buffer, mtl.ResourceUsage.read);
    e.useResource(m.suh.buffer, mtl.ResourceUsage.read);
    e.useResource(m.svh.buffer, mtl.ResourceUsage.read);
    if (m.bias) |b| e.useResource(b.buffer, mtl.ResourceUsage.read);
}

/// Common K,N; descriptor weights must remain resident and disjoint from writable views through completion.
pub const Experts = struct { descriptors: View, count: usize, k: usize, n: usize };

/// The three codebooks as one table, filled from the format's own reference decoder.
pub fn codebookTable(device: mtl.Device) ?mtl.Buffer {
    const entries = 3 * 65536;
    const buffer = device.buffer(entries * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked) catch return null;
    const out: [*]u16 = @ptrCast(@alignCast(buffer.contents()));
    for (0..3) |cb| {
        const codebook: format.Codebook = @fromBackingInt(@intCast(cb));
        for (0..65536) |state| {
            out[cb * 65536 + state] = @bitCast(format.codebookValue(codebook, @intCast(state)));
        }
    }
    std.log.info("exl3: codebook table {d} entries, first {d} {d} {d}", .{ entries, out[0], out[1], out[65536] });
    return buffer;
}

/// The one-row table body.
pub fn tableDecodeOn() bool {
    return knobs.table_decode;
}

/// The codebook read from a 1024-entry threadgroup LUT in the register-only routed decode body.
pub fn fmaLutOn() bool {
    return knobs.fma_lut;
}

/// The static state selection in the register-only routed decode body.
pub fn fmaStaticOn() bool {
    return knobs.fma_static;
}

/// The register-only body on the dense one-row path.
pub fn fmaDenseOn() bool {
    return knobs.fma_dense;
}

/// The dense prompt attention projections on the double-buffered 128x64 body.
pub fn denseDb() bool {
    return knobs.dense_db;
}

/// The routed prompt tile skips the grouping's padded rows.
pub fn skipPadOn() bool {
    return knobs.skip_pad;
}

/// The routed prompt tile takes the funnel path as its only path.
pub fn funnelOnlyOn() bool {
    return knobs.funnel_only;
}

/// The 128-row routed tile with the thirty-two-row warp.
pub fn warp32On() bool {
    return knobs.warp32;
}

/// The routed prompt tile's funnel read.
pub fn funnelPrefillOn() bool {
    return knobs.funnel_prefill;
}

/// The one-row split count pinned for the whole binary; null keeps the per-shape default.
pub fn splitOverride() ?u32 {
    return knobs.splits_override;
}

/// Split-K count for the routed-expert decode matmuls.
pub fn moeSplitCount() usize {
    return knobs.moe_splits;
}

pub const Args = extern struct {
    k: u32,
    n: u32,
    rows: u32,
    halves: u32,
    codebook: u32,
    su_signs: u32,
    sv_signs: u32,
    has_bias: u32,
    output_f32: u32,
    input_ids: u32,
    output_ids: u32,
    routed: u32,
    expert_id: u32,
    repeat: u32,
    indexed: u32 = 0,
    experts: u32 = 0,
    splits: u32 = 1,
    x_stride: u64,
    y_stride: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
};

pub fn shape(k: usize, n: usize) !void {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or
        k > std.math.maxInt(u32) or n > std.math.maxInt(u32)) return error.UnsupportedExl3Shape;
}
pub fn rotatedBytes(k: usize, rows: usize) !usize {
    const bytes = try std.math.mul(usize, try std.math.mul(usize, k, rows), 2);
    return (try std.math.add(usize, bytes, 15)) & ~@as(usize, 15);
}

/// Caller allocates this once for its largest window; scratch never holds a dequantized full weight matrix.
pub fn scratchBytes(k: usize, n: usize, rows: usize) !usize {
    try shape(k, n);
    if (rows == 0 or rows > std.math.maxInt(u32) - 7) return error.InvalidExl3Rows;
    var bytes = std.math.add(usize, try rotatedBytes(k, rows), try std.math.mul(usize, try std.math.mul(usize, n, rows), 4)) catch return error.UnsupportedExl3Shape;
    // The split dispatch's partials, one partial row a split a row of the call.
    if (rows <= 8) bytes = std.math.add(usize, bytes, try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, one_row_splits, rows), n), 4)) catch return error.UnsupportedExl3Shape;
    // A decode tile of a few routed rows can split its k range too: room for every split's rows by n.
    if (rows <= moe_split_rows) bytes = std.math.add(usize, bytes, try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, moeSplitCount(), @max(rows, 8)), n), 4)) catch return error.UnsupportedExl3Shape;
    return bytes;
}

/// Split-K count for a one-row projection: one threadgroup a column tile leaves the GPU idle.
pub const one_row_splits: usize = 16;

/// Routed rows the decode tile can carry in one call with its k range split: eight shipped rows.
pub const moe_split_rows: usize = 64;

pub fn check(v: View, bytes: usize, alignment: usize) !void {
    const len = v.buffer.length();
    if (v.offset % alignment != 0 or v.offset > len or bytes > len - v.offset) return error.InvalidExl3View;
}

pub fn disjoint(a: View, a_bytes: usize, b: View, b_bytes: usize) !void {
    if (a.buffer.id == b.buffer.id and a.offset < b.offset + b_bytes and b.offset < a.offset + a_bytes)
        return error.AliasedExl3View;
}

pub fn readonly(v: View, bytes: usize, y: View, y_bytes: usize, scratch: View, scratch_bytes: usize) !void {
    try disjoint(v, bytes, y, y_bytes);
    try disjoint(v, bytes, scratch, scratch_bytes);
}
pub fn rowsArgs(k: usize, n: usize, r: Rows, x: View, y: View, scratch: View) !Args {
    const scratch_size = try scratchBytes(k, n, r.count);
    if (r.x_stride < k or r.y_stride < n) return error.InvalidExl3Stride;
    const repeat = if (r.routes) |v| v.repeat else 1;
    if (repeat == 0 or repeat > std.math.maxInt(u32) or r.count % repeat != 0) return error.InvalidExl3Repeat;
    if ((r.input_ids != null and r.input_rows == 0) or
        (r.output_ids != null and r.output_rows == 0)) return error.InvalidExl3Capacity;
    const xrows = if (r.input_ids != null) r.input_rows else r.count / repeat;
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const xend = try std.math.add(usize, try std.math.mul(usize, xrows - 1, r.x_stride), k);
    const yend = try std.math.add(usize, try std.math.mul(usize, yrows - 1, r.y_stride), n);
    const xbytes = try std.math.mul(usize, xend, 2);
    const ybytes = try std.math.mul(usize, yend, if (r.output == .f32) @as(usize, 4) else 2);
    try check(x, xbytes, 2);
    try check(y, ybytes, if (r.output == .f32) @as(usize, 4) else 2);
    try check(scratch, scratch_size, 16);
    // A prerotated call's `x` is the scratch buffer's rotated base by construction, so only the write guards apply.
    if (r.prerotated) {
        try check(x, xbytes, 2);
        try disjoint(x, xbytes, y, ybytes);
    } else try readonly(x, xbytes, y, ybytes, scratch, scratch_size);
    try disjoint(y, ybytes, scratch, scratch_size);
    const id_bytes = try std.math.mul(usize, r.count, 4);
    for ([_]?View{ r.input_ids, r.output_ids, if (r.routes) |v| v.ids else null }) |optional| {
        if (optional) |v| {
            try check(v, id_bytes, 4);
            try readonly(v, id_bytes, y, ybytes, scratch, scratch_size);
        }
    }
    return .{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(r.count),
        .halves = 0,
        .codebook = 0,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = @intFromBool(r.output == .f32),
        .input_ids = @intFromBool(r.input_ids != null),
        .output_ids = @intFromBool(r.output_ids != null),
        .routed = @intFromBool(r.routes != null),
        .expert_id = if (r.routes) |v| v.expert_id else 0,
        .repeat = @intCast(repeat),
        .x_stride = r.x_stride,
        .y_stride = r.y_stride,
        .tile_k_stride = 0,
        .tile_n_stride = 0,
    };
}
pub fn bind(e: mtl.ComputeEncoder, v: View, index: usize) void {
    e.setBuffer(v.buffer, v.offset, index);
}

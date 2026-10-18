//! Borrowed resident EXL3 projections; fixed tiles keep each row independent of its surrounding window.
const std = @import("std");
pub const checkpoint = @import("exl3_load.zig");
pub const load = @import("exl3_metal_load.zig");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const format = @import("exl3_format.zig");
const decoder = @import("exl3_projection_decode.zig");

pub const View = struct {
    buffer: mtl.Buffer,
    offset: usize = 0,
};
pub const ScaleFormat = enum { f16, packed_signs };
pub const Output = enum { bf16, f32 };

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
    fn checkStorage(m: Matrix) !usize {
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
    };
}

fn address(v: View) !u64 {
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

pub const Pipelines = struct {
    rotate_in: mtl.Pipeline,
    matmul: mtl.Pipeline,
    rotate_out: mtl.Pipeline,
    prefill: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_exl3_projection, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const ri = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_in", false);
        errdefer ri.deinit();
        const mm = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_hoisted", false);
        errdefer mm.deinit();
        const ro = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_out", false);
        errdefer ro.deinit();
        for ([_]mtl.Pipeline{ ri, mm, ro }) |p|
            if (p.simdWidth() != 32 or p.maxThreads() < 32) return error.UnsupportedExl3Simd;
        var prefill = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill", false) catch null;
        if (prefill) |pf| {
            if (pf.simdWidth() != 32 or pf.maxThreads() < 128) {
                pf.deinit();
                prefill = null;
            }
        }
        return .{ .rotate_in = ri, .matmul = mm, .rotate_out = ro, .prefill = prefill };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rotate_in.deinit();
        p.matmul.deinit();
        p.rotate_out.deinit();
        if (p.prefill) |pf| pf.deinit();
    }
};

const Args = extern struct {
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
    x_stride: u64,
    y_stride: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
};

fn shape(k: usize, n: usize) !void {
    if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or
        k > std.math.maxInt(u32) or n > std.math.maxInt(u32)) return error.UnsupportedExl3Shape;
}
fn rotatedBytes(k: usize, rows: usize) !usize {
    const bytes = try std.math.mul(usize, try std.math.mul(usize, k, rows), 2);
    return (try std.math.add(usize, bytes, 15)) & ~@as(usize, 15);
}

/// Caller allocates this once for its largest window; scratch never holds a dequantized full weight matrix.
pub fn scratchBytes(k: usize, n: usize, rows: usize) !usize {
    try shape(k, n);
    if (rows == 0 or rows > std.math.maxInt(u32) - 7) return error.InvalidExl3Rows;
    return std.math.add(usize, try rotatedBytes(k, rows), try std.math.mul(usize, try std.math.mul(usize, n, rows), 4));
}
fn check(v: View, bytes: usize, alignment: usize) !void {
    const len = v.buffer.length();
    if (v.offset % alignment != 0 or v.offset > len or bytes > len - v.offset) return error.InvalidExl3View;
}

fn disjoint(a: View, a_bytes: usize, b: View, b_bytes: usize) !void {
    if (a.buffer.id == b.buffer.id and a.offset < b.offset + b_bytes and b.offset < a.offset + a_bytes)
        return error.AliasedExl3View;
}

fn readonly(v: View, bytes: usize, y: View, y_bytes: usize, scratch: View, scratch_bytes: usize) !void {
    try disjoint(v, bytes, y, y_bytes);
    try disjoint(v, bytes, scratch, scratch_bytes);
}
fn rowsArgs(k: usize, n: usize, r: Rows, x: View, y: View, scratch: View) !Args {
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
    try readonly(x, xbytes, y, ybytes, scratch, scratch_size);
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
fn bind(e: mtl.ComputeEncoder, v: View, index: usize) void {
    e.setBuffer(v.buffer, v.offset, index);
}

/// No allocation or commit; retain all views until completion, disjoint from writes, with valid GPU IDs.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, x: View, y: View, scratch: View, r: Rows) !void {
    const storage_bytes = try m.checkStorage();
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    try matrixWrites(m, storage_bytes, y, scratch, r);
    a.halves = m.width.halves;
    a.codebook = @intCast(@backingInt(m.codebook));
    a.su_signs = @intFromBool(m.suh_format == .packed_signs);
    a.sv_signs = @intFromBool(m.svh_format == .packed_signs);
    a.has_bias = @intFromBool(m.bias != null);
    a.tile_k_stride = try m.kStride();
    a.tile_n_stride = m.nStride();
    stages(p, e, a, m.trellis, m.suh, m.svh, m.bias orelse m.svh, scratch, x, y, scratch, r);
}

/// GPU routes choose caller-validated resident descriptors; grouped padding uses maxInt(u32).
pub fn encodeIndexed(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows) !void {
    if (r.routes == null or experts.count == 0 or experts.count > std.math.maxInt(u32)) return error.InvalidExl3Experts;
    if (r.grouped and (r.input_ids == null or r.output_ids == null or r.count % 8 != 0))
        return error.InvalidExl3GroupedRows;
    try shape(experts.k, experts.n);
    try check(experts.descriptors, try std.math.mul(usize, experts.count, @sizeOf(ExpertDescriptor)), 8);
    var a = try rowsArgs(experts.k, experts.n, r, x, y, scratch);
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + experts.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    try readonly(experts.descriptors, experts.count * @sizeOf(ExpertDescriptor), y, ybytes, scratch, try scratchBytes(experts.k, experts.n, r.count));
    a.indexed = if (r.grouped) 2 else 1;
    a.experts = @intCast(experts.count);
    stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, r);
}

fn matrixWrites(m: Matrix, storage_bytes: usize, y: View, scratch: View, r: Rows) !void {
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + m.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    const scratch_bytes = try scratchBytes(m.k, m.n, r.count);
    try readonly(m.trellis, storage_bytes, y, ybytes, scratch, scratch_bytes);
    try readonly(m.suh, m.k * 2 / (if (m.suh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    try readonly(m.svh, m.n * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, ybytes, scratch, scratch_bytes);
}

fn stages(p: *const Pipelines, e: mtl.ComputeEncoder, a: Args, trellis: View, su: View, sv: View, bias: View, descriptors: View, x: View, y: View, scratch: View, r: Rows) void {
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + ((@as(usize, a.rows) * a.k * 2 + 15) & ~@as(usize, 15)) };
    const routes = if (r.routes) |v| v.ids else scratch;
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, su, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, r.input_ids orelse scratch, 4);
    bind(e, routes, 5);
    bind(e, descriptors, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    const shared = p.prefill != null and a.rows >= 32 and a.indexed == 0 and a.routed == 0;
    e.setPipeline(if (shared) p.prefill.? else p.matmul);
    bind(e, scratch, 0);
    bind(e, trellis, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, routes, 4);
    bind(e, descriptors, 5);
    const group_rows: usize = if (shared) 32 else 8;
    e.dispatchGroups(.{ .width = a.n / 16, .height = if (a.indexed == 1) a.rows else (@as(usize, a.rows) + group_rows - 1) / group_rows }, .{ .width = if (shared) @as(usize, 128) else 32 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, sv, 1);
    bind(e, bias, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, r.output_ids orelse scratch, 5);
    bind(e, routes, 6);
    bind(e, descriptors, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

test "EXL3 scratch is row compact and bounded independently of full matrices" {
    try std.testing.expectEqual(@as(usize, 214016), try scratchBytes(5120, 2304, 11));
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, 0));
    try std.testing.expectError(error.UnsupportedExl3Shape, scratchBytes(129, 128, 1));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(ExpertDescriptor));
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Args));
}

test "stored per-matrix width and rectangular strides include public target shapes" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 5120, .n = 2304, .width = .{ .halves = 6 } };
    try m.validate();
    try std.testing.expectEqual(@as(usize, 4423680), try m.storageBytes());
    m.width.halves = 5;
    try m.validate();
    m.codebook = .mcg;
    try std.testing.expectError(error.UnsupportedExl3Width, m.validate());
    m.width.halves = 16;
    m.k = 1280;
    m.n = 4096;
    try m.validate();
    m.tile_n_stride = 258;
    try m.validate();
    m.tile_k_stride = 256;
    try std.testing.expectError(error.InvalidExl3Stride, m.validate());
}

test "invalid and overflowing extents fail before storage arithmetic can wrap" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 0, .n = 128, .width = .{ .halves = 6 } };
    try std.testing.expectError(error.UnsupportedExl3Shape, m.storageBytes());
    m.k = 128;
    m.n = 256;
    m.tile_n_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.validate());
    m.tile_n_stride = m.tileBytes();
    m.tile_k_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.storageBytes());
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, std.math.maxInt(u32)));
    try std.testing.expectError(error.Overflow, scratchBytes(0xffffff80, 0xffffff80, 0xffffff80));
}

test {
    _ = decoder;
    _ = @import("exl3_rect_test.zig");
}

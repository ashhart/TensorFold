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

    /// Borrow whole output rotation blocks, retaining the parent's K stride and complete reduction.
    pub fn columns(m: Matrix, start: usize, count: usize) !Matrix {
        try m.validate();
        if (start % 128 != 0 or count == 0 or count % 128 != 0 or
            start > m.n or count > m.n - start) return error.InvalidExl3Partition;
        var shard = m;
        shard.tile_k_stride = try m.kStride();
        shard.trellis.offset = try std.math.add(usize, m.trellis.offset, try std.math.mul(usize, start / 16, m.nStride()));
        const scale_bytes = start * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1);
        shard.svh.offset = try std.math.add(usize, m.svh.offset, scale_bytes);
        if (m.bias) |bias| shard.bias.?.offset = try std.math.add(usize, bias.offset, start * 2);
        shard.n = count;
        return shard;
    }

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

    /// Apply the same output-block cut to an already validated resident expert descriptor.
    pub fn columns(d: ExpertDescriptor, n: usize, start: usize, count: usize) !ExpertDescriptor {
        if (n == 0 or n % 128 != 0 or start % 128 != 0 or count == 0 or count % 128 != 0 or
            start > n or count > n - start) return error.InvalidExl3Partition;
        var out = d;
        out.trellis = try std.math.add(u64, d.trellis, try std.math.mul(u64, start / 16, d.tile_n_stride));
        const scale_bytes = try std.math.mul(u64, start / (if (d.svh_signs != 0) @as(usize, 16) else 1), 2);
        out.svh = try std.math.add(u64, d.svh, scale_bytes);
        if (d.has_bias != 0) out.bias = try std.math.add(u64, d.bias, try std.math.mul(u64, start, 2));
        return out;
    }
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
    dense: mtl.Pipeline,
    /// Two columns per SIMD group, for products too small to fill the GPU with sixteen-column groups.
    matmul_pairs: mtl.Pipeline,
    /// Maximum sixteen-column grid for matmul_pairs (0 disables); see docs/exl3-projection.md for tuning.
    pairs_up_to_groups: usize = 128,

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
        const dn = try mtl.Pipeline.init(device, lib, "tf_exl3_dense", false);
        errdefer dn.deinit();
        const mp = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_pairs", false);
        errdefer mp.deinit();
        for ([_]mtl.Pipeline{ ri, mm, ro, mp }) |p|
            if (p.simdWidth() != 32 or p.maxThreads() < 32) return error.UnsupportedExl3Simd;
        var prefill = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill", false) catch null;
        if (prefill) |pf| {
            if (pf.simdWidth() != 32 or pf.maxThreads() < 128) {
                pf.deinit();
                prefill = null;
            }
        }
        errdefer if (prefill) |pf| pf.deinit();
        var out: Pipelines = .{ .rotate_in = ri, .matmul = mm, .rotate_out = ro, .prefill = prefill, .dense = dn, .matmul_pairs = mp };
        if (std.c.getenv("TF_EXL3_PAIRS_UP_TO_GROUPS")) |v|
            out.pairs_up_to_groups = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.InvalidExl3PairsGroups;
        return out;
    }
    pub fn deinit(p: *Pipelines) void {
        p.rotate_in.deinit();
        p.matmul.deinit();
        p.rotate_out.deinit();
        if (p.prefill) |pf| pf.deinit();
        p.dense.deinit();
        p.matmul_pairs.deinit();
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
    stages(p, e, try prepare(m, x, y, scratch, r));
}

fn prepare(m: Matrix, x: View, y: View, scratch: View, r: Rows) !Staged {
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
    return .{ .a = a, .trellis = m.trellis, .su = m.suh, .sv = m.svh, .bias = m.bias orelse m.svh, .descriptors = scratch, .x = x, .y = y, .scratch = scratch, .r = r };
}

/// One product of a concurrent group: a plain matrix or GPU-routed experts.
pub const Job = struct {
    weights: union(enum) { matrix: Matrix, experts: Experts },
    x: View,
    y: View,
    scratch: View,
    rows: Rows,
};

/// Dispatch each stage for all independent jobs before its barrier; reject cross-job read/write aliasing first.
pub fn encodeJobs(p: *const Pipelines, e: mtl.ComputeEncoder, jobs: []const Job) !void {
    if (jobs.len == 0 or jobs.len > max_jobs) return error.InvalidExl3Jobs;
    var staged: [max_jobs]Staged = undefined;
    for (jobs, staged[0..jobs.len]) |job, *out| out.* = switch (job.weights) {
        .matrix => |m| try prepare(m, job.x, job.y, job.scratch, job.rows),
        .experts => |x| try prepareIndexed(x, job.x, job.y, job.scratch, job.rows),
    };
    for (staged[0..jobs.len], 0..) |a, i| for (staged[0..jobs.len], 0..) |b, j| {
        if (i == j) continue;
        // a's writes against everything b touches.
        for ([_]Extent{ a.output(), a.scratchExtent() }) |w| for (b.reads() ++ [_]Extent{ b.output(), b.scratchExtent() }) |o| {
            if (o.bytes != 0) try disjoint(w.view, w.bytes, o.view, o.bytes);
        };
    };
    // Stage barriers: the matmul reads every job's rotated rows, rotate_out every job's sums.
    for (0..3) |stage| {
        for (staged[0..jobs.len]) |job| dispatchStage(p, e, job, stage);
        e.barrier();
    }
}
const max_jobs = 6;

/// A kept-dense EXL3 matrix: a bf16 weight beside the input/output rotation scales, no trellis.
pub const DenseMatrix = struct {
    weight: View,
    suh: View,
    svh: View,
    bias: ?View = null,
    k: usize,
    n: usize,
};

/// Rotate as in encode, but use a dense f32 product in sequential K order, matching the fp64 reference.
pub fn encodeDense(p: *const Pipelines, e: mtl.ComputeEncoder, m: DenseMatrix, x: View, y: View, scratch: View, r: Rows) !void {
    try shape(m.k, m.n);
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    a.su_signs = 0;
    a.sv_signs = 0;
    a.has_bias = @intFromBool(m.bias != null);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    const sums_bytes = try std.math.mul(usize, try std.math.mul(usize, m.n, r.count), 4);
    try readonly(m.weight, try std.math.mul(usize, try std.math.mul(usize, m.n, m.k), 2), y, 0, scratch, sums_bytes);
    try readonly(m.suh, m.k * 2, y, 0, scratch, sums_bytes);
    try readonly(m.svh, m.n * 2, y, 0, scratch, sums_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, 0, scratch, sums_bytes);
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.dense);
    bind(e, scratch, 0);
    bind(e, m.weight, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    e.dispatchThreads(.{ .width = a.n, .height = a.rows }, .{ .width = 64 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

/// GPU routes choose caller-validated resident descriptors; grouped padding uses maxInt(u32).
pub fn encodeIndexed(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows) !void {
    stages(p, e, try prepareIndexed(experts, x, y, scratch, r));
}

fn prepareIndexed(experts: Experts, x: View, y: View, scratch: View, r: Rows) !Staged {
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
    return .{ .a = a, .trellis = scratch, .su = scratch, .sv = scratch, .bias = scratch, .descriptors = experts.descriptors, .x = x, .y = y, .scratch = scratch, .r = r };
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

const Extent = struct { view: View, bytes: usize };

/// A validated product: its arguments and every view its three stages bind.
const Staged = struct {
    a: Args,
    trellis: View,
    su: View,
    sv: View,
    bias: View,
    descriptors: View,
    x: View,
    y: View,
    scratch: View,
    r: Rows,

    fn output(s: Staged) Extent {
        const yrows: usize = if (s.r.output_ids != null) s.r.output_rows else s.r.count;
        return .{ .view = s.y, .bytes = ((yrows - 1) * s.r.y_stride + s.a.n) * (if (s.r.output == .f32) @as(usize, 4) else 2) };
    }
    fn scratchExtent(s: Staged) Extent {
        return .{ .view = s.scratch, .bytes = scratchBytes(s.a.k, s.a.n, s.r.count) catch unreachable };
    }
    /// The input rows and ID maps; weights are read-only resident tensors for every job.
    fn reads(s: Staged) [4]Extent {
        const repeat: usize = if (s.r.routes) |v| v.repeat else 1;
        const xrows: usize = if (s.r.input_ids != null) s.r.input_rows else s.r.count / repeat;
        const ids = s.r.count * 4;
        return .{
            .{ .view = s.x, .bytes = ((xrows - 1) * s.r.x_stride + s.a.k) * 2 },
            .{ .view = s.r.input_ids orelse s.x, .bytes = if (s.r.input_ids != null) ids else 0 },
            .{ .view = s.r.output_ids orelse s.x, .bytes = if (s.r.output_ids != null) ids else 0 },
            .{ .view = if (s.r.routes) |v| v.ids else s.x, .bytes = if (s.r.routes != null) ids else 0 },
        };
    }
};

fn stages(p: *const Pipelines, e: mtl.ComputeEncoder, s: Staged) void {
    for (0..3) |stage| {
        dispatchStage(p, e, s, stage);
        e.barrier();
    }
}

/// Stages rotate input, multiply, then rotate sums out; the caller places barriers between stages.
fn dispatchStage(p: *const Pipelines, e: mtl.ComputeEncoder, s: Staged, stage: usize) void {
    const a = s.a;
    const r = s.r;
    const scratch = s.scratch;
    const descriptors = s.descriptors;
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + ((@as(usize, a.rows) * a.k * 2 + 15) & ~@as(usize, 15)) };
    const routes = if (r.routes) |v| v.ids else scratch;
    switch (stage) {
        0 => {
            e.setPipeline(p.rotate_in);
            bind(e, s.x, 0);
            bind(e, s.su, 1);
            bind(e, scratch, 2);
            e.setValue(a, 3);
            bind(e, r.input_ids orelse scratch, 4);
            bind(e, routes, 5);
            bind(e, descriptors, 6);
            e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
        },
        1 => {
            const shared = p.prefill != null and a.rows >= 32 and a.indexed == 0 and a.routed == 0;
            const row_groups: usize = if (a.indexed == 1) a.rows else (@as(usize, a.rows) + 7) / 8;
            const pairs = !shared and a.n / 16 * row_groups <= p.pairs_up_to_groups;
            e.setPipeline(if (shared) p.prefill.? else if (pairs) p.matmul_pairs else p.matmul);
            bind(e, scratch, 0);
            bind(e, s.trellis, 1);
            bind(e, sums, 2);
            e.setValue(a, 3);
            bind(e, routes, 4);
            bind(e, descriptors, 5);
            const group_rows: usize = if (shared) 32 else 8;
            e.dispatchGroups(.{ .width = if (pairs) a.n / 2 else a.n / 16, .height = if (a.indexed == 1) a.rows else (@as(usize, a.rows) + group_rows - 1) / group_rows }, .{ .width = if (shared) @as(usize, 128) else 32 });
        },
        else => {
            e.setPipeline(p.rotate_out);
            bind(e, sums, 0);
            bind(e, s.sv, 1);
            bind(e, s.bias, 2);
            bind(e, s.y, 3);
            e.setValue(a, 4);
            bind(e, r.output_ids orelse scratch, 5);
            bind(e, routes, 6);
            bind(e, descriptors, 7);
            e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
        },
    }
}

test "EXL3 scratch is row compact and bounded independently of full matrices" {
    try std.testing.expectEqual(@as(usize, 214016), try scratchBytes(5120, 2304, 11));
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, 0));
    try std.testing.expectError(error.UnsupportedExl3Shape, scratchBytes(129, 128, 1));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(ExpertDescriptor));
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Args));
}

test {
    _ = @import("exl3_projection_test.zig");
    _ = decoder;
    _ = @import("exl3_rect_test.zig");
}

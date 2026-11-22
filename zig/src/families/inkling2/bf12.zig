//! TF_INKLING2_BF12=1: lossless 12-bit bf16 ("BF12") copies of Mini's dense rows that decode reads through a rowdot (the
//! attention output wo, the relative-bias rows wr, the global layers' dynamic-convolution rows and the output head). The
//! host packs each matrix at load (sm121a_pack.packBf12) and the bf16 copy is never uploaded.
//! Decode rows (up to window_rows) run bf12.cu's tensor-core matmul on the packed bytes (~25% less weight traffic); wider
//! calls unpack the matrix into a scratch (bit-exact bf16) and run the plain rowdot as before. The matmul's sums run in
//! MMA order, not rowdot's lane order: the weights are exact, the fp32 sums are not bit-equal to rowdot's.
//! Format: per-row base B in [1, 240], a weight with exponent in [B, B + 14] keeps ec = exp - B and sm = sign | mantissa;
//! zeros are ec 15 with sm = sign << 7; the rest are ec 15 with sm 0x01 and sit in a CSR side table of (col << 16 | bits).
//! Tiles of 8 rows x 64 columns (768 bytes), in blocks of 64 rows: [block][column group][tile] (sm121a_pack.tileOffset).
const std = @import("std");
const cuda = @import("cuda");
const pack_ = @import("sm121a_pack.zig");

pub const tile_bytes = pack_.tile_bytes;

/// A packed matrix [n, k] on the device.
pub const Mat = struct {
    data: u64,
    row_base: u64,
    row_ptr: u64,
    entries: u64,
    n: u32,
    k: u32,
    nnz: u32,
    plan: Plan,

    /// bf16 bytes of the matrix this one replaces.
    pub fn bf16Bytes(m: Mat) usize {
        return @as(usize, m.n) * m.k * 2;
    }

    /// Device bytes of the four buffers.
    pub fn bytes(m: Mat) usize {
        const total_n8 = (@as(usize, m.n) + 7) / 8;
        return total_n8 * (m.k / 64) * tile_bytes + total_n8 * 8 + (@as(usize, m.n) + 1) * 4 + @max(@as(usize, m.nnz) * 4, 4);
    }
};

/// bf12.cuh's plan, by shape alone (never by the row count).
pub const Plan = struct { bn: u32, sk: u32, stages: u32, kg_step: u32, l2: bool };

pub fn planOf(n: usize, k: usize) Plan {
    const w12 = ((n + 7) / 8) * (k / 64) * tile_bytes;
    const l2 = w12 <= 10 << 20;
    const kg = k / 64;
    const tiles = (n + 63) / 64;
    if (tiles >= 96) return if (l2) .{ .bn = 32, .sk = 1, .stages = if (kg <= 8) 2 else 3, .kg_step = 1, .l2 = true } else .{ .bn = 64, .sk = 1, .stages = 4, .kg_step = 1, .l2 = false };
    if (n <= 1024 and k <= 4096) return .{ .bn = 32, .sk = 1, .stages = 4, .kg_step = if (kg % 2 == 0) 2 else 1, .l2 = l2 };
    const max_sk: usize = if (tiles >= 36) 2 else 16;
    var sk: usize = 1;
    while (sk < max_sk and tiles * sk < 160 and kg % (sk * 2) == 0 and kg / (sk * 2) >= 2) sk *= 2;
    return .{ .bn = 64, .sk = @intCast(sk), .stages = if (sk >= 16) 4 else 3, .kg_step = 1, .l2 = false };
}

/// The bf12.cu entry points: [variant][fp32 out]. `wm` warp rows of four warps each (128 threads a warp row).
const Variant = struct { bm: u32, bn: u32, st: u32, ks: u32, split: bool, cw: bool, wm: u32 = 1 };
const variants = [_]Variant{
    .{ .bm = 16, .bn = 32, .st = 4, .ks = 2, .split = false, .cw = true },
    .{ .bm = 16, .bn = 32, .st = 3, .ks = 1, .split = false, .cw = true },
    .{ .bm = 16, .bn = 32, .st = 2, .ks = 1, .split = false, .cw = true },
    .{ .bm = 16, .bn = 64, .st = 3, .ks = 1, .split = true, .cw = true },
    .{ .bm = 16, .bn = 64, .st = 3, .ks = 1, .split = false, .cw = true },
    .{ .bm = 16, .bn = 64, .st = 4, .ks = 1, .split = true, .cw = false },
    .{ .bm = 16, .bn = 64, .st = 3, .ks = 1, .split = true, .cw = false },
    .{ .bm = 32, .bn = 64, .st = 3, .ks = 1, .split = true, .cw = false },
    .{ .bm = 48, .bn = 64, .st = 3, .ks = 1, .split = true, .cw = false },
    .{ .bm = 64, .bn = 64, .st = 3, .ks = 1, .split = true, .cw = false },
    .{ .bm = 16, .bn = 64, .st = 4, .ks = 1, .split = false, .cw = false },
    .{ .bm = 32, .bn = 64, .st = 4, .ks = 1, .split = false, .cw = false },
    .{ .bm = 48, .bn = 64, .st = 3, .ks = 1, .split = false, .cw = false },
    .{ .bm = 64, .bn = 64, .st = 3, .ks = 1, .split = false, .cw = false },
    .{ .bm = 128, .bn = 64, .st = 2, .ks = 1, .split = false, .cw = false, .wm = 2 },
};

/// Wide prefill calls (m >= 128 rows on a matrix of 64 rows or more): BM 128 over 2 x 4 warps, no split, two stages
/// (44 KB: two CTAs an SM; three stages' 66 KB fit one and ran 5-15% slower than BM 64 on GB10).
const wide = find(.{ .bm = 128, .bn = 64, .st = 2, .ks = 1, .split = false, .cw = false, .wm = 2 });

fn find(v: Variant) usize {
    for (variants, 0..) |x, i| if (std.meta.eql(x, v)) return i;
    unreachable;
}

/// launch_bf12_mm_plan's kernel choice: BM by the row count only on DRAM-resident matrices, stages by the plan.
fn variantFor(p: Plan, k: usize, m: usize) usize {
    if (p.l2) {
        if (p.bn == 32 and p.sk == 1) {
            return if (p.kg_step == 2 and (k / 64) % 2 == 0) find(.{ .bm = 16, .bn = 32, .st = 4, .ks = 2, .split = false, .cw = true }) else find(.{ .bm = 16, .bn = 32, .st = if (p.stages == 2) 2 else 3, .ks = 1, .split = false, .cw = true });
        }
        return find(.{ .bm = 16, .bn = 64, .st = 3, .ks = 1, .split = p.sk > 1, .cw = true });
    }
    const bm: u32 = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 48) 48 else 64;
    if (p.sk > 1) return find(.{ .bm = bm, .bn = 64, .st = if (bm == 16 and p.stages == 4) 4 else 3, .ks = 1, .split = true, .cw = false });
    return find(.{ .bm = bm, .bn = 64, .st = if (bm >= 48) 3 else 4, .ks = 1, .split = false, .cw = false });
}

fn smemOf(v: Variant) u32 {
    const sub = ((v.bm * 128 + (v.bn / 8) * tile_bytes) + 127) / 128 * 128;
    return @intCast(v.st * v.ks * sub);
}

/// Floats of split-K partials a call of `rows` rows on `n` rows' matrix needs (zero when the plan never splits).
pub fn partFloats(n: usize, k: usize, rows: usize) usize {
    return planOf(n, k).sk * rows * n;
}

pub const Kernels = struct {
    module: cuda.Module,
    mm: [variants.len][2]cuda.Function,
    reduce: [2]cuda.Function,
    unpack_fn: cuda.Function,
    scatter_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available or cuda.kernels.inkling2_bf12.len == 0) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .module = try cuda.Module.load(d, cuda.kernels.inkling2_bf12), .mm = undefined, .reduce = undefined, .unpack_fn = undefined, .scatter_fn = undefined };
        errdefer k.module.unload();
        inline for (variants, 0..) |v, i| inline for (0..2) |f| {
            const name = std.fmt.comptimePrint("tf_ink2_bf12_mm_{d}_{d}_{d}_{d}_{d}_{d}_{d}", .{ v.bm, v.bn, v.st, v.ks, f, @intFromBool(v.split), @intFromBool(v.cw) });
            k.mm[i][f] = try k.module.function(name);
            if (smemOf(v) > 48 * 1024) try k.mm[i][f].allowDynamicShared(smemOf(v));
        };
        k.reduce = .{ try k.module.function("tf_ink2_bf12_reduce_csr_f0"), try k.module.function("tf_ink2_bf12_reduce_csr_f1") };
        k.unpack_fn = try k.module.function("tf_ink2_bf12_unpack_tiles_kernel");
        k.scatter_fn = try k.module.function("tf_ink2_bf12_scatter_outliers_kernel");
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }

    /// x [m, k] bf16 (16-byte aligned, rows `x_stride` apart) @ mat.T -> out [m, n] (fp32 with `fp32`, else bf16). Split-K
    /// partials go to `part` (partFloats(n, k, m) floats). No host sync or allocation: it captures into a graph.
    pub fn matmul(k: *const Kernels, st: cuda.Stream, x: u64, x_stride: usize, mat: *const Mat, out: u64, fp32: bool, part: u64, m: usize) !void {
        return k.matmulStore(st, x, x_stride, mat, out, if (fp32) .f32 else .bf16, part, m);
    }

    /// How `matmulStore` writes its sums: bf16, fp32, or fp32 holding the bf16 value (ops' roundBf16 in the store).
    pub const Store = enum { bf16, f32, f32_bf16 };

    pub fn matmulStore(k: *const Kernels, st: cuda.Stream, x: u64, x_stride: usize, mat: *const Mat, out: u64, store: Store, part: u64, m: usize) !void {
        const fp32 = store != .bf16;
        const round: c_int = @intFromBool(store == .f32_bf16);
        if (x % 16 != 0 or (m > 1 and x_stride % 8 != 0)) return error.Bf12Unaligned;
        var p = mat.plan;
        // Prefill-width calls skip split-K (no partials/reduce): full-width BM=64, sk=1, split=false.
        if (m >= 64 and mat.n >= 64) {
            p.sk = 1;
            p.l2 = false;
        }
        const vi = if (m >= 128 and mat.n >= 64) wide else variantFor(p, mat.k, m);
        const v = variants[vi];
        const stride: usize = if (m == 1) mat.k else x_stride;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intCast(stride)));
        a.add(mat.data);
        a.add(mat.row_base);
        a.add(mat.row_ptr);
        a.add(mat.entries);
        a.add(out);
        a.add(part);
        for ([_]usize{ m, mat.n, mat.k, p.sk }) |q| a.add(@as(c_int, @intCast(q)));
        a.add(round);
        try cuda.launch.launch(k.mm[vi][@intFromBool(fp32)], .{
            .grid = .{ .x = @intCast((m + v.bm - 1) / v.bm), .y = @intCast((mat.n + v.bn - 1) / v.bn), .z = p.sk },
            .block = .{ .x = 128 * v.wm },
            .shared = smemOf(v),
        }, st, &a);
        if (p.sk > 1) {
            const total = m * mat.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(x);
            r.add(@as(c_int, @intCast(stride)));
            r.add(mat.row_ptr);
            r.add(mat.entries);
            r.add(out);
            for ([_]usize{ m, mat.n, p.sk }) |q| r.add(@as(c_int, @intCast(q)));
            r.add(round);
            try cuda.launch.launch(k.reduce[@intFromBool(fp32)], .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, st, &r);
        }
    }

    /// The exact bf16 rows [n, k] of `mat` into `dst` (device, n * k * 2 bytes).
    pub fn unpack(k: *const Kernels, st: cuda.Stream, mat: *const Mat, dst: u64) !void {
        const tiles: usize = ((@as(usize, mat.n) + 7) / 8) * (mat.k / 64);
        var a: cuda.Args = .{};
        a.add(mat.data);
        a.add(mat.row_base);
        a.add(dst);
        a.add(@as(c_int, @intCast(mat.n)));
        a.add(@as(c_int, @intCast(mat.k)));
        try cuda.launch.launch(k.unpack_fn, .{ .grid = .{ .x = @intCast((tiles + 3) / 4) }, .block = .{ .x = 128 } }, st, &a);
        if (mat.nnz > 0) {
            var b: cuda.Args = .{};
            b.add(mat.row_ptr);
            b.add(mat.entries);
            b.add(dst);
            b.add(@as(c_int, @intCast(mat.n)));
            b.add(@as(c_int, @intCast(mat.k)));
            try cuda.launch.launch(k.scatter_fn, .{ .grid = .{ .x = @intCast((mat.n + 255) / 256) }, .block = .{ .x = 256 } }, st, &b);
        }
    }
};

/// TF_INKLING2_BF12=1 and the kernel set is in this build; false (with a warning) when asked for and absent.
pub fn enabled() bool {
    const want = @import("sys").environment.read(bool, "TF_INKLING2_BF12", false) catch false;
    if (!want) return false;
    if (!cuda.kernels.available or cuda.kernels.inkling2_bf12.len == 0) {
        std.log.warn("TF_INKLING2_BF12=1: this build has no BF12 kernel set; the dense rows stay bf16", .{});
        return false;
    }
    return true;
}

/// Whether a bf16 [n, k] matrix can be held: k a multiple of 64 and under 65536 (a column is a u16 in the escape table).
pub fn eligible(n: usize, k: usize) bool {
    return n > 0 and k % 64 == 0 and k <= 65535;
}

/// Packs `bf16_le` ([n][k] little-endian bf16, any alignment) on the host and uploads it, each buffer appended to
/// `bufs` (the caller frees them). error.Bf12EntriesOverflow: too many escapes to be worth packing, keep the bf16 rows.
pub fn pack(gpa: std.mem.Allocator, d: *const cuda.Driver, bufs: *std.ArrayList(cuda.DeviceBuffer), bf16_le: []const u8, n: usize, k: usize) !Mat {
    if (!eligible(n, k) or bf16_le.len != n * k * 2) return error.Bf12Shape;
    const total_n8 = (n + 7) / 8;
    const kg_n = k / 64;
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    for (rows, 0..) |*v, i| v.* = std.mem.readInt(u16, bf16_le[i * 2 ..][0..2], .little);
    const data = try gpa.alloc(u8, total_n8 * kg_n * tile_bytes);
    defer gpa.free(data);
    const bases = try gpa.alloc(u8, total_n8 * 8);
    defer gpa.free(bases);
    @memset(bases, 112);
    const row_ptr = try gpa.alloc(u32, n + 1);
    defer gpa.free(row_ptr);
    // escapes are rare (an exponent outside a row's 15-wide window); a matrix past 1.5% of its elements stays bf16
    const entries = try gpa.alloc(u32, @max(n * k / 64, 1 << 12));
    defer gpa.free(entries);
    const nnz = try pack_.packBf12(rows, n, k, .{ .data = data, .row_base = bases, .row_ptr = row_ptr, .entries = entries, .n_entries = 0 });
    var m: Mat = .{ .data = 0, .row_base = 0, .row_ptr = 0, .entries = 0, .n = @intCast(n), .k = @intCast(k), .nnz = @intCast(nnz), .plan = planOf(n, k) };
    for ([_]struct { *u64, []const u8 }{
        .{ &m.data, data },
        .{ &m.row_base, bases },
        .{ &m.row_ptr, std.mem.sliceAsBytes(row_ptr) },
        .{ &m.entries, std.mem.sliceAsBytes(entries[0..nnz]) },
    }) |item| {
        var b = try cuda.DeviceBuffer.alloc(d, @max(item[1].len, 4));
        errdefer b.free();
        if (item[1].len > 0) try b.upload(0, item[1]);
        try bufs.append(gpa, b);
        item[0].* = b.ptr;
    }
    return m;
}

test "bf12 plans: the output head never splits and the small matrices do or stay cached" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 1), planOf(200_000, 2304).sk); // a tall matrix has tiles enough
    try t.expect(planOf(2304, 3072).sk >= 1);
    try t.expectEqual(@as(usize, 0), partFloats(0, 64, 16));
    try t.expect(eligible(2304, 3072) and !eligible(24, 100) and !eligible(24, 65600));
    // every plan lands on a table entry
    for ([_][2]usize{ .{ 2304, 3072 }, .{ 384, 2304 }, .{ 4096, 2304 }, .{ 200_000, 2304 }, .{ 24, 2304 } }) |nk| {
        for ([_]usize{ 1, 16, 17, 33, 49, 2048 }) |m| _ = variantFor(planOf(nk[0], nk[1]), nk[1], m);
    }
    // M10.5: a split-K call's partials (sk*m*n) always fit partFloats(n, k, 64): split calls take at
    // most 64 rows (the zig chunk cap), and m>=64&&n>=64 forces sk=1 (zero partials). Mirrors the
    // zig-side chunking (pin only ever shrinks chunks to 64) and matmulStore's m>=64 override.
    for ([_][2]usize{ .{ 2304, 3072 }, .{ 1024, 2304 }, .{ 24, 4224 } }) |nk| {
        const p = planOf(nk[0], nk[1]);
        for ([_]bool{ false, true }) |pin| {
            for ([_]usize{ 1, 63, 64, 65, 128, 2048 }) |rows| {
                const zsk: usize = if (rows >= 64 and nk[0] >= 64 and !pin) 1 else p.sk;
                const chunk: usize = if (zsk > 1 or (pin and rows > 64)) 64 else rows;
                var r0: usize = 0;
                while (r0 < rows) {
                    const take = @min(rows - r0, chunk);
                    const ksk: usize = if (take >= 64 and nk[0] >= 64) 1 else p.sk;
                    // SPLITK kernels write ksk*take*n partials; sk=1 kernels write none.
                    const need: usize = if (ksk > 1) ksk * take * nk[0] else 0;
                    try t.expect(need <= partFloats(nk[0], nk[1], 64));
                    r0 += take;
                }
            }
        }
    }
}

//! DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 prompt kernels on Apple M5 Ultra Metal 4.1.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };
pub const Call = struct { k: u32, n: u32, rows: u32, out: Out = .f32, reuse_x: bool = false };
pub const Batch = struct { n: u32 = 1, xs: i64 = 0, ws: i64 = 0, ss: i64 = 0, ys: i64 = 0 };

const F32Args = extern struct { K: u32, N: u32, R: u32 };
const CertArgs = extern struct { K: u32, N: u32, R: u32, col_cap: u32, c_hw: f32 };
const FastArgs = extern struct { K: u32, N: u32, R: u32, B: u32, xs: i64, ws: i64, ss: i64, ys: i64 };
const PrepSmallArgs = extern struct { K: u32, R: u32, B: u32, pad: u32 = 0, xs: i64 };
const SmallArgs = extern struct { K: u32, N: u32, R: u32, B: u32, xs: i64, ws: i64, ss: i64, ys: i64, nws: i64, x_shared: u32, c_hw: f32 };

const max_norm_entries = 2048;
const norm_cache_bytes: usize = 64 << 20;
const max_scratch_rows: usize = 4096;
const max_scratch_k: usize = 8192;
const max_scratch_n: usize = 32768;
const col_cap_default: u32 = 512;

/// Whether a Metal device name identifies Apple M5 Ultra.
pub fn chipMatches(name: []const u8) bool {
    return std.mem.indexOf(u8, name, "M5 Ultra") != null;
}

/// Whether this Mac satisfies the strict Apple M5 Ultra + Metal 4.1 (macOS 27+) gate for TF_DSV41_FP64_BF16=1.
pub fn supported(device: mtl.Device) bool {
    return device.tensorUnits() and device.generation() == 17 and mtl.objc.osVersion().major >= 27 and chipMatches(std.mem.span(device.name()));
}

const NormKey = struct { w_addr: u64, s_addr: u64, k: u32, n: u32, norm_off: usize };

const State = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    x_half: mtl.Buffer,
    xs: mtl.Buffer,
    norm_x: mtl.Buffer,
    batch_x_half: ?mtl.Buffer = null,
    batch_xs: ?mtl.Buffer = null,
    batch_norm_x: ?mtl.Buffer = null,
    batch_xh_cap: usize = 0,
    batch_xs_cap: usize = 0,
    batch_nx_cap: usize = 0,
    col_count: mtl.Buffer,
    cross_rows: mtl.Buffer,
    norm_w_pool: mtl.Buffer,
    norm_w_used: usize = 0,
    keys: [max_norm_entries]NormKey = undefined,
    nkeys: usize = 0,
    tmp_norm_w: mtl.Buffer,
    small_xh: mtl.Buffer,
    small_xs: mtl.Buffer,
    small_nx: mtl.Buffer,

    fn init(a: std.mem.Allocator, device: mtl.Device) !*State {
        const st = try a.create(State);
        errdefer a.destroy(st);
        const opt = mtl.ResourceOptions.shared;
        const x_half = try device.buffer(max_scratch_rows * max_scratch_k * 2, opt);
        errdefer x_half.deinit();
        const xs = try device.buffer(max_scratch_rows * (max_scratch_k / 32) * 4, opt);
        errdefer xs.deinit();
        const norm_x = try device.buffer(max_scratch_rows * 16, opt);
        errdefer norm_x.deinit();
        const col_count = try device.buffer(max_scratch_n * 4, opt);
        errdefer col_count.deinit();
        const cross_rows = try device.buffer(max_scratch_n * col_cap_default * 2, opt);
        errdefer cross_rows.deinit();
        const norm_w_pool = try device.buffer(norm_cache_bytes, opt);
        errdefer norm_w_pool.deinit();
        const tmp_norm_w = try device.buffer(max_scratch_n * 16, opt);
        errdefer tmp_norm_w.deinit();
        const small_xh = try device.buffer(8 * 16 * max_scratch_k * 2, opt);
        errdefer small_xh.deinit();
        const small_xs = try device.buffer(8 * 16 * (max_scratch_k / 32) * 4, opt);
        errdefer small_xs.deinit();
        const small_nx = try device.buffer(8 * 16 * 16, opt);
        st.* = .{
            .a = a,
            .device = device,
            .x_half = x_half,
            .xs = xs,
            .norm_x = norm_x,
            .col_count = col_count,
            .cross_rows = cross_rows,
            .norm_w_pool = norm_w_pool,
            .tmp_norm_w = tmp_norm_w,
            .small_xh = small_xh,
            .small_xs = small_xs,
            .small_nx = small_nx,
        };
        return st;
    }

    fn deinit(st: *State) void {
        st.x_half.deinit();
        st.xs.deinit();
        st.norm_x.deinit();
        if (st.batch_x_half) |b| b.deinit();
        if (st.batch_xs) |b| b.deinit();
        if (st.batch_norm_x) |b| b.deinit();
        st.col_count.deinit();
        st.cross_rows.deinit();
        st.norm_w_pool.deinit();
        st.tmp_norm_w.deinit();
        st.small_xh.deinit();
        st.small_xs.deinit();
        st.small_nx.deinit();
        st.a.destroy(st);
    }

    fn ensureBatch(st: *State, need_xh: usize, need_xs: usize, need_nx: usize) !void {
        const opt = mtl.ResourceOptions.shared;
        if (st.batch_xh_cap < need_xh) {
            if (st.batch_x_half) |b| b.deinit();
            st.batch_x_half = null;
            st.batch_xh_cap = 0;
            st.batch_x_half = try st.device.buffer(need_xh, opt);
            st.batch_xh_cap = need_xh;
        }
        if (st.batch_xs_cap < need_xs) {
            if (st.batch_xs) |b| b.deinit();
            st.batch_xs = null;
            st.batch_xs_cap = 0;
            st.batch_xs = try st.device.buffer(need_xs, opt);
            st.batch_xs_cap = need_xs;
        }
        if (st.batch_nx_cap < need_nx) {
            if (st.batch_norm_x) |b| b.deinit();
            st.batch_norm_x = null;
            st.batch_nx_cap = 0;
            st.batch_norm_x = try st.device.buffer(need_nx, opt);
            st.batch_nx_cap = need_nx;
        }
    }

    fn clearCache(st: *State) void {
        st.nkeys = 0;
        st.norm_w_used = 0;
    }

    fn getNormW(st: *State, p: *const Pipelines, e: mtl.ComputeEncoder, w: Ref, s: Ref, k: u32, n: u32) Ref {
        const wa = w.buf.gpuAddress() + w.off;
        const sa = s.buf.gpuAddress() + s.off;
        for (st.keys[0..st.nkeys]) |entry| {
            if (entry.w_addr == wa and entry.s_addr == sa and entry.k == k and entry.n == n) {
                return .{ .buf = st.norm_w_pool, .off = entry.norm_off };
            }
        }
        const need = std.mem.alignForward(usize, @as(usize, n) * 16, 256);
        var dst: Ref = undefined;
        if (st.nkeys < max_norm_entries and st.norm_w_used + need <= norm_cache_bytes) {
            dst = .{ .buf = st.norm_w_pool, .off = st.norm_w_used };
            st.keys[st.nkeys] = .{ .w_addr = wa, .s_addr = sa, .k = k, .n = n, .norm_off = st.norm_w_used };
            st.nkeys += 1;
            st.norm_w_used += need;
        } else {
            dst = .{ .buf = st.tmp_norm_w, .off = 0 };
        }
        e.setPipeline(p.norm_w);
        e.setBuffer(w.buf, w.off, 0);
        e.setBuffer(s.buf, s.off, 1);
        e.setBuffer(dst.buf, dst.off, 2);
        e.setValue([2]u32{ n, k }, 3);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        return dst;
    }
};

pub const Pipelines = struct {
    f32: mtl.Pipeline,
    norm_w: mtl.Pipeline,
    prep_x: mtl.Pipeline,
    zero_cnt: mtl.Pipeline,
    gemm: mtl.Pipeline,
    repair: mtl.Pipeline,
    fast_bf16: mtl.Pipeline,
    fast_f32: mtl.Pipeline,
    prep_x_small: mtl.Pipeline,
    small_gemm: mtl.Pipeline,
    st: *State,

    /// Buffers held by the certified norm cache and small-M scratch (for joining the target's residency set).
    pub fn normBufs(p: *const Pipelines) [5]mtl.Buffer {
        return .{ p.st.norm_w_pool, p.st.tmp_norm_w, p.st.small_xh, p.st.small_xs, p.st.small_nx };
    }

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Pipelines {
        return loadSource(a, device, sources.dsv41.fp8_mma_cert_all);
    }

    pub fn loadSource(a: std.mem.Allocator, device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const st = try State.init(a, device);
        errdefer st.deinit();
        var p_f32 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_f32", false);
        errdefer p_f32.deinit();
        var norm_w = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_norm_w", false);
        errdefer norm_w.deinit();
        var prep_x = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_prep_x", false);
        errdefer prep_x.deinit();
        var zero_cnt = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_zero_cnt", false);
        errdefer zero_cnt.deinit();
        var gemm = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_gemm", false);
        errdefer gemm.deinit();
        var repair = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_repair_bf16", false);
        errdefer repair.deinit();
        var fast_bf16 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_fast_nax_bf16", false);
        errdefer fast_bf16.deinit();
        var fast_f32 = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_fast_nax_f32", false);
        errdefer fast_f32.deinit();
        var prep_x_small = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_prep_x_small", false);
        errdefer prep_x_small.deinit();
        const small_gemm = try mtl.Pipeline.init(device, lib, "tf_dsv41_fp8_cert_small_gemm", false);
        return .{ .f32 = p_f32, .norm_w = norm_w, .prep_x = prep_x, .zero_cnt = zero_cnt, .gemm = gemm, .repair = repair, .fast_bf16 = fast_bf16, .fast_f32 = fast_f32, .prep_x_small = prep_x_small, .small_gemm = small_gemm, .st = st };
    }

    pub fn clearNormCache(p: *Pipelines) void {
        p.st.clearCache();
    }

    pub fn deinit(p: *Pipelines) void {
        p.f32.deinit();
        p.norm_w.deinit();
        p.prep_x.deinit();
        p.zero_cnt.deinit();
        p.gemm.deinit();
        p.repair.deinit();
        p.fast_bf16.deinit();
        p.fast_f32.deinit();
        p.prep_x_small.deinit();
        p.small_gemm.deinit();
        p.st.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.k % 32 != 0 or c.n == 0 or c.n % 32 != 0) return error.BadFp8Shape;
}

/// X [rows][K] bf16, W [N][K] E4M3, S [N/32][K/32] E8M0, Y [rows][N]: certified 0.0 ULP vs FP64.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call) !void {
    try check(c);
    if (c.out == .f32) {
        const args = F32Args{ .K = c.k, .N = c.n, .R = c.rows };
        e.setPipeline(p.f32);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(s.buf, s.off, 2);
        e.setBuffer(y.buf, y.off, 3);
        e.setValue(args, 4);
        e.dispatchThreads(mtl.Size.of(c.n, c.rows, 1), mtl.Size.of(16, 16, 1));
        return;
    }
    if (c.k > max_scratch_k or c.n > max_scratch_n) return error.BadFp8Shape;
    const nw = p.st.getNormW(p, e, w, s, c.k, c.n);
    var r0: u32 = 0;
    while (r0 < c.rows) : (r0 += max_scratch_rows) {
        const m: u32 = @min(@as(u32, max_scratch_rows), c.rows - r0);
        const pad_m: u32 = (m + 31) & ~@as(u32, 31);
        const xr = Ref{ .buf = x.buf, .off = x.off + @as(usize, r0) * c.k * 2 };
        const yr = Ref{ .buf = y.buf, .off = y.off + @as(usize, r0) * c.n * 2 };

        if (!(c.reuse_x and c.rows <= max_scratch_rows)) {
            e.setPipeline(p.prep_x);
            e.setBuffer(xr.buf, xr.off, 0);
            e.setBuffer(p.st.x_half, 0, 1);
            e.setBuffer(p.st.xs, 0, 2);
            e.setBuffer(p.st.norm_x, 0, 3);
            e.setValue([2]u32{ m, c.k }, 4);
            e.dispatchGroups(mtl.Size.of(pad_m, 1, 1), mtl.Size.of(128, 1, 1));
        }

        e.setPipeline(p.zero_cnt);
        e.setBuffer(p.st.col_count, 0, 0);
        e.setValue(c.n, 1);
        e.dispatchThreads(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();

        const cap: u32 = @min(col_cap_default, m);
        const ca_args = CertArgs{ .K = c.k, .N = c.n, .R = m, .col_cap = cap, .c_hw = 2.0e-7 };
        e.setPipeline(p.gemm);
        e.setBuffer(p.st.x_half, 0, 0);
        e.setBuffer(p.st.xs, 0, 1);
        e.setBuffer(p.st.norm_x, 0, 2);
        e.setBuffer(w.buf, w.off, 3);
        e.setBuffer(s.buf, s.off, 4);
        e.setBuffer(nw.buf, nw.off, 5);
        e.setBuffer(yr.buf, yr.off, 6);
        e.setBuffer(p.st.col_count, 0, 7);
        e.setBuffer(p.st.cross_rows, 0, 8);
        e.setValue(ca_args, 9);
        e.dispatchGroups(mtl.Size.of((c.n + 63) / 64, (m + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        e.barrier();

        e.setPipeline(p.repair);
        e.setBuffer(xr.buf, xr.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(s.buf, s.off, 2);
        e.setBuffer(p.st.col_count, 0, 3);
        e.setBuffer(p.st.cross_rows, 0, 4);
        e.setBuffer(yr.buf, yr.off, 5);
        e.setValue(ca_args, 6);
        e.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
    }
}

/// Batched BF16 FP8 projection for `batch` contiguous groups in 4 dispatches and 3 barriers.
pub fn encodeBatched(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, batch: u32, c: Call) !void {
    try check(c);
    if (c.out != .bf16 or batch <= 1 or c.rows > max_scratch_rows or c.k > max_scratch_k or @as(usize, batch) * c.n > max_scratch_n) {
        const g_k: usize = c.k / 32;
        const g_n: usize = c.n / 32;
        for (0..batch) |bi| {
            try encode(
                p,
                e,
                .{ .buf = x.buf, .off = x.off + bi * @as(usize, c.rows) * c.k * 2 },
                .{ .buf = w.buf, .off = w.off + bi * @as(usize, c.n) * c.k },
                .{ .buf = s.buf, .off = s.off + bi * g_n * g_k },
                .{ .buf = y.buf, .off = y.off + bi * @as(usize, c.rows) * c.n * 2 },
                c,
            );
        }
        return;
    }
    const pad_m: u32 = (c.rows + 31) & ~@as(u32, 31);
    const total_n: u32 = batch * c.n;
    const nw = p.st.getNormW(p, e, w, s, c.k, total_n);
    const need_xh: usize = @as(usize, batch) * pad_m * c.k * 2;
    const need_xs: usize = @as(usize, batch) * pad_m * (c.k / 32) * 4;
    const need_nx: usize = @as(usize, batch) * pad_m * 16;
    try p.st.ensureBatch(need_xh, need_xs, need_nx);
    const xh_buf = p.st.batch_x_half.?;
    const xs_buf = p.st.batch_xs.?;
    const nx_buf = p.st.batch_norm_x.?;

    e.setPipeline(p.prep_x);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(xh_buf, 0, 1);
    e.setBuffer(xs_buf, 0, 2);
    e.setBuffer(nx_buf, 0, 3);
    e.setValue([2]u32{ c.rows, c.k }, 4);
    e.dispatchGroups(mtl.Size.of(pad_m, batch, 1), mtl.Size.of(128, 1, 1));

    e.setPipeline(p.zero_cnt);
    e.setBuffer(p.st.col_count, 0, 0);
    e.setValue(total_n, 1);
    e.dispatchThreads(mtl.Size.of(total_n, 1, 1), mtl.Size.of(256, 1, 1));
    e.barrier();

    const cap: u32 = @min(col_cap_default, c.rows);
    const ca_args = CertArgs{ .K = c.k, .N = c.n, .R = c.rows, .col_cap = cap, .c_hw = 2.0e-7 };
    e.setPipeline(p.gemm);
    e.setBuffer(xh_buf, 0, 0);
    e.setBuffer(xs_buf, 0, 1);
    e.setBuffer(nx_buf, 0, 2);
    e.setBuffer(w.buf, w.off, 3);
    e.setBuffer(s.buf, s.off, 4);
    e.setBuffer(nw.buf, nw.off, 5);
    e.setBuffer(y.buf, y.off, 6);
    e.setBuffer(p.st.col_count, 0, 7);
    e.setBuffer(p.st.cross_rows, 0, 8);
    e.setValue(ca_args, 9);
    e.dispatchGroups(mtl.Size.of((c.n + 63) / 64, (c.rows + 31) / 32, batch), mtl.Size.of(128, 1, 1));
    e.barrier();

    e.setPipeline(p.repair);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(p.st.col_count, 0, 3);
    e.setBuffer(p.st.cross_rows, 0, 4);
    e.setBuffer(y.buf, y.off, 5);
    e.setValue(ca_args, 6);
    e.dispatchGroups(mtl.Size.of(c.n, batch, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
}

/// Fast non-dd M5 tensor-unit FP8 projection for DSpark drafting (1..128 rows, batched via `b`).
pub fn encodeFast(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch) !void {
    try check(c);
    if (b.n == 0) return error.BadFp8Shape;
    const args = FastArgs{ .K = c.k, .N = c.n, .R = c.rows, .B = b.n, .xs = b.xs, .ws = b.ws, .ss = b.ss, .ys = b.ys };
    e.setPipeline(if (c.out == .f32) p.fast_f32 else p.fast_bf16);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBuffer(s.buf, s.off, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of((c.n + 31) / 32, (c.rows + 15) / 16, b.n), mtl.Size.of(128, 1, 1));
}

/// Whether exact small-M tensor-unit verify (`encodeSmall`) beats scalar `fp8_rows_cert` for this shape on M5 Ultra.
pub fn takesSmall(c: Call, launches: u32) bool {
    if (c.out != .bf16 or c.rows == 0 or c.rows > 16 or launches == 0 or launches > 8) return false;
    if (c.k == 0 or c.k % 32 != 0 or c.k > max_scratch_k or c.n == 0 or c.n % 32 != 0) return false;
    const cols = c.n * launches;
    if (cols >= 1024) return c.rows >= 2;
    if (cols >= 512) return c.rows >= 3;
    return c.rows >= 4;
}

/// Hoist small-M (R = 1..16) BF16 activation prep + 4-segment L2 norms into L2-resident scratch (`small_xh/xs/nx`).
pub fn prepSmall(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, rows: u32, k: u32, batch: u32, xs: i64) void {
    const b_prep: u32 = if (batch > 1 and xs == 0) 1 else batch;
    const args = PrepSmallArgs{ .K = k, .R = rows, .B = b_prep, .xs = xs };
    e.setPipeline(p.prep_x_small);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(p.st.small_xh, 0, 1);
    e.setBuffer(p.st.small_xs, 0, 2);
    e.setBuffer(p.st.small_nx, 0, 3);
    e.setValue(args, 4);
    e.dispatchGroups(mtl.Size.of(16, b_prep, 1), mtl.Size.of(256, 1, 1));
}

/// Exact FP64-certified BF16 small-M (R = 1..16) M5 tensor-unit projection with hoisted `prepSmall` and inline repair.
pub fn encodeSmall(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, c: Call, b: Batch) !void {
    try check(c);
    if (c.out != .bf16 or c.rows > 16 or c.k > max_scratch_k or b.n == 0 or b.n > 8) return error.BadFp8Shape;
    const x_shared: u32 = if (b.n > 1 and b.xs == 0) 1 else 0;
    const contig = b.n == 1 or (b.ws == @as(i64, c.n) * c.k and b.ss == @as(i64, c.n / 32) * (c.k / 32));
    if (contig) {
        const nw = p.st.getNormW(p, e, w, s, c.k, b.n * c.n);
        if (!c.reuse_x) {
            prepSmall(p, e, x, c.rows, c.k, b.n, b.xs);
            e.barrier();
        }
        const args = SmallArgs{ .K = c.k, .N = c.n, .R = c.rows, .B = b.n, .xs = b.xs, .ws = b.ws, .ss = b.ss, .ys = b.ys, .nws = @as(i64, c.n) * 16, .x_shared = x_shared, .c_hw = 2.0e-7 };
        e.setPipeline(p.small_gemm);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(p.st.small_xh, 0, 1);
        e.setBuffer(p.st.small_xs, 0, 2);
        e.setBuffer(p.st.small_nx, 0, 3);
        e.setBuffer(w.buf, w.off, 4);
        e.setBuffer(s.buf, s.off, 5);
        e.setBuffer(nw.buf, nw.off, 6);
        e.setBuffer(y.buf, y.off, 7);
        e.setValue(args, 8);
        e.dispatchGroups(mtl.Size.of((c.n + 31) / 32, 1, b.n), mtl.Size.of(128, 1, 1));
        return;
    }
    if (b.n == 2) {
        const w1 = Ref{ .buf = w.buf, .off = @intCast(@as(i64, @intCast(w.off)) + b.ws) };
        const s1 = Ref{ .buf = s.buf, .off = @intCast(@as(i64, @intCast(s.off)) + b.ss) };
        const nw0 = p.st.getNormW(p, e, w, s, c.k, c.n);
        const nw1 = p.st.getNormW(p, e, w1, s1, c.k, c.n);
        if (nw0.buf.id == p.st.norm_w_pool.id and nw1.buf.id == p.st.norm_w_pool.id) {
            if (!c.reuse_x) {
                prepSmall(p, e, x, c.rows, c.k, 2, b.xs);
                e.barrier();
            }
            const args = SmallArgs{ .K = c.k, .N = c.n, .R = c.rows, .B = 2, .xs = b.xs, .ws = b.ws, .ss = b.ss, .ys = b.ys, .nws = @as(i64, @intCast(nw1.off)) - @as(i64, @intCast(nw0.off)), .x_shared = x_shared, .c_hw = 2.0e-7 };
            e.setPipeline(p.small_gemm);
            e.setBuffer(x.buf, x.off, 0);
            e.setBuffer(p.st.small_xh, 0, 1);
            e.setBuffer(p.st.small_xs, 0, 2);
            e.setBuffer(p.st.small_nx, 0, 3);
            e.setBuffer(w.buf, w.off, 4);
            e.setBuffer(s.buf, s.off, 5);
            e.setBuffer(nw0.buf, nw0.off, 6);
            e.setBuffer(y.buf, y.off, 7);
            e.setValue(args, 8);
            e.dispatchGroups(mtl.Size.of((c.n + 31) / 32, 1, 2), mtl.Size.of(128, 1, 1));
            return;
        }
    }
    for (0..b.n) |bi| {
        const i: i64 = @intCast(bi);
        var ci = c;
        ci.reuse_x = (x_shared == 1) and (c.reuse_x or bi > 0);
        try encodeSmall(p, e, .{ .buf = x.buf, .off = @intCast(@as(i64, @intCast(x.off)) + i * b.xs) }, .{ .buf = w.buf, .off = @intCast(@as(i64, @intCast(w.off)) + i * b.ws) }, .{ .buf = s.buf, .off = @intCast(@as(i64, @intCast(s.off)) + i * b.ss) }, .{ .buf = y.buf, .off = @intCast(@as(i64, @intCast(y.off)) + i * b.ys) }, ci, .{});
    }
}

test "M5 Ultra chip guard matches only M5 Ultra" {
    try std.testing.expect(chipMatches("Apple M5 Ultra"));
    try std.testing.expect(!chipMatches("Apple M5 Max"));
    try std.testing.expect(!chipMatches("Apple M5 Pro"));
    try std.testing.expect(!chipMatches("Apple M5"));
    try std.testing.expect(!chipMatches("Apple M4 Ultra"));
    try std.testing.expect(!chipMatches("Apple M3 Ultra"));
}

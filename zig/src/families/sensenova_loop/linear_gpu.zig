//! SenseNova-Loop BF16 and AWQ W4A16 g128 Metal linear wrapper, pipeline cache, and sweep benchmark.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("core").frags;
const awq = @import("awq.zig");

const quant = @import("linear_quant.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Sums = struct { ref: Ref, stride: u32 };
pub const OutSplits = struct {
    split1: u32 = 0,
    y1: ?Ref = null,
    ldy1: u32 = 0,
    split2: u32 = 0,
    y2: ?Ref = null,
    ldy2: u32 = 0,
};

pub const Bf16Matrix = struct {
    buf: mtl.Buffer,
    off: usize = 0,
    n: u32,
    k: u32,
};

pub const AwqMatrix = struct {
    w: mtl.Buffer,
    w_off: usize = 0,
    sb: mtl.Buffer,
    sb_off: usize = 0,
    input_scale: ?mtl.Buffer = null,
    input_scale_off: usize = 0,
    n: u32,
    k: u32,
    m5_layout: bool,
    owns: bool = false,

    pub fn deinit(m: *AwqMatrix) void {
        if (m.owns) {
            m.w.deinit();
            m.sb.deinit();
            if (m.input_scale) |b| b.deinit();
        }
        m.* = undefined;
    }
};

pub const Kind = enum {
    bf16_mv,
    bf16_gemm,
    bf16_splitk,
    bf16_splitk_sum,
    awq_m5_lane,
    awq_m5_mv,
    awq_m5_gemm,
    awq_m3_mv,
    awq_m3_qmm,
    awq_xsum,
    awq_scale_input,
    awq_embed,
    awq_quant,
};

pub const Key = struct {
    kind: Kind,
    n: u32 = 0,
    k: u32 = 0,
    sk: u32 = 1,
    rows_tier: u32 = 1,
    prompt: bool = false,
    precompute_sums: bool = false,
    edge_m: bool = false,
    edge_n: bool = false,
};

pub const Bf16Args = extern struct {
    rows: u32,
    n: u32,
    k: u32,
    ldy: u32,
    y1_addr: u64 = 0,
    y2_addr: u64 = 0,
    split1: u32 = 0,
    ldy1: u32 = 0,
    split2: u32 = 0,
    ldy2: u32 = 0,
};
pub const AwqArgs = extern struct {
    m: i32,
    n: i32,
    k: i32,
    mp: i32,
    ldy: i32,
    split1: u32 = 0,
    y1_addr: u64 = 0,
    y2_addr: u64 = 0,
    ldy1: u32 = 0,
    split2: u32 = 0,
    ldy2: u32 = 0,
    _pad: u32 = 0,
};
pub const Bf16GemmArgs = extern struct {
    p: [16]i32 = @splat(0),
    y1_addr: u64 = 0,
    y2_addr: u64 = 0,
    split1: u32 = 0,
    ldy1: u32 = 0,
    split2: u32 = 0,
    ldy2: u32 = 0,
};
pub const Bf16SumArgs = extern struct {
    p: [6]i32,
    y1_addr: u64 = 0,
    y2_addr: u64 = 0,
    split1: u32 = 0,
    ldy1: u32 = 0,
    split2: u32 = 0,
    ldy2: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Bf16Args) == 48 and @offsetOf(Bf16Args, "y1_addr") == 16);
    std.debug.assert(@sizeOf(AwqArgs) == 56 and @offsetOf(AwqArgs, "y1_addr") == 24);
    std.debug.assert(@sizeOf(Bf16GemmArgs) == 96 and @offsetOf(Bf16GemmArgs, "y1_addr") == 64);
    std.debug.assert(@sizeOf(Bf16SumArgs) == 56 and @offsetOf(Bf16SumArgs, "y1_addr") == 24);
}

pub fn rowsTier(rows: u32) u32 {
    if (rows <= 1) return 1;
    if (rows <= 2) return 2;
    if (rows <= 4) return 4;
    if (rows <= 8) return 8;
    if (rows <= 16) return 16;
    return 32;
}

pub fn bf16MvCols(tier: u32) u32 {
    return if (tier == 1) 1 else if (tier <= 4) 2 else 1;
}

pub fn awqM3MvCols(tier: u32) u32 {
    return if (tier >= 2 and tier <= 8) 2 else 1;
}

pub fn decodeSk(tensor_units: bool, is_awq: bool, n: u32, k: u32) u32 {
    if (tensor_units and is_awq) {
        if (k >= 8192) return 8;
        if (n <= 4096) return 8;
        if (n == 12288) return 2;
        return 4;
    }
    if (is_awq) {
        if (k >= 8192) return 4;
        if (n >= 32768) return 1;
        return 2;
    }
    if (n >= 32768) return 2;
    return 4;
}

pub const LinearGpu = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    tensor_units: bool,
    bf16_lib: ?mtl.Library = null,
    awq_lib: ?mtl.Library = null,
    cache: std.AutoHashMapUnmanaged(Key, mtl.Pipeline) = .empty,
    xsum_ring: [2]?mtl.Buffer = .{ null, null },
    xsum_caps: [2]usize = .{ 0, 0 },
    xsum_turn: u1 = 0,
    splitk_buf: ?mtl.Buffer = null,
    splitk_cap: usize = 0,
    scaled_buf: ?mtl.Buffer = null,
    scaled_cap: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) LinearGpu {
        return .{ .allocator = a, .device = device, .tensor_units = device.tensorUnits() };
    }

    pub fn deinit(g: *LinearGpu) void {
        var it = g.cache.valueIterator();
        while (it.next()) |p| p.deinit();
        g.cache.deinit(g.allocator);
        if (g.bf16_lib) |l| l.deinit();
        if (g.awq_lib) |l| l.deinit();
        for (g.xsum_ring) |slot| if (slot) |b| b.deinit();
        if (g.splitk_buf) |b| b.deinit();
        if (g.scaled_buf) |b| b.deinit();
        g.* = undefined;
    }

    /// Preallocate untracked scratch buffers up to `max_rows` before residency hold.
    pub fn reserve(g: *LinearGpu, max_rows: u32, is_awq: bool) !void {
        const stride = std.mem.alignForward(usize, max_rows, 16);
        _ = try ensureScratch(g.device, &g.splitk_buf, &g.splitk_cap, 4 * @as(usize, max_rows) * 4160 * 4);
        if (is_awq) {
            const xb = @as(usize, 12288 / 128) * stride * 4;
            for (&g.xsum_ring, &g.xsum_caps) |*slot, *cap| _ = try ensureScratch(g.device, slot, cap, xb);
            _ = try ensureScratch(g.device, &g.scaled_buf, &g.scaled_cap, @as(usize, max_rows) * 12288 * 2);
        }
    }

    pub fn appendBuffers(g: *const LinearGpu, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        for (g.xsum_ring) |slot| if (slot) |b| try list.append(a, b);
        if (g.splitk_buf) |b| try list.append(a, b);
        if (g.scaled_buf) |b| try list.append(a, b);
    }

    fn bf16Lib(g: *LinearGpu) !mtl.Library {
        if (g.bf16_lib) |l| return l;
        const text = try frags.source(g.device, g.allocator, ks.qwen27_bf16_linear);
        defer g.allocator.free(text);
        const l = try mtl.Library.fromSource(g.device, text, mtl.CompileOptions.mlx());
        g.bf16_lib = l;
        return l;
    }

    fn awqLib(g: *LinearGpu) !mtl.Library {
        if (g.awq_lib) |l| return l;
        const text = try frags.source(g.device, g.allocator, ks.qwen27_awq_w4a16);
        defer g.allocator.free(text);
        const l = try mtl.Library.fromSource(g.device, text, mtl.CompileOptions.mlx());
        g.awq_lib = l;
        return l;
    }

    pub fn pipeline(g: *LinearGpu, key: Key) !mtl.Pipeline {
        if (g.cache.get(key)) |p| return p;
        var buf: [96]u8 = undefined;
        const fun: [:0]const u8 = switch (key.kind) {
            .bf16_mv => try std.fmt.bufPrintSentinel(&buf, "sn_bf16_mv_r{d}_sk{d}", .{ key.rows_tier, key.sk }, 0),
            .bf16_gemm => try std.fmt.bufPrintSentinel(&buf, "sn_bf16_gemm_{d}{d}", .{ @intFromBool(!key.edge_m), @intFromBool(!key.edge_n) }, 0),
            .bf16_splitk => try std.fmt.bufPrintSentinel(&buf, "sn_bf16_gemm_splitk_{d}{d}", .{ @intFromBool(!key.edge_m), @intFromBool(!key.edge_n) }, 0),
            .bf16_splitk_sum => "sn_bf16_gemm_splitk_sum",
            .awq_m5_lane => try std.fmt.bufPrintSentinel(&buf, "sn_awq_lane_sk{d}_p{d}_r{d}", .{ key.sk, @intFromBool(key.precompute_sums), @intFromBool(key.prompt) }, 0),
            .awq_m5_mv => try std.fmt.bufPrintSentinel(&buf, "sn_awq_m5_mv_r{d}_sk{d}", .{ key.rows_tier, key.sk }, 0),
            .awq_m5_gemm => "sn_awq_gemm_m5",
            .awq_m3_mv => try std.fmt.bufPrintSentinel(&buf, "sn_awq_row_mv_r{d}_sk{d}", .{ key.rows_tier, key.sk }, 0),
            .awq_m3_qmm => "sn_awq_qmm_m3",
            .awq_xsum => "sn_awq_xsum_128",
            .awq_scale_input => "sn_awq_scale_input",
            .awq_embed => "sn_awq_int4_embed",
            .awq_quant => "sn_awq_quantize_bf16",
        };
        const lib = switch (key.kind) {
            .bf16_mv, .bf16_gemm, .bf16_splitk, .bf16_splitk_sum => try g.bf16Lib(),
            else => try g.awqLib(),
        };
        const p = try mtl.Pipeline.init(g.device, lib, fun, false);
        errdefer p.deinit();
        try g.cache.put(g.allocator, key, p);
        return p;
    }

    fn ensureScratch(device: mtl.Device, slot: *?mtl.Buffer, cap: *usize, bytes: usize) !mtl.Buffer {
        if (slot.* != null and cap.* >= bytes) return slot.*.?;
        if (slot.*) |b| b.deinit();
        const alloc_bytes = @max(bytes, 256);
        const b = try device.buffer(alloc_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        slot.* = b;
        cap.* = alloc_bytes;
        return b;
    }

    pub fn prepareBf16(buf: mtl.Buffer, off: usize, n: u32, k: u32) !Bf16Matrix {
        if (n == 0 or k == 0 or k % 8 != 0 or off % 16 != 0) return error.InvalidMatrix;
        if (buf.length() < off + @as(usize, n) * k * 2) return error.BufferTooSmall;
        return .{ .buf = buf, .off = off, .n = n, .k = k };
    }

    pub fn prepareAwqFromPacked(g: *const LinearGpu, b: awq.PackedBundle) !AwqMatrix {
        return @import("linear_quant.zig").prepareAwqFromPacked(g, b);
    }

    pub fn prepareAwq(g: *const LinearGpu, parts: []const awq.Part) ![]AwqMatrix {
        return @import("linear_quant.zig").prepareAwq(g, parts);
    }

    pub fn xsum128(g: *LinearGpu, e: mtl.ComputeEncoder, x: Ref, rows: u32, k: u32, stride: u32, xs: Ref) !void {
        if (rows == 0 or k == 0 or k % 128 != 0 or stride < rows) return error.InvalidShape;
        const p = try g.pipeline(.{ .kind = .awq_xsum });
        const kg = k / 128;
        e.setPipeline(p);
        e.setBuffer(x.buf, x.off, 0);
        e.setValue([4]i32{ @intCast(rows), @intCast(stride), @intCast(k), 0 }, 1);
        e.setBuffer(xs.buf, xs.off, 2);
        e.dispatchThreads(mtl.Size.of(stride * kg, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn scaleInput(g: *LinearGpu, e: mtl.ComputeEncoder, x: Ref, scale: Ref, rows: u32, k: u32, out: Ref) !void {
        if (rows == 0 or k == 0) return error.InvalidShape;
        const p = try g.pipeline(.{ .kind = .awq_scale_input });
        e.setPipeline(p);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(scale.buf, scale.off, 1);
        e.setValue([2]i32{ @intCast(rows), @intCast(k) }, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchThreads(mtl.Size.of(k, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub const Dims = struct { rows: u32, n: u32, k: u32, ldy: u32 };

    fn makeBf16Args(rows: u32, n: u32, k: u32, ldy: u32, splits: OutSplits, addrs: [2]u64, done: u32) Bf16Args {
        return .{
            .rows = rows,
            .n = n,
            .k = k,
            .ldy = ldy,
            .y1_addr = if (addrs[0] != 0) addrs[0] + @as(u64, done) * splits.ldy1 * 2 else 0,
            .y2_addr = if (addrs[1] != 0) addrs[1] + @as(u64, done) * splits.ldy2 * 2 else 0,
            .split1 = splits.split1,
            .ldy1 = splits.ldy1,
            .split2 = splits.split2,
            .ldy2 = splits.ldy2,
        };
    }

    fn makeAwqArgs(m: u32, n: u32, k: u32, mp: u32, ldy: u32, splits: OutSplits, addrs: [2]u64, done: u32) AwqArgs {
        return .{
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .mp = @intCast(mp),
            .ldy = @intCast(ldy),
            .split1 = splits.split1,
            .y1_addr = if (addrs[0] != 0) addrs[0] + @as(u64, done) * splits.ldy1 * 2 else 0,
            .y2_addr = if (addrs[1] != 0) addrs[1] + @as(u64, done) * splits.ldy2 * 2 else 0,
            .ldy1 = splits.ldy1,
            .split2 = splits.split2,
            .ldy2 = splits.ldy2,
        };
    }

    pub fn dispatchBf16SplitK(kp: mtl.Pipeline, sp: mtl.Pipeline, e: mtl.ComputeEncoder, x: Ref, w: Ref, scratch: Ref, y: Ref, d: Dims, splits: u32) void {
        quant.dispatchBf16SplitKSplit(kp, sp, e, x, w, scratch, y, d, splits, .{});
    }

    pub fn dispatchBf16Gemm(gp: mtl.Pipeline, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, d: Dims) void {
        quant.dispatchBf16GemmSplit(gp, e, x, w, y, d, .{});
    }

    pub fn encodeBf16(g: *LinearGpu, e: mtl.ComputeEncoder, mat: Bf16Matrix, x: Ref, y: Ref, rows: u32, ldy: u32, prompt: bool, sk_override: ?u32) !void {
        return g.encodeBf16Split(e, mat, x, y, rows, ldy, prompt, sk_override, .{});
    }

    pub fn encodeBf16Split(
        g: *LinearGpu,
        e: mtl.ComputeEncoder,
        mat: Bf16Matrix,
        x: Ref,
        y: Ref,
        rows: u32,
        ldy: u32,
        prompt: bool,
        sk_override: ?u32,
        splits: OutSplits,
    ) !void {
        if (rows == 0) return error.InvalidShape;
        try quant.validateSplits(mat.n, ldy, splits);
        const addrs = quant.splitAddrs(e, splits);
        if (sk_override != null or !prompt) {
            const sk = sk_override orelse decodeSk(g.tensor_units, false, mat.n, mat.k);
            const max_take: u32 = if (sk_override != null) 32 else if (g.tensor_units) 16 else 8;
            var done: u32 = 0;
            while (done < rows) {
                const take = @min(rows - done, max_take);
                const tier: u32 = if (g.tensor_units and take < 4 and sk_override == null) 4 else rowsTier(take);
                const p = try g.pipeline(.{ .kind = .bf16_mv, .n = mat.n, .k = mat.k, .sk = sk, .rows_tier = tier });
                const no = (8 / sk) * bf16MvCols(tier);
                e.setPipeline(p);
                e.setBuffer(x.buf, x.off + @as(usize, done) * mat.k * 2, 0);
                e.setBuffer(mat.buf, mat.off, 1);
                e.setValue(makeBf16Args(take, mat.n, mat.k, ldy, splits, addrs, done), 2);
                e.setBuffer(y.buf, y.off + @as(usize, done) * ldy * 2, 3);
                e.dispatchGroups(mtl.Size.of((mat.n + no - 1) / no, 1, 1), mtl.Size.of(256, 1, 1));
                done += take;
            }
            return;
        }
        if (!g.tensor_units and mat.n >= 512 and mat.n <= 8192) {
            const k_splits: u32 = if (mat.n <= 4160 and mat.k >= 12288) 4 else 2;
            const sc = try ensureScratch(g.device, &g.splitk_buf, &g.splitk_cap, @as(usize, k_splits) * rows * mat.n * 4);
            const kp = try g.pipeline(.{
                .kind = .bf16_splitk,
                .n = mat.n,
                .k = mat.k,
                .prompt = true,
                .edge_m = rows % 64 != 0,
                .edge_n = mat.n % 64 != 0,
            });
            const sp = try g.pipeline(.{ .kind = .bf16_splitk_sum });
            quant.dispatchBf16SplitKSplit(kp, sp, e, x, .{ .buf = mat.buf, .off = mat.off }, .{ .buf = sc }, y, .{ .rows = rows, .n = mat.n, .k = mat.k, .ldy = ldy }, k_splits, splits);
            return;
        }
        const gp = try g.pipeline(.{
            .kind = .bf16_gemm,
            .n = mat.n,
            .k = mat.k,
            .prompt = true,
            .edge_m = rows % 64 != 0,
            .edge_n = mat.n % 128 != 0,
        });
        quant.dispatchBf16GemmSplit(gp, e, x, .{ .buf = mat.buf, .off = mat.off }, y, .{ .rows = rows, .n = mat.n, .k = mat.k, .ldy = ldy }, splits);
    }

    fn resolveXsum(g: *LinearGpu, e: mtl.ComputeEncoder, x: Ref, rows: u32, k: u32, eff_xs: ?Sums) !Sums {
        if (eff_xs) |s| {
            if (s.stride < rows) return error.InvalidShape;
            return s;
        }
        const stride = std.mem.alignForward(u32, rows, 16);
        const xs_bytes = @as(usize, k / 128) * stride * 4;
        const turn = g.xsum_turn;
        g.xsum_turn +%= 1;
        const xb = try ensureScratch(g.device, &g.xsum_ring[turn], &g.xsum_caps[turn], xs_bytes);
        const ref = Ref{ .buf = xb };
        try g.xsum128(e, x, rows, k, stride, ref);
        e.barrier();
        return .{ .ref = ref, .stride = stride };
    }

    pub fn encodeAwq(g: *LinearGpu, e: mtl.ComputeEncoder, mat: AwqMatrix, x_in: Ref, xs_opt: ?Sums, y: Ref, rows: u32, ldy: u32, prompt: bool, sk_override: ?u32) !void {
        return g.encodeAwqSplit(e, mat, x_in, xs_opt, y, rows, ldy, prompt, sk_override, .{});
    }

    pub fn encodeAwqSplit(
        g: *LinearGpu,
        e: mtl.ComputeEncoder,
        mat: AwqMatrix,
        x_in: Ref,
        xs_opt: ?Sums,
        y: Ref,
        rows: u32,
        ldy: u32,
        prompt: bool,
        sk_override: ?u32,
        splits: OutSplits,
    ) !void {
        if (rows == 0) return error.InvalidShape;
        try quant.validateSplits(mat.n, ldy, splits);
        const addrs = quant.splitAddrs(e, splits);
        var x = x_in;
        if (mat.input_scale) |is_buf| {
            const need = @as(usize, rows) * mat.k * 2;
            const sc = try ensureScratch(g.device, &g.scaled_buf, &g.scaled_cap, need);
            try g.scaleInput(e, x, .{ .buf = is_buf, .off = mat.input_scale_off }, rows, mat.k, .{ .buf = sc });
            e.barrier();
            x = .{ .buf = sc };
        }
        if (mat.m5_layout and g.tensor_units) {
            const eff_xs = if (mat.input_scale != null) null else xs_opt;
            if ((sk_override != null or !prompt) and eff_xs == null and mat.n <= 24576 and rows <= 8) {
                const sk = sk_override orelse decodeSk(true, true, mat.n, mat.k);
                const p = try g.pipeline(.{ .kind = .awq_m5_lane, .n = mat.n, .k = mat.k, .sk = sk, .precompute_sums = false, .prompt = false });
                e.setPipeline(p);
                e.setBuffer(x.buf, x.off, 0);
                e.setBuffer(mat.w, mat.w_off, 1);
                e.setBuffer(mat.sb, mat.sb_off, 2);
                e.setValue(makeAwqArgs(rows, mat.n, mat.k, 0, ldy, splits, addrs, 0), 3);
                e.setBuffer(y.buf, y.off, 4);
                e.setBuffer(x.buf, x.off, 5);
                e.dispatchGroups(mtl.Size.of((mat.n + 31) / 32, 1, 1), mtl.Size.of(sk * 32, 1, 1));
                return;
            }
            const sums = try g.resolveXsum(e, x, rows, mat.k, eff_xs);
            const args = makeAwqArgs(rows, mat.n, mat.k, sums.stride, ldy, splits, addrs, 0);
            if (sk_override != null or !prompt) {
                const sk = sk_override orelse decodeSk(true, true, mat.n, mat.k);
                const rb = rows > 16;
                const p = try g.pipeline(.{ .kind = .awq_m5_lane, .n = mat.n, .k = mat.k, .sk = sk, .precompute_sums = true, .prompt = rb });
                e.setPipeline(p);
                e.setBuffer(x.buf, x.off, 0);
                e.setBuffer(mat.w, mat.w_off, 1);
                e.setBuffer(mat.sb, mat.sb_off, 2);
                e.setValue(args, 3);
                e.setBuffer(y.buf, y.off, 4);
                e.setBuffer(sums.ref.buf, sums.ref.off, 5);
                e.dispatchGroups(mtl.Size.of((mat.n + 31) / 32, if (rb) (rows + 15) / 16 else 1, 1), mtl.Size.of(sk * 32, 1, 1));
                return;
            }
            const p = try g.pipeline(.{ .kind = .awq_m5_gemm, .n = mat.n, .k = mat.k, .prompt = true });
            const tn = (mat.n + 127) / 128;
            const tm = (rows + 63) / 64;
            e.setPipeline(p);
            e.setBuffer(x.buf, x.off, 0);
            e.setBuffer(mat.w, mat.w_off, 1);
            e.setBuffer(mat.sb, mat.sb_off, 2);
            e.setValue(args, 3);
            e.setBuffer(y.buf, y.off, 4);
            e.setBuffer(sums.ref.buf, sums.ref.off, 5);
            e.dispatchGroups(mtl.Size.of(tn * 4, (tm + 3) / 4, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        if (sk_override != null or !prompt) {
            const sk = sk_override orelse decodeSk(g.tensor_units, true, mat.n, mat.k);
            const max_take: u32 = if (sk_override != null) 32 else 8;
            var done: u32 = 0;
            while (done < rows) {
                const take = @min(rows - done, max_take);
                const tier = rowsTier(take);
                const p = try g.pipeline(.{ .kind = .awq_m3_mv, .n = mat.n, .k = mat.k, .sk = sk, .rows_tier = tier });
                const no = (8 / sk) * awqM3MvCols(tier);
                e.setPipeline(p);
                e.setBuffer(x.buf, x.off + @as(usize, done) * mat.k * 2, 0);
                e.setBuffer(mat.w, mat.w_off, 1);
                e.setBuffer(mat.sb, mat.sb_off, 2);
                e.setValue(makeAwqArgs(take, mat.n, mat.k, 0, ldy, splits, addrs, done), 3);
                e.setBuffer(y.buf, y.off + @as(usize, done) * ldy * 2, 4);
                e.dispatchGroups(mtl.Size.of((mat.n + no - 1) / no, 1, 1), mtl.Size.of(256, 1, 1));
                done += take;
            }
            return;
        }
        const p = try g.pipeline(.{ .kind = .awq_m3_qmm, .n = mat.n, .k = mat.k, .prompt = true });
        e.setPipeline(p);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(mat.w, mat.w_off, 1);
        e.setBuffer(mat.sb, mat.sb_off, 2);
        e.setValue(makeAwqArgs(rows, mat.n, mat.k, 0, ldy, splits, addrs, 0), 3);
        e.setBuffer(y.buf, y.off, 4);
        e.dispatchGroups(mtl.Size.of((mat.n + 63) / 64, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn quantizeBf16ToAwq(g: *LinearGpu, queue: mtl.Queue, src: Bf16Matrix) !AwqMatrix {
        return quant.quantizeBf16ToAwq(g, queue, src);
    }

    /// Sweeps SK in {1,2,4,8} for decode and prefill across SenseNova shapes.
    pub fn bench(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
        return @import("linear_bench.zig").bench(a, device, queue);
    }
};

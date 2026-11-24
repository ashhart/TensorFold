//! Cross-platform CUDA / HIP Device Bridge for Folded Memory Weights v6.
//! Allows any CUDA or HIP engine (Nemotron-H CUDA, Qwen3.5 CUDA, Qwen3/Llama CUDA) to attach the
//! full v6 Folded Memory Weights engine (exact BF16 KV, spectral SVD, temporal belief revision,
//! ClusterIndex, CRC delta journal, in-flight learning, sleep-time consolidation) by supplying
//! ONLY device-side launch + DtoH/HtoD copy callbacks — zero duplicated compression or routing logic!
const std = @import("std");
const quant = @import("quant.zig");
const bank = @import("bank.zig");
const Allocator = std.mem.Allocator;

/// KV layout convention of a CUDA/HIP engine's device buffers.
pub const DeviceKvLayout = enum {
    /// `[kv_heads, seq_len, head_dim]` BF16 (direct match to `RawState` — zero transpose).
    head_major,
    /// `[seq_len, kv_heads, head_dim]` BF16 (automatically transposed during HtoD/DtoH staging).
    token_major,
};

/// Device-side hooks supplied by a CUDA/HIP family engine (e.g. `nemotron/cuda_engine.zig`, `qwen3_5/cuda_engine.zig`).
pub const DeviceOps = struct {
    ctx: *anyopaque,
    kv_layout: DeviceKvLayout = .head_major,
    /// Run layers `0..=layer_sem` on device for `ids` and copy BF16 residuals `[ids.len * hidden]` to host buffers.
    extractHiddenBf16: *const fn (
        ctx: *anyopaque,
        ids: []const u32,
        layer_lex: usize,
        layer_sem: usize,
        out_lex_bf16: []u16,
        out_sem_bf16: []u16,
    ) anyerror!void,
    /// Upload optional `base` KV + recurrent state to a scratch device slot, run `ids`, and download ONLY
    /// the new `ids.len` KV rows (in `kv_layout`) + final recurrent state into host buffers.
    compileDeviceState: *const fn (
        ctx: *anyopaque,
        base_len: u32,
        base_kv_k: []const []const u16,
        base_kv_v: []const []const u16,
        base_rec_state: []const []const f32,
        base_rec_conv: []const []const u16,
        ids: []const u32,
        out_kv_k: [][]u16,
        out_kv_v: [][]u16,
        out_rec_state: [][]f32,
        out_rec_conv: [][]u16,
    ) anyerror!void,
    /// Upload composed KV + recurrent state to a device snapshot slot, run `suffix_ids`, and return opaque device Snap handle.
    buildDeviceSnap: *const fn (
        ctx: *anyopaque,
        len: u32,
        kv_k: []const []const u16,
        kv_v: []const []const u16,
        rec_state: []const []const f32,
        rec_conv: []const []const u16,
        suffix_ids: []const u32,
    ) anyerror!bank.SnapHandle,
    /// Free a device snapshot previously created by `buildDeviceSnap`.
    dropDeviceSnap: *const fn (ctx: *anyopaque, snap: *anyopaque) void,
};

pub const CudaBridge = struct {
    gpa: Allocator,
    arch: bank.ArchSpec,
    ops: DeviceOps,

    pub fn init(gpa: Allocator, arch: bank.ArchSpec, ops: DeviceOps) !*CudaBridge {
        const b = try gpa.create(CudaBridge);
        b.* = .{ .gpa = gpa, .arch = arch, .ops = ops };
        return b;
    }

    pub fn deinit(self: *CudaBridge) void {
        self.gpa.destroy(self);
    }

    pub fn deinitOpaque(ptr: *anyopaque, gpa: Allocator) void {
        const self: *CudaBridge = @ptrCast(@alignCast(ptr));
        gpa.destroy(self);
    }

    pub fn backend(self: *CudaBridge) bank.Backend {
        return .{
            .ctx = self,
            .arch = self.arch,
            .extractHidden = extractHiddenFn,
            .compileState = compileStateFn,
            .buildSnap = buildSnapFn,
            .dropSnap = dropSnapFn,
        };
    }

    fn of(ctx: *anyopaque) *CudaBridge {
        return @ptrCast(@alignCast(ctx));
    }

    fn extractHiddenFn(
        ctx: *anyopaque,
        ids: []const u32,
        layer_lex: usize,
        layer_sem: usize,
        out_lex: []f32,
        out_sem: []f32,
    ) anyerror!void {
        const self = of(ctx);
        const elems = ids.len * self.arch.hidden;
        const lex_bf16 = try self.gpa.alloc(u16, elems);
        defer self.gpa.free(lex_bf16);
        const sem_bf16 = try self.gpa.alloc(u16, elems);
        defer self.gpa.free(sem_bf16);
        try self.ops.extractHiddenBf16(self.ops.ctx, ids, layer_lex, layer_sem, lex_bf16, sem_bf16);
        quant.bf16SliceToF32(lex_bf16, out_lex[0..elems]);
        quant.bf16SliceToF32(sem_bf16, out_sem[0..elems]);
    }

    fn compileStateFn(
        ctx: *anyopaque,
        base: ?*const bank.RawState,
        ids: []const u32,
        out: *bank.RawState,
    ) anyerror!void {
        const self = of(ctx);
        const arch = self.arch;
        const base_len: u32 = if (base) |b| b.len else 0;

        if (self.ops.kv_layout == .head_major) {
            const bk: []const []const u16 = if (base) |b| @ptrCast(b.kv_k) else &.{};
            const bv: []const []const u16 = if (base) |b| @ptrCast(b.kv_v) else &.{};
            const brs: []const []const f32 = if (base) |b| @ptrCast(b.rec_state) else &.{};
            const brc: []const []const u16 = if (base) |b| @ptrCast(b.rec_conv) else &.{};
            return self.ops.compileDeviceState(self.ops.ctx, base_len, bk, bv, brs, brc, ids, out.kv_k, out.kv_v, out.rec_state, out.rec_conv);
        }

        // Token-major device layout: transpose base to token-major and transpose downloaded output back to head-major
        const base_tm_k = try allocLayerBufs(self.gpa, arch.num_attn, if (base != null) arch.kvLayerElems(base_len) else 0);
        defer freeLayerBufs(self.gpa, base_tm_k);
        const base_tm_v = try allocLayerBufs(self.gpa, arch.num_attn, if (base != null) arch.kvLayerElems(base_len) else 0);
        defer freeLayerBufs(self.gpa, base_tm_v);
        if (base) |b| {
            for (0..arch.num_attn) |l| {
                transposeHeadToToken(b.kv_k[l], base_tm_k[l], arch.kv_heads, base_len, arch.head_dim);
                transposeHeadToToken(b.kv_v[l], base_tm_v[l], arch.kv_heads, base_len, arch.head_dim);
            }
        }
        const out_tm_k = try allocLayerBufs(self.gpa, arch.num_attn, arch.kvLayerElems(ids.len));
        defer freeLayerBufs(self.gpa, out_tm_k);
        const out_tm_v = try allocLayerBufs(self.gpa, arch.num_attn, arch.kvLayerElems(ids.len));
        defer freeLayerBufs(self.gpa, out_tm_v);

        const bk: []const []const u16 = if (base != null) @ptrCast(base_tm_k) else &.{};
        const bv: []const []const u16 = if (base != null) @ptrCast(base_tm_v) else &.{};
        const brs: []const []const f32 = if (base) |b| @ptrCast(b.rec_state) else &.{};
        const brc: []const []const u16 = if (base) |b| @ptrCast(b.rec_conv) else &.{};
        try self.ops.compileDeviceState(self.ops.ctx, base_len, bk, bv, brs, brc, ids, out_tm_k, out_tm_v, out.rec_state, out.rec_conv);

        for (0..arch.num_attn) |l| {
            transposeTokenToHead(out_tm_k[l], out.kv_k[l], arch.kv_heads, ids.len, arch.head_dim);
            transposeTokenToHead(out_tm_v[l], out.kv_v[l], arch.kv_heads, ids.len, arch.head_dim);
        }
    }

    fn buildSnapFn(
        ctx: *anyopaque,
        composed: *const bank.RawState,
        suffix_ids: []const u32,
    ) anyerror!bank.SnapHandle {
        const self = of(ctx);
        const arch = self.arch;
        const crs: []const []const f32 = @ptrCast(composed.rec_state);
        const crc: []const []const u16 = @ptrCast(composed.rec_conv);

        if (self.ops.kv_layout == .head_major) {
            const ck: []const []const u16 = @ptrCast(composed.kv_k);
            const cv: []const []const u16 = @ptrCast(composed.kv_v);
            return self.ops.buildDeviceSnap(self.ops.ctx, composed.len, ck, cv, crs, crc, suffix_ids);
        }

        const tm_k = try allocLayerBufs(self.gpa, arch.num_attn, arch.kvLayerElems(composed.len));
        defer freeLayerBufs(self.gpa, tm_k);
        const tm_v = try allocLayerBufs(self.gpa, arch.num_attn, arch.kvLayerElems(composed.len));
        defer freeLayerBufs(self.gpa, tm_v);
        for (0..arch.num_attn) |l| {
            transposeHeadToToken(composed.kv_k[l], tm_k[l], arch.kv_heads, composed.len, arch.head_dim);
            transposeHeadToToken(composed.kv_v[l], tm_v[l], arch.kv_heads, composed.len, arch.head_dim);
        }
        return self.ops.buildDeviceSnap(self.ops.ctx, composed.len, @ptrCast(tm_k), @ptrCast(tm_v), crs, crc, suffix_ids);
    }

    fn dropSnapFn(ctx: *anyopaque, snap: *anyopaque) void {
        const self = of(ctx);
        self.ops.dropDeviceSnap(self.ops.ctx, snap);
    }
};

fn allocLayerBufs(gpa: Allocator, n_layers: usize, elems: usize) ![][]u16 {
    const arr = try gpa.alloc([]u16, n_layers);
    var made: usize = 0;
    errdefer {
        for (arr[0..made]) |s| gpa.free(s);
        gpa.free(arr);
    }
    for (0..n_layers) |i| {
        arr[i] = try gpa.alloc(u16, elems);
        made += 1;
    }
    return arr;
}

fn freeLayerBufs(gpa: Allocator, arr: [][]u16) void {
    for (arr) |s| gpa.free(s);
    gpa.free(arr);
}

fn transposeHeadToToken(src: []const u16, dst: []u16, kv_heads: usize, seq_len: usize, head_dim: usize) void {
    for (0..seq_len) |t| {
        for (0..kv_heads) |h| {
            const s_off = (h * seq_len + t) * head_dim;
            const d_off = (t * kv_heads + h) * head_dim;
            @memcpy(dst[d_off..][0..head_dim], src[s_off..][0..head_dim]);
        }
    }
}

fn transposeTokenToHead(src: []const u16, dst: []u16, kv_heads: usize, seq_len: usize, head_dim: usize) void {
    for (0..seq_len) |t| {
        for (0..kv_heads) |h| {
            const s_off = (t * kv_heads + h) * head_dim;
            const d_off = (h * seq_len + t) * head_dim;
            @memcpy(dst[d_off..][0..head_dim], src[s_off..][0..head_dim]);
        }
    }
}

test "cuda_bridge(v6): token-major CUDA/HIP bridge round-trips end-to-end with Bank" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const FakeCuda = struct {
        gpa: Allocator,
        arch: bank.ArchSpec,
        snaps: usize = 0,

        fn extractHiddenBf16(ctx: *anyopaque, ids: []const u32, _: usize, _: usize, out_lex: []u16, out_sem: []u16) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const d = self.arch.hidden;
            @memset(out_lex, 0);
            @memset(out_sem, 0);
            for (ids, 0..) |id, t| {
                out_lex[t * d + 0] = quant.f32ToBf16(4.0);
                out_lex[t * d + 1] = quant.f32ToBf16(-3.0);
                out_lex[t * d + (4 + (id % (d - 4)))] = quant.f32ToBf16(5.0);
                out_sem[t * d + 0] = quant.f32ToBf16(3.0);
                out_sem[t * d + 2] = quant.f32ToBf16(-2.0);
                out_sem[t * d + (4 + (id % (d - 4)))] = quant.f32ToBf16(5.0);
            }
        }

        fn compileDeviceState(
            ctx: *anyopaque,
            _: u32,
            _: []const []const u16,
            _: []const []const u16,
            _: []const []const f32,
            _: []const []const u16,
            ids: []const u32,
            out_kv_k: [][]u16,
            out_kv_v: [][]u16,
            _: [][]f32,
            _: [][]u16,
        ) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (0..self.arch.num_attn) |l| {
                for (0..ids.len) |t| {
                    for (0..self.arch.kv_heads) |h| {
                        const off = (t * self.arch.kv_heads + h) * self.arch.head_dim;
                        for (0..self.arch.head_dim) |c| {
                            out_kv_k[l][off + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c) % 13)) * 0.1);
                            out_kv_v[l][off + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c + 1) % 11)) * 0.1);
                        }
                    }
                }
            }
        }

        fn buildDeviceSnap(
            ctx: *anyopaque,
            len: u32,
            _: []const []const u16,
            _: []const []const u16,
            _: []const []const f32,
            _: []const []const u16,
            suffix_ids: []const u32,
        ) !bank.SnapHandle {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const p = try self.gpa.create(u32);
            p.* = len + @as(u32, @intCast(suffix_ids.len));
            self.snaps += 1;
            return .{ .saved = p, .at = p.* };
        }

        fn dropDeviceSnap(ctx: *anyopaque, snap: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const p: *u32 = @ptrCast(@alignCast(snap));
            self.gpa.destroy(p);
            self.snaps -= 1;
        }
    };

    const arch = bank.ArchSpec{
        .family = .cuda_attn,
        .hidden = 64,
        .num_layers = 4,
        .layer_lex = 1,
        .layer_sem = 2,
        .num_attn = 2,
        .kv_heads = 2,
        .head_dim = 32,
        .rotary_dim = 16,
        .rope_theta = 10_000.0,
        .num_rec = 0,
        .rec_heads = 0,
        .rec_rows = 0,
        .rec_cols = 0,
        .conv_elems = 0,
        .svd_rank = 0,
        .svd_rank_short = 0,
    };

    var dev = FakeCuda{ .gpa = gpa, .arch = arch };
    var bridge = try CudaBridge.init(gpa, arch, .{
        .ctx = &dev,
        .kv_layout = .token_major,
        .extractHiddenBf16 = FakeCuda.extractHiddenBf16,
        .compileDeviceState = FakeCuda.compileDeviceState,
        .buildDeviceSnap = FakeCuda.buildDeviceSnap,
        .dropDeviceSnap = FakeCuda.dropDeviceSnap,
    });
    defer bridge.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    var b = try bank.Bank.init(gpa, io, dir, bridge.backend());
    defer b.deinit();

    const cal_a = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const cal_b = [_]u32{ 6, 7, 8, 9, 10, 11 };
    const cal_ids = [_][]const u32{ &cal_a, &cal_b };
    const pref_ids = [_]u32{ 1, 2, 3 };
    const suf_ids = [_]u32{ 4, 5 };

    try b.ensureCalibrated(&cal_ids, &pref_ids, &suf_ids);
    const f_ids = [_]u32{ 30, 31, 32, 33 };
    _ = try b.teachChunk(.{ .parent_id = 1, .body_ids = &f_ids, .gate_ids = &f_ids, .is_small_fact = true });
    try std.testing.expect((try b.save()) > 0);

    const snap = (try b.routeAndCompose(&[_]u32{ 30, 31, 32 })) orelse return error.ExpectedCudaHit;
    try std.testing.expectEqual(@as(u32, 3 + 4 + 2), snap.at);
    bridge.backend().dropSnap(bridge.backend().ctx, snap.saved);
    try std.testing.expectEqual(@as(usize, 0), dev.snaps);
}

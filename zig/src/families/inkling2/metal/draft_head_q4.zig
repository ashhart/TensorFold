//! Draft-only 4-bit/64 affine unembedding copy shared across Inkling 2 Metal drafters when TF_INKLING2_DRAFT_HEAD_Q4=1.
const std = @import("std");
const bf = @import("bf16_round");
const mtl = @import("metal");
const sys = @import("sys");
const sources = @import("inkling2_metal");

pub const env_name: [:0]const u8 = "TF_INKLING2_DRAFT_HEAD_Q4";
pub const Ref = @import("inkling2_metal_ref").Ref;
pub const LogitsArgs = extern struct { rows: u32, vocab: u32, k: u32, softcap: u32 };
pub const symbol = "tf_ink2_dspark_logits_q4";
pub const legacy_threads: u32 = 256;
pub const batch_symbol = "tf_ink2_draft_head_q4_logits";
pub const batch_threads: u32 = 128;
pub const group_size: usize = 64;
pub const max_workers: usize = 16;

pub const QuantizeError = error{ NonFiniteAffineInput, NonFiniteAffineRange };
pub const GeometryError = error{BadDraftHeadGeometry};

pub const Layout = struct {
    words_bytes: usize,
    scales_bytes: usize,
    scales_off: usize,
    biases_off: usize,
    total_bytes: usize,
};

pub fn layout(n: u32, k: u32) GeometryError!Layout {
    if (n == 0 or k == 0 or k % group_size != 0) return error.BadDraftHeadGeometry;
    const rows: usize = @intCast(n);
    const width: usize = @intCast(k);
    const words_bytes = rows * (width / 8) * @sizeOf(u32);
    const scales_bytes = rows * (width / group_size) * @sizeOf(u16);
    const scales_off = std.mem.alignForward(usize, words_bytes, 256);
    const biases_off = std.mem.alignForward(usize, scales_off + scales_bytes, 256);
    const total_bytes = std.mem.alignForward(usize, biases_off + scales_bytes, 256);
    return .{
        .words_bytes = words_bytes,
        .scales_bytes = scales_bytes,
        .scales_off = scales_off,
        .biases_off = biases_off,
        .total_bytes = total_bytes,
    };
}

pub fn bytesNeeded(n: u32, k: u32) GeometryError!u64 {
    return @intCast((try layout(n, k)).total_bytes);
}

pub fn enabled() sys.environment.Error!bool {
    return sys.environment.read(bool, env_name, false);
}

pub const DraftHeadQ4 = struct {
    buf: mtl.Buffer,
    scales_off: usize,
    biases_off: usize,
    n: u32,
    k: u32,

    pub fn deinit(h: *DraftHeadQ4) void {
        h.buf.deinit();
        h.* = undefined;
    }

    pub fn fixedBytes(h: *const DraftHeadQ4) u64 {
        return h.buf.length();
    }

    pub fn wordsRef(h: *const DraftHeadQ4) Ref {
        return .{ .buf = h.buf, .off = 0 };
    }

    pub fn scalesRef(h: *const DraftHeadQ4) Ref {
        return .{ .buf = h.buf, .off = h.scales_off };
    }

    pub fn biasesRef(h: *const DraftHeadQ4) Ref {
        return .{ .buf = h.buf, .off = h.biases_off };
    }
};

pub fn asFloat(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn init(device: mtl.Device, unembed_bf16: []const u16, n: u32, k: u32) !DraftHeadQ4 {
    const l = try layout(n, k);
    const need_elems = @as(usize, @intCast(n)) * @as(usize, @intCast(k));
    if (unembed_bf16.len < need_elems) return error.BadDraftHeadGeometry;
    const buf = try device.buffer(l.total_bytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const groups = (@as(usize, @intCast(n)) * @as(usize, @intCast(k))) / group_size;
    const dst_w: [*]u32 = @ptrCast(@alignCast(buf.contents()));
    const dst_s: [*]u16 = @ptrCast(@alignCast(buf.contents() + l.scales_off));
    const dst_b: [*]u16 = @ptrCast(@alignCast(buf.contents() + l.biases_off));
    try quantizeParallel(unembed_bf16.ptr, dst_w, dst_s, dst_b, groups);
    return .{ .buf = buf, .scales_off = l.scales_off, .biases_off = l.biases_off, .n = n, .k = k };
}

pub fn maybeInit(device: mtl.Device, unembed_bf16: []const u16, n: u32, k: u32, used_bytes: u64, limit_bytes: u64) !?DraftHeadQ4 {
    if (!try enabled()) return null;
    const need = try bytesNeeded(n, k);
    if (used_bytes > limit_bytes or need > limit_bytes - used_bytes) return null;
    return try init(device, unembed_bf16, n, k);
}

fn loadNamedPipeline(device: mtl.Device, text: []const u8, sym: [:0]const u8, min_threads: u32) !mtl.Pipeline {
    const opts = mtl.CompileOptions{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
    const lib = try mtl.Library.fromSource(device, text, opts);
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, sym, false);
    errdefer pipe.deinit();
    if (pipe.maxThreads() < min_threads) return error.MetalThreadgroupTooSmall;
    return pipe;
}

pub fn loadPipeline(device: mtl.Device) !mtl.Pipeline {
    const text = if (device.tensorUnits()) sources.inkling2_dspark else sources.inkling2_dspark_pre_m5;
    return loadNamedPipeline(device, text, symbol, legacy_threads);
}

pub fn loadBatchPipeline(device: mtl.Device) !mtl.Pipeline {
    const text = if (device.tensorUnits()) sources.inkling2_dflash2_fast else sources.inkling2_dflash2_fast_pre_m5;
    return loadNamedPipeline(device, text, batch_symbol, batch_threads);
}

fn encodeDispatch(
    pipeline: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    w: Ref,
    scales: Ref,
    biases: Ref,
    y: Ref,
    out: Ref,
    rows: u32,
    vocab_count: u32,
    k: u32,
    softcap: bool,
    tile_n: usize,
    tile_m: usize,
    threads: usize,
) void {
    std.debug.assert(rows > 0 and vocab_count > 0 and k > 0 and k % group_size == 0);
    e.setPipeline(pipeline);
    const refs = [_]Ref{ w, scales, biases, y, out };
    for (refs, 0..) |item, slot| e.setBuffer(item.buf, item.off, slot);
    e.setValue(LogitsArgs{ .rows = rows, .vocab = vocab_count, .k = k, .softcap = @intFromBool(softcap) }, 5);
    const groups_x = (@as(usize, vocab_count) + tile_n - 1) / tile_n;
    const groups_y = (@as(usize, rows) + tile_m - 1) / tile_m;
    e.dispatchGroups(mtl.Size.of(groups_x, groups_y, 1), mtl.Size.of(threads, 1, 1));
}

pub fn dispatch(
    pipeline: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    w: Ref,
    scales: Ref,
    biases: Ref,
    y: Ref,
    out: Ref,
    rows: u32,
    vocab_count: u32,
    k: u32,
    softcap: bool,
) void {
    encodeDispatch(pipeline, e, w, scales, biases, y, out, rows, vocab_count, k, softcap, 8, 8, legacy_threads);
}

/// Dispatch the 32-row batched Q4 logits kernel (single weight read across up to 32 rows).
pub fn dispatchBatch(
    pipeline: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    w: Ref,
    scales: Ref,
    biases: Ref,
    y: Ref,
    out: Ref,
    rows: u32,
    vocab_count: u32,
    k: u32,
    softcap: bool,
) void {
    encodeDispatch(pipeline, e, w, scales, biases, y, out, rows, vocab_count, k, softcap, 64, 32, batch_threads);
}

pub fn tf_ink2_draft_head_q4_logits(
    pipeline: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    head: *const DraftHeadQ4,
    y: Ref,
    out: Ref,
    rows: u32,
    softcap: bool,
) void {
    dispatch(pipeline, e, head.wordsRef(), head.scalesRef(), head.biasesRef(), y, out, rows, head.n, head.k, softcap);
}

pub fn encodeBatchHead(
    pipeline: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    head: *const DraftHeadQ4,
    y: Ref,
    out: Ref,
    rows: u32,
    softcap: bool,
) void {
    dispatchBatch(pipeline, e, head.wordsRef(), head.scalesRef(), head.biasesRef(), y, out, rows, head.n, head.k, softcap);
}

pub fn quantizeParallel(src_bf16: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, groups: usize) QuantizeError!void {
    if (groups == 0) return;
    const cpu = std.Thread.getCpuCount() catch 1;
    const workers = @min(max_workers, cpu, @max(1, groups / 4096));
    const per = (groups + workers - 1) / workers;
    var results: [max_workers]QuantizeError!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |t| {
        const first = @min(groups, t * per);
        const last = @min(groups, (t + 1) * per);
        if (first < last) threads[t] = std.Thread.spawn(.{}, runRange, .{ &results[t], src_bf16, dst_w, dst_s, dst_b, first, last }) catch null;
    }
    runRange(&results[0], src_bf16, dst_w, dst_s, dst_b, 0, @min(groups, per));
    for (1..workers) |t| {
        const first = @min(groups, t * per);
        const last = @min(groups, (t + 1) * per);
        if (threads[t]) |th| th.join() else if (first < last) runRange(&results[t], src_bf16, dst_w, dst_s, dst_b, first, last);
    }
    for (results[0..workers]) |res| try res;
}

fn runRange(out: *QuantizeError!void, src_bf16: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, first: usize, last: usize) void {
    out.* = quantizeGroups(src_bf16, dst_w, dst_s, dst_b, first, last);
}

/// Pack 64-element BF16 groups into uint32 nibbles with BF16 scales and biases matching s * q + b.
pub fn quantizeGroups(src_bf16: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, first: usize, last: usize) QuantizeError!void {
    for (first..last) |g| {
        var vals: [group_size]f32 = undefined;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = -std.math.inf(f32);
        const src = src_bf16 + g * group_size;
        for (0..group_size) |i| {
            const v = asFloat(src[i]);
            if (!std.math.isFinite(v)) return error.NonFiniteAffineInput;
            vals[i] = v;
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        const b_bits = bf.bits(lower);
        const b: f32 = asFloat(b_bits);
        const s_raw: f32 = @floatCast(if (upper > b) (@as(f64, upper) - @as(f64, b)) / 15.0 else 0.0);
        const s_bits = bf.bits(s_raw);
        const s: f32 = asFloat(s_bits);
        if (!std.math.isFinite(b) or !std.math.isFinite(s)) return error.NonFiniteAffineRange;
        dst_s[g] = s_bits;
        dst_b[g] = b_bits;
        const w = dst_w[g * 8 ..][0..8];
        @memset(w, 0);
        if (s > 0) {
            for (vals, 0..) |v, idx| {
                const q: u32 = @intFromFloat(@min(@as(f32, 15.0), @max(@as(f32, 0.0), @round((v - b) / s))));
                w[idx / 8] |= q << @as(u5, @intCast(4 * (idx % 8)));
            }
        }
    }
}

fn roundBf(v: f64) f64 {
    return asFloat(bf.bits(@floatCast(v)));
}

test "Q4 draft head layout matches Mini 200058x2304 byte accounting" {
    const l = try layout(200058, 2304);
    try std.testing.expectEqual(@as(usize, 230_466_816), l.words_bytes);
    try std.testing.expectEqual(@as(usize, 14_404_176), l.scales_bytes);
    try std.testing.expectEqual(@as(usize, 230_466_816), l.scales_off);
    try std.testing.expectEqual(@as(usize, 244_871_168), l.biases_off);
    try std.testing.expectEqual(@as(u64, 259_275_520), try bytesNeeded(200058, 2304));
    try std.testing.expectError(error.BadDraftHeadGeometry, layout(0, 64));
    try std.testing.expectError(error.BadDraftHeadGeometry, layout(8, 96));
}

test "parallel quantization matches serial bit-for-bit across constant, signed and mixed groups" {
    const a = std.testing.allocator;
    const rows = 4096;
    const ng = 2;
    const groups = rows * ng;
    const src = try a.alloc(u16, groups * group_size);
    defer a.free(src);
    var prng = std.Random.DefaultPrng.init(4961);
    for (0..groups) |g| {
        const base = src[g * group_size ..][0..group_size];
        switch (g % 5) {
            0 => @memset(base, 0),
            1 => @memset(base, bf.bits(-3.25)),
            2 => for (base) |*dst| {
                dst.* = bf.bits(-1.0 - prng.random().float(f32) * 3.0);
            },
            3 => for (base) |*dst| {
                dst.* = bf.bits(0.5 + prng.random().float(f32) * 2.5);
            },
            else => for (base) |*dst| {
                dst.* = bf.bits((prng.random().float(f32) * 2.0 - 1.0) * 1.5);
            },
        }
    }
    const w0 = try a.alloc(u32, groups * 8);
    defer a.free(w0);
    const s0 = try a.alloc(u16, groups);
    defer a.free(s0);
    const b0 = try a.alloc(u16, groups);
    defer a.free(b0);
    const w1 = try a.alloc(u32, groups * 8);
    defer a.free(w1);
    const s1 = try a.alloc(u16, groups);
    defer a.free(s1);
    const b1 = try a.alloc(u16, groups);
    defer a.free(b1);
    try quantizeGroups(src.ptr, w0.ptr, s0.ptr, b0.ptr, 0, groups);
    try quantizeParallel(src.ptr, w1.ptr, s1.ptr, b1.ptr, groups);
    try std.testing.expectEqualSlices(u32, w0, w1);
    try std.testing.expectEqualSlices(u16, s0, s1);
    try std.testing.expectEqualSlices(u16, b0, b1);
    for (0..groups) |g| {
        const s = asFloat(s0[g]);
        const b = asFloat(b0[g]);
        const tol = @max(s * 0.55, @abs(b) * 0.01) + 1e-6;
        for (0..group_size) |i| {
            const want = asFloat(src[g * group_size + i]);
            const code = (w0[g * 8 + i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 0xf;
            const got = s * @as(f32, @floatFromInt(code)) + b;
            try std.testing.expect(@abs(want - got) <= tol);
        }
    }
    src[17] = 0x7f80;
    try std.testing.expectError(error.NonFiniteAffineInput, quantizeGroups(src.ptr, w0.ptr, s0.ptr, b0.ptr, 0, 1));
    src[17] = 0x7fc0;
    try std.testing.expectError(error.NonFiniteAffineInput, quantizeGroups(src.ptr, w0.ptr, s0.ptr, b0.ptr, 0, 1));
}

test "maybeInit respects opt-in flag, malformed refusal, budget and both Metal Q4 kernels" {
    const saved = sys.environment.rawText(env_name);
    const saved_copy = if (saved) |val| try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}", .{val}, 0) else null;
    defer {
        if (saved_copy) |val| {
            _ = sys.env.setenv(env_name, val.ptr, 1);
            std.testing.allocator.free(val);
        } else _ = sys.env.unsetenv(env_name);
    }
    _ = sys.env.unsetenv(env_name);
    try std.testing.expect(!try enabled());
    var dummy: [128]u16 = @splat(0);
    try std.testing.expect((try maybeInit(undefined, &dummy, 2, 64, 0, 1 << 40)) == null);
    _ = sys.env.setenv(env_name, "0", 1);
    try std.testing.expect(!try enabled());
    _ = sys.env.setenv(env_name, "invalid", 1);
    try std.testing.expectError(error.InvalidEnvironment, enabled());
    _ = sys.env.setenv(env_name, "1", 1);
    try std.testing.expect(try enabled());
    const need_mini = try bytesNeeded(200058, 2304);
    try std.testing.expect((try maybeInit(undefined, &dummy, 200058, 2304, 1024, 1024 + need_mini - 1)) == null);
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const pipe = try loadPipeline(dev);
    defer pipe.deinit();
    const batch_pipe = try loadBatchPipeline(dev);
    defer batch_pipe.deinit();
    const pre_batch_pipe = try loadNamedPipeline(dev, sources.inkling2_dflash2_fast_pre_m5, batch_symbol, batch_threads);
    defer pre_batch_pipe.deinit();
    const n: u32 = 67;
    const k: u32 = 128;
    const rows: u32 = 33;
    const w_bf = try std.testing.allocator.alloc(u16, n * k);
    defer std.testing.allocator.free(w_bf);
    var prng = std.Random.DefaultPrng.init(77);
    for (w_bf) |*slot| slot.* = bf.bits((prng.random().float(f32) * 2.0 - 1.0) * 0.5);
    const need_small = try bytesNeeded(n, k);
    var head = (try maybeInit(dev, w_bf, n, k, 64, 64 + need_small)).?;
    defer head.deinit();
    const y_buf = try dev.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer y_buf.deinit();
    const out_buf = try dev.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer out_buf.deinit();
    const y_slice = y_buf.slice(u16, rows * k);
    for (y_slice) |*slot| slot.* = bf.bits((prng.random().float(f32) * 2.0 - 1.0) * 2.5);
    const dst_s = @as([*]const u16, @ptrCast(@alignCast(head.buf.contents() + head.scales_off)));
    const dst_b = @as([*]const u16, @ptrCast(@alignCast(head.buf.contents() + head.biases_off)));
    const dst_w = @as([*]const u32, @ptrCast(@alignCast(head.buf.contents())));
    for ([_]mtl.Pipeline{ pipe, batch_pipe, pre_batch_pipe }) |p| {
        const use_batch = (p.id != pipe.id);
        for ([_]bool{ false, true }) |softcap| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = q.commandBuffer();
            const e = cb.compute(.serial);
            if (use_batch) {
                encodeBatchHead(p, e, &head, .{ .buf = y_buf }, .{ .buf = out_buf }, rows, softcap);
            } else {
                tf_ink2_draft_head_q4_logits(p, e, &head, .{ .buf = y_buf }, .{ .buf = out_buf }, rows, softcap);
            }
            e.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
            const got_slice = out_buf.slice(u16, rows * n);
            const ng = k / 64;
            for (0..rows) |r_idx| {
                for (0..n) |v_idx| {
                    var acc: f64 = 0;
                    for (0..k) |d_idx| {
                        const g = v_idx * ng + d_idx / 64;
                        const q4 = (dst_w[v_idx * (k / 8) + d_idx / 8] >> @as(u5, @intCast(4 * (d_idx % 8)))) & 0xf;
                        const wv = asFloat(dst_s[g]) * @as(f32, @floatFromInt(q4)) + asFloat(dst_b[g]);
                        acc += @as(f64, wv) * @as(f64, asFloat(y_slice[r_idx * k + d_idx]));
                    }
                    const raw = roundBf(acc);
                    const want = if (softcap) roundBf(30.0 * std.math.tanh(raw / 30.0)) else raw;
                    const got: f64 = asFloat(got_slice[r_idx * n + v_idx]);
                    try std.testing.expect(@abs(want - got) <= @max(@abs(want) * 1e-2, 0.05));
                }
            }
        }
    }
}

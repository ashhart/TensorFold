//! GPU BF16 -> AWQ W4A16 g128 quantizer, packed AWQ upload, and split-output helpers.
const std = @import("std");
const mtl = @import("metal");
const awq = @import("awq.zig");
const linear = @import("linear_gpu.zig");

const LinearGpu = linear.LinearGpu;
const Bf16Matrix = linear.Bf16Matrix;
const AwqMatrix = linear.AwqMatrix;
const OutSplits = linear.OutSplits;
const Ref = linear.Ref;
const Dims = linear.LinearGpu.Dims;

const QuantArgs = extern struct { n: u32, k: u32, m5_layout: u32, n_eff: u32 };

pub fn prepareAwqFromPacked(g: *const LinearGpu, b: awq.PackedBundle) !AwqMatrix {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const w_bytes = b.w.len * 4;
    const sb_bytes = b.sb.len * @sizeOf(awq.Float2);
    const wb = try g.device.buffer(w_bytes, opts);
    errdefer wb.deinit();
    @memcpy(wb.contents()[0..w_bytes], std.mem.sliceAsBytes(b.w));
    const sbb = try g.device.buffer(sb_bytes, opts);
    errdefer sbb.deinit();
    @memcpy(sbb.contents()[0..sb_bytes], std.mem.sliceAsBytes(b.sb));
    var is_buf: ?mtl.Buffer = null;
    if (b.input_scale) |is| {
        const ib = try g.device.buffer(is.len * 4, opts);
        errdefer ib.deinit();
        @memcpy(ib.contents()[0 .. is.len * 4], std.mem.sliceAsBytes(is));
        is_buf = ib;
    }
    return .{
        .w = wb,
        .sb = sbb,
        .input_scale = is_buf,
        .n = @intCast(b.n),
        .k = @intCast(b.k),
        .m5_layout = b.layout == .m5_tile,
        .owns = true,
    };
}

pub fn prepareAwq(g: *const LinearGpu, parts: []const awq.Part) ![]AwqMatrix {
    const layout: awq.Layout = if (g.tensor_units) .m5_tile else .m3_row;
    const bundles = try awq.packBundles(g.allocator, parts, layout);
    defer {
        for (bundles) |*b| b.deinit(g.allocator);
        g.allocator.free(bundles);
    }
    const out = try g.allocator.alloc(AwqMatrix, bundles.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*m| m.deinit();
        g.allocator.free(out);
    }
    for (bundles) |b| {
        out[made] = try prepareAwqFromPacked(g, b);
        made += 1;
    }
    return out;
}

/// Quantize row-major BF16 weight matrix [N, K] on `queue` into AWQ W4A16 g128 `AwqMatrix`.
pub fn quantizeBf16ToAwq(g: *LinearGpu, queue: mtl.Queue, src: Bf16Matrix) !AwqMatrix {
    if (src.n == 0 or src.n % 8 != 0 or src.k == 0 or src.k % 128 != 0) return error.InvalidShape;
    if (src.buf.length() < src.off + @as(usize, src.n) * src.k * 2) return error.BufferTooSmall;
    const layout: awq.Layout = if (g.tensor_units) .m5_tile else .m3_row;
    const w_words = awq.weightWords(src.n, src.k, layout);
    const sb_count = awq.metaCount(src.n, src.k);
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const wb = try g.device.buffer(w_words * 4, opts);
    errdefer wb.deinit();
    const sbb = try g.device.buffer(sb_count * @sizeOf(awq.Float2), opts);
    errdefer sbb.deinit();
    const p = try g.pipeline(.{ .kind = .awq_quant });
    const n_eff: u32 = if (g.tensor_units) std.mem.alignForward(u32, src.n, 32) else src.n;
    const kg: u32 = src.k / 128;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(p);
    e.setBuffer(src.buf, src.off, 0);
    e.setValue(QuantArgs{ .n = src.n, .k = src.k, .m5_layout = @intFromBool(g.tensor_units), .n_eff = n_eff }, 1);
    e.setBuffer(wb, 0, 2);
    e.setBuffer(sbb, 0, 3);
    e.dispatchGroups(mtl.Size.of((n_eff + 7) / 8, kg, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("quantizeBf16ToAwq gpu err: {s}", .{why});
        return error.GpuFailed;
    }
    return .{ .w = wb, .sb = sbb, .n = src.n, .k = src.k, .m5_layout = g.tensor_units, .owns = true };
}

pub fn validateSplits(n: u32, ldy: u32, splits: OutSplits) !void {
    if (splits.y1 != null) {
        if (splits.split1 == 0 or splits.split1 >= n or splits.split1 % 128 != 0) return error.InvalidShape;
        if (ldy < splits.split1) return error.InvalidShape;
        const end1 = if (splits.y2 != null) splits.split2 else n;
        if (splits.y2 != null) {
            if (splits.split2 <= splits.split1 or splits.split2 >= n or splits.split2 % 128 != 0) return error.InvalidShape;
            if (splits.ldy2 < n - splits.split2) return error.InvalidShape;
        }
        if (splits.ldy1 < end1 - splits.split1) return error.InvalidShape;
    } else {
        if (splits.y2 != null or splits.split1 != 0 or splits.split2 != 0) return error.InvalidShape;
        if (ldy < n) return error.InvalidShape;
    }
}

pub fn splitAddrs(e: mtl.ComputeEncoder, splits: OutSplits) [2]u64 {
    if (splits.y1) |y1| e.useResource(y1.buf, mtl.ResourceUsage.write);
    if (splits.y2) |y2| e.useResource(y2.buf, mtl.ResourceUsage.write);
    return .{
        if (splits.y1) |y1| y1.buf.gpuAddress() + y1.off else 0,
        if (splits.y2) |y2| y2.buf.gpuAddress() + y2.off else 0,
    };
}

pub fn dispatchBf16SplitKSplit(
    kp: mtl.Pipeline,
    sp: mtl.Pipeline,
    e: mtl.ComputeEncoder,
    x: Ref,
    w: Ref,
    scratch: Ref,
    y: Ref,
    d: Dims,
    splits: u32,
    out_splits: OutSplits,
) void {
    const tn = (d.n + 63) / 64;
    const tm = (d.rows + 63) / 64;
    const kp_k = ((d.k / splits + 255) / 256) * 256;
    var prm = [_]i32{
        @intCast(d.rows), @intCast(d.n),          @intCast(d.k),  @intCast(d.k),
        @intCast(d.k),    @intCast(d.n),          @intCast(tn),   @intCast(tm),
        0,                @intCast(d.rows * d.n), @intCast(kp_k), 2,
        0,                0,                      0,              0,
    };
    e.setPipeline(kp);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setBytes(std.mem.asBytes(&prm), 2);
    e.setBuffer(scratch.buf, scratch.off, 3);
    e.dispatchGroups(mtl.Size.of(splits * (tn << 2) * ((tm + 3) >> 2), 1, 1), mtl.Size.of(32, 2, 2));
    e.barrier();
    const addrs = splitAddrs(e, out_splits);
    const sprm = linear.Bf16SumArgs{
        .p = .{ @intCast(splits), @intCast(d.rows * d.n), @intCast(d.n), @intCast(d.n), @intCast(d.rows), @intCast(d.ldy) },
        .y1_addr = addrs[0],
        .y2_addr = addrs[1],
        .split1 = out_splits.split1,
        .ldy1 = out_splits.ldy1,
        .split2 = out_splits.split2,
        .ldy2 = out_splits.ldy2,
    };
    e.setPipeline(sp);
    e.setBuffer(scratch.buf, scratch.off, 0);
    e.setValue(sprm, 1);
    e.setBuffer(y.buf, y.off, 2);
    e.dispatchGroups(mtl.Size.of((d.n + 31) / 32, (d.rows + 7) / 8, 1), mtl.Size.of(32, 8, 1));
}

pub fn dispatchBf16GemmSplit(gp: mtl.Pipeline, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, d: Dims, splits: OutSplits) void {
    const tn = (d.n + 127) / 128;
    const tm = (d.rows + 63) / 64;
    const addrs = splitAddrs(e, splits);
    var args = linear.Bf16GemmArgs{
        .y1_addr = addrs[0],
        .y2_addr = addrs[1],
        .split1 = splits.split1,
        .ldy1 = splits.ldy1,
        .split2 = splits.split2,
        .ldy2 = splits.ldy2,
    };
    for ([_]usize{ d.rows, d.n, d.k, d.k, d.k, d.ldy, tn, tm, 2, d.k / 256 }, 0..) |v, i| args.p[i] = @intCast(v);
    e.setPipeline(gp);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setValue(args, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.dispatchGroups(mtl.Size.of(tn * 4, (tm + 3) / 4, 1), mtl.Size.of(32, 4, 2));
}

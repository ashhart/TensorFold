//! Layer 0's second hyper-connection: the mixer branch written back, then the same read-out on the mlp weights.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");

const branch_sk: usize = 8;

/// `slices` are the output face's unreduced K slices. `h` is the four streams, `inj` the first connection's gates.
pub fn connect(comptime Tri: type, gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, mapped: *const embed.Mapped, tri: Tri, h: u64, inj: u64, hidden: []const u16, slices: []const f32) !u32 {
    if (hidden.len == 0 or hidden.len % 4 != 0) return error.UnexpectedTensor;
    const streams: usize = 4;
    const dims = hidden.len / streams;
    if (slices.len != branch_sk * dims) return error.UnexpectedTensor;
    const inj_bytes = try gpa.alloc(u8, streams * 2);
    defer gpa.free(inj_bytes);
    try stream.synchronize();
    var inj_buf: cuda.DeviceBuffer = .{ .d = driver, .ptr = inj, .len = inj_bytes.len };
    try inj_buf.download(0, inj_bytes);
    const inj_u16 = try gpa.alloc(u16, streams);
    defer gpa.free(inj_u16);
    for (inj_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, inj_bytes[2 * i ..][0..2], .little);
    const branch = try gpa.alloc(u16, dims);
    defer gpa.free(branch);
    try hc.sumSlices(slices, branch_sk, dims, branch);
    const host_h = try gpa.alloc(u16, hidden.len);
    defer gpa.free(host_h);
    try hc.applyBranch(hidden, branch, inj_u16, dims, streams, host_h);

    var part = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(slices));
    defer part.free();
    const nc = dims / 256;
    var pss_b = try cuda.DeviceBuffer.alloc(driver, nc * streams * 4);
    defer pss_b.free();
    try tri.hcWriteback(h, h, pss_b.ptr, part.ptr, "*fp32", inj, h, "*bf16", h, "*bf16", dims, 1, dims, streams, 4, 1, 1, branch_sk);
    const got_h = try gpa.alloc(u8, hidden.len * 2);
    defer gpa.free(got_h);
    const pss_bytes = try gpa.alloc(u8, nc * streams * 4);
    defer gpa.free(pss_bytes);
    try stream.synchronize();
    var h_buf: cuda.DeviceBuffer = .{ .d = driver, .ptr = h, .len = got_h.len };
    try h_buf.download(0, got_h);
    try pss_b.download(0, pss_bytes);
    var h_off: usize = 0;
    var h_steps: u32 = 0;
    for (host_h, 0..) |want, i| {
        const got = std.mem.readInt(u16, got_h[2 * i ..][0..2], .little);
        const dist = hc.mixedSteps(got, want);
        if (dist != 0) h_off += 1;
        if (dist > h_steps) h_steps = dist;
    }
    const want_pss = try gpa.alloc(f32, nc * streams);
    defer gpa.free(want_pss);
    try hc.partialSums(host_h, dims, streams, want_pss);
    var pss_ulp: u32 = 0;
    for (want_pss, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, pss_bytes[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > pss_ulp) pss_ulp = dist;
    }
    std.debug.print("mlp writeback off {d} max_steps {d} pss_ulp {d} head {x:0>4} host {x:0>4} branch {x:0>4}\n", .{ h_off, h_steps, pss_ulp, std.mem.readInt(u16, got_h[0..2], .little), host_h[0], branch[0] });
    if (h_steps != 0 or pss_ulp > 4) return 1;

    const prefix = "language_model.model.layers.0.mlp_hyper_connection.";
    var name: [180]u8 = undefined;
    const down_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.weight", .{prefix}), .u32);
    const down_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.scales", .{prefix}), .bf16);
    const down_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.biases", .{prefix}), .bf16);
    const inj_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.weight", .{prefix}), .u32);
    const inj_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.scales", .{prefix}), .bf16);
    const inj_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.biases", .{prefix}), .bf16);
    const gamma = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}hc_norm.weight", .{prefix}), .bf16);
    const up_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.weight", .{prefix}), .u32);
    const up_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.scales", .{prefix}), .bf16);
    const up_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.biases", .{prefix}), .bf16);
    const n = down_w.dim(0) + inj_w.dim(0);
    const k8 = down_w.dim(1);
    const kg = k8 / 4;
    if (n != 324 or k8 != 1280 or gamma.dim(0) != streams * dims) return error.UnexpectedTensor;
    const stacked_w = try gpa.alloc(u8, down_w.bytes.len + inj_w.bytes.len);
    defer gpa.free(stacked_w);
    @memcpy(stacked_w[0..down_w.bytes.len], down_w.bytes);
    @memcpy(stacked_w[down_w.bytes.len..], inj_w.bytes);
    const stacked_s = try gpa.alloc(u8, down_s.bytes.len + inj_s.bytes.len);
    defer gpa.free(stacked_s);
    @memcpy(stacked_s[0..down_s.bytes.len], down_s.bytes);
    @memcpy(stacked_s[down_s.bytes.len..], inj_s.bytes);
    const stacked_b = try gpa.alloc(u8, down_b.bytes.len + inj_b.bytes.len);
    defer gpa.free(stacked_b);
    @memcpy(stacked_b[0..down_b.bytes.len], down_b.bytes);
    @memcpy(stacked_b[down_b.bytes.len..], inj_b.bytes);
    const tiled = try hc.tileWords(gpa, stacked_w, n, k8);
    defer gpa.free(tiled);
    const scales_t = try hc.transposeBf16(gpa, stacked_s, n, kg);
    defer gpa.free(scales_t);
    const biases_t = try hc.transposeBf16(gpa, stacked_b, n, kg);
    defer gpa.free(biases_t);
    const scale = try gpa.alloc(f32, gamma.dim(0));
    defer gpa.free(scale);
    for (scale, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, gamma.bytes[2 * i ..][0..2], .little)) << 16);
    var w_b = try cuda.DeviceBuffer.fromHost(driver, tiled);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales_t);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases_t);
    defer b_b.free();
    var scale_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(scale));
    defer scale_b.free();
    var norm_b = try cuda.DeviceBuffer.alloc(driver, streams * dims * 2);
    defer norm_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, n * 2);
    defer out_b.free();
    const down_sk: usize = 32;
    var down_part = try cuda.DeviceBuffer.alloc(driver, down_sk * n * 4);
    defer down_part.free();
    try tri.hcDown(h, pss_b.ptr, scale_b.ptr, norm_b.ptr, w_b.ptr, s_b.ptr, b_b.ptr, out_b.ptr, down_part.ptr, 1e-6, 1, n, k8 * 8, dims, nc, streams, down_sk);
    const got_norm = try gpa.alloc(u8, streams * dims * 2);
    defer gpa.free(got_norm);
    try stream.synchronize();
    try norm_b.download(0, got_norm);
    const pss_f = try gpa.alloc(f32, nc * streams);
    defer gpa.free(pss_f);
    for (pss_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, pss_bytes[4 * i ..][0..4], .little));
    const gpu_h = try gpa.alloc(u16, hidden.len);
    defer gpa.free(gpu_h);
    for (gpu_h, 0..) |*o, i| o.* = std.mem.readInt(u16, got_h[2 * i ..][0..2], .little);
    const want_norm = try gpa.alloc(u16, hidden.len);
    defer gpa.free(want_norm);
    try hc.normed(gpu_h, pss_f, scale, dims, streams, 1e-6, want_norm);
    var norm_off: usize = 0;
    var norm_steps: u32 = 0;
    var norm_at: usize = 0;
    for (want_norm, 0..) |want, i| {
        const got = std.mem.readInt(u16, got_norm[2 * i ..][0..2], .little);
        if (got != want) {
            const dist = hc.mixedSteps(got, want);
            if (norm_off == 0 or dist > norm_steps) {
                norm_steps = dist;
                norm_at = i;
            }
            norm_off += 1;
        }
    }
    var part4: [4]u8 = undefined;
    try down_part.download(0, &part4);
    const gpu_part: f32 = @bitCast(std.mem.readInt(u32, &part4, .little));
    const gpu_norm = try gpa.alloc(u16, 320);
    defer gpa.free(gpu_norm);
    for (gpu_norm, 0..) |*o, i| o.* = std.mem.readInt(u16, got_norm[2 * i ..][0..2], .little);
    const host_part = try hc.firstPartial(gpu_norm, stacked_w[0 .. k8 * 4], stacked_s[0 .. kg * 2], stacked_b[0 .. kg * 2], 10);
    const col_ulp = hc.ulps(gpu_part, host_part);
    std.debug.print("mlp hcdown normed off {d} max_steps {d} at {d} col0 gpu {d} host {d} ulp {d}\n", .{ norm_off, norm_steps, norm_at, gpu_part, host_part, col_ulp });
    if (norm_steps > 1 or col_ulp > 8) return 1;

    const low: usize = 320;
    var act_b = try cuda.DeviceBuffer.alloc(driver, low * 2);
    defer act_b.free();
    var xs_b = try cuda.DeviceBuffer.alloc(driver, (low / 32) * 4);
    defer xs_b.free();
    var inj_out = try cuda.DeviceBuffer.alloc(driver, streams * 2);
    defer inj_out.free();
    try tri.hcReduceAct(down_part.ptr, act_b.ptr, xs_b.ptr, inj_out.ptr, down_sk, 1, streams, low, n, 1);
    const part_bytes = try gpa.alloc(u8, down_sk * n * 4);
    defer gpa.free(part_bytes);
    try stream.synchronize();
    try down_part.download(0, part_bytes);
    const part_f = try gpa.alloc(f32, down_sk * n);
    defer gpa.free(part_f);
    for (part_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, part_bytes[4 * i ..][0..4], .little));
    const act_want = try gpa.alloc(u16, low);
    defer gpa.free(act_want);
    const inj_want = try gpa.alloc(u16, streams);
    defer gpa.free(inj_want);
    const xs_want = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_want);
    try hc.reduceAct(part_f, down_sk, n, low, streams, act_want, inj_want, xs_want);
    const act_bytes = try gpa.alloc(u8, low * 2);
    defer gpa.free(act_bytes);
    const inj_out_bytes = try gpa.alloc(u8, streams * 2);
    defer gpa.free(inj_out_bytes);
    const xs_bytes = try gpa.alloc(u8, (low / 32) * 4);
    defer gpa.free(xs_bytes);
    try act_b.download(0, act_bytes);
    try inj_out.download(0, inj_out_bytes);
    try xs_b.download(0, xs_bytes);
    var act_off: usize = 0;
    for (act_want, 0..) |want, i| {
        if (std.mem.readInt(u16, act_bytes[2 * i ..][0..2], .little) != want) act_off += 1;
    }
    var inj_off: usize = 0;
    for (inj_want, 0..) |want, i| {
        if (std.mem.readInt(u16, inj_out_bytes[2 * i ..][0..2], .little) != want) inj_off += 1;
    }
    var xs_ulp: u32 = 0;
    for (xs_want, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > xs_ulp) xs_ulp = dist;
    }
    std.debug.print("mlp reduce act mismatch {d} inject mismatch {d} xs max_ulp {d}\n", .{ act_off, inj_off, xs_ulp });
    if (act_off != 0 or inj_off != 0 or xs_ulp > 4) return 1;

    const up_n = up_w.dim(0);
    const up_k8 = up_w.dim(1);
    if (up_n != streams * dims or up_k8 != 40) return error.UnexpectedTensor;
    const up_tiled = try hc.tileWords(gpa, up_w.bytes, up_n, up_k8);
    defer gpa.free(up_tiled);
    const up_scales = try hc.transposeBf16(gpa, up_s.bytes, up_n, up_k8 / 4);
    defer gpa.free(up_scales);
    const up_biases = try hc.transposeBf16(gpa, up_b.bytes, up_n, up_k8 / 4);
    defer gpa.free(up_biases);
    var up_wb = try cuda.DeviceBuffer.fromHost(driver, up_tiled);
    defer up_wb.free();
    var up_sb = try cuda.DeviceBuffer.fromHost(driver, up_scales);
    defer up_sb.free();
    var up_bb = try cuda.DeviceBuffer.fromHost(driver, up_biases);
    defer up_bb.free();
    var mixed_b = try cuda.DeviceBuffer.alloc(driver, dims * 2);
    defer mixed_b.free();
    var xsm_b = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
    defer xsm_b.free();
    try tri.hcUpmix(act_b.ptr, xs_b.ptr, up_wb.ptr, up_sb.ptr, up_bb.ptr, norm_b.ptr, mixed_b.ptr, xsm_b.ptr, 1, up_n, up_k8 * 8, dims, streams);
    const mixed_bytes = try gpa.alloc(u8, dims * 2);
    defer gpa.free(mixed_bytes);
    try stream.synchronize();
    try mixed_b.download(0, mixed_bytes);
    const act_u16 = try gpa.alloc(u16, low);
    defer gpa.free(act_u16);
    for (act_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, act_bytes[2 * i ..][0..2], .little);
    const xs_f = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_f);
    for (xs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
    const norm_u16 = try gpa.alloc(u16, hidden.len);
    defer gpa.free(norm_u16);
    for (norm_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, got_norm[2 * i ..][0..2], .little);
    const host_mixed = try gpa.alloc(u16, dims);
    defer gpa.free(host_mixed);
    try hc.mix(act_u16, xs_f, norm_u16, up_w.bytes, up_s.bytes, up_b.bytes, up_n, up_k8, dims, streams, host_mixed);
    var mixed_off: usize = 0;
    var max_steps: u32 = 0;
    var mixed_at: usize = 0;
    for (host_mixed, 0..) |want, i| {
        const got = std.mem.readInt(u16, mixed_bytes[2 * i ..][0..2], .little);
        const dist = hc.mixedSteps(got, want);
        if (dist != 0) mixed_off += 1;
        if (dist > max_steps) {
            max_steps = dist;
            mixed_at = i;
        }
    }
    std.debug.print("mlp upmix dims {d} off {d} max_steps {d} at {d} head {x:0>4} host {x:0>4}\n", .{ dims, mixed_off, max_steps, mixed_at, std.mem.readInt(u16, mixed_bytes[0..2], .little), host_mixed[0] });
    if (max_steps > 2) return 1;
    return 0;
}

//! The final mixer and the word scores for one decode token: four embedding copies, no inject gate.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hc = @import("cuda_hc.zig");
const qmm = @import("cuda_qmm.zig");

const dims: usize = 2560;
const streams: usize = 4;
const low: usize = 320;
const wide: usize = streams * dims;
const sk: usize = 32;
const per: usize = 10;
const eps: f32 = 1e-6;
const vocab: usize = 248320;
const prefix = "language_model.model.hyper_connection_mixer.";

const Tensor = core.safetensors.Tensor;
const Triple = struct { w: Tensor, s: Tensor, b: Tensor };

const Gap = struct { off: usize = 0, steps: u32 = 0, at: usize = 0 };

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn roundBf16(v: f32) f32 {
    return promote(toBf16(v));
}

fn tree32(buf: *[32]f32) f32 {
    var n: usize = 32;
    while (n > 1) {
        n /= 2;
        for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
    }
    return buf[0];
}

fn readU16(raw: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, raw[2 * i ..][0..2], .little);
}

fn shardPath(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const file = parsed.value.object.get("weight_map").?.object.get(prefix ++ "hc_norm.weight").?.string;
    return std.fs.path.join(gpa, &.{ dir, file });
}

fn triple(file: *const core.safetensors.File, name: []const u8, rows: usize, k8: usize) !Triple {
    var buf: [180]u8 = undefined;
    const w = file.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name})) orelse return error.MissingTensor;
    const s = file.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{name})) orelse return error.MissingTensor;
    const b = file.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{name})) orelse return error.MissingTensor;
    if (!w.is(.u32, &.{ rows, k8 }) or !s.is(.bf16, &.{ rows, k8 / 4 }) or !b.is(.bf16, &.{ rows, k8 / 4 })) return error.UnexpectedTensor;
    return .{ .w = w, .s = s, .b = b };
}

/// Split-K sum, one bf16 round, divide by the stream count, SiLU. No inject columns.
fn siluRow(part: []const f32, slices: usize, n: usize, nstreams: usize, act: []u16, xs: []f32) !void {
    if (n == 0 or n % 32 != 0 or slices == 0 or part.len != slices * n or act.len != n or xs.len != n / 32) return error.UnexpectedTensor;
    for (0..n) |i| {
        var sum = part[i];
        for (1..slices) |s| sum += part[s * n + i];
        const y = roundBf16(roundBf16(sum) / @as(f32, @floatFromInt(nstreams)));
        act[i] = toBf16(y / (1.0 + @exp(-y)));
    }
    for (xs, 0..) |*o, g| {
        var buf: [32]f32 = undefined;
        for (0..32) |j| buf[j] = promote(act[g * 32 + j]);
        o.* = tree32(&buf);
    }
}

fn gap(got: []const u16, want: []const u16) Gap {
    var g = Gap{};
    for (got, want, 0..) |a, b, i| {
        const d = hc.mixedSteps(a, b);
        if (d != 0) g.off += 1;
        if (d > g.steps) {
            g.steps = d;
            g.at = i;
        }
    }
    return g;
}

fn u16s(dst: []u16, raw: []const u8) void {
    for (dst, 0..) |*o, i| o.* = readU16(raw, i);
}

test "a zero down sum without an inject gate is SiLU zero" {
    var part: [32]f32 = @splat(0);
    var act: [32]u16 = undefined;
    var xs: [1]f32 = undefined;
    try siluRow(&part, 1, 32, 1, &act, &xs);
    try std.testing.expectEqual(@as(u16, 0), act[0]);
    try std.testing.expectEqual(@as(f32, 0), xs[0]);
}

/// Mixer read-out of `hidden` (four stream copies) and the head's scores of that mixed row.
pub fn scores(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, hidden: []const u16) !u32 {
    if (hidden.len != wide) return error.UnexpectedTensor;
    const path = try shardPath(gpa, io, model_dir);
    defer gpa.free(path);
    var file = try core.safetensors.File.open(gpa, io, path);
    defer file.close(io);
    const gamma = file.get(prefix ++ "hc_norm.weight") orelse return error.MissingTensor;
    if (!gamma.is(.bf16, &.{wide})) return error.UnexpectedTensor;
    const down = try triple(&file, prefix ++ "input_mix_weight_down", low, wide / 8);
    const up = try triple(&file, prefix ++ "input_mix_weight_up", wide, low / 8);
    const head = try triple(&file, "language_model.lm_head", vocab, dims / 8);

    const scale = try gpa.alloc(f32, wide);
    defer gpa.free(scale);
    for (scale, 0..) |*o, i| o.* = promote(readU16(gamma.bytes, i));
    const pss = try gpa.alloc(f32, (dims / 256) * streams);
    defer gpa.free(pss);
    try hc.partialSums(hidden, dims, streams, pss);
    const host_norm = try gpa.alloc(u16, wide);
    defer gpa.free(host_norm);
    try hc.normed(hidden, pss, scale, dims, streams, eps, host_norm);

    var h_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(hidden));
    defer h_b.free();
    var pss_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(pss));
    defer pss_b.free();
    var scale_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(scale));
    defer scale_b.free();
    const down_w = try hc.tileWords(gpa, down.w.bytes, low, wide / 8);
    defer gpa.free(down_w);
    const down_s = try hc.transposeBf16(gpa, down.s.bytes, low, wide / 32);
    defer gpa.free(down_s);
    const down_b = try hc.transposeBf16(gpa, down.b.bytes, low, wide / 32);
    defer gpa.free(down_b);
    var dw = try cuda.DeviceBuffer.fromHost(driver, down_w);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(driver, down_s);
    defer ds.free();
    var db = try cuda.DeviceBuffer.fromHost(driver, down_b);
    defer db.free();
    var norm_b = try cuda.DeviceBuffer.alloc(driver, wide * 2);
    defer norm_b.free();
    var dn_b = try cuda.DeviceBuffer.alloc(driver, low * 2);
    defer dn_b.free();
    var part_b = try cuda.DeviceBuffer.alloc(driver, sk * low * 4);
    defer part_b.free();
    try tri.hcDown(h_b.ptr, pss_b.ptr, scale_b.ptr, norm_b.ptr, dw.ptr, ds.ptr, db.ptr, dn_b.ptr, part_b.ptr, eps, 1, low, wide, dims, dims / 256, streams, sk);

    var act_b = try cuda.DeviceBuffer.alloc(driver, low * 2);
    defer act_b.free();
    var xs_b = try cuda.DeviceBuffer.alloc(driver, (low / 32) * 4);
    defer xs_b.free();
    try tri.hcReduceAct(part_b.ptr, act_b.ptr, xs_b.ptr, act_b.ptr, sk, 1, streams, low, low, 0);

    const up_w = try hc.tileWords(gpa, up.w.bytes, wide, low / 8);
    defer gpa.free(up_w);
    const up_s = try hc.transposeBf16(gpa, up.s.bytes, wide, low / 32);
    defer gpa.free(up_s);
    const up_b = try hc.transposeBf16(gpa, up.b.bytes, wide, low / 32);
    defer gpa.free(up_b);
    var uw = try cuda.DeviceBuffer.fromHost(driver, up_w);
    defer uw.free();
    var us = try cuda.DeviceBuffer.fromHost(driver, up_s);
    defer us.free();
    var ub = try cuda.DeviceBuffer.fromHost(driver, up_b);
    defer ub.free();
    var mixed_b = try cuda.DeviceBuffer.alloc(driver, dims * 2);
    defer mixed_b.free();
    var xsm_b = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
    defer xsm_b.free();
    try tri.hcUpmix(act_b.ptr, xs_b.ptr, uw.ptr, us.ptr, ub.ptr, norm_b.ptr, mixed_b.ptr, xsm_b.ptr, 1, wide, low, dims, streams);

    const norm_raw = try gpa.alloc(u8, wide * 2);
    defer gpa.free(norm_raw);
    const part_raw = try gpa.alloc(u8, sk * low * 4);
    defer gpa.free(part_raw);
    const act_raw = try gpa.alloc(u8, low * 2);
    defer gpa.free(act_raw);
    const xs_raw = try gpa.alloc(u8, (low / 32) * 4);
    defer gpa.free(xs_raw);
    const mixed_raw = try gpa.alloc(u8, dims * 2);
    defer gpa.free(mixed_raw);
    const xsm_raw = try gpa.alloc(u8, (dims / 32) * 4);
    defer gpa.free(xsm_raw);
    try stream.synchronize();
    try norm_b.download(0, norm_raw);
    try part_b.download(0, part_raw);
    try act_b.download(0, act_raw);
    try xs_b.download(0, xs_raw);
    try mixed_b.download(0, mixed_raw);
    try xsm_b.download(0, xsm_raw);

    var norm_off: usize = 0;
    for (host_norm, 0..) |want, i| {
        if (readU16(norm_raw, i) != want) norm_off += 1;
    }
    const gpu_norm = try gpa.alloc(u16, per * 32);
    defer gpa.free(gpu_norm);
    u16s(gpu_norm, norm_raw);
    const host_part = try hc.firstPartial(gpu_norm, down.w.bytes[0 .. per * 16], down.s.bytes[0 .. per * 2], down.b.bytes[0 .. per * 2], per);
    const gpu_part: f32 = @bitCast(std.mem.readInt(u32, part_raw[0..4], .little));
    const col_ulp = hc.ulps(gpu_part, host_part);

    const part_f = try gpa.alloc(f32, sk * low);
    defer gpa.free(part_f);
    for (part_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, part_raw[4 * i ..][0..4], .little));
    const act_want = try gpa.alloc(u16, low);
    defer gpa.free(act_want);
    const xs_want = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_want);
    try siluRow(part_f, sk, low, streams, act_want, xs_want);
    var act_off: usize = 0;
    for (act_want, 0..) |want, i| {
        if (readU16(act_raw, i) != want) act_off += 1;
    }
    var xs_ulp: u32 = 0;
    for (xs_want, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, xs_raw[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > xs_ulp) xs_ulp = dist;
    }

    const act_u = try gpa.alloc(u16, low);
    defer gpa.free(act_u);
    u16s(act_u, act_raw);
    const xs_f = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_f);
    for (xs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_raw[4 * i ..][0..4], .little));
    const norm_u = try gpa.alloc(u16, wide);
    defer gpa.free(norm_u);
    u16s(norm_u, norm_raw);
    const host_mixed = try gpa.alloc(u16, dims);
    defer gpa.free(host_mixed);
    try hc.mix(act_u, xs_f, norm_u, up.w.bytes, up.s.bytes, up.b.bytes, wide, low / 8, dims, streams, host_mixed);
    const got_mixed = try gpa.alloc(u16, dims);
    defer gpa.free(got_mixed);
    u16s(got_mixed, mixed_raw);
    const mixed_gap = gap(got_mixed, host_mixed);
    std.debug.print("mix norm {d} col_ulp {d} act {d} xs_ulp {d} off {d} max_steps {d} at {d} got {x:0>4} host {x:0>4} head {x:0>4}\n", .{ norm_off, col_ulp, act_off, xs_ulp, mixed_gap.off, mixed_gap.steps, mixed_gap.at, got_mixed[mixed_gap.at], host_mixed[mixed_gap.at], got_mixed[0] });

    var lane = try qmm.pack(gpa, head.w.bytes, head.s.bytes, head.b.bytes, vocab, dims / 8);
    defer lane.deinit(gpa);
    var hw = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    defer hw.free();
    var hs = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    defer hs.free();
    var hb = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    defer hb.free();
    var logits_b = try cuda.DeviceBuffer.alloc(driver, vocab * 2);
    defer logits_b.free();
    try qmm.matmul(driver, stream.*, mixed_b.ptr, xsm_b.ptr, hw.ptr, hs.ptr, hb.ptr, logits_b.ptr, 1, vocab, dims);
    const xsm_f = try gpa.alloc(f32, dims / 32);
    defer gpa.free(xsm_f);
    for (xsm_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xsm_raw[4 * i ..][0..4], .little));
    const host_logits = try gpa.alloc(u16, vocab);
    defer gpa.free(host_logits);
    try qmm.dotRow(got_mixed, xsm_f, head.w.bytes, head.s.bytes, head.b.bytes, vocab, dims, host_logits);
    const logits_raw = try gpa.alloc(u8, vocab * 2);
    defer gpa.free(logits_raw);
    try stream.synchronize();
    try logits_b.download(0, logits_raw);
    const got_logits = try gpa.alloc(u16, vocab);
    defer gpa.free(got_logits);
    u16s(got_logits, logits_raw);
    const score_gap = gap(got_logits, host_logits);
    std.debug.print("scores n {d} off {d} max_steps {d} at {d} got {x:0>4} host {x:0>4} head {x:0>4}\n", .{ vocab, score_gap.off, score_gap.steps, score_gap.at, got_logits[score_gap.at], host_logits[score_gap.at], got_logits[0] });
    if (score_gap.off > 0 and score_gap.off <= 8) {
        for (got_logits, host_logits, 0..) |a, b, i| {
            const d = hc.mixedSteps(a, b);
            if (d != 0) std.debug.print("score {d} gpu {x:0>4} host {x:0>4} steps {d}\n", .{ i, a, b, d });
        }
    }

    if (norm_off != 0 or col_ulp > 8 or act_off != 0 or xs_ulp > 4 or mixed_gap.steps > 2 or score_gap.steps > 1 or score_gap.off > 3) return 1;
    return 0;
}

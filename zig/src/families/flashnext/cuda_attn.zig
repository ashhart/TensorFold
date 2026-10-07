//! Layer 15's full attention for one decode token at position 0: project, prepare, attend, gate, project out.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hc = @import("cuda_hc.zig");
const qmm = @import("cuda_qmm.zig");

const dims: usize = 2560;
const q_heads: usize = 24;
const kv_heads: usize = 2;
const head_dim: usize = 256;
const index_heads: usize = 4;
const index_dim: usize = 128;
const half: usize = 32;
const eps: f32 = 1e-6;
const attn_scale: f32 = 0.0625;
const proj_n: usize = q_heads * 2 * head_dim + 2 * kv_heads * head_dim + (index_heads + 1) * index_dim;
const out_k: usize = q_heads * head_dim;
const nch: usize = 3;
const prefix = "language_model.model.layers.15.self_attn.";

const Gap = struct { off: usize, steps: u32, at: usize };

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn readBf(dst: []u16, raw: []const u8) void {
    for (dst, 0..) |*o, i| o.* = std.mem.readInt(u16, raw[2 * i ..][0..2], .little);
}

fn treeSum(buf: []f32) f32 {
    var n = buf.len;
    while (n > 1) {
        n /= 2;
        for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
    }
    return buf[0];
}

fn sumSq(x: []const u16) f32 {
    var buf: [head_dim]f32 = undefined;
    for (x, 0..) |bits, i| {
        const v = promote(bits);
        buf[i] = v * v;
    }
    return treeSum(buf[0..x.len]);
}

/// glue._attn_prep at position 0: RMSNorm, round, rotate-half. A zero angle stores the normalized row.
fn normRot(x: []const u16, scale_w: []const f32, inv: []const f32, pos: f32, out: []u16) void {
    const width = x.len;
    const rinv = 1.0 / @sqrt(sumSq(x) / @as(f32, @floatFromInt(width)) + eps);
    var xn: [head_dim]f32 = undefined;
    for (0..width) |d| xn[d] = promote(toBf16(promote(x[d]) * rinv * scale_w[d]));
    for (0..width) |d| {
        const partner = if (d < half) d + half else if (d < 2 * half) d - half else d;
        const axis_i = if (d < half) d else if (d < 2 * half) d - half else 0;
        const ang = pos * inv[axis_i];
        const c = @cos(ang);
        const s = @sin(ang);
        const a = xn[d];
        const b = xn[partner];
        const rot = if (d < half) a * c - b * s else if (d < 2 * half) b * s + a * c else a;
        out[d] = toBf16(rot);
    }
}

fn copyRow(src: []const u16, dst: []u16) void {
    @memcpy(dst, src);
}

/// One key: the softmax weight is 1, so each query head copies its KV head's value.
fn attend(q: []const u16, k: []const u16, v: []const u16, out: []u16) void {
    _ = q;
    _ = k;
    copyRow(v, out);
}

fn applyGate(o: []const u16, g: []const u16, out: []u16, xs: []f32) void {
    for (o, g, out) |ov, gv, *y| y.* = toBf16(promote(ov) / (1.0 + @exp(-promote(gv))));
    for (xs, 0..) |*s, i| {
        var buf: [32]f32 = undefined;
        for (0..32) |j| buf[j] = promote(out[i * 32 + j]);
        s.* = treeSum(buf[0..]);
    }
}

fn gap(got: []const u16, want: []const u16) Gap {
    var off: usize = 0;
    var steps: u32 = 0;
    var at: usize = 0;
    for (got, want, 0..) |a, b, i| {
        const d = hc.mixedSteps(a, b);
        if (d != 0) off += 1;
        if (d > steps) {
            steps = d;
            at = i;
        }
    }
    return .{ .off = off, .steps = steps, .at = at };
}

fn shardPath(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const file = parsed.value.object.get("weight_map").?.object.get(prefix ++ "q_proj.weight").?.string;
    return std.fs.path.join(gpa, &.{ dir, file });
}

const Triple = struct { w: core.safetensors.Tensor, s: core.safetensors.Tensor, b: core.safetensors.Tensor };

fn triple(file: *const core.safetensors.File, name: []const u8, rows: usize, k8: usize) !Triple {
    var buf: [180]u8 = undefined;
    const w = file.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name})) orelse return error.MissingTensor;
    const s = file.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{name})) orelse return error.MissingTensor;
    const b = file.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{name})) orelse return error.MissingTensor;
    if (!w.is(.u32, &.{ rows, k8 }) or !s.is(.bf16, &.{ rows, k8 / 4 }) or !b.is(.bf16, &.{ rows, k8 / 4 })) return error.UnexpectedTensor;
    return .{ .w = w, .s = s, .b = b };
}

fn stitch(gpa: std.mem.Allocator, parts: []const Triple) !struct { words: []u8, scales: []u8, biases: []u8 } {
    var wb: usize = 0;
    var sb: usize = 0;
    for (parts) |p| {
        wb += p.w.bytes.len;
        sb += p.s.bytes.len;
    }
    const words = try gpa.alloc(u8, wb);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u8, sb);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u8, sb);
    var wo: usize = 0;
    var so: usize = 0;
    for (parts) |p| {
        @memcpy(words[wo..][0..p.w.bytes.len], p.w.bytes);
        wo += p.w.bytes.len;
        @memcpy(scales[so..][0..p.s.bytes.len], p.s.bytes);
        @memcpy(biases[so..][0..p.b.bytes.len], p.b.bytes);
        so += p.s.bytes.len;
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}

fn promoteScale(bytes: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| o.* = promote(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

fn invFreq(out: []f32) void {
    const n: f64 = @floatFromInt(out.len);
    for (out, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / n));
}

fn prepare(pa: []const u16, q_scale: []const f32, k_scale: []const f32, i_scale: []const f32, inv: []const f32, q: []u16, k: []u16, v: []u16, iq: []u16, ik: []u16) void {
    const q_pairs = q_heads * 2 * head_dim;
    const kv = kv_heads * head_dim;
    for (0..q_heads) |h| normRot(pa[h * 2 * head_dim ..][0..head_dim], q_scale, inv, 0, q[h * head_dim ..][0..head_dim]);
    for (0..kv_heads) |h| {
        normRot(pa[q_pairs + h * head_dim ..][0..head_dim], k_scale, inv, 0, k[h * head_dim ..][0..head_dim]);
        copyRow(pa[q_pairs + kv + h * head_dim ..][0..head_dim], v[h * head_dim ..][0..head_dim]);
    }
    const iq0 = q_pairs + 2 * kv;
    for (0..index_heads) |h| normRot(pa[iq0 + h * index_dim ..][0..index_dim], i_scale, inv, 0, iq[h * index_dim ..][0..index_dim]);
    copyRow(pa[iq0 + index_heads * index_dim ..][0..index_dim], ik);
}

fn hostAttend(q: []const u16, k: []const u16, v: []const u16, out: []u16) void {
    const g = q_heads / kv_heads;
    for (0..q_heads) |h| {
        const hk = h / g;
        attend(q[h * head_dim ..][0..head_dim], k[hk * head_dim ..][0..head_dim], v[hk * head_dim ..][0..head_dim], out[h * head_dim ..][0..head_dim]);
    }
}

fn gatesOf(pa: []const u16, out: []u16) void {
    for (0..q_heads) |h| copyRow(pa[h * 2 * head_dim + head_dim ..][0..head_dim], out[h * head_dim ..][0..head_dim]);
}

/// The projection, prep, one-key attention, gate and output projection, against the host of the same mixed row.
pub fn decode(comptime Tri: type, gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, io: std.Io, model_dir: []const u8, tri: Tri, mixed: u64, xs: u64) !u32 {
    const path = try shardPath(gpa, io, model_dir);
    defer gpa.free(path);
    var file = try core.safetensors.File.open(gpa, io, path);
    defer file.close(io);
    const parts = [_]Triple{
        try triple(&file, prefix ++ "q_proj", q_heads * 2 * head_dim, dims / 8),
        try triple(&file, prefix ++ "k_proj", kv_heads * head_dim, dims / 8),
        try triple(&file, prefix ++ "v_proj", kv_heads * head_dim, dims / 8),
        try triple(&file, prefix ++ "indexer.index_qk_proj", (index_heads + 1) * index_dim, dims / 8),
    };
    const out_w = try triple(&file, prefix ++ "o_proj", dims, out_k / 8);
    const qn = file.get(prefix ++ "q_norm.weight") orelse return error.MissingTensor;
    const kn = file.get(prefix ++ "k_norm.weight") orelse return error.MissingTensor;
    const iqn = file.get(prefix ++ "indexer.q_layernorm.weight") orelse return error.MissingTensor;
    if (!qn.is(.bf16, &.{head_dim}) or !kn.is(.bf16, &.{head_dim}) or !iqn.is(.bf16, &.{index_dim})) return error.UnexpectedTensor;
    if (proj_n != 13952) return error.UnexpectedTensor;

    const stacked = try stitch(gpa, &parts);
    defer gpa.free(stacked.words);
    defer gpa.free(stacked.scales);
    defer gpa.free(stacked.biases);
    var lane = try qmm.pack(gpa, stacked.words, stacked.scales, stacked.biases, proj_n, dims / 8);
    defer lane.deinit(gpa);
    var ow = try qmm.pack(gpa, out_w.w.bytes, out_w.s.bytes, out_w.b.bytes, dims, out_k / 8);
    defer ow.deinit(gpa);

    var q_scale: [head_dim]f32 = undefined;
    var k_scale: [head_dim]f32 = undefined;
    var i_scale: [index_dim]f32 = undefined;
    var inv: [half]f32 = undefined;
    promoteScale(qn.bytes, &q_scale);
    promoteScale(kn.bytes, &k_scale);
    promoteScale(iqn.bytes, &i_scale);
    invFreq(&inv);

    var w_b = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    defer b_b.free();
    var pa_b = try cuda.DeviceBuffer.alloc(driver, proj_n * 2);
    defer pa_b.free();
    try qmm.matmul(driver, stream.*, mixed, xs, w_b.ptr, s_b.ptr, b_b.ptr, pa_b.ptr, 1, proj_n, dims);

    var qw_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&q_scale));
    defer qw_b.free();
    var kw_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&k_scale));
    defer kw_b.free();
    var iw_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&i_scale));
    defer iw_b.free();
    var inv_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&inv));
    defer inv_b.free();
    var pos_host: i32 = 0;
    var pos_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.asBytes(&pos_host));
    defer pos_b.free();
    var q_b = try cuda.DeviceBuffer.alloc(driver, q_heads * head_dim * 2);
    defer q_b.free();
    var kc_b = try cuda.DeviceBuffer.alloc(driver, kv_heads * head_dim * 2);
    defer kc_b.free();
    var vc_b = try cuda.DeviceBuffer.alloc(driver, kv_heads * head_dim * 2);
    defer vc_b.free();
    var iq_b = try cuda.DeviceBuffer.alloc(driver, index_heads * index_dim * 2);
    defer iq_b.free();
    var ik_b = try cuda.DeviceBuffer.alloc(driver, index_dim * 2);
    defer ik_b.free();
    var dummy = try cuda.DeviceBuffer.alloc(driver, 256);
    defer dummy.free();
    try tri.attnPrep(pa_b.ptr, pos_b.ptr, qw_b.ptr, kw_b.ptr, iw_b.ptr, inv_b.ptr, q_b.ptr, kc_b.ptr, vc_b.ptr, dummy.ptr, dummy.ptr, iq_b.ptr, ik_b.ptr, dummy.ptr, dummy.ptr, eps, 1, 0);

    var po_b = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * head_dim * 4);
    defer po_b.free();
    var pm_b = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
    defer pm_b.free();
    var pl_b = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
    defer pl_b.free();
    var o_b = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
    defer o_b.free();
    try tri.attnChunks(q_b.ptr, kc_b.ptr, vc_b.ptr, dummy.ptr, dummy.ptr, pos_b.ptr, po_b.ptr, pm_b.ptr, pl_b.ptr, dummy.ptr, dummy.ptr, dummy.ptr, 1, kv_heads, 1);
    try tri.attnMerge(po_b.ptr, pm_b.ptr, pl_b.ptr, pos_b.ptr, o_b.ptr, dummy.ptr, dummy.ptr, 1, kv_heads);

    var gated_b = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
    defer gated_b.free();
    var gxs_b = try cuda.DeviceBuffer.alloc(driver, (out_k / 32) * 4);
    defer gxs_b.free();
    try tri.attnGate(o_b.ptr, pa_b.ptr, gated_b.ptr, gxs_b.ptr, 1, proj_n, q_heads, head_dim);

    var ow_b = try cuda.DeviceBuffer.fromHost(driver, ow.weight);
    defer ow_b.free();
    var os_b = try cuda.DeviceBuffer.fromHost(driver, ow.scales);
    defer os_b.free();
    var ob_b = try cuda.DeviceBuffer.fromHost(driver, ow.biases);
    defer ob_b.free();
    var branch_b = try cuda.DeviceBuffer.alloc(driver, dims * 2);
    defer branch_b.free();
    try qmm.matmul(driver, stream.*, gated_b.ptr, gxs_b.ptr, ow_b.ptr, os_b.ptr, ob_b.ptr, branch_b.ptr, 1, dims, out_k);

    const row_raw = try gpa.alloc(u8, dims * 2);
    defer gpa.free(row_raw);
    const xs_raw = try gpa.alloc(u8, (dims / 32) * 4);
    defer gpa.free(xs_raw);
    const pa_raw = try gpa.alloc(u8, proj_n * 2);
    defer gpa.free(pa_raw);
    const q_raw = try gpa.alloc(u8, q_heads * head_dim * 2);
    defer gpa.free(q_raw);
    const k_raw = try gpa.alloc(u8, kv_heads * head_dim * 2);
    defer gpa.free(k_raw);
    const v_raw = try gpa.alloc(u8, kv_heads * head_dim * 2);
    defer gpa.free(v_raw);
    const iq_raw = try gpa.alloc(u8, index_heads * index_dim * 2);
    defer gpa.free(iq_raw);
    const ik_raw = try gpa.alloc(u8, index_dim * 2);
    defer gpa.free(ik_raw);
    const o_raw = try gpa.alloc(u8, out_k * 2);
    defer gpa.free(o_raw);
    const gated_raw = try gpa.alloc(u8, out_k * 2);
    defer gpa.free(gated_raw);
    const gxs_raw = try gpa.alloc(u8, (out_k / 32) * 4);
    defer gpa.free(gxs_raw);
    const branch_raw = try gpa.alloc(u8, dims * 2);
    defer gpa.free(branch_raw);
    try stream.synchronize();
    var mixed_view: cuda.DeviceBuffer = .{ .d = driver, .ptr = mixed, .len = row_raw.len };
    var xs_view: cuda.DeviceBuffer = .{ .d = driver, .ptr = xs, .len = xs_raw.len };
    try mixed_view.download(0, row_raw);
    try xs_view.download(0, xs_raw);
    try pa_b.download(0, pa_raw);
    try q_b.download(0, q_raw);
    try kc_b.download(0, k_raw);
    try vc_b.download(0, v_raw);
    try iq_b.download(0, iq_raw);
    try ik_b.download(0, ik_raw);
    try o_b.download(0, o_raw);
    try gated_b.download(0, gated_raw);
    try gxs_b.download(0, gxs_raw);
    try branch_b.download(0, branch_raw);

    const row = try gpa.alloc(u16, dims);
    defer gpa.free(row);
    readBf(row, row_raw);
    const xs_f = try gpa.alloc(f32, dims / 32);
    defer gpa.free(xs_f);
    for (xs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_raw[4 * i ..][0..4], .little));
    const host_pa = try gpa.alloc(u16, proj_n);
    defer gpa.free(host_pa);
    try qmm.dotRow(row, xs_f, stacked.words, stacked.scales, stacked.biases, proj_n, dims, host_pa);
    const gpu_pa = try gpa.alloc(u16, proj_n);
    defer gpa.free(gpu_pa);
    readBf(gpu_pa, pa_raw);
    const proj = gap(gpu_pa, host_pa);

    const host_q = try gpa.alloc(u16, q_heads * head_dim);
    defer gpa.free(host_q);
    const host_k = try gpa.alloc(u16, kv_heads * head_dim);
    defer gpa.free(host_k);
    const host_v = try gpa.alloc(u16, kv_heads * head_dim);
    defer gpa.free(host_v);
    const host_iq = try gpa.alloc(u16, index_heads * index_dim);
    defer gpa.free(host_iq);
    const host_ik = try gpa.alloc(u16, index_dim);
    defer gpa.free(host_ik);
    prepare(gpu_pa, &q_scale, &k_scale, &i_scale, &inv, host_q, host_k, host_v, host_iq, host_ik);
    const gpu_q = try gpa.alloc(u16, host_q.len);
    defer gpa.free(gpu_q);
    const gpu_k = try gpa.alloc(u16, host_k.len);
    defer gpa.free(gpu_k);
    const gpu_v = try gpa.alloc(u16, host_v.len);
    defer gpa.free(gpu_v);
    const gpu_iq = try gpa.alloc(u16, host_iq.len);
    defer gpa.free(gpu_iq);
    const gpu_ik = try gpa.alloc(u16, host_ik.len);
    defer gpa.free(gpu_ik);
    readBf(gpu_q, q_raw);
    readBf(gpu_k, k_raw);
    readBf(gpu_v, v_raw);
    readBf(gpu_iq, iq_raw);
    readBf(gpu_ik, ik_raw);
    const qg = gap(gpu_q, host_q);
    const kg = gap(gpu_k, host_k);
    const vg = gap(gpu_v, host_v);
    const iqg = gap(gpu_iq, host_iq);
    const ikg = gap(gpu_ik, host_ik);

    const host_o = try gpa.alloc(u16, out_k);
    defer gpa.free(host_o);
    hostAttend(gpu_q, gpu_k, gpu_v, host_o);
    const gpu_o = try gpa.alloc(u16, out_k);
    defer gpa.free(gpu_o);
    readBf(gpu_o, o_raw);
    const og = gap(gpu_o, host_o);

    const gate_row = try gpa.alloc(u16, out_k);
    defer gpa.free(gate_row);
    gatesOf(gpu_pa, gate_row);
    const host_gated = try gpa.alloc(u16, out_k);
    defer gpa.free(host_gated);
    const host_gxs = try gpa.alloc(f32, out_k / 32);
    defer gpa.free(host_gxs);
    applyGate(gpu_o, gate_row, host_gated, host_gxs);
    const gpu_gated = try gpa.alloc(u16, out_k);
    defer gpa.free(gpu_gated);
    readBf(gpu_gated, gated_raw);
    const gg = gap(gpu_gated, host_gated);
    var xs_ulp: u32 = 0;
    for (host_gxs, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, gxs_raw[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > xs_ulp) xs_ulp = dist;
    }

    const gxs_f = try gpa.alloc(f32, out_k / 32);
    defer gpa.free(gxs_f);
    for (gxs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, gxs_raw[4 * i ..][0..4], .little));
    const host_branch = try gpa.alloc(u16, dims);
    defer gpa.free(host_branch);
    try qmm.dotRow(gpu_gated, gxs_f, out_w.w.bytes, out_w.s.bytes, out_w.b.bytes, dims, out_k, host_branch);
    const gpu_branch = try gpa.alloc(u16, dims);
    defer gpa.free(gpu_branch);
    readBf(gpu_branch, branch_raw);
    const bg = gap(gpu_branch, host_branch);

    std.debug.print("attn proj n {d} off {d} max_steps {d} at {d} got {x:0>4} host {x:0>4} head {x:0>4}\n", .{ proj_n, proj.off, proj.steps, proj.at, gpu_pa[proj.at], host_pa[proj.at], gpu_pa[0] });
    std.debug.print("attn prep q {d}/{d} k {d}/{d} v {d}/{d} iq {d}/{d} ik {d}/{d} head {x:0>4} host {x:0>4}\n", .{ qg.off, qg.steps, kg.off, kg.steps, vg.off, vg.steps, iqg.off, iqg.steps, ikg.off, ikg.steps, gpu_q[0], host_q[0] });
    std.debug.print("attn out off {d} max_steps {d} head {x:0>4} host {x:0>4}\n", .{ og.off, og.steps, gpu_o[0], host_o[0] });
    std.debug.print("attn gate off {d} max_steps {d} xs_ulp {d} head {x:0>4} host {x:0>4}\n", .{ gg.off, gg.steps, xs_ulp, gpu_gated[0], host_gated[0] });
    std.debug.print("attn oproj n {d} off {d} max_steps {d} head {x:0>4} host {x:0>4}\n", .{ dims, bg.off, bg.steps, gpu_branch[0], host_branch[0] });
    if (proj.off > 1 or proj.steps > 1 or qg.steps != 0 or kg.steps != 0 or vg.steps != 0 or iqg.steps != 0 or ikg.steps != 0 or og.steps != 0 or gg.steps != 0 or xs_ulp != 0 or bg.steps != 0) return 1;
    return 0;
}

test "position zero stores the normalized query" {
    var x: [head_dim]u16 = @splat(0);
    x[0] = 0x3f80;
    var scale_w: [head_dim]f32 = @splat(1);
    var inv: [half]f32 = @splat(1);
    var out: [head_dim]u16 = undefined;
    normRot(&x, &scale_w, &inv, 0, &out);
    try std.testing.expectEqual(@as(u16, 0x4180), out[0]);
    try std.testing.expectEqual(@as(u16, 0), out[half]);
}

test "the gate rounds one times sigmoid of zero to one half" {
    var o: [head_dim]u16 = @splat(0);
    var g: [head_dim]u16 = @splat(0);
    o[0] = 0x3f80;
    var out: [head_dim]u16 = undefined;
    var xs: [head_dim / 32]f32 = undefined;
    applyGate(&o, &g, &out, &xs);
    try std.testing.expectEqual(@as(u16, 0x3f00), out[0]);
}

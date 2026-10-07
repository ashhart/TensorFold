//! Layer 0's experts on one mixed row: router, top 10 plus the shared expert, then each slot's gate, up and down.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");

const count: usize = 512;
const top_k: usize = 10;
const slots: usize = top_k + 1;
const dims: usize = 2560;
const width: usize = 640;
const routed_rows: usize = count + 1;
const tile: usize = 16;
const block_bytes: usize = 160 * 4;

const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiii";
const plan_symbol: [:0]const u8 = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
const up_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
const down_symbol: [:0]const u8 = "_ZN10tf_experts13expert_kernelILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";

const Proj = struct {
    routed_w: []const u8,
    routed_s: []const u8,
    routed_b: []const u8,
    shared_w: []const u8,
    shared_s: []const u8,
    shared_b: []const u8,
    n: usize,
    k: usize,
};

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn roundTrip(v: f32) f32 {
    return promote(toBf16(v));
}

/// The expert kernel's group sum: each lane adds its eight inputs, then the quad adds as (0+1)+(2+3).
fn groupSum(x: []const u16, g: usize) f32 {
    var pair: [2]f32 = .{ 0, 0 };
    for (0..4) |t| {
        var s: f32 = 0;
        const base = g * 32 + t * 8;
        for (0..8) |j| s += promote(x[base + j]);
        if (t % 2 == 0) pair[t / 2] = s else pair[t / 2] += s;
    }
    return pair[0] + pair[1];
}

/// One expert matrix: groups in order, `p * scale + xs * bias`, fp32. `xs` is `groupSum` of `x`.
fn project(x: []const u16, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k: usize, out: []f32) void {
    const kg = k / 32;
    const k8 = k / 8;
    var xs: [80]f32 = undefined;
    for (0..kg) |g| xs[g] = groupSum(x, g);
    for (0..n) |col| {
        var acc: f32 = 0;
        const wb = col * k8 * 4;
        const sb = col * kg * 2;
        for (0..kg) |g| {
            var p: f32 = 0;
            for (0..4) |i| {
                const word = std.mem.readInt(u32, words[wb + (g * 4 + i) * 4 ..][0..4], .little);
                const base = g * 32 + i * 8;
                inline for (0..8) |j| {
                    const q: f32 = @floatFromInt((word >> (j * 4)) & 0xF);
                    p += promote(x[base + j]) * q;
                }
            }
            const s = promote(std.mem.readInt(u16, scales[sb + g * 2 ..][0..2], .little));
            const b = promote(std.mem.readInt(u16, biases[sb + g * 2 ..][0..2], .little));
            acc = @mulAdd(f32, xs[g], b, @mulAdd(f32, p, s, acc));
        }
        out[col] = acc;
    }
}

/// bf16(SiLU(bf16(gate)) * bf16(up)), the SwiGLU epilogue with no clip.
fn swiglu(gate: f32, up: f32) u16 {
    const gv = roundTrip(gate);
    const uv = roundTrip(up);
    const silu = roundTrip(gv / (1 + @exp(-gv)));
    return toBf16(silu * uv);
}

/// Router rows in K tiles of 256, the same tile the captured program uses, added in tile order.
fn routerDots(x: []const u16, w: []const u16, out: []f32) void {
    const k = x.len;
    for (out, 0..) |*o, e| {
        var acc: f32 = 0;
        const row = w[e * k ..][0..k];
        var t: usize = 0;
        while (t < k) : (t += 256) {
            var part: f32 = 0;
            for (0..256) |i| part += promote(x[t + i]) * promote(row[t + i]);
            acc += part;
        }
        o.* = acc;
    }
}

/// Largest `k` of the first `n` logits (lower id on a tie), weights exp(l - l0) / sum rounded to bf16, then the shared gate.
fn choose(logits: []const f32, n: usize, k: usize, picks: []i32, wts: []f32) void {
    var scratch: [count]f32 = undefined;
    for (0..n) |i| scratch[i] = logits[i];
    var top = scratch[0];
    for (scratch[1..n]) |v| top = @max(top, v);
    var total: f32 = 0;
    var exs: [16]f32 = undefined;
    for (0..k) |slot| {
        var m = scratch[0];
        var idx: usize = 0;
        for (scratch[1..n], 1..) |v, i| if (v > m) {
            m = v;
            idx = i;
        };
        const ex = @exp(m - top);
        picks[slot] = @intCast(idx);
        exs[slot] = ex;
        total += ex;
        scratch[idx] = -std.math.inf(f32);
    }
    for (0..k) |slot| wts[slot] = roundTrip(exs[slot] / total);
    const shared = roundTrip(logits[n]);
    picks[k] = @intCast(n);
    wts[k] = roundTrip(1 / (1 + @exp(-shared)));
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn loadProj(mapped: *const embed.Mapped, gpa: std.mem.Allocator, name: []const u8, n: usize, k: usize) !Proj {
    const prefix = "language_model.model.layers.0.mlp.";
    var buf: [180]u8 = undefined;
    const k8 = k / 8;
    const kg = k / 32;
    const rw = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}switch_mlp.{s}.weight", .{ prefix, name }), .u32);
    const rs = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}switch_mlp.{s}.scales", .{ prefix, name }), .bf16);
    const rb = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}switch_mlp.{s}.biases", .{ prefix, name }), .bf16);
    const sw = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}shared_expert.{s}.weight", .{ prefix, name }), .u32);
    const ss = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}shared_expert.{s}.scales", .{ prefix, name }), .bf16);
    const sb = try mapped.lookup(gpa, try std.fmt.bufPrint(&buf, "{s}shared_expert.{s}.biases", .{ prefix, name }), .bf16);
    if (rw.dim(0) != count or rw.dim(1) != n or rw.dim(2) != k8) return error.UnexpectedTensor;
    if (rs.dim(0) != count or rs.dim(1) != n or rs.dim(2) != kg) return error.UnexpectedTensor;
    if (rb.dim(0) != count or rb.dim(1) != n or rb.dim(2) != kg) return error.UnexpectedTensor;
    if (sw.dim(0) != n or sw.dim(1) != k8 or ss.dim(0) != n or ss.dim(1) != kg) return error.UnexpectedTensor;
    if (sb.dim(0) != n or sb.dim(1) != kg) return error.UnexpectedTensor;
    return .{ .routed_w = rw.bytes, .routed_s = rs.bytes, .routed_b = rb.bytes, .shared_w = sw.bytes, .shared_s = ss.bytes, .shared_b = sb.bytes, .n = n, .k = k };
}

fn copyExpert(dst_w: []u8, dst_s: []u8, dst_b: []u8, p: Proj, id: i32, slot: usize) void {
    const k8 = p.k / 8;
    const kg = p.k / 32;
    const wb = p.n * k8 * 4;
    const sb = p.n * kg * 2;
    const routed = id != count;
    const wsrc = if (routed) p.routed_w[@as(usize, @intCast(id)) * wb ..][0..wb] else p.shared_w;
    const ssrc = if (routed) p.routed_s[@as(usize, @intCast(id)) * sb ..][0..sb] else p.shared_s;
    const bsrc = if (routed) p.routed_b[@as(usize, @intCast(id)) * sb ..][0..sb] else p.shared_b;
    @memcpy(dst_w[slot * wb ..][0..wb], wsrc);
    @memcpy(dst_s[slot * sb ..][0..sb], ssrc);
    @memcpy(dst_b[slot * sb ..][0..sb], bsrc);
}

fn packDownload(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, pack_fn: cuda.Function, words: []const u8, scales: []const u8, biases: []const u8, e: usize, n: usize, k: usize) ![]u8 {
    const kg = k / 32;
    const nb = n / 32;
    const out_len = e * nb * kg * block_bytes;
    var w_b = try cuda.DeviceBuffer.fromHost(driver, words);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, out_len);
    defer out_b.free();
    var a: cuda.Args = .{};
    a.add(w_b.ptr);
    a.add(s_b.ptr);
    a.add(b_b.ptr);
    a.add(out_b.ptr);
    a.add(cint(n));
    a.add(cint(k / 8));
    a.add(cint(kg));
    a.add(cint(nb));
    try cuda.launch.launch(pack_fn, .{ .grid = .{ .x = @intCast(kg), .y = @intCast(nb), .z = @intCast(e) }, .block = .{ .x = 160 } }, stream, &a);
    const out = try gpa.alloc(u8, out_len);
    errdefer gpa.free(out);
    try stream.synchronize();
    try out_b.download(0, out);
    return out;
}

fn stackHalves(gpa: std.mem.Allocator, gate: []const u8, up: []const u8) ![]u8 {
    if (gate.len != up.len or gate.len % block_bytes != 0) return error.UnexpectedTensor;
    const nblk = gate.len / block_bytes;
    const out = try gpa.alloc(u8, gate.len + up.len);
    for (0..nblk) |b| {
        @memcpy(out[b * 2 * block_bytes ..][0..block_bytes], gate[b * block_bytes ..][0..block_bytes]);
        @memcpy(out[b * 2 * block_bytes + block_bytes ..][0..block_bytes], up[b * block_bytes ..][0..block_bytes]);
    }
    return out;
}

fn maxUlp(got: []const f32, want: []const f32) struct { off: usize, ulp: u32 } {
    var off: usize = 0;
    var ulp: u32 = 0;
    for (got, want) |g, w| {
        const dist = hc.ulps(g, w);
        if (dist != 0) off += 1;
        if (dist > ulp) ulp = dist;
    }
    return .{ .off = off, .ulp = ulp };
}

/// The 512 routed experts and the shared expert, on the mixed row already at `mixed`.
pub fn experts(comptime Tri: type, gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, mapped: *const embed.Mapped, tri: Tri, mixed: u64) !u32 {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    const prefix = "language_model.model.layers.0.mlp.";
    var name: [180]u8 = undefined;
    const gate = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}gate.weight", .{prefix}), .bf16);
    const sg_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}shared_expert_gate.weight", .{prefix}), .u32);
    const sg_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}shared_expert_gate.scales", .{prefix}), .bf16);
    const sg_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}shared_expert_gate.biases", .{prefix}), .bf16);
    if (gate.dim(0) != count or gate.dim(1) != dims) return error.UnexpectedTensor;
    if (sg_w.dim(0) != 1 or sg_w.dim(1) != dims / 8) return error.UnexpectedTensor;

    const row_raw = try gpa.alloc(u8, dims * 2);
    defer gpa.free(row_raw);
    try stream.synchronize();
    var mixed_buf: cuda.DeviceBuffer = .{ .d = driver, .ptr = mixed, .len = row_raw.len };
    try mixed_buf.download(0, row_raw);
    const row = try gpa.alloc(u16, dims);
    defer gpa.free(row);
    for (row, 0..) |*o, i| o.* = std.mem.readInt(u16, row_raw[2 * i ..][0..2], .little);

    const router = try gpa.alloc(u16, routed_rows * dims);
    defer gpa.free(router);
    @memcpy(std.mem.sliceAsBytes(router[0 .. count * dims]), gate.bytes);
    try embed.dequant(sg_w.bytes, sg_s.bytes, sg_b.bytes, dims, 0, router[count * dims ..][0..dims]);
    const host_l = try gpa.alloc(f32, routed_rows);
    defer gpa.free(host_l);
    routerDots(row, router, host_l);

    var w_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(router));
    defer w_b.free();
    var logits_b = try cuda.DeviceBuffer.alloc(driver, routed_rows * 4);
    defer logits_b.free();
    var pick_b = try cuda.DeviceBuffer.alloc(driver, slots * 4);
    defer pick_b.free();
    var wts_b = try cuda.DeviceBuffer.alloc(driver, slots * 4);
    defer wts_b.free();
    try tri.router(mixed, w_b.ptr, logits_b.ptr, dims);
    try tri.topkRows(logits_b.ptr, pick_b.ptr, wts_b.ptr);

    const logit_raw = try gpa.alloc(u8, routed_rows * 4);
    defer gpa.free(logit_raw);
    const pick_raw = try gpa.alloc(u8, slots * 4);
    defer gpa.free(pick_raw);
    const wts_raw = try gpa.alloc(u8, slots * 4);
    defer gpa.free(wts_raw);
    try stream.synchronize();
    try logits_b.download(0, logit_raw);
    try pick_b.download(0, pick_raw);
    try wts_b.download(0, wts_raw);
    const gpu_l = try gpa.alloc(f32, routed_rows);
    defer gpa.free(gpu_l);
    for (gpu_l, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, logit_raw[4 * i ..][0..4], .little));
    var gpu_pick: [slots]i32 = undefined;
    var gpu_w: [slots]f32 = undefined;
    for (0..slots) |i| {
        gpu_pick[i] = std.mem.readInt(i32, pick_raw[4 * i ..][0..4], .little);
        gpu_w[i] = @bitCast(std.mem.readInt(u32, wts_raw[4 * i ..][0..4], .little));
    }
    const route = maxUlp(gpu_l, host_l);
    var from_gpu: [slots]i32 = undefined;
    var w_gpu: [slots]f32 = undefined;
    choose(gpu_l, count, top_k, &from_gpu, &w_gpu);
    var from_host: [slots]i32 = undefined;
    var w_host: [slots]f32 = undefined;
    choose(host_l, count, top_k, &from_host, &w_host);
    var pick_off: usize = 0;
    var host_pick_off: usize = 0;
    for (gpu_pick, from_gpu, from_host) |g, a, b| {
        if (g != a) pick_off += 1;
        if (g != b) host_pick_off += 1;
    }
    const wts_same = maxUlp(&gpu_w, &w_gpu);
    var host_wts_off: usize = 0;
    var host_wts_steps: u32 = 0;
    for (gpu_w, w_host) |g, h| {
        const steps = hc.mixedSteps(toBf16(g), toBf16(h));
        if (steps != 0) host_wts_off += 1;
        if (steps > host_wts_steps) host_wts_steps = steps;
    }
    std.debug.print("router off {d} max_ulp {d} head {x:0>8} host {x:0>8}\n", .{ route.off, route.ulp, @as(u32, @bitCast(gpu_l[0])), @as(u32, @bitCast(host_l[0])) });
    std.debug.print("topk pick_off {d} host_pick_off {d} wts_ulp {d} host_wts_off {d} host_wts_steps {d} picks", .{ pick_off, host_pick_off, wts_same.ulp, host_wts_off, host_wts_steps });
    for (gpu_pick) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});

    var packed_ids: [slots]i32 = undefined;
    var npack: usize = 0;
    var local: [slots]i32 = undefined;
    for (gpu_pick, 0..) |id, s| {
        if (id < 0 or id > count) return error.UnexpectedTensor;
        var found: ?usize = null;
        for (packed_ids[0..npack], 0..) |prev, i| if (prev == id) {
            found = i;
        };
        if (found) |i| {
            local[s] = @intCast(i);
        } else {
            packed_ids[npack] = id;
            local[s] = @intCast(npack);
            npack += 1;
        }
    }

    const gate_p = try loadProj(mapped, gpa, "gate_proj", width, dims);
    const up_p = try loadProj(mapped, gpa, "up_proj", width, dims);
    const down_p = try loadProj(mapped, gpa, "down_proj", dims, width);
    const gathered_w = try gpa.alloc(u8, npack * width * (dims / 8) * 4);
    defer gpa.free(gathered_w);
    const gathered_s = try gpa.alloc(u8, npack * width * (dims / 32) * 2);
    defer gpa.free(gathered_s);
    const gathered_b = try gpa.alloc(u8, gathered_s.len);
    defer gpa.free(gathered_b);
    const up_w = try gpa.alloc(u8, gathered_w.len);
    defer gpa.free(up_w);
    const up_s = try gpa.alloc(u8, gathered_s.len);
    defer gpa.free(up_s);
    const up_b = try gpa.alloc(u8, gathered_s.len);
    defer gpa.free(up_b);
    const down_w = try gpa.alloc(u8, npack * dims * (width / 8) * 4);
    defer gpa.free(down_w);
    const down_s = try gpa.alloc(u8, npack * dims * (width / 32) * 2);
    defer gpa.free(down_s);
    const down_b = try gpa.alloc(u8, down_s.len);
    defer gpa.free(down_b);
    for (packed_ids[0..npack], 0..) |id, slot| {
        copyExpert(gathered_w, gathered_s, gathered_b, gate_p, id, slot);
        copyExpert(up_w, up_s, up_b, up_p, id, slot);
        copyExpert(down_w, down_s, down_b, down_p, id, slot);
    }

    var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
    defer pack_mod.unload();
    var exp_mod = try cuda.Module.load(driver, cuda.kernels.experts);
    defer exp_mod.unload();
    const pack_fn = try pack_mod.function(pack_symbol);
    const packed_g = try packDownload(gpa, driver, stream.*, pack_fn, gathered_w, gathered_s, gathered_b, npack, width, dims);
    defer gpa.free(packed_g);
    const packed_u = try packDownload(gpa, driver, stream.*, pack_fn, up_w, up_s, up_b, npack, width, dims);
    defer gpa.free(packed_u);
    const packed_d = try packDownload(gpa, driver, stream.*, pack_fn, down_w, down_s, down_b, npack, dims, width);
    defer gpa.free(packed_d);
    const stacked = try stackHalves(gpa, packed_g, packed_u);
    defer gpa.free(stacked);

    var up_wb = try cuda.DeviceBuffer.fromHost(driver, stacked);
    defer up_wb.free();
    var down_wb = try cuda.DeviceBuffer.fromHost(driver, packed_d);
    defer down_wb.free();
    var local_pick: [slots]i32 = undefined;
    for (0..slots) |s| local_pick[s] = local[s];
    var picks_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&local_pick));
    defer picks_b.free();
    var members_b = try cuda.DeviceBuffer.alloc(driver, slots * 4);
    defer members_b.free();
    var items_b = try cuda.DeviceBuffer.alloc(driver, slots * 3 * 4);
    defer items_b.free();
    var counts_b = try cuda.DeviceBuffer.alloc(driver, 8);
    defer counts_b.free();
    var act_b = try cuda.DeviceBuffer.alloc(driver, slots * width * 2);
    defer act_b.free();
    var y_b = try cuda.DeviceBuffer.alloc(driver, slots * dims * 4);
    defer y_b.free();
    const plan_fn = try exp_mod.function(plan_symbol);
    var plan_a: cuda.Args = .{};
    plan_a.add(picks_b.ptr);
    plan_a.add(cint(slots));
    plan_a.add(cint(npack));
    plan_a.add(cint(tile));
    plan_a.add(members_b.ptr);
    plan_a.add(items_b.ptr);
    plan_a.add(counts_b.ptr);
    try cuda.launch.launch(plan_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream.*, &plan_a);

    const up_fn = try exp_mod.function(up_symbol);
    const down_fn = try exp_mod.function(down_symbol);
    try launchExpert(up_fn, stream.*, mixed, dims, slots, up_wb.ptr, dims / 32, width / 32, items_b.ptr, counts_b.ptr, members_b.ptr, act_b.ptr, width, slots * (width / 32));
    try launchExpert(down_fn, stream.*, act_b.ptr, width, 0, down_wb.ptr, width / 32, dims / 32, items_b.ptr, counts_b.ptr, members_b.ptr, y_b.ptr, dims, slots * (dims / 32));

    const act_raw = try gpa.alloc(u8, slots * width * 2);
    defer gpa.free(act_raw);
    const y_raw = try gpa.alloc(u8, slots * dims * 4);
    defer gpa.free(y_raw);
    try stream.synchronize();
    try act_b.download(0, act_raw);
    try y_b.download(0, y_raw);

    var act_off: usize = 0;
    var act_steps: u32 = 0;
    var y_off: usize = 0;
    var y_ulp: u32 = 0;
    var y_steps: u32 = 0;
    var head_g: u32 = 0;
    var head_h: u32 = 0;
    const gate_col = try gpa.alloc(f32, width);
    defer gpa.free(gate_col);
    const up_col = try gpa.alloc(f32, width);
    defer gpa.free(up_col);
    const act_row = try gpa.alloc(u16, width);
    defer gpa.free(act_row);
    const down_col = try gpa.alloc(f32, dims);
    defer gpa.free(down_col);
    const gw = width * (dims / 8) * 4;
    const gs = width * (dims / 32) * 2;
    const dw = dims * (width / 8) * 4;
    const ds = dims * (width / 32) * 2;
    for (0..slots) |s| {
        const src: usize = @intCast(local[s]);
        project(row, gathered_w[src * gw ..][0..gw], gathered_s[src * gs ..][0..gs], gathered_b[src * gs ..][0..gs], width, dims, gate_col);
        project(row, up_w[src * gw ..][0..gw], up_s[src * gs ..][0..gs], up_b[src * gs ..][0..gs], width, dims, up_col);
        for (act_row, gate_col, up_col) |*o, gv, uv| o.* = swiglu(gv, uv);
        for (0..width) |c| {
            const got = std.mem.readInt(u16, act_raw[(s * width + c) * 2 ..][0..2], .little);
            const dist = hc.mixedSteps(got, act_row[c]);
            if (dist != 0) act_off += 1;
            if (dist > act_steps) act_steps = dist;
        }
        project(act_row, down_w[src * dw ..][0..dw], down_s[src * ds ..][0..ds], down_b[src * ds ..][0..ds], dims, width, down_col);
        for (0..dims) |c| {
            const got: f32 = @bitCast(std.mem.readInt(u32, y_raw[(s * dims + c) * 4 ..][0..4], .little));
            const dist = hc.ulps(got, down_col[c]);
            const steps = hc.mixedSteps(toBf16(got), toBf16(down_col[c]));
            if (dist != 0) y_off += 1;
            if (dist > y_ulp) y_ulp = dist;
            if (steps > y_steps) y_steps = steps;
            if (s == 0 and c == 0) {
                head_g = @bitCast(got);
                head_h = @bitCast(down_col[c]);
            }
        }
    }
    std.debug.print("experts slots {d} act_off {d} act_steps {d} y_off {d} y_ulp {d} y_steps {d} head {x:0>8} host {x:0>8}\n", .{ slots, act_off, act_steps, y_off, y_ulp, y_steps, head_g, head_h });
    if (pick_off != 0 or host_pick_off != 0 or wts_same.ulp != 0 or host_wts_steps > 1 or act_steps != 0 or y_steps != 0) return 1;
    return 0;
}

fn launchExpert(f: cuda.Function, stream: cuda.Stream, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, items: u64, counts: u64, members: u64, out: u64, n: usize, units: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(cint(x_stride));
    a.add(cint(slot_n));
    a.add(w);
    a.add(cint(kg));
    a.add(cint(nb));
    a.add(items);
    a.add(counts);
    a.add(members);
    a.add(out);
    a.add(cint(n));
    a.add(@as(f32, 0));
    const grid = (units + 3) / 4;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, stream, &a);
}

test "a tied logit keeps the lower expert id and the shared gate rounds to one half" {
    const logits = [_]f32{ 1, 3, 3, 0 };
    var picks: [3]i32 = undefined;
    var wts: [3]f32 = undefined;
    choose(&logits, 3, 2, &picks, &wts);
    try std.testing.expectEqual([3]i32{ 1, 2, 3 }, picks);
    try std.testing.expectEqual(@as(f32, 0.5), wts[0]);
    try std.testing.expectEqual(@as(f32, 0.5), wts[1]);
    try std.testing.expectEqual(@as(f32, 0.5), wts[2]);
}

test "each lane of the quad contributes to the group sum" {
    var x: [32]u16 = @splat(0);
    x[0] = 0x3f80;
    x[8] = 0x3f80;
    x[16] = 0x3f80;
    x[24] = 0x3f80;
    try std.testing.expectEqual(@as(f32, 4), groupSum(&x, 0));
}

test "scale times the nibble dot plus the group bias is five" {
    var words: [16]u8 = @splat(0);
    std.mem.writeInt(u32, words[0..4], 0x21, .little);
    var scale: [2]u8 = undefined;
    std.mem.writeInt(u16, &scale, 0x3f80, .little);
    var bias: [2]u8 = undefined;
    std.mem.writeInt(u16, &bias, 0x3f80, .little);
    var x: [32]u16 = @splat(0);
    x[0] = 0x3f80;
    x[1] = 0x3f80;
    var out: [1]f32 = undefined;
    project(&x, &words, &scale, &bias, 1, 32, &out);
    try std.testing.expectEqual(@as(f32, 5), out[0]);
}

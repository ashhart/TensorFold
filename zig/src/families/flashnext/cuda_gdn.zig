//! One-row Gated DeltaNet: depthwise conv, the delta-rule state, the gated norm, then the output projection.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");
const qmm = @import("cuda_qmm.zig");

pub const nk: usize = 16;
pub const nv: usize = 48;
pub const dk: usize = 128;
pub const dv: usize = 128;
pub const taps: usize = 4;
pub const conv_dim: usize = 2 * nk * dk + nv * dv;
pub const proj_width: usize = conv_dim + nv * dv + 2 * nv;
pub const value_dim: usize = nv * dv;
pub const groups: usize = value_dim / qmm.group;

const symbol: [:0]const u8 = "flashnext_gdn_chain";

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn bf(v: f32) f32 {
    return promote(toBf16(v));
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn softplus(x: f32) f32 {
    return if (x > 20) x else @log(1.0 + @exp(x));
}

fn warpSum(v: *[32]f32) void {
    var span: usize = 16;
    while (span != 0) : (span >>= 1) {
        var next: [32]f32 = undefined;
        for (0..32) |lane| next[lane] = v[lane] + v[lane ^ span];
        v.* = next;
    }
}

fn convAct(proj: []const u16, cs: []const u16, cw: []const u16, row: usize, channel: usize) f32 {
    var acc: f32 = 0;
    for (0..taps) |tap| {
        const at = row + tap;
        const x: f32 = if (at < taps - 1) promote(cs[at * conv_dim + channel]) else promote(proj[(at - (taps - 1)) * proj_width + channel]);
        acc = acc + promote(cw[channel * taps + tap]) * x;
    }
    return bf(acc / (1.0 + @exp(-acc)));
}

fn normPair(x: []f32, scale_dk: bool) void {
    var part: [32]f32 = undefined;
    for (0..32) |lane| {
        var ss: f32 = 0;
        for (0..4) |i| {
            const v = x[lane * 4 + i];
            ss = ss + v * v;
        }
        part[lane] = ss;
    }
    warpSum(&part);
    var inv = 1.0 / @sqrt(part[0] + 1e-6);
    if (scale_dk) inv = inv * (1.0 / @sqrt(@as(f32, @floatFromInt(dk))));
    for (0..dk) |i| x[i] = x[i] * inv;
}

fn update(state: []f32, base: usize, ks: []const f32, vs: []const f32, g: f32, beta: f32) void {
    for (0..32) |warp| {
        var kk: [32][4]f32 = undefined;
        for (0..32) |lane| {
            for (0..4) |i| kk[lane][i] = ks[lane * 4 + i];
        }
        for (0..4) |j| {
            var kv: [32]f32 = undefined;
            const row = warp * 4 + j;
            for (0..32) |lane| {
                var acc: f32 = 0;
                for (0..4) |i| {
                    const at = base + row * dk + lane * 4 + i;
                    state[at] = state[at] * g;
                    acc = acc + state[at] * kk[lane][i];
                }
                kv[lane] = acc;
            }
            warpSum(&kv);
            const delta = (vs[row] - kv[0]) * beta;
            for (0..32) |lane| {
                for (0..4) |i| {
                    const at = base + row * dk + lane * 4 + i;
                    state[at] = state[at] + kk[lane][i] * delta;
                }
            }
        }
    }
}

fn readout(state: []const f32, base: usize, qs: []const f32, ys: []f32) void {
    for (0..32) |warp| {
        for (0..4) |j| {
            var part: [32]f32 = undefined;
            const row = warp * 4 + j;
            for (0..32) |lane| {
                var acc: f32 = 0;
                for (0..4) |i| acc = acc + state[base + row * dk + lane * 4 + i] * qs[lane * 4 + i];
                part[lane] = acc;
            }
            warpSum(&part);
            ys[row] = bf(part[0]);
        }
    }
}

/// One fresh or continued row. `state` is read and written in place, layout [value head, row, column].
pub fn chain(proj: []const u16, cs: []const u16, cw: []const u16, state: []f32, a_log: []const f32, dt_bias: []const f32, norm_w: []const u16, eps: f32, rows: usize, out: []u16, xs: []f32) !void {
    if (rows == 0 or proj.len < rows * proj_width or cs.len < (taps - 1) * conv_dim or cw.len < conv_dim * taps) return error.UnexpectedTensor;
    if (state.len < nv * dv * dk or a_log.len < nv or dt_bias.len < nv or norm_w.len < dv) return error.UnexpectedTensor;
    if (out.len < rows * value_dim or xs.len < rows * groups) return error.UnexpectedTensor;
    for (0..rows) |r| {
        for (0..nv) |hv| {
            const hk = hv / (nv / nk);
            var qs: [dk]f32 = undefined;
            var ks: [dk]f32 = undefined;
            var vs: [dv]f32 = undefined;
            for (0..dk) |t| qs[t] = convAct(proj, cs, cw, r, hk * dk + t);
            for (0..dk) |t| ks[t] = convAct(proj, cs, cw, r, nk * dk + hk * dk + t);
            for (0..dv) |t| vs[t] = convAct(proj, cs, cw, r, 2 * nk * dk + hv * dv + t);
            normPair(&qs, true);
            normPair(&ks, false);
            const b = promote(proj[r * proj_width + conv_dim + nv * dv + hv]);
            const a = promote(proj[r * proj_width + conv_dim + nv * dv + nv + hv]);
            const g = @exp(-@exp(a_log[hv]) * softplus(a + dt_bias[hv]));
            const beta = bf(sigmoid(b));
            const base = hv * dv * dk;
            update(state, base, &ks, &vs, g, beta);
            var ys: [dv]f32 = undefined;
            readout(state, base, &qs, &ys);
            var part: [32]f32 = undefined;
            for (0..32) |lane| {
                var ss: f32 = 0;
                for (0..4) |i| {
                    const y = ys[lane * 4 + i];
                    ss = ss + y * y;
                }
                part[lane] = ss;
            }
            warpSum(&part);
            const rinv = 1.0 / @sqrt(part[0] / @as(f32, @floatFromInt(dv)) + eps);
            var gated: [dv]f32 = undefined;
            for (0..dv) |t| {
                const yn = bf(bf(ys[t] * rinv) * promote(norm_w[t]));
                const z = promote(proj[r * proj_width + conv_dim + hv * dv + t]);
                gated[t] = bf(yn * sigmoid(z));
                out[r * value_dim + hv * dv + t] = toBf16(gated[t]);
            }
            for (0..dv / 32) |warp| {
                for (0..32) |lane| part[lane] = gated[warp * 32 + lane];
                warpSum(&part);
                xs[r * groups + hv * (dv / 32) + warp] = part[0];
            }
        }
    }
}

fn orderedUlps(a: f32, b: f32) u32 {
    const ai: i32 = @bitCast(a);
    const bi: i32 = @bitCast(b);
    const ao: i32 = if (ai < 0) std.math.minInt(i32) - ai else ai;
    const bo: i32 = if (bi < 0) std.math.minInt(i32) - bi else bi;
    const d = ao - bo;
    return @intCast(if (d < 0) -d else d);
}

/// The one-row kernel. Conv state is [3, conv channels], recurrence state is [heads, 128, 128].
pub fn launch(d: *const cuda.Driver, stream: cuda.Stream, proj: u64, cs: u64, cw: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, eps: f32, out: u64, xs: u64, state_out: u64) !void {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    var module = try cuda.Module.load(d, cuda.kernels.flashnext_gdn);
    defer module.unload();
    const f = try module.function(symbol);
    var args: cuda.Args = .{};
    args.add(proj);
    args.add(cs);
    args.add(cw);
    args.add(state_in);
    args.add(a_log);
    args.add(dt_bias);
    args.add(norm_w);
    args.add(eps);
    args.add(@as(c_int, 1));
    args.add(out);
    args.add(xs);
    args.add(state_out);
    try cuda.launch.launch(f, .{ .grid = .{ .x = nv }, .block = .{ .x = 1024 } }, stream, &args);
}

fn f32FromBf16(bytes: []const u8) f32 {
    return promote(std.mem.readInt(u16, bytes[0..2], .little));
}

/// Layer 0's conv, recurrence and output projection. A match also returns the output face's eight unreduced K slices.
pub fn memory(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, mapped: *const embed.Mapped, proj_dev: cuda.DeviceBuffer, proj_bytes: []const u8) !?[]f32 {
    const prefix = "language_model.model.layers.0.linear_attn.";
    var name: [160]u8 = undefined;
    const conv = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}conv1d.weight", .{prefix}), .bf16);
    const a_bits = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}A_log", .{prefix}), .bf16);
    const dt_bits = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}dt_bias", .{prefix}), .bf16);
    const norm = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}norm.weight", .{prefix}), .bf16);
    const out_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}out_proj.weight", .{prefix}), .u32);
    const out_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}out_proj.scales", .{prefix}), .bf16);
    const out_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}out_proj.biases", .{prefix}), .bf16);
    if (conv.dim(0) != conv_dim or conv.dim(1) != taps or norm.dim(0) != dv) return error.UnexpectedTensor;
    if (a_bits.dim(0) != nv or dt_bits.dim(0) != nv) return error.UnexpectedTensor;
    if (out_w.dim(0) != 2560 or out_w.dim(1) != value_dim / 8) return error.UnexpectedTensor;
    if (proj_bytes.len != proj_width * 2) return error.UnexpectedTensor;

    const proj = try gpa.alloc(u16, proj_width);
    defer gpa.free(proj);
    for (proj, 0..) |*o, i| o.* = std.mem.readInt(u16, proj_bytes[2 * i ..][0..2], .little);
    const cs_bytes = try gpa.alloc(u8, (taps - 1) * conv_dim * 2);
    defer gpa.free(cs_bytes);
    @memset(cs_bytes, 0);
    const a_log = try gpa.alloc(f32, nv);
    defer gpa.free(a_log);
    const dt_bias = try gpa.alloc(f32, nv);
    defer gpa.free(dt_bias);
    for (0..nv) |i| {
        a_log[i] = f32FromBf16(a_bits.bytes[2 * i ..]);
        dt_bias[i] = f32FromBf16(dt_bits.bytes[2 * i ..]);
    }
    const state_n = nv * dv * dk;
    const state = try gpa.alloc(f32, state_n);
    defer gpa.free(state);
    @memset(state, 0);
    var st_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(state));
    defer st_b.free();
    const cs = try gpa.alloc(u16, (taps - 1) * conv_dim);
    defer gpa.free(cs);
    @memset(cs, 0);
    const cw = try gpa.alloc(u16, conv_dim * taps);
    defer gpa.free(cw);
    for (cw, 0..) |*o, i| o.* = std.mem.readInt(u16, conv.bytes[2 * i ..][0..2], .little);
    const norm_w = try gpa.alloc(u16, dv);
    defer gpa.free(norm_w);
    for (norm_w, 0..) |*o, i| o.* = std.mem.readInt(u16, norm.bytes[2 * i ..][0..2], .little);
    const host_out = try gpa.alloc(u16, value_dim);
    defer gpa.free(host_out);
    const host_xs = try gpa.alloc(f32, groups);
    defer gpa.free(host_xs);
    const eps: f32 = 1e-6;
    try chain(proj, cs, cw, state, a_log, dt_bias, norm_w, eps, 1, host_out, host_xs);

    var cs_b = try cuda.DeviceBuffer.fromHost(driver, cs_bytes);
    defer cs_b.free();
    var cw_b = try cuda.DeviceBuffer.fromHost(driver, conv.bytes);
    defer cw_b.free();
    var a_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(a_log));
    defer a_b.free();
    var dt_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(dt_bias));
    defer dt_b.free();
    var norm_b = try cuda.DeviceBuffer.fromHost(driver, norm.bytes);
    defer norm_b.free();
    var st_out = try cuda.DeviceBuffer.alloc(driver, state_n * 4);
    defer st_out.free();
    var gout = try cuda.DeviceBuffer.alloc(driver, value_dim * 2);
    defer gout.free();
    var gxs = try cuda.DeviceBuffer.alloc(driver, groups * 4);
    defer gxs.free();
    try launch(driver, stream.*, proj_dev.ptr, cs_b.ptr, cw_b.ptr, st_b.ptr, a_b.ptr, dt_b.ptr, norm_b.ptr, eps, gout.ptr, gxs.ptr, st_out.ptr);

    const got_bytes = try gpa.alloc(u8, value_dim * 2);
    defer gpa.free(got_bytes);
    const xs_bytes = try gpa.alloc(u8, groups * 4);
    defer gpa.free(xs_bytes);
    const st_bytes = try gpa.alloc(u8, state_n * 4);
    defer gpa.free(st_bytes);
    try stream.synchronize();
    try gout.download(0, got_bytes);
    try gxs.download(0, xs_bytes);
    try st_out.download(0, st_bytes);

    var out_off: usize = 0;
    var out_steps: u32 = 0;
    for (host_out, 0..) |want, i| {
        const got = std.mem.readInt(u16, got_bytes[2 * i ..][0..2], .little);
        const dist = hc.mixedSteps(got, want);
        if (dist != 0) out_off += 1;
        if (dist > out_steps) out_steps = dist;
    }
    var xs_off: usize = 0;
    var xs_ulp: u32 = 0;
    for (host_xs, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
        const dist = orderedUlps(got, want);
        if (dist != 0) xs_off += 1;
        if (dist > xs_ulp) xs_ulp = dist;
    }
    var st_off: usize = 0;
    var st_ulp: u32 = 0;
    for (state, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, st_bytes[4 * i ..][0..4], .little));
        const dist = orderedUlps(got, want);
        if (dist != 0) st_off += 1;
        if (dist > st_ulp) st_ulp = dist;
    }
    const got_head = std.mem.readInt(u16, got_bytes[0..2], .little);
    std.debug.print("gdn out {d} off {d} max_steps {d} xs_off {d} xs_ulp {d} state_off {d} state_ulp {d} head {x:0>4} host {x:0>4}\n", .{ value_dim, out_off, out_steps, xs_off, xs_ulp, st_off, st_ulp, got_head, host_out[0] });

    const got_u16 = try gpa.alloc(u16, value_dim);
    defer gpa.free(got_u16);
    for (got_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, got_bytes[2 * i ..][0..2], .little);
    const got_xs = try gpa.alloc(f32, groups);
    defer gpa.free(got_xs);
    for (got_xs, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
    const host_proj = try gpa.alloc(u16, 2560);
    defer gpa.free(host_proj);
    try qmm.dotRow(got_u16, got_xs, out_w.bytes, out_s.bytes, out_b.bytes, 2560, value_dim, host_proj);
    var lane = try qmm.pack(gpa, out_w.bytes, out_s.bytes, out_b.bytes, 2560, value_dim / 8);
    defer lane.deinit(gpa);
    var w_b = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    defer b_b.free();
    var branch = try cuda.DeviceBuffer.alloc(driver, 2560 * 2);
    defer branch.free();
    try qmm.matmul(driver, stream.*, gout.ptr, gxs.ptr, w_b.ptr, s_b.ptr, b_b.ptr, branch.ptr, 1, 2560, value_dim);
    const branch_bytes = try gpa.alloc(u8, 2560 * 2);
    defer gpa.free(branch_bytes);
    try stream.synchronize();
    try branch.download(0, branch_bytes);
    var proj_off: usize = 0;
    var proj_steps: u32 = 0;
    for (host_proj, 0..) |want, i| {
        const got = std.mem.readInt(u16, branch_bytes[2 * i ..][0..2], .little);
        const dist = hc.mixedSteps(got, want);
        if (dist != 0) proj_off += 1;
        if (dist > proj_steps) proj_steps = dist;
    }
    std.debug.print("gdn out_proj n {d} off {d} max_steps {d} head {x:0>4} host {x:0>4}\n", .{ 2560, proj_off, proj_steps, std.mem.readInt(u16, branch_bytes[0..2], .little), host_proj[0] });
    if (out_off != 0 or xs_off != 0 or st_off != 0 or proj_off != 0) return null;
    // Eight K slices, left unreduced for the write-back.
    const sk: usize = 8;
    const n: usize = 2560;
    var part_b = try cuda.DeviceBuffer.alloc(driver, sk * n * 4);
    defer part_b.free();
    try qmm.partials(driver, stream.*, gout.ptr, gxs.ptr, w_b.ptr, s_b.ptr, b_b.ptr, branch.ptr, part_b.ptr, 1, n, value_dim, sk);
    const raw = try gpa.alloc(u8, sk * n * 4);
    defer gpa.free(raw);
    try stream.synchronize();
    try part_b.download(0, raw);
    const slices = try gpa.alloc(f32, sk * n);
    errdefer gpa.free(slices);
    for (slices, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
    return slices;
}

test "warp sum is the sum of the lanes" {
    var v: [32]f32 = @splat(0);
    v[0] = 1;
    v[31] = 3;
    warpSum(&v);
    try std.testing.expectEqual(@as(f32, 4), v[0]);
    try std.testing.expectEqual(@as(f32, 4), v[17]);
}

test "a zero projection and zero weights stay a zero row" {
    const gpa = std.testing.allocator;
    const proj = try gpa.alloc(u16, proj_width);
    defer gpa.free(proj);
    @memset(proj, 0);
    const cs = try gpa.alloc(u16, (taps - 1) * conv_dim);
    defer gpa.free(cs);
    @memset(cs, 0);
    const cw = try gpa.alloc(u16, conv_dim * taps);
    defer gpa.free(cw);
    @memset(cw, 0);
    const state = try gpa.alloc(f32, nv * dv * dk);
    defer gpa.free(state);
    @memset(state, 0);
    const a_log = try gpa.alloc(f32, nv);
    defer gpa.free(a_log);
    @memset(a_log, 0);
    const dt = try gpa.alloc(f32, nv);
    defer gpa.free(dt);
    @memset(dt, 0);
    const norm = try gpa.alloc(u16, dv);
    defer gpa.free(norm);
    @memset(norm, 0);
    const out = try gpa.alloc(u16, value_dim);
    defer gpa.free(out);
    const xs = try gpa.alloc(f32, groups);
    defer gpa.free(xs);
    try chain(proj, cs, cw, state, a_log, dt, norm, 1e-6, 1, out, xs);
    try std.testing.expectEqual(@as(u16, 0), out[0]);
    try std.testing.expectEqual(@as(u16, 0), out[value_dim - 1]);
    try std.testing.expectEqual(@as(f32, 0), xs[0]);
    try std.testing.expectEqual(@as(f32, 0), state[0]);
}

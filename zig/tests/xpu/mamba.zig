//! Mamba2 mixer decode step on Nemotron layer 0: each op on fixture inputs, then the whole chain over 6 tokens.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");
const fixture = @import("fixture.zig");

const spv = xpu.kernels.mamba;
var f_w_in: []const u8 = &.{};
var f_s_in: []const u8 = &.{};
var f_b_in: []const u8 = &.{};
var f_w_out: []const u8 = &.{};
var f_s_out: []const u8 = &.{};
var f_b_out: []const u8 = &.{};
var f_conv_w: []const u8 = &.{};
var f_conv_b: []const u8 = &.{};
var f_a_log: []const u8 = &.{};
var f_d: []const u8 = &.{};
var f_dtb: []const u8 = &.{};
var f_norm_w: []const u8 = &.{};
var f_x: []const u8 = &.{};
var f_proj: []const u8 = &.{};
var f_xc: []const u8 = &.{};
var f_y: []const u8 = &.{};
var f_yn: []const u8 = &.{};
var f_out: []const u8 = &.{};
var f_conv_state: []const u8 = &.{};
var f_ssm_state: []const u8 = &.{};

const T = 6;
const hidden = 2688;
const heads = 64;
const head_dim = 64;
const state_dim = 128;
const n_groups = 8;
const xd = heads * head_dim; // 4096
const cd = xd + 2 * n_groups * state_dim; // 6144
const proj_dim = xd + cd + heads; // 10304
const eps: f32 = 1e-5;
const inf = std.math.inf(f32);

fn at(b: rt.Buffer, off: usize) rt.Buffer {
    return b.at(off);
}

/// Embedded bytes carry no alignment guarantee; copies them into u16 storage.
fn u16s(bytes: []const u8) ![]u16 {
    const out = try std.heap.page_allocator.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn f32s(bytes: []const u8) ![]f32 {
    const out = try std.heap.page_allocator.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Distance in bf16 steps along the number line (sign-magnitude ordered).
fn ulps(a: u16, b: u16) u32 {
    const oa: i32 = if (a & 0x8000 != 0) -@as(i32, a & 0x7fff) else a;
    const ob: i32 = if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) else b;
    return @abs(oa - ob);
}

const Stat = struct { differ: usize, worst: u32, max_abs: f32, max_ref: f32 };

fn stat(got: []const u16, want: []const u16) Stat {
    var s: Stat = .{ .differ = 0, .worst = 0, .max_abs = 0, .max_ref = 0 };
    for (got, want) |g, w| {
        const d = ulps(g, w);
        if (d > 0) s.differ += 1;
        s.worst = @max(s.worst, d);
        s.max_abs = @max(s.max_abs, @abs(bf(g) - bf(w)));
        s.max_ref = @max(s.max_ref, @abs(bf(w)));
    }
    return s;
}

/// Accumulates per-token stats for one op and prints the verdict; `max_ulp` is the allowed worst bf16 distance.
const Check = struct {
    name: []const u8,
    total: usize = 0,
    differ: usize = 0,
    worst: u32 = 0,
    max_abs: f32 = 0,
    max_ref: f32 = 0,

    fn add(self: *Check, got: []const u16, want: []const u16) void {
        const s = stat(got, want);
        self.total += got.len;
        self.differ += s.differ;
        self.worst = @max(self.worst, s.worst);
        self.max_abs = @max(self.max_abs, s.max_abs);
        self.max_ref = @max(self.max_ref, s.max_ref);
    }

    fn report(self: Check, max_ulp: u32, max_frac: f32) !void {
        const frac = @as(f32, @floatFromInt(self.differ)) / @as(f32, @floatFromInt(self.total));
        std.debug.print("  {s}: {d} values, {d} differ ({d:.4}%), worst {d} ulp, max abs err {e} (max |ref| {e})\n", .{ self.name, self.total, self.differ, frac * 100, self.worst, self.max_abs, self.max_ref });
        if (self.worst > max_ulp or frac > max_frac) return error.TooInaccurate;
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Mixer = struct {
    r: *rt.Runtime,
    w_in: rt.Buffer,
    s_in: rt.Buffer,
    b_in: rt.Buffer,
    w_out: rt.Buffer,
    s_out: rt.Buffer,
    b_out: rt.Buffer,
    conv_w: rt.Buffer,
    conv_b: rt.Buffer,
    a_log: rt.Buffer,
    d: rt.Buffer,
    dtb: rt.Buffer,
    norm_w: rt.Buffer,
    k_qmv: rt.Kernel,
    k_conv: rt.Kernel,
    k_ssm: rt.Kernel,
    k_norm: rt.Kernel,

    fn upload(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
        const b = try r.alloc(bytes.len);
        try r.upload(b, bytes);
        return b;
    }

    fn init(r: *rt.Runtime, m: *rt.Module) !Mixer {
        return .{
            .r = r,
            .w_in = try upload(r, f_w_in),
            .s_in = try upload(r, f_s_in),
            .b_in = try upload(r, f_b_in),
            .w_out = try upload(r, f_w_out),
            .s_out = try upload(r, f_s_out),
            .b_out = try upload(r, f_b_out),
            .conv_w = try upload(r, f_conv_w),
            .conv_b = try upload(r, f_conv_b),
            .a_log = try upload(r, f_a_log),
            .d = try upload(r, f_d),
            .dtb = try upload(r, f_dtb),
            .norm_w = try upload(r, f_norm_w),
            .k_qmv = try m.kernel("qmv4_bf16", .{ 16, 1, 1 }),
            .k_conv = try m.kernel("conv1d_step", .{ 64, 1, 1 }),
            .k_ssm = try m.kernel("ssm_step", .{ 64, 1, 1 }),
            .k_norm = try m.kernel("group_rmsnorm", .{ 64, 1, 1 }),
        };
    }

    /// y[rows] = W x (bf16 in, bf16 out) with the in_proj or out_proj tables.
    fn qmv(self: *Mixer, out_proj: bool, x: rt.Buffer, y: rt.Buffer) !void {
        const k = &self.k_qmv;
        try k.setBuffer(0, if (out_proj) self.w_out else self.w_in);
        try k.setBuffer(1, if (out_proj) self.s_out else self.s_in);
        try k.setBuffer(2, if (out_proj) self.b_out else self.b_in);
        try k.setBuffer(3, x);
        try k.setBuffer(4, y);
        try k.setU32(5, if (out_proj) xd else hidden);
        try k.launch(.{ if (out_proj) hidden else proj_dim, 1, 1 });
    }

    fn conv(self: *Mixer, proj: rt.Buffer, state: rt.Buffer, out: rt.Buffer) !void {
        const k = &self.k_conv;
        try k.setBuffer(0, proj);
        try k.setU32(1, xd);
        try k.setBuffer(2, state);
        try k.setBuffer(3, self.conv_w);
        try k.setBuffer(4, self.conv_b);
        try k.setBuffer(5, out);
        try k.setU32(6, cd);
        try k.launch(.{ cd / 64, 1, 1 });
    }

    fn ssm(self: *Mixer, proj: rt.Buffer, xc: rt.Buffer, state: rt.Buffer, y: rt.Buffer) !void {
        const k = &self.k_ssm;
        try k.setBuffer(0, proj);
        try k.setU32(1, xd + cd);
        try k.setBuffer(2, xc);
        try k.setBuffer(3, state);
        try k.setBuffer(4, self.a_log);
        try k.setBuffer(5, self.d);
        try k.setBuffer(6, self.dtb);
        try k.setBuffer(7, y);
        try k.setU32(8, head_dim);
        try k.setU32(9, xd);
        try k.setU32(10, n_groups);
        try k.setU32(11, heads / n_groups);
        try k.setU32(12, @bitCast(@as(f32, 0.0))); // time_step_limit absent in config: (0, inf)
        try k.setU32(13, @bitCast(inf));
        try k.launch(.{ heads, head_dim / 4, 1 });
    }

    fn norm(self: *Mixer, x: rt.Buffer, y: rt.Buffer) !void {
        const k = &self.k_norm;
        try k.setBuffer(0, x);
        try k.setBuffer(1, self.norm_w);
        try k.setBuffer(2, y);
        try k.setU32(3, xd / n_groups);
        try k.setU32(4, @bitCast(eps));
        try k.launch(.{ n_groups, 1, 1 });
    }
};

fn zeros(r: *rt.Runtime, bytes: usize) !rt.Buffer {
    const host = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(host);
    @memset(host, 0);
    const b = try r.alloc(bytes);
    try r.upload(b, host);
    try r.sync();
    return b;
}

fn fetch16(r: *rt.Runtime, b: rt.Buffer, out: []u16) !void {
    try r.download(std.mem.sliceAsBytes(out), b);
    try r.sync();
}

/// Largest |got - ref| relative to max(|ref|, 1e-3 * max |ref|) over an fp32 state.
fn stateError(got: []const f32, want: []const f32) f32 {
    var mx: f32 = 0;
    for (want) |v| mx = @max(mx, @abs(v));
    var worst: f32 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(g - w) / @max(@abs(w), 1e-3 * mx));
    return worst;
}

pub fn run() !void {
    try loadFixtures();
    var r = try rt.Runtime.init();
    defer r.deinit();
    var m = try r.module(spv);
    defer m.deinit();
    var mx = try Mixer.init(r, &m);
    const x = try Mixer.upload(r, f_x);
    const proj_f = try Mixer.upload(r, f_proj);
    const xc_f = try Mixer.upload(r, f_xc);
    const y_f = try Mixer.upload(r, f_y);
    const yn_f = try Mixer.upload(r, f_yn);
    try r.sync();
    const want_proj = try u16s(f_proj);
    const want_xc = try u16s(f_xc);
    const want_y = try u16s(f_y);
    const want_yn = try u16s(f_yn);
    const want_out = try u16s(f_out);
    const want_conv_state = try u16s(f_conv_state);
    const want_ssm = try f32s(f_ssm_state);

    const proj = try r.alloc(proj_dim * 2);
    const xc = try r.alloc(cd * 2);
    const y = try r.alloc(xd * 2);
    const yn = try r.alloc(xd * 2);
    const out = try r.alloc(hidden * 2);
    var h_proj: [proj_dim]u16 = undefined;
    var h_xc: [cd]u16 = undefined;
    var h_y: [xd]u16 = undefined;
    var h_out: [hidden]u16 = undefined;

    // Per-op checks on fixture inputs: bf16 results within 1 ulp (fp32 order / exp differences only flip roundings).
    std.debug.print("per-op, fixture inputs, {d} tokens:\n", .{T});
    {
        var c: Check = .{ .name = "in_proj  " };
        for (0..T) |t| {
            try mx.qmv(false, at(x, t * hidden * 2), proj);
            try fetch16(r, proj, &h_proj);
            c.add(&h_proj, want_proj[t * proj_dim ..][0..proj_dim]);
        }
        try c.report(1, 0.01);
    }
    {
        const state = try zeros(r, 3 * cd * 2);
        var c: Check = .{ .name = "conv+silu " };
        for (0..T) |t| {
            try mx.conv(at(proj_f, t * proj_dim * 2), state, xc);
            try fetch16(r, xc, &h_xc);
            c.add(&h_xc, want_xc[t * cd ..][0..cd]);
        }
        try c.report(1, 0.01);
        var h_state: [3 * cd]u16 = undefined;
        try fetch16(r, state, &h_state);
        var sc: Check = .{ .name = "conv state" };
        sc.add(&h_state, want_conv_state);
        try sc.report(0, 0.0);
    }
    {
        const state = try zeros(r, heads * head_dim * state_dim * 4);
        var c: Check = .{ .name = "ssm y     " };
        for (0..T) |t| {
            try mx.ssm(at(proj_f, t * proj_dim * 2), at(xc_f, t * cd * 2), state, y);
            try fetch16(r, y, &h_y);
            c.add(&h_y, want_y[t * xd ..][0..xd]);
        }
        try c.report(1, 0.01);
        const got = try std.heap.page_allocator.alloc(f32, want_ssm.len);
        try r.download(std.mem.sliceAsBytes(got), state);
        try r.sync();
        const e = stateError(got, want_ssm);
        std.debug.print("  ssm state: {d} values, max relative error {e}\n", .{ got.len, e });
        if (e > 1e-4) return error.TooInaccurate;
    }
    {
        var c: Check = .{ .name = "group norm" };
        for (0..T) |t| {
            try mx.norm(at(y_f, t * xd * 2), yn);
            try fetch16(r, yn, &h_y);
            c.add(&h_y, want_yn[t * xd ..][0..xd]);
        }
        try c.report(1, 0.01);
    }
    {
        var c: Check = .{ .name = "out_proj  " };
        for (0..T) |t| {
            try mx.qmv(true, at(yn_f, t * xd * 2), out);
            try fetch16(r, out, &h_out);
            c.add(&h_out, want_out[t * hidden ..][0..hidden]);
        }
        try c.report(1, 0.01);
    }

    // Chain: GPU-only carry of conv and SSM state across the 6 tokens, every intermediate compared with the fixtures.
    std.debug.print("chain, {d} tokens, states carried on the device:\n", .{T});
    {
        const cstate = try zeros(r, 3 * cd * 2);
        const sstate = try zeros(r, heads * head_dim * state_dim * 4);
        var cp: Check = .{ .name = "proj" };
        var cx: Check = .{ .name = "xc  " };
        var cy: Check = .{ .name = "y   " };
        var co: Check = .{ .name = "out " };
        var worst_rel: f32 = 0;
        for (0..T) |t| {
            try mx.qmv(false, at(x, t * hidden * 2), proj);
            try mx.conv(proj, cstate, xc);
            try mx.ssm(proj, xc, sstate, y);
            try mx.norm(y, yn);
            try mx.qmv(true, yn, out);
            try fetch16(r, proj, &h_proj);
            try fetch16(r, xc, &h_xc);
            try fetch16(r, y, &h_y);
            try fetch16(r, out, &h_out);
            cp.add(&h_proj, want_proj[t * proj_dim ..][0..proj_dim]);
            cx.add(&h_xc, want_xc[t * cd ..][0..cd]);
            cy.add(&h_y, want_y[t * xd ..][0..xd]);
            co.add(&h_out, want_out[t * hidden ..][0..hidden]);
            const s = stat(&h_out, want_out[t * hidden ..][0..hidden]);
            worst_rel = @max(worst_rel, s.max_abs / s.max_ref);
        }
        // bf16 flips in proj (fp32 accumulation order) cascade through the recurrence; at most 1 ulp on a few values.
        try cp.report(1, 0.001);
        try cx.report(1, 0.001);
        try cy.report(1, 0.001);
        try co.report(1, 0.001);
        std.debug.print("  out: worst per-token max abs error / max |ref| = {e}\n", .{worst_rel});
        if (worst_rel > 2e-3) return error.TooInaccurate;
        const got = try std.heap.page_allocator.alloc(f32, want_ssm.len);
        try r.download(std.mem.sliceAsBytes(got), sstate);
        try r.sync();
        const e = stateError(got, want_ssm);
        std.debug.print("  ssm state after chain: max relative error {e}\n", .{e});
    }

    // Timing: repeated launches of each op on token 0, states left to drift.
    std.debug.print("timing (50 launches each, per launch):\n", .{});
    {
        const cstate = try zeros(r, 3 * cd * 2);
        const sstate = try zeros(r, heads * head_dim * state_dim * 4);
        try mx.qmv(false, x, proj);
        try r.sync();
        const iters = 50;
        var t0 = nowNs();
        for (0..iters) |_| try mx.qmv(false, x, proj);
        try r.sync();
        const dt_in = (nowNs() - t0) / iters;
        t0 = nowNs();
        for (0..iters) |_| try mx.conv(proj, cstate, xc);
        try r.sync();
        const dt_conv = (nowNs() - t0) / iters;
        t0 = nowNs();
        for (0..iters) |_| try mx.ssm(proj, xc, sstate, y);
        try r.sync();
        const dt_ssm = (nowNs() - t0) / iters;
        t0 = nowNs();
        for (0..iters) |_| try mx.norm(y, yn);
        try r.sync();
        const dt_norm = (nowNs() - t0) / iters;
        t0 = nowNs();
        for (0..iters) |_| try mx.qmv(true, yn, out);
        try r.sync();
        const dt_out = (nowNs() - t0) / iters;
        t0 = nowNs();
        for (0..iters) |_| {
            try mx.qmv(false, x, proj);
            try mx.conv(proj, cstate, xc);
            try mx.ssm(proj, xc, sstate, y);
            try mx.norm(y, yn);
            try mx.qmv(true, yn, out);
        }
        try r.sync();
        const dt_all = (nowNs() - t0) / iters;
        const gb_in = @as(f64, @floatFromInt(f_w_in.len + f_s_in.len + f_b_in.len)) / @as(f64, @floatFromInt(dt_in));
        const gb_out = @as(f64, @floatFromInt(f_w_out.len + f_s_out.len + f_b_out.len)) / @as(f64, @floatFromInt(dt_out));
        const gb_ssm = @as(f64, @floatFromInt(heads * head_dim * state_dim * 8)) / @as(f64, @floatFromInt(dt_ssm));
        std.debug.print("  in_proj {d} us ({d:.1} GB/s), conv {d} us, ssm {d} us ({d:.1} GB/s state), norm {d} us, out_proj {d} us ({d:.1} GB/s)\n", .{ dt_in / 1000, gb_in, dt_conv / 1000, dt_ssm / 1000, gb_ssm, dt_norm / 1000, dt_out / 1000, gb_out });
        std.debug.print("  whole mixer step {d} us\n", .{dt_all / 1000});
    }
    std.debug.print("mamba mixer OK\n", .{});
}

fn loadFixtures() !void {
    f_w_in = try fixture.load("mamba_w_in");
    f_s_in = try fixture.load("mamba_s_in");
    f_b_in = try fixture.load("mamba_b_in");
    f_w_out = try fixture.load("mamba_w_out");
    f_s_out = try fixture.load("mamba_s_out");
    f_b_out = try fixture.load("mamba_b_out");
    f_conv_w = try fixture.load("mamba_conv_w");
    f_conv_b = try fixture.load("mamba_conv_b");
    f_a_log = try fixture.load("mamba_a_log");
    f_d = try fixture.load("mamba_d");
    f_dtb = try fixture.load("mamba_dt_bias");
    f_norm_w = try fixture.load("mamba_norm_w");
    f_x = try fixture.load("mamba_x");
    f_proj = try fixture.load("mamba_proj");
    f_xc = try fixture.load("mamba_xc");
    f_y = try fixture.load("mamba_y");
    f_yn = try fixture.load("mamba_yn");
    f_out = try fixture.load("mamba_out");
    f_conv_state = try fixture.load("mamba_conv_state");
    f_ssm_state = try fixture.load("mamba_ssm_state");
}

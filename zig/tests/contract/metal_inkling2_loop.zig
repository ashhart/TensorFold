//! Independent FP64 checks of Mini's original loop math and source-level fault controls.
const std = @import("std");
const mtl = @import("metal");
const loop = @import("tensorfold").inkling2_metal.loop;
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const rig = @import("rig.zig");
const math = @import("inkling2_math_loop.zig");
const c = u.c;
const g = u.g;

const Kind = enum { messages, mix_f32, down, qkv, mlp };
const Control = struct { kind: Kind, name: []const u8, from: []const u8, to: []const u8 };
const controls = [_]Control{
    .{ .kind = .messages, .name = "control: inkling2 loop messages use unnormalized values", .from = "coef[q][s] = (score[s] / l) * inv_tg[s];", .to = "coef[q][s] = score[s] / l;" },
    .{ .kind = .down, .name = "control: inkling2 loop controller drops message half of down projection", .from = "fma(float(wr[ulong(j) * 2 * H + H + i]), xm[e], acc[j])", .to = "fma(float(wr[ulong(j) * 2 * H + i]), xm[e], acc[j])" },
    .{ .kind = .qkv, .name = "control: inkling2 loop controller adds forget gate", .from = "1.0f - metal::precise::tanh(f)", .to = "1.0f + metal::precise::tanh(f)" },
    .{ .kind = .mlp, .name = "control: inkling2 loop MLP reads query message", .from = "return g == 0 ? a.slot0", .to = "return g == 0 ? 0" },
    .{ .kind = .messages, .name = "control: inkling2 loop statistics dot the third query with the second", .from = "acc[2] = fma(q1[i], v, acc[2]);", .to = "acc[2] = fma(q0[i], v, acc[2]);" },
    .{ .kind = .down, .name = "control: inkling2 loop controller norms the message with the state's scale", .from = "dst[H + i] = bfloat(xm[e] * inv_m);", .to = "dst[H + i] = bfloat(xm[e] * inv_h);" },
    .{ .kind = .messages, .name = "control: inkling2 loop multi-group messages use unnormalized values", .from = "coef[q][s] = inv_tg[s] * (score[s] / l);", .to = "coef[q][s] = score[s] / l;" },
    .{ .kind = .down, .name = "control: inkling2 loop multi-group controller drops message half", .from = "fma(float(wr[ulong(j) * 2 * H + H + i]), xm, acc[j])", .to = "fma(float(wr[ulong(j) * 2 * H + i]), xm, acc[j])" },
    .{ .kind = .qkv, .name = "control: inkling2 loop multi-group controller adds forget gate", .from = "(1.0f - forget)", .to = "(1.0f + forget)" },
    .{ .kind = .qkv, .name = "control: inkling2 loop gate output skips the norm weight", .from = "dst[i] = bfloat(float(nw[i]) * (v[e] * inv));", .to = "dst[i] = bfloat(v[e] * inv);" },
    .{ .kind = .qkv, .name = "control: inkling2 loop MMA gate states skip the first rank step", .from = "for (uint k = 0; k < R; k += 8u) {", .to = "for (uint k = 8u; k < R; k += 8u) {" },
    .{ .kind = .qkv, .name = "control: inkling2 loop MMA gate states take the gain from the forget weights", .from = "gb[j].thread_elements()[0] = wg[c * R + k + fm];", .to = "gb[j].thread_elements()[0] = wf[c * R + k + fm];" },
    .{ .kind = .down, .name = "control: inkling2 loop MMA controller drops the last chain step", .from = "for (uint e = steps & ~7u; e < steps; ++e) {", .to = "for (uint e = steps & ~7u; e + 1u < steps; ++e) {" },
    .{ .kind = .down, .name = "control: inkling2 loop MMA controller keeps only the last slice", .from = "total[t] = s == 0 ? slice_sum : total[t] + slice_sum;", .to = "total[t] = slice_sum;" },
    .{ .kind = .down, .name = "control: inkling2 loop MMA controller reads the state for the message half", .from = "return e < steps_half ? 256u * e : hidden + 256u * (e - steps_half);", .to = "return 256u * (e % steps_half);" },
};

// Rows past loop.small_rows run the one-group kernels (gate states and down on MMA), the rest the multi-group ones.
const mix_shapes = [_]c.Shape{ .{ .rows = 1, .n = 2304, .k = 1 }, .{ .rows = 3, .n = 96, .k = 8 }, .{ .rows = 2, .n = 128, .k = 0 }, .{ .rows = 17, .n = 128, .k = 3 } };
const gate_shapes = [_]c.Shape{ .{ .rows = 1, .n = 2304, .k = 64 }, .{ .rows = 2, .n = 96, .k = 7 }, .{ .rows = 3, .n = 128, .k = 128 }, .{ .rows = 17, .n = 96, .k = 9 }, .{ .rows = 17, .n = 96, .k = 64 }, .{ .rows = 33, .n = 130, .k = 8 }, .{ .rows = 40, .n = 2304, .k = 64 }, .{ .rows = 19, .n = 33, .k = 128 }, .{ .rows = 17, .n = 768, .k = 16 }, .{ .rows = 19, .n = 512, .k = 128 }, .{ .rows = 33, .n = 2560, .k = 64 } };
fn noSums(_: c.Shape) usize {
    return 0;
}

fn shifted(ref: rig.Ref, offset: usize) loop.View {
    return .{ .buffer = ref.buf, .offset = ref.off + offset };
}

const Probe = struct {
    arena: std.heap.ArenaAllocator,
    rig: rig.Rig,
    ops: loop.Ops,
    kind: Kind,
    name: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    // CPU copies of uploaded input, in the same plane order but referenced independently.
    blocks: []u16 = &.{},
    partial16: []u16 = &.{},
    partial32: []f32 = &.{},
    queries: []f32 = &.{},
    states: []f32 = &.{},
    messages: []f32 = &.{},
    dw: []u16 = &.{},
    fw: []u16 = &.{},
    gw: []u16 = &.{},
    nw: []u16 = &.{},
    x: rig.Ref = undefined,
    y: rig.Ref = undefined,
    z: rig.Ref = undefined,
    down_buf: rig.Ref = undefined,
    forget: rig.Ref = undefined,
    gain: rig.Ref = undefined,
    norm: rig.Ref = undefined,
    out: rig.Ref = undefined,

    fn isMix(p: *const Probe) bool {
        return p.kind == .messages or p.kind == .mix_f32;
    }
    fn lanes(p: *const Probe) usize {
        return switch (p.kind) {
            .messages => 4,
            .mix_f32, .qkv, .down => 3,
            .mlp => 1,
        };
    }
    fn plane(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }
    fn slot(p: *const Probe, i: usize) usize {
        return if (p.kind == .mlp) 3 else i;
    }
    fn upload(p: *Probe, data: anytype) !rig.Ref {
        return p.rig.input(std.mem.sliceAsBytes(data));
    }

    fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.ops.deinit();
    }

    const vt = g.hooks(Probe);
    fn kernel(p: *Probe, control: bool) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (control) .shapes else null, .spec = .{
            .name = p.name,
            .accum = .none,
            .depth = noSums,
            .out = if (p.isMix()) .f32 else .bf16,
            .served = if (p.isMix()) &mix_shapes else &gate_shapes,
        } };
    }

    pub fn count(p: *Probe) usize {
        return p.lanes() * p.shape.rows * (if (p.kind == .down) p.shape.k else p.shape.n);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const h = shape.n;
        const k = shape.k;
        if (rows == 0 or rows > 40 or h == 0 or h > loop.max_hidden) return false;
        if (p.isMix()) {
            if (k >= loop.max_sources) return false;
        } else if (k == 0 or k > loop.max_rank) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        const plane_n = p.plane();
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        p.out = try p.rig.output(p.count() * (if (p.isMix()) @as(usize, 4) else 2));
        if (p.isMix()) {
            p.blocks = try a.alloc(u16, k * plane_n);
            p.partial16 = try a.alloc(u16, plane_n);
            p.partial32 = try a.alloc(f32, plane_n);
            p.queries = try a.alloc(f32, p.lanes() * h);
            for (p.blocks, 0..) |*v, i| v.* = u.uniform(rnd, if ((i / plane_n) % 2 == 0) 4 else 0.0009);
            for (p.partial16, p.partial32) |*w, *v| {
                v.* = rnd.float(f32) * 2 - 1;
                w.* = c.toBf16(v.*);
            }
            for (p.queries) |*v| v.* = rnd.float(f32) * 2 - 1;
            if (fill == .nonfinite) {
                p.partial16[0] = u.nan_bits;
                p.partial32[0] = std.math.nan(f32);
            }
            p.x = try p.upload(p.blocks);
            p.y = if (p.kind == .mix_f32) try p.upload(p.partial32) else try p.upload(p.partial16);
            p.z = try p.upload(p.queries);
        } else {
            const gates = p.lanes();
            p.states = try a.alloc(f32, gates * plane_n);
            p.messages = try a.alloc(f32, 4 * plane_n);
            p.dw = try a.alloc(u16, gates * k * 2 * h);
            p.fw = try a.alloc(u16, gates * h * k);
            p.gw = try a.alloc(u16, gates * h * k);
            p.nw = try a.alloc(u16, gates * h);
            for (p.states) |*v| v.* = 2 * rnd.float(f32) - 1;
            for (p.messages, 0..) |*v, i| v.* = (2 * rnd.float(f32) - 1) * (if (i / plane_n == 3) @as(f32, 0.125) else 1);
            u.fillUniform(rnd, p.dw, 0.9 / @sqrt(@as(f32, @floatFromInt(h))));
            u.fillUniform(rnd, p.fw, 1.5 / @sqrt(@as(f32, @floatFromInt(k))));
            u.fillUniform(rnd, p.gw, 1.0 / @sqrt(@as(f32, @floatFromInt(k))));
            for (p.nw) |*v| v.* = c.toBf16(0.5 + rnd.float(f32));
            if (fill == .nonfinite) p.states[0] = std.math.nan(f32);
            p.x = try p.upload(p.states);
            p.y = try p.upload(p.messages);
            p.z = try p.upload(p.dw);
            p.forget = try p.upload(p.fw);
            p.gain = try p.upload(p.gw);
            p.norm = try p.upload(p.nw);
            p.down_buf = if (p.kind == .down) p.out else try p.rig.scratch(gates * rows * k * 2);
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const h = p.shape.n;
        const k = p.shape.k;
        const rows: u32 = @intCast(p.shape.rows);
        const stride: u32 = @intCast(p.plane());
        const pass = p.rig.begin();
        if (p.isMix()) {
            var queries: [4]loop.View = undefined;
            for (&queries, 0..) |*q, i| q.* = shifted(p.z, (i % p.lanes()) * h * 4);
            try p.ops.mix(pass.e, .{
                .blocks = u.view(p.x),
                .block_stride = stride,
                .nb = @intCast(k),
                .partial = u.view(p.y),
                .partial_f32 = p.kind == .mix_f32,
                .queries = queries,
                .nq = @intCast(p.lanes()),
                .out = u.view(p.out),
                .out_stride = stride,
                .rows = rows,
                .hidden = @intCast(h),
            });
        } else {
            var x: [3]loop.View = undefined;
            var ds: [3]loop.View = undefined;
            var fs: [3]loop.View = undefined;
            var gs: [3]loop.View = undefined;
            var ns: [3]loop.View = undefined;
            var slots: [3]u32 = undefined;
            for (0..3) |i| {
                const j = i % p.lanes();
                x[i] = shifted(p.x, j * stride * 4);
                ds[i] = shifted(p.z, j * k * 2 * h * 2);
                fs[i] = shifted(p.forget, j * h * k * 2);
                gs[i] = shifted(p.gain, j * h * k * 2);
                ns[i] = shifted(p.norm, j * h * 2);
                slots[i] = @intCast(p.slot(j));
            }
            try p.ops.gateDown(pass.e, .{
                .h = x,
                .msg = u.view(p.y),
                .msg_stride = stride,
                .slots = slots,
                .gates = @intCast(p.lanes()),
                .down_w = ds,
                .down = u.view(p.down_buf),
                .rows = rows,
                .hidden = @intCast(h),
                .rank = @intCast(k),
            });
            if (p.kind != .down) try p.ops.gateMix(pass.e, .{
                .h = x,
                .msg = u.view(p.y),
                .msg_stride = stride,
                .slots = slots,
                .gates = @intCast(p.lanes()),
                .down = u.view(p.down_buf),
                .forget = fs,
                .gain = gs,
                .norms = ns,
                .out = u.view(p.out),
                .out_stride = stride,
                .rows = rows,
                .hidden = @intCast(h),
                .rank = @intCast(k),
                .eps = 1e-6,
            });
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const h = p.shape.n;
        const k = p.shape.k;
        const rows = p.shape.rows;
        const plane_n = p.plane();
        const a = p.arena.allocator();
        @memset(mag, 0);
        if (p.isMix()) {
            const src = try a.alloc(f64, (k + 1) * h);
            defer a.free(src);
            for (0..rows) |r| {
                for (0..k) |b| for (src[b * h ..][0..h], p.blocks[b * plane_n + r * h ..][0..h]) |*v, w| {
                    v.* = u.val(w);
                };
                for (src[k * h ..], 0..) |*v, i| v.* = if (p.kind == .mix_f32) p.partial32[r * h + i] else u.val(p.partial16[r * h + i]);
                for (0..p.lanes()) |q| math.mix(src, p.queries[q * h ..][0..h], want[q * plane_n + r * h ..][0..h]);
            }
        } else {
            const x = try a.alloc(f64, h);
            defer a.free(x);
            const m = try a.alloc(f64, h);
            defer a.free(m);
            const d = try a.alloc(f64, k);
            defer a.free(d);
            for (0..p.lanes()) |i| for (0..rows) |r| {
                for (x, p.states[i * plane_n + r * h ..][0..h]) |*v, in| v.* = in;
                for (m, p.messages[p.slot(i) * plane_n + r * h ..][0..h]) |*v, in| v.* = in;
                math.down(x, m, p.dw[i * k * 2 * h ..][0 .. k * 2 * h], d);
                if (p.kind == .down) {
                    @memcpy(want[(i * rows + r) * k ..][0..k], d);
                } else {
                    math.gate(x, m, d, p.fw[i * h * k ..][0 .. h * k], p.gw[i * h * k ..][0 .. h * k], p.nw[i * h ..][0..h], want[i * plane_n + r * h ..][0..h]);
                }
            };
        }
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        // FP32 sums/scales; controller boundary ties may differ by one BF16 step in the down output, so propagate it.
        for (allow, want) |*e, v| e.* = if (p.isMix()) 5e-6 * (1 + @abs(v)) else 0.003 + 0x1p-7 * @abs(v);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, only: ?[]const u8, name: []const u8, kind: Kind, control: ?Control) !void {
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        const ops = if (control) |ctl| blk: {
            const source = try u.mutate(s.a, sources.inkling2_loop_ops, ctl.from, ctl.to);
            defer s.a.free(source);
            break :blk try loop.Ops.loadSource(s.device, .{}, source);
        } else try loop.Ops.load(s.device, .{});
        p.* = .{ .arena = .init(s.a), .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .ops = ops, .kind = kind, .name = name };
        errdefer p.deinit();
        try s.held.ensureUnusedCapacity(s.a, 1);
        try s.kernels.append(s.a, p.kernel(control != null));
        s.held.appendAssumeCapacity(p);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(only, "metal inkling2 loop four cross-pass messages", .messages, null);
        try s.add(only, "metal inkling2 loop FP32 partial input mix", .mix_f32, null);
        try s.add(only, "metal inkling2 loop BF16 controller down", .down, null);
        try s.add(only, "metal inkling2 loop distinct Q K V controllers", .qkv, null);
        try s.add(only, "metal inkling2 loop MLP controller", .mlp, null);
        for (controls) |ctl| try s.add(only, ctl.name, ctl.kind, ctl);
    }

    pub fn deinit(s: *Suite) void {
        for (s.held.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.held.deinit(s.a);
    }
};

//! Byte lanes and external group sums retain the pinned staged shader's output at identical K splits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const base = @import("metal_affine.zig");
const variants = @import("metal_lane_variants.zig");
const rig_ = @import("rig.zig");
const lane = tf.lane_projection;
const c = ct.spec;
const g = ct.gates;
pub const Bad = enum { none, group, stride };
const Case = struct { reg: base.Reg, name: []const u8, bad: Bad = .none, pf: usize = 1, given: bool = true };
const small = [_]c.Shape{ .{ .rows = 1, .n = 512, .k = 512 }, .{ .rows = 16, .n = 64, .k = 512 }, .{ .rows = 5, .n = 48, .k = 512 } };
const cases = [_]Case{
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 1, .f32_out = true, .served = &small }, .name = "qwen27 q8 exact fp32 sk1" }, .name = "qwen27 q8 exact fp32 sk1" },
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 8, .f32_out = true, .served = &small }, .name = "qwen27 q8 exact fp32 sk8" }, .name = "qwen27 q8 exact fp32 sk8" },
    .{ .reg = variants.q8[0], .name = "qwen27 q8 exact g64 sk1" },
    .{ .reg = variants.q8[1], .name = "qwen27 q8 exact g64 sk2" },
    .{ .reg = variants.q8[2], .name = "qwen27 q8 exact g64 sk4" },
    .{ .reg = variants.q8[3], .name = "qwen27 q8 exact g64 sk8" },
    .{ .reg = variants.q8[2], .name = "control: qwen27 q8 exact g64 wrong group", .bad = .group },
    .{ .reg = variants.q8[2], .name = "control: qwen27 q8 exact g64 wrong sums stride", .bad = .stride },
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 2, .served = &small }, .name = "qwen27 q8 exact g64 fallback" }, .name = "qwen27 q8 exact g64 fallback", .given = false },
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 32, .served = &small }, .name = "qwen27 q8 exact g32 sk1" }, .name = "qwen27 q8 exact g32 sk1" },
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 128, .sk = 4, .served = &small }, .name = "qwen27 q8 exact g128 sk4" }, .name = "qwen27 q8 exact g128 sk4" },
    .{ .reg = .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 2, .served = &small }, .name = "qwen27 q8 exact g64 sk2 pf2" }, .name = "qwen27 q8 exact g64 sk2 pf2", .pf = 2 },
};
fn depth(_: c.Shape) usize {
    return 0;
}
pub fn text(a: std.mem.Allocator, d: mtl.Device, l: lane.Layout, frozen: bool, bad: Bad) ![]u8 {
    const current = try lane.source(a, l);
    defer a.free(current);
    const body = if (frozen) try std.mem.concat(a, u8, &.{ current[0 .. current.len - @import("kernel_sources").core_lane_projection.len], @embedFile("lane_q8_staged.metal") }) else try a.dupe(u8, current);
    defer a.free(body);
    const changed = switch (bad) {
        .none => try a.dupe(u8, body),
        .group => try std.mem.replaceOwned(u8, a, body, "((size_t)tile * KG + (g)) * (NT * GS)", "((size_t)tile * KG + (((g) + 1) % KG)) * (NT * GS)"),
        .stride => try std.mem.replaceOwned(u8, a, body, "XS[(g) * mdims[1] + (row)]", "XS[(g) * 16 + (row)]"),
    };
    defer a.free(changed);
    if (bad != .none and std.mem.eql(u8, changed, body)) return error.ControlNotPlanted;
    return tf.frags.source(d, a, changed);
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [cases.len]Probe = undefined,
    made: usize = 0,
    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*p| p.deinit();
    }
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (cases) |v| {
            if (only) |t| if (std.mem.indexOf(u8, v.name, t) == null) continue;
            const p = &s.probes[s.made];
            p.* = .{ .inner = try base.Proj.init(s.a, s.device, s.queue, v.reg.v, v.name), .which = v };
            s.made += 1;
            try s.kernels.append(s.a, p.kernel());
        }
    }
};
const Probe = struct {
    inner: base.Proj,
    which: Case,
    candidate: ?lane.Projection = null,
    previous: rig_.Ref = undefined,
    sums: rig_.Ref = undefined,
    const stride = 128;
    const vt = blk: {
        var v = g.hooks(base.Proj);
        v.prepare = prepare;
        v.run = run;
        v.count = count;
        v.reference = reference;
        v.simulate = null;
        v.replaced = replaced;
        v.check = check;
        break :blk v;
    };
    fn parent(ctx: *anyopaque) *Probe {
        return @fieldParentPtr("inner", @as(*base.Proj, @ptrCast(@alignCast(ctx))));
    }
    fn deinit(p: *Probe) void {
        if (p.candidate) |x| x.deinit();
        p.inner.deinit();
    }
    fn kernel(p: *Probe) g.Kernel {
        var spec = base.spec(p.which.reg.v, p.which.name);
        spec.out = .raw;
        spec.accum = .none;
        spec.depth = depth;
        spec.replaces = if (p.which.bad == .none) "the pinned staged byte lane with identical packed inputs and splits" else null;
        spec.sums = .{ .why = "the original FP32 dot, group sums and split reduction are byte-identical; FP64 entries remain" };
        spec.nan = .{ .stated = "every output encoding matches the staged kernel, including nonfinite classes and payloads" };
        return .{ .ctx = &p.inner, .vt = &vt, .control = if (p.which.bad == .none) null else .shapes, .spec = spec };
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (p.candidate) |x| x.deinit();
        p.candidate = null;
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const v = p.which.reg.v;
        var l = lane.Layout{ .n = shape.n, .k = shape.k, .format = .{ .bits = 8, .group = v.group }, .sk = v.sk, .pf = p.which.pf, .output = if (v.f32_out) .f32 else .bf16 };
        const old = try text(p.inner.a, p.inner.rig.device, l, true, .none);
        defer p.inner.a.free(old);
        const original = try rig_.pipeline(p.inner.rig.device, old, "tf_lane");
        p.inner.lane_p.?.pipe.deinit();
        p.inner.lane_p.?.pipe = original;
        l.precompute_sums = v.group == 64;
        p.candidate = try lane.Projection.init(p.inner.a, p.inner.rig.device, l);
        if (p.which.bad != .none) {
            const bad = try text(p.inner.a, p.inner.rig.device, l, false, p.which.bad);
            defer p.inner.a.free(bad);
            const pipe = try rig_.pipeline(p.inner.rig.device, bad, "tf_lane");
            p.candidate.?.pipe.deinit();
            p.candidate.?.pipe = pipe;
        }
        p.previous = try p.inner.rig.scratch(p.byteCount());
        p.sums = try p.inner.rig.scratch(shape.k / v.group * stride * 4);
        return true;
    }
    fn ref(r: rig_.Ref) lane.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const pass = p.inner.rig.begin();
        const rows: u32 = @intCast(p.inner.f.?.shape.rows);
        const cl = p.candidate.?;
        if (if (p.which.given) cl.sums_pipe else null) |pipe| {
            pass.e.setPipeline(pipe);
            pass.e.setBuffer(p.inner.x.buf, p.inner.x.off, 0);
            pass.e.setValue(@as(i32, @intCast(rows)), 1);
            pass.e.setBuffer(p.sums.buf, p.sums.off, 2);
            pass.e.setValue(@as(i32, stride), 3);
            pass.e.dispatchThreads(.{ .width = cl.group_count * stride }, .{ .width = 256 });
            pass.e.barrier();
            try cl.encodeSums(pass.e, ref(p.inner.x), ref(p.inner.w), ref(p.inner.s), .{ .ref = ref(p.sums), .stride = stride }, ref(p.inner.out), rows);
        } else try cl.encode(pass.e, ref(p.inner.x), ref(p.inner.w), ref(p.inner.s), ref(p.inner.out), rows);
        try p.inner.rig.finish(pass);
    }
    fn byteCount(p: *Probe) usize {
        return p.inner.f.?.outputs() * @as(usize, if (p.which.reg.v.f32_out) 4 else 2);
    }
    fn count(ctx: *anyopaque) usize {
        return parent(ctx).byteCount();
    }
    fn reference(_: *anyopaque, out: []f64, mag: []f64) !void {
        @memset(out, 0);
        @memset(mag, 0);
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const output = p.inner.out;
        p.inner.out = p.previous;
        defer p.inner.out = output;
        try p.inner.run();
        return p.inner.rig.bytes(p.previous, p.byteCount());
    }
    fn check(ctx: *anyopaque, _: g.Gate) !?[]const u8 {
        const p = parent(ctx);
        return if (std.mem.eql(u8, p.inner.rig.bytes(p.inner.out, p.byteCount()), try replaced(ctx))) null else "byte lane or producer sums differ from the pinned staged output";
    }
};

//! Metal kernel contract probes and planted controls for q27_add_ane_bf16.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
const q = tf.qwen27;

const D: u32 = 256;

fn noDepth(_: c.Shape) usize {
    return 0;
}

fn ctlOf(ctl: u8) ?g.Gate {
    return if (ctl == 1) .nonfinite else if (ctl > 1) .shapes else null;
}

fn isNonFiniteBf(bits: u16) bool {
    return (bits & 0x7f80) == 0x7f80;
}

fn randBf(rng: std.Random, buf: []u16, sc: f32) void {
    for (buf) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * sc);
}

fn compileCtl(dev: mtl.Device, a: std.mem.Allocator, base: []const u8, ctl: u8, name: [:0]const u8) !mtl.Pipeline {
    if (ctl == 0) return rr.pipeline(dev, base, name);
    const hdr = try std.fmt.allocPrint(a, "#define Q27_ANE_CTL {d}\n", .{ctl});
    defer a.free(hdr);
    const full = try std.mem.concat(a, u8, &.{ hdr, base });
    defer a.free(full);
    return rr.pipeline(dev, full, name);
}

pub const AddAneProbe = struct {
    a: std.mem.Allocator,
    rig: rr.Rig,
    has_gpu: bool,
    ctl: u8,
    pipe: mtl.Pipeline,
    name: []const u8,
    rows: u32 = 0,
    hidden: u32 = 0,
    base: u32 = 2,
    planes_h: []u16 = &.{},
    gpu_h: []u16 = &.{},
    pl_ref: rr.Ref = undefined,
    gpu_ref: rr.Ref = undefined,
    y_ref: rr.Ref = undefined,
    const vt = g.hooks(AddAneProbe);

    pub fn init(a: std.mem.Allocator, dev: mtl.Device, q_: mtl.Queue, has_gpu: bool, ctl: u8, name: []const u8) !AddAneProbe {
        return .{ .a = a, .rig = .{ .a = a, .device = dev, .queue = q_ }, .has_gpu = has_gpu, .ctl = ctl, .pipe = try compileCtl(dev, a, q.ane_awq.add_ane_bf16_source, ctl, "q27_add_ane_bf16"), .name = name };
    }
    pub fn deinit(p: *AddAneProbe) void {
        p.a.free(p.planes_h);
        p.a.free(p.gpu_h);
        p.pipe.deinit();
        p.rig.deinit();
    }
    pub fn kernel(p: *AddAneProbe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = ctlOf(p.ctl), .spec = .{ .name = p.name, .accum = .none, .depth = noDepth, .out = .raw, .nan = .{ .stated = "nonfinite ANE/GPU inputs propagate" }, .served = &.{ .{ .rows = 6, .n = 1, .k = D }, .{ .rows = 1, .n = 1, .k = 512 }, .{ .rows = 17, .n = 1, .k = D } }, .tiles = .{ .{ .size = 8, .min = 1 }, .{ .size = 1, .min = 1 }, .{ .size = D, .step = 128, .min = 128 } } } };
    }
    pub fn prepare(p: *AddAneProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n != 1 or shape.k == 0 or shape.k % 128 != 0) return false;
        p.a.free(p.planes_h);
        p.a.free(p.gpu_h);
        p.rig.reset();
        p.rows = @intCast(shape.rows);
        p.hidden = @intCast(shape.k);
        const tot_r = p.base + p.rows + 2;
        p.planes_h = try p.a.alloc(u16, p.hidden * 128);
        p.gpu_h = try p.a.alloc(u16, tot_r * p.hidden);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.planes_h) |*v| v.* = @bitCast(@as(f16, @floatCast((rng.float(f32) * 2.0 - 1.0) * 4.0)));
        randBf(rng, p.gpu_h, 4.0);
        if (fill == .nonfinite) {
            p.planes_h[0] = 0x7e00;
            p.planes_h[1] = 0x7c00;
            p.planes_h[2] = 0xfc00;
            if (p.has_gpu) p.gpu_h[p.base * p.hidden] = 0x7fc5;
        }
        p.pl_ref = try p.rig.input(std.mem.sliceAsBytes(p.planes_h));
        p.gpu_ref = try p.rig.input(std.mem.sliceAsBytes(p.gpu_h));
        p.y_ref = try p.rig.output(tot_r * p.hidden * 2);
        @memset(p.rig.bytes(p.y_ref, tot_r * p.hidden * 2), 0x5a);
        return true;
    }
    pub fn run(p: *AddAneProbe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.gpu_ref.buf, p.gpu_ref.off, 0);
        e.setBuffer(p.pl_ref.buf, p.pl_ref.off, 1);
        e.setBuffer(p.y_ref.buf, p.y_ref.off, 2);
        e.setValue([4]u32{ p.hidden, p.rows, p.base, @intFromBool(p.has_gpu) }, 3);
        e.dispatchThreads(.of(p.hidden, p.rows, 1), .of(256, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn count(_: *AddAneProbe) usize {
        return 0;
    }
    pub fn reference(_: *AddAneProbe, _: []f64, _: []f64) !void {}
    pub fn check(p: *AddAneProbe, gate: g.Gate) !?[]const u8 {
        const y = std.mem.bytesAsSlice(u16, p.rig.bytes(p.y_ref, (p.base + p.rows + 2) * p.hidden * 2));
        if (gate == .nonfinite) {
            var nf: usize = 0;
            for (0..@min(@as(usize, 4), p.rows)) |r| if (isNonFiniteBf(y[(p.base + r) * p.hidden])) {
                nf += 1;
            };
            if (nf < 3 or isNonFiniteBf(y[(p.base + p.rows - 1) * p.hidden])) return "nonfinite propagation or isolation wrong";
            return null;
        }
        for (0..p.base) |r| for (0..p.hidden) |x| if (y[r * p.hidden + x] != 0x5a5a) return "y prefix sentinel touched";
        for (p.base + p.rows..p.base + p.rows + 2) |r| for (0..p.hidden) |x| if (y[r * p.hidden + x] != 0x5a5a) return "y suffix sentinel touched";
        for (0..p.rows) |r| for (0..p.hidden) |x| {
            const a_v: f32 = @floatCast(@as(f16, @bitCast(p.planes_h[x * 128 + r])));
            const g_v: f32 = if (p.has_gpu) @floatCast(c.fromBf16(p.gpu_h[(p.base + r) * p.hidden + x])) else 0.0;
            if (y[(p.base + r) * p.hidden + x] != c.toBf16(g_v + a_v)) return "q27_add_ane_bf16 value differs";
        };
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    add: std.ArrayList(*AddAneProbe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]struct { gpu: bool, ctl: u8, n: []const u8 }{
            .{ .gpu = false, .ctl = 0, .n = "metal q27 ane awq add ane only" },
            .{ .gpu = true, .ctl = 0, .n = "metal q27 ane awq add gpu+ane" },
            .{ .gpu = true, .ctl = 1, .n = "control: q27 ane awq add nonfinite" },
            .{ .gpu = true, .ctl = 2, .n = "control: q27 ane awq add row shift" },
        }) |it| {
            if (only != null and std.mem.indexOf(u8, it.n, only.?) == null) continue;
            const p = try s.a.create(AddAneProbe);
            errdefer s.a.destroy(p);
            p.* = try AddAneProbe.init(s.a, s.device, s.queue, it.gpu, it.ctl, it.n);
            errdefer p.deinit();
            try s.add.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.add.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.add.deinit(s.a);
    }
};

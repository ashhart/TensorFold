//! GLM-5.3-Flash DFlash2 tap stream mean+scatter kernel glm_tap_mean_ring under contract gates 1-4 and controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const base = "glm tap ";

const TapRingArgs = extern struct {
    hidden: u32,
    rows: u32,
    skip: u32,
    pos: u32,
    window: u32,
    plane_off: u32,
    ring_stride: u32,
};

const planes: usize = 5;

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = [_]c.Shape{
    .{ .rows = 4, .n = 16, .k = 64 },
    .{ .rows = 1, .n = 16, .k = 64 },
    .{ .rows = 8, .n = 16, .k = 128 },
    .{ .rows = 6, .n = 32, .k = 4096 },
};

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = "fp32 stream-order sum * 0.25 -> bf16; nonfinite propagates into written plane, unwritten bytes stay untouched" },
        .tiles = .{ .{}, .{ .size = 16 }, .{ .size = 64 } },
        .served = &served,
    };
}

pub const TapRing = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: u8,
    shape: c.Shape = undefined,
    skip: u32 = 1,
    pos: u32 = 0,
    plane: usize = 2,
    x_host: []u16 = &.{},
    x: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8) !TapRing {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL_TAP {d}\n{s}", .{ ctl, sources.glm_glue });
        defer a.free(text);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, text, "glm_tap_mean_ring"),
            .ctl = ctl,
        };
    }

    pub fn deinit(t: *TapRing) void {
        t.a.free(t.x_host);
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *TapRing, name: []const u8) g.Kernel {
        const control: ?g.Gate = switch (t.ctl) {
            1 => .repeat,
            2, 3 => .shapes,
            else => null,
        };
        return .{ .spec = spec(name), .ctx = t, .vt = &vt, .control = control };
    }

    const vt = g.hooks(TapRing);

    pub fn prepare(t: *TapRing, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.n < 2 or shape.k < 1 or shape.rows > shape.n) return false;
        t.rig.reset();
        t.shape = shape;
        t.skip = 1;
        t.pos = @intCast(shape.n - 1);
        t.plane = 2;
        const total_in_rows = t.skip + shape.rows;
        const elems = total_in_rows * 4 * shape.k;
        t.a.free(t.x_host);
        t.x_host = try t.a.alloc(u16, elems);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x_host) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * 4.0);
        if (fill == .nonfinite and shape.k >= 4) {
            const base_off = t.skip * 4 * shape.k;
            t.x_host[base_off + 0] = c.toBf16(std.math.nan(f32));
            t.x_host[base_off + 1] = c.toBf16(std.math.inf(f32));
            t.x_host[base_off + 2] = c.toBf16(-std.math.inf(f32));
            t.x_host[base_off + 3] = c.toBf16(std.math.inf(f32));
            t.x_host[base_off + shape.k + 3] = c.toBf16(-std.math.inf(f32));
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.x_host));
        t.out = try t.rig.output(shape.n * planes * shape.k * 2);
        return true;
    }

    pub fn run(t: *TapRing) !void {
        @memset(t.rig.bytes(t.out, t.count()), 0xff);
        const pass = t.rig.begin();
        const D: u32 = @intCast(t.shape.k);
        const args = TapRingArgs{
            .hidden = D,
            .rows = @intCast(t.shape.rows),
            .skip = t.skip,
            .pos = t.pos,
            .window = @intCast(t.shape.n),
            .plane_off = @intCast(t.plane * D),
            .ring_stride = @intCast(planes * D),
        };
        pass.e.setPipeline(t.pipe);
        pass.e.setBuffer(t.x.buf, t.x.off, 0);
        pass.e.setBuffer(t.out.buf, t.out.off, 1);
        pass.e.setValue(args, 2);
        pass.e.dispatchThreads(mtl.Size.of(D, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(pass);
    }

    pub fn count(t: *TapRing) usize {
        return t.shape.n * planes * t.shape.k * 2;
    }

    pub fn reference(_: *TapRing, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn simulate(_: *TapRing, _: c.Accum, _: g.Order, out: []f64) !void {
        @memset(out, 0);
    }

    pub fn check(t: *TapRing, at: g.Gate) !?[]const u8 {
        const D = t.shape.k;
        const win = t.shape.n;
        const got = std.mem.bytesAsSlice(u16, t.rig.bytes(t.out, t.count()));
        var written = try t.a.alloc(bool, win);
        defer t.a.free(written);
        @memset(written, false);
        for (0..t.shape.rows) |r| {
            const slot = (t.pos + t.skip + r) % win;
            written[slot] = true;
            const src = t.x_host[(t.skip + r) * 4 * D ..];
            const dst = got[slot * planes * D + t.plane * D ..][0..D];
            for (0..D) |col| {
                var raw: f32 = c.fromBf16(src[col]);
                for (1..4) |s| raw += c.fromBf16(src[s * D + col]);
                const want = c.toBf16(raw * 0.25);
                if (at == .nonfinite) {
                    if (c.classOf(c.fromBf16(dst[col])) != c.classOf(c.fromBf16(want))) {
                        return try std.fmt.bufPrint(&t.msg, "nonfinite class mismatch row {d} col {d}", .{ r, col });
                    }
                } else if (dst[col] != want) {
                    return try std.fmt.bufPrint(&t.msg, "row {d} slot {d} col {d}: got 0x{x}, want 0x{x}", .{ r, slot, col, dst[col], want });
                }
            }
        }
        for (0..win) |slot| for (0..planes) |pl| {
            if (written[slot] and pl == t.plane) continue;
            const seg = got[slot * planes * D + pl * D ..][0..D];
            for (seg, 0..) |v, col| if (v != 0xffff) {
                return try std.fmt.bufPrint(&t.msg, "unwritten slot {d} plane {d} col {d} overwritten: 0x{x}", .{ slot, pl, col, v });
            };
        };
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*TapRing) = .empty,

    const list = [_]struct { ctl: u8, name: []const u8 }{
        .{ .ctl = 0, .name = base ++ "glm_tap_mean_ring" },
        .{ .ctl = 1, .name = base ++ "control: writes into input" },
        .{ .ctl = 2, .name = base ++ "control: misses ring wrap" },
        .{ .ctl = 3, .name = base ++ "control: wrong scale 0.5" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(TapRing);
            p.* = try TapRing.init(a, device, queue, entry.ctl);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel(entry.name));
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};

//! GLM's DFlash2 block after a GPU acceptance under the contract: its ids, positions and mask equal the host's.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const fol = tf.glm.metal_dflash_follow;
const config = tf.qwen27.dflash.config.Config{};

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "ids, positions and mask bytes are compared as words: none is a float, so none is non-finite";

/// Random windows; adversarial ones (contexts a full window long, nothing kept or every draft kept).
const Data = enum { random, adversarial };
/// Served; a control whose sliding window reaches one key too far back.
const Case = enum { plain, reach };
const Variant = struct { data: Data, case: Case = .plain };
pub const registered = [_]Variant{ .{ .data = .random }, .{ .data = .adversarial }, .{ .data = .adversarial, .case = .reach } };

pub fn name(v: Variant) []const u8 {
    return switch (v.case) {
        .plain => if (v.data == .random) "metal glm dflash2 follow, random windows" else "metal glm dflash2 follow, adversarial windows",
        .reach => "control: metal glm dflash2 follow, a sliding window one key too long",
    };
}

fn edit(case: Case) ?[2][]const u8 {
    return switch (case) {
        .plain => null,
        .reach => .{ "const ulong reach = end + q + 1;", "const ulong reach = end + q;" },
    };
}

/// Streams (rows of the shape), block rows (n) and window rows a stream (k).
pub fn spec(v: Variant) c.Spec {
    return .{
        .name = name(v),
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .replaces = "the drafter's host-planned block ids, positions and mask",
        .tiles = .{ .{ .size = 32 }, .{}, .{} },
        .served = &.{ .{ .rows = 1, .n = 8, .k = 8 }, .{ .rows = 3, .n = 8, .k = 8 }, .{ .rows = 3, .n = 16, .k = 8 }, .{ .rows = 8, .n = 16, .k = 16 } },
    };
}

pub const Follow = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    k: fol.Follow,
    plan: fol.Plan = undefined,
    accepted: rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    sizes: [3]usize = .{ 0, 0, 0 },
    host: []u8 = &.{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Follow {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .k = try fol.Follow.init(a, device, edit(v.case)) };
    }

    pub fn deinit(t: *Follow) void {
        t.a.free(t.host);
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Follow) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = if (t.v.case == .plain) null else .shapes };
    }

    const vt = g.hooks(Follow);

    pub fn prepare(t: *Follow, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        t.rig.reset();
        const streams: usize = shape.rows;
        const block: u32 = @intCast(shape.n);
        const rows: u32 = @intCast(shape.k);
        if (streams == 0 or streams > 8 or block < 2 or block > 16 or rows == 0 or rows > 16) return false;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var windows: [8]fol.Window = undefined;
        var accepted: [16]u32 = undefined;
        for (0..streams) |s| {
            const full = t.v.data == .adversarial or rng.boolean();
            const first: u64 = if (full) config.window + rng.uintLessThan(u64, 1 << 20) else rng.uintLessThan(u64, config.window);
            const begin: u64 = if (full) rng.uintLessThan(u64, 64) else rng.uintLessThan(u64, first + 1);
            windows[s] = .{ .begin = begin, .first = first, .rows = rows };
            const kept = if (t.v.data == .adversarial) (if (s % 2 == 0) 0 else rows - 1) else rng.uintLessThan(u32, rows);
            accepted[2 * s] = kept;
            accepted[2 * s + 1] = rng.uintLessThan(u32, config.vocab);
        }
        t.plan = try fol.plan(config, windows[0..streams], block, config.window + 16);
        const n = t.plan.args.rows;
        t.sizes = .{ @as(usize, n) * 4, @as(usize, n) * 8, @as(usize, n) * t.plan.args.mask_stride };
        t.a.free(t.host);
        t.host = try t.a.alloc(u8, t.sizes[0] + t.sizes[1] + t.sizes[2]);
        const ids = try t.a.alloc(u32, n);
        defer t.a.free(ids);
        const positions = try t.a.alloc(u64, n);
        defer t.a.free(positions);
        fol.fill(t.plan, accepted[0 .. 2 * streams], ids, positions, t.host[t.sizes[0] + t.sizes[1] ..]);
        @memcpy(t.host[0..t.sizes[0]], std.mem.sliceAsBytes(ids));
        @memcpy(t.host[t.sizes[0]..][0..t.sizes[1]], std.mem.sliceAsBytes(positions));
        t.accepted = try t.rig.input(std.mem.sliceAsBytes(accepted[0 .. 2 * streams]));
        for (&t.outs, t.sizes) |*o, size| o.* = try t.rig.output(size);
        return true;
    }

    pub fn run(t: *Follow) !void {
        const pass = t.rig.begin();
        const o = t.outs;
        try t.k.encode(pass.e, t.plan, .{ .buf = t.accepted.buf, .off = t.accepted.off }, .{ .buf = o[0].buf, .off = o[0].off }, .{ .buf = o[1].buf, .off = o[1].off }, .{ .buf = o[2].buf, .off = o[2].off });
        try t.rig.finish(pass);
    }

    pub fn count(t: *Follow) usize {
        return t.sizes[0] + t.sizes[1] + t.sizes[2];
    }

    pub fn reference(_: *Follow, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(t: *Follow, _: g.Gate) !?[]const u8 {
        var at: usize = 0;
        for (t.outs, t.sizes, [_][]const u8{ "ids", "positions", "mask" }) |o, size, what| {
            const got = t.rig.bytes(o, size);
            if (std.mem.indexOfDiff(u8, got, t.host[at..][0..size]) != null) return what;
            at += size;
        }
        return null;
    }

    /// The host's plan filled at the same counts: the bytes gate 3 requires, in the outputs' order.
    pub fn replaced(t: *Follow) ![]const u8 {
        return t.host;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    items: std.ArrayList(*Follow) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (registered) |v| {
            if (only) |o| if (std.mem.indexOf(u8, name(v), o) == null) continue;
            const p = try s.a.create(Follow);
            errdefer s.a.destroy(p);
            p.* = try Follow.init(s.a, s.device, s.queue, v);
            errdefer p.deinit();
            try @import("contract").register(s.a, &s.items, &s.kernels, p, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.items.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.items.deinit(s.a);
    }
};

//! GLM's DFlash2 chain acceptance on the GPU under the contract: every stream's words equal the host's acceptance.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const acc = tf.glm.metal_dflash_accept;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "token ids are compared as words: none is non-finite, and any id (vocabulary or not) is accepted alike";

/// Random rows; adversarial rows (picks equal to the drafts but shifted by one, every draft kept, none kept).
const Data = enum { random, adversarial };
/// Served; a control comparing a draft with its own row's pick instead of the row before.
const Case = enum { plain, shifted };
const Variant = struct { data: Data, case: Case = .plain };
pub const registered = [_]Variant{ .{ .data = .random }, .{ .data = .adversarial }, .{ .data = .adversarial, .case = .shifted } };

pub fn name(v: Variant) []const u8 {
    return switch (v.case) {
        .plain => if (v.data == .random) "metal glm dflash2 accept, random rows" else "metal glm dflash2 accept, adversarial rows",
        .shifted => "control: metal glm dflash2 accept, a draft compared with its own row's pick",
    };
}

fn edit(case: Case) ?[2][]const u8 {
    return switch (case) {
        .plain => null,
        .shifted => .{ "tokens[at + kept + 1] == picks[at + kept]", "tokens[at + kept + 1] == picks[at + kept + 1]" },
    };
}

/// Streams (rows of the shape) and window rows a stream (n): 1 to 8 streams of up to 16 rows.
pub fn spec(v: Variant) c.Spec {
    return .{
        .name = name(v),
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .replaces = "the lane core's chain acceptance on the host",
        .tiles = .{ .{ .size = acc.threads }, .{}, .{} },
        .served = &.{ .{ .rows = 1, .n = 8, .k = 1 }, .{ .rows = 3, .n = 8, .k = 1 }, .{ .rows = 3, .n = 16, .k = 1 }, .{ .rows = 8, .n = 2, .k = 1 } },
    };
}

pub const Accept = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    k: acc.Accept,
    streams: u32 = 0,
    n: u32 = 0,
    tokens: []u32 = &.{},
    picks: []u32 = &.{},
    out: rig_.Ref = undefined,
    refs: [4]rig_.Ref = undefined,
    host: []u32 = &.{},
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Accept {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .k = try acc.Accept.init(a, device, edit(v.case)) };
    }

    pub fn deinit(t: *Accept) void {
        t.a.free(t.tokens);
        t.a.free(t.picks);
        t.a.free(t.host);
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Accept) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = if (t.v.case == .plain) null else .shapes };
    }

    const vt = g.hooks(Accept);

    pub fn prepare(t: *Accept, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.streams = @intCast(shape.rows);
        t.n = @intCast(shape.n);
        const total = shape.rows * shape.n;
        for ([_]*[]u32{ &t.tokens, &t.picks, &t.host }) |x| {
            t.a.free(x.*);
            x.* = &.{};
        }
        t.tokens = try t.a.alloc(u32, total);
        t.picks = try t.a.alloc(u32, total);
        t.host = try t.a.alloc(u32, shape.rows * 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var offsets = try t.a.alloc(u32, shape.rows);
        defer t.a.free(offsets);
        var rows = try t.a.alloc(u32, shape.rows);
        defer t.a.free(rows);
        for (0..shape.rows) |s| {
            const at = s * shape.n;
            offsets[s] = @intCast(at);
            rows[s] = @intCast(1 + rng.uintLessThan(usize, shape.n)); // a stream's window holds 0 to n - 1 drafts
            const keep = rng.uintLessThan(usize, rows[s]); // the drafts its picks agree with
            for (0..shape.n) |j| {
                t.tokens[at + j] = rng.uintLessThan(u32, 154880);
                t.picks[at + j] = rng.uintLessThan(u32, 154880);
            }
            for (0..keep) |j| t.picks[at + j] = t.tokens[at + j + 1];
            if (t.v.data == .adversarial) switch (s % 3) {
                0 => for (0..shape.n) |j| { // each pick equal to its own row's token: a shifted compare keeps them all
                    t.picks[at + j] = t.tokens[at + j];
                },
                1 => for (0..shape.n - 1) |j| { // every draft agrees: the last pick is the next token
                    t.picks[at + j] = t.tokens[at + j + 1];
                },
                else => {},
            };
            const h = acc.accept(t.tokens[at..][0..rows[s]], t.picks[at..][0..rows[s]]);
            t.host[2 * s] = h[0];
            t.host[2 * s + 1] = h[1];
        }
        t.refs = .{ try t.rig.input(std.mem.sliceAsBytes(t.tokens)), try t.rig.input(std.mem.sliceAsBytes(t.picks)), try t.rig.input(std.mem.sliceAsBytes(offsets)), try t.rig.input(std.mem.sliceAsBytes(rows)) };
        t.out = try t.rig.output(shape.rows * 8);
        return true;
    }

    pub fn run(t: *Accept) !void {
        const pass = t.rig.begin();
        const r = t.refs;
        try t.k.encode(pass.e, .{ .buf = r[0].buf, .off = r[0].off }, .{ .buf = r[1].buf, .off = r[1].off }, .{ .buf = r[2].buf, .off = r[2].off }, .{ .buf = r[3].buf, .off = r[3].off }, .{ .buf = t.out.buf, .off = t.out.off }, t.streams);
        try t.rig.finish(pass);
    }

    pub fn count(t: *Accept) usize {
        return @as(usize, t.streams) * 8;
    }

    pub fn reference(_: *Accept, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(t: *Accept, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u32, t.rig.bytes(t.out, t.count()));
        for (0..t.streams) |s| if (got[2 * s] != t.host[2 * s] or got[2 * s + 1] != t.host[2 * s + 1])
            return try std.fmt.bufPrint(&t.msg, "stream {d} of {d}: kept {d}, next {d}; the host kept {d}, next {d}", .{ s, t.streams, got[2 * s], got[2 * s + 1], t.host[2 * s], t.host[2 * s + 1] });
        return null;
    }

    /// The host's acceptance, as words: the bytes gate 3 requires.
    pub fn replaced(t: *Accept) ![]const u8 {
        return std.mem.sliceAsBytes(t.host);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    items: std.ArrayList(*Accept) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (registered) |v| {
            if (only) |o| if (std.mem.indexOf(u8, name(v), o) == null) continue;
            const p = try s.a.create(Accept);
            errdefer s.a.destroy(p);
            p.* = try Accept.init(s.a, s.device, s.queue, v);
            errdefer p.deinit();
            try s.items.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
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

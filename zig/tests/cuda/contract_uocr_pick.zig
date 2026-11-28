//! tf_uocr_argmax under the contract: the first maximum (NaN skipped, EOS barred on request), as float64 picks it.
const std = @import("std");
const rig_ = @import("contract_rig.zig");
const hub = @import("contract_uocr.zig");
const v = @import("contract_uocr_vision.zig");

const c = v.c;
const g = v.g;
const ks = hub.kernels;

pub const Kind = enum { argmax };

pub fn function(_: Kind) []const u8 {
    return "argmax";
}

/// The planted control (contract_uocr_glue.cuh).
pub const controls = [_]struct { Kind, v.Control }{
    .{ .argmax, .{ .function = "tf_ctl_uocr_argmax_high", .name = "control: cuda uocr tf_uocr_argmax, ties go to the higher id", .gate = .shapes } },
};

const rule = "the pick is the lowest id of the largest non-NaN logit (EOS, id 1, skipped when barred), 0 when none is above -inf";

/// Rows: one pick; n: the logits; k: 1 when EOS is barred.
pub fn spec(_: Kind) c.Spec {
    return .{ .name = "cuda uocr tf_uocr_argmax (first maximum)", .accum = .none, .depth = v.noSums, .out = .raw, .nan = .{ .stated = rule }, .served = &.{ .{ .rows = 1, .n = hub.vocab, .k = 0 }, .{ .rows = 1, .n = hub.vocab, .k = 1 }, .{ .rows = 1, .n = 300, .k = 1 } } };
}

pub const Pick = struct {
    rig: rig_.Rig,
    k: ks.Kernels,
    kind: Kind,
    control: ?v.Control,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    want: u32 = 0,
    in: u64 = 0,
    out: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn init(rig: rig_.Rig, k: ks.Kernels, kind: Kind, ctl: ?v.Control) Pick {
        return .{ .rig = rig, .k = k, .kind = kind, .control = ctl };
    }
    pub fn deinit(t: *Pick) void {
        t.rig.gpu.gpa.free(t.logits);
        t.rig.deinit();
    }
    pub fn kernel(t: *Pick) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.control);
    }
    const vt = g.hooks(Pick);

    fn barred(t: *const Pick) bool {
        return t.shape.k != 0;
    }

    pub fn prepare(t: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n < 4) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.logits);
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n;
        t.logits = try a.alloc(u16, n);
        // whole numbers -20 to 20: many ties at the top; EOS one above them all
        for (t.logits) |*x| x.* = c.toBf16(@floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 41))) - 20));
        t.logits[1] = c.toBf16(21);
        if (fill == .nonfinite) {
            const P = v.planted;
            t.logits[0] = c.toBf16(P[0]);
            t.logits[n / 2] = c.toBf16(P[1]);
            t.logits[n - 1] = c.toBf16(P[1]);
            t.logits[2] = c.toBf16(P[2]);
        }
        t.want = oracle(t.logits, t.barred());
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output(4);
        return true;
    }

    fn oracle(logits: []const u16, bar: bool) u32 {
        var best = -std.math.inf(f64);
        var at: ?u32 = null;
        for (logits, 0..) |h, i| {
            if (bar and i == 1) continue;
            const x: f64 = c.fromBf16(h);
            if (x > best) {
                best = x;
                at = @intCast(i);
            }
        }
        return at orelse 0;
    }

    pub fn run(t: *Pick) !void {
        try t.k.run("argmax", t.rig.s, .{ 1, 1, 1 }, 1024, .{ t.in, t.shape.n, @as(c_int, @intFromBool(t.barred())), t.out });
        try t.rig.s.synchronize();
    }

    pub fn count(_: *Pick) usize {
        return 4;
    }

    pub fn reference(_: *Pick, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Gates 1 to 3's stated rule: the pick equals the oracle's.
    pub fn check(t: *Pick, _: g.Gate) !?[]const u8 {
        if (!try t.rig.intact()) return "tf_uocr_argmax wrote outside its output";
        const got = std.mem.readInt(u32, (try t.rig.outputBytes())[0..4], .little);
        if (got != t.want) return try std.fmt.bufPrint(&t.msg, "picked id {d} where the oracle picks {d}", .{ got, t.want });
        return null;
    }
};

/// The pick and its control, appended to `list`.
pub fn register(ctx: hub.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !hub.Held {
    return v.register(@This(), Pick, ctx, a, list);
}

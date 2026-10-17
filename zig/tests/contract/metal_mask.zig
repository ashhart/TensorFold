//! Flash Next's grammar mask under the contract on Metal: fz_mask_rows, disallowed bf16 logits to -inf in place.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const grammar = tf.flashnext_grammar;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "an allowed token keeps its bits (NaN, +Inf and -Inf included), a disallowed one becomes -inf (0xff80)";

const neg_inf: u16 = 0xff80;

/// Rows a round masks (up to MAXR) by the vocabulary (n); its edges: 256-thread groups and n ending inside a mask word.
pub fn spec() c.Spec {
    return .{
        .name = "metal flash next fz_mask_rows",
        .accum = .none,
        .depth = noSums,
        .out = .bf16,
        .in_place = true,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &.{ .{ .rows = 16, .n = 248320, .k = 1 }, .{ .rows = 1, .n = 248320, .k = 1 }, .{ .rows = 5, .n = 248320, .k = 1 } },
    };
}

pub const Mask = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    n: usize = 0,
    words: usize = 0,
    host: []u16 = &.{}, // the logits as prepared, before the mask
    bits: []u32 = &.{},
    logits: rig_.Ref = undefined,
    mask: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Mask {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, grammar.source, "fz_mask_rows") };
    }

    pub fn deinit(t: *Mask) void {
        t.a.free(t.host);
        t.a.free(t.bits);
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Mask) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Mask);

    fn allowed(t: *const Mask, r: usize, i: usize) bool {
        return (t.bits[r * t.words + i / 32] >> @intCast(i % 32)) & 1 != 0;
    }

    /// Random bf16 logits and a mask per row: about half allowed, whole words on and off, and words cut by n's end.
    pub fn prepare(t: *Mask, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.rows, t.n = .{ shape.rows, shape.n };
        t.words = (t.n + 31) / 32;
        t.a.free(t.host);
        t.a.free(t.bits);
        t.host = &.{};
        t.bits = &.{};
        t.host = try t.a.alloc(u16, t.rows * t.n);
        t.bits = try t.a.alloc(u32, t.rows * t.words);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.host) |*w| w.* = c.toBf16(rng.floatNorm(f32) * 4);
        for (t.bits, 0..) |*w, j| w.* = switch (j % 5) {
            0 => 0,
            1 => 0xffff_ffff,
            else => rng.int(u32),
        };
        if (fill == .nonfinite) { // each class on an allowed token and on a disallowed one, in the first rows
            const put = [_]u16{ 0x7fc1, 0x7f80, neg_inf };
            for (put, 0..) |v, r| if (r < t.rows) {
                var on = false;
                var off = false;
                for (0..t.n) |i| {
                    if (on and off) break;
                    const keep = t.allowed(r, i);
                    if ((keep and on) or (!keep and off)) continue;
                    t.host[r * t.n + i] = v;
                    if (keep) on = true else off = true;
                }
            };
        }
        t.mask = try t.rig.input(std.mem.sliceAsBytes(t.bits));
        t.logits = try t.rig.output(t.rows * t.n * 2);
        @memcpy(t.rig.bytes(t.logits, t.rows * t.n * 2), std.mem.sliceAsBytes(t.host));
        return true;
    }

    /// As Masks.encode launches it: one thread a token, groups of 256, rows on the grid's second axis.
    pub fn run(t: *Mask) !void {
        const pass = t.rig.begin();
        const cfg = [2]u32{ @intCast(t.n), @intCast(t.words) };
        pass.e.setPipeline(t.pipe);
        pass.e.setBuffer(t.logits.buf, t.logits.off, 0);
        pass.e.setBuffer(t.mask.buf, t.mask.off, 1);
        pass.e.setBytes(std.mem.asBytes(&cfg), 2);
        pass.e.dispatchThreads(mtl.Size.of(t.n, t.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(pass);
    }

    pub fn count(t: *Mask) usize {
        return t.rows * t.n;
    }

    pub fn reference(t: *Mask, want: []f64, mag: []f64) !void {
        for (want, t.host, 0..) |*w, h, i| w.* = if (t.allowed(i / t.n, i % t.n)) c.fromBf16(h) else -std.math.inf(f64);
        @memset(mag, 0);
    }

    /// Every token's bits against the rule; at the non-finite gate, also that each planted class met both outcomes.
    pub fn check(t: *Mask, at: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u16, t.rig.bytes(t.logits, t.rows * t.n * 2));
        var kept_nonfinite: usize = 0;
        var masked_nonfinite: usize = 0;
        for (t.host, 0..) |h, i| {
            const keep = t.allowed(i / t.n, i % t.n);
            const want = if (keep) h else neg_inf;
            if (got[i] != want) return try std.fmt.bufPrint(&t.msg, "row {d} token {d} ({s}): 0x{x:0>4} where the rule gives 0x{x:0>4}", .{ i / t.n, i % t.n, if (keep) "allowed" else "disallowed", got[i], want });
            if (h & 0x7f80 == 0x7f80) {
                if (keep) kept_nonfinite += 1 else masked_nonfinite += 1;
            }
        }
        if (at == .nonfinite and (kept_nonfinite < @min(t.rows, 3) or masked_nonfinite < @min(t.rows, 3)))
            return try std.fmt.bufPrint(&t.msg, "{d} allowed and {d} disallowed non-finite inputs: no proof", .{ kept_nonfinite, masked_nonfinite });
        return null;
    }
};

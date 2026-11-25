//! rows.metal's tf_hy4_argmax (DFly2's greedy draft column on the GPU) by its stated rule, over Hy4's vocabulary.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");

const c = contract.spec;
const g = contract.gates;

/// The id a row with no logit above -inf gets: the kernel's starting id, which no logit replaced.
const none: u32 = 0xffffffff;

/// The rule, as a scan in id order: a logit replaces the best only when strictly larger, so NaN never does.
fn rule(row: []const u16) u32 {
    var best = -std.math.inf(f32);
    var at: u32 = none;
    for (row, 0..) |w, i| {
        const v = c.fromBf16(w);
        if (v > best) {
            best = v;
            at = @intCast(i);
        }
    }
    return at;
}

pub const Pick = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    fault: ?text.Fault,
    shapes: [1]c.Shape,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    at: u32 = 0,
    logits: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [192]u8 = undefined,

    pub const base = "metal hy4 rows tf_hy4_argmax";
    const stated = "the lowest id of the row's largest logit (-0 equals +0); NaN never wins; a row with no logit above -inf gives 0xffffffff";

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, fault: ?text.Fault, vocab: usize) !Pick {
        const src = try text.rows(a, fault);
        defer a.free(src);
        const pipe = try text.pipeline(device, src, "tf_hy4_argmax");
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .fault = fault, .shapes = .{.{ .rows = 1, .n = vocab, .k = 1 }} };
    }
    pub fn deinit(t: *Pick) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *Pick) g.Kernel {
        t.name = h.fl.named(base, if (t.fault) |f| f.drops else null, &t.label);
        const s: c.Spec = .{
            .name = t.name,
            .accum = .none,
            .depth = h.fl.noSums,
            .out = .raw,
            .nan = .{ .stated = stated },
            // n: the vocabulary, a thread every 1,024 logits; serve_draft.zig picks one row a column
            .tiles = .{ .{}, .{ .size = 1024, .min = 1 }, .{} },
            .served = &t.shapes,
        };
        const at: ?g.Gate = if (t.fault) |f| (if (std.mem.eql(u8, f.token, text.nan_wins.token)) .nonfinite else .shapes) else null;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = at };
    }
    const vt = g.hooks(Pick);

    pub fn prepare(t: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.logits = &.{};
        t.shape = shape;
        // the draft column the pick writes (serve_draft.zig's `col`)
        t.at = @intCast(seed % 4);
        const n = shape.n;
        t.logits = try t.a.alloc(u16, shape.rows * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // 64 levels, so the largest is tied many times; every third row at most 0, where -0 and +0 tie for the largest
        for (t.logits, 0..) |*v, i| {
            const level: f32 = @floatFromInt(rng.uintLessThan(u32, 64));
            const x = level / 4 - 8;
            const row = i / n;
            const zero: f32 = if (rng.boolean()) 0.0 else -0.0;
            v.* = c.toBf16(if (row % 3 == 2) (if (rng.boolean()) -@abs(x) else zero) else x);
        }
        if (fill == .nonfinite and shape.rows >= 5) {
            const nan = c.toBf16(h.fl.planted[0]);
            const inf = c.toBf16(h.fl.planted[1]);
            const ninf = c.toBf16(h.fl.planted[2]);
            const r = t.logits;
            // a NaN first (serve_metal.zig's host scan would keep it) and inside; two infinities, the first wins
            r[0] = nan;
            r[n / 3] = nan;
            r[n + n / 2] = inf;
            r[2 * n - 1] = inf;
            r[2 * n] = ninf;
            r[3 * n - 1] = nan;
            r[3 * n] = ninf;
            r[4 * n - 1] = inf;
            // a row with nothing above -inf
            for (r[4 * n .. 5 * n], 0..) |*v, i| v.* = if (i % 2 == 0) ninf else nan;
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output((t.at + shape.rows) * 4);
        return true;
    }

    /// metal_linears.zig's Rows.pick: a group of 1,024 threads a row, (vocab, at) as its uint2.
    pub fn run(t: *Pick) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue([2]u32{ @intCast(t.shape.n), t.at }, 2);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Pick) usize {
        return t.shape.rows;
    }
    pub fn reference(_: *Pick, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    /// Each row's id against the rule, and the ids before the draft column left as poisoned.
    pub fn check(t: *Pick, _: g.Gate) !?[]const u8 {
        const n = t.shape.n;
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.out, (t.at + t.shape.rows) * 4));
        for (ids[0..t.at], 0..) |v, i| if (v != none) return try std.fmt.bufPrint(&t.msg, "wrote id {d} at slot {d}, before its column {d}", .{ v, i, t.at });
        for (0..t.shape.rows) |r| {
            const want = rule(t.logits[r * n ..][0..n]);
            const got = ids[t.at + r];
            if (got != want) return try std.fmt.bufPrint(&t.msg, "row {d} drew id {d} where the rule picks {d} of {d}", .{ r, got, want, n });
        }
        return null;
    }
};

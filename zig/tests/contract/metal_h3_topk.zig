//! FastH3's tile top-k (h3_topk) under the contract: its bytes against a CPU oracle with the same comparisons.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const h3k = @import("h3_kernels");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN score ranks as -inf, ties go to the lower tile, so every row keeps exactly n tiles";

/// Rows: query video tiles; n: tiles kept a row; k: video tiles. Compiled for 288 tiles, FastH3's at 864x480.
pub fn spec() c.Spec {
    return .{
        .name = "metal h3 tile top-k h3_topk",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{}, .{ .size = 32 } },
        .served = &.{ .{ .rows = 280, .n = 56, .k = 280 }, .{ .rows = 8, .n = 3, .k = 12 } },
    };
}

const heads = 2;
const prefix = 8;

fn rank(v: f32) f32 {
    return if (std.math.isNan(v)) -std.math.inf(f32) else v;
}

pub const TopK = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    s: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !TopK {
        const src = try std.mem.concat(a, u8, &.{ "#define H3_HEADS 2\n#define H3_SLOTS 18432\n#define H3_TILES 288\n", h3k.source });
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "h3_topk") };
    }

    pub fn deinit(t: *TopK) void {
        t.a.free(t.scores);
        t.pipe.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *TopK) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(TopK);

    fn tiles(t: *const TopK) usize {
        return prefix + t.shape.k;
    }

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n > shape.k or shape.rows > shape.k or prefix + shape.k > 288) return false;
        t.rig.reset();
        t.shape = shape;
        const nt = t.tiles();
        t.a.free(t.scores);
        t.scores = &.{};
        t.scores = try t.a.alloc(f32, heads * nt * nt);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // eighths: many ties
        for (t.scores) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 8 - 4;
        if (fill == .nonfinite) {
            const at = [_]usize{ (prefix + 1) * nt + prefix + 2, (prefix + 2) * nt + nt - 1, (prefix + 3) * nt + prefix };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| t.scores[p] = v;
        }
        t.s = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.out = try t.rig.output(t.count());
        return true;
    }

    pub fn run(t: *TopK) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.s.buf, t.s.off, 0);
        p.e.setValue([3]i32{ @intCast(t.tiles()), prefix, @intCast(t.shape.n) }, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, heads, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *TopK) usize {
        return heads * t.shape.k * (prefix + t.shape.n) * 4;
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each launched row's chosen tiles against the oracle: the prefix, then each tile fewer than `n` score above.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const nt = t.tiles();
        const nv = t.shape.k;
        const keep = t.shape.n;
        const got = std.mem.bytesAsSlice(i32, t.rig.bytes(t.out, t.count()));
        var planted: usize = 0;
        for (0..heads) |h| for (0..t.shape.rows) |q| {
            const row = t.scores[(h * nt + prefix + q) * nt + prefix ..][0..nv];
            const out = got[(h * nv + q) * (prefix + keep) ..][0 .. prefix + keep];
            for (0..prefix) |i| if (out[i] != i) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: prefix slot {d} holds {d}", .{ h, q, i, out[i] });
            var placed: usize = prefix;
            for (row, 0..) |raw, j| {
                if (!std.math.isFinite(raw)) planted += 1;
                const mine = rank(raw);
                var above: usize = 0;
                for (row, 0..) |other, i| above += @intFromBool(rank(other) > mine or (rank(other) == mine and i < j));
                if (above >= keep) continue;
                if (placed >= out.len or out[placed] != prefix + j) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: tile {d} kept, slot {d} differs", .{ h, q, j, placed });
                placed += 1;
            }
            if (placed != out.len) return try std.fmt.bufPrint(&t.msg, "head {d} row {d}: {d} tiles kept, not {d}", .{ h, q, placed - prefix, keep });
        };
        if (at == .nonfinite and planted == 0) return try std.fmt.bufPrint(&t.msg, "no planted score in a launched row: no proof", .{});
        return null;
    }
};

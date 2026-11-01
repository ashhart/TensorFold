//! Host tests of the exact decoder suffix rules: the cone's row counts and the oracle that proves them exact.
const std = @import("std");
const testing = std.testing;
const ced = @import("ced.zig");
const Config = @import("config.zig").Config;
const Ids = @import("config.zig").Ids;

/// A toy model where every decoder layer's row p reads rows [p - (w - 1), p] of the layer below.
const Toy = struct {
    cfg: Config,
    d: u32,
    layers: u32,
    w: u32,
    kept: u32,
    n: u64,

    /// The toy config: the decoder starts at layer d, every decoder layer reads a w-row window below it.
    fn init(d: u32, layers: u32, w: u32, kept: u32, n: u64) Toy {
        var ratios: [64]u32 = @splat(1);
        for (0..d) |i| ratios[i] = 0;
        var cfg: Config = .{};
        cfg.layers = layers;
        cfg.window = w;
        cfg.compress_ratios = Ids.of(ratios[0..layers]);
        cfg.kv_sources = Ids.of(&.{d});
        cfg.index_sources = Ids.of(&.{d});
        cfg.engram_layers = Ids.of(&.{});
        return .{ .cfg = cfg, .d = d, .layers = layers, .w = w, .kept = kept, .n = n };
    }

    /// Which rows of each layer a cone run of `suffix` rows makes exact; a row is exact iff its window below is.
    fn oracle(self: Toy, a: std.mem.Allocator, suffix: u32) ![]bool {
        const exact = try a.alloc(bool, @intCast(self.layers * self.n));
        @memset(exact, false);
        // the encoder's layers below the boundary run whole
        for (0..self.d - 1) |L| {
            for (0..self.n) |p| exact[L * self.n + p] = true;
        }
        // the boundary layer keeps the cone's rows
        const rows_e = @min(@as(u64, suffix), self.n);
        for (0..rows_e) |p| exact[(self.d - 1) * self.n + (self.n - rows_e + p)] = true;
        for (self.d..self.layers) |L| {
            const rows_l = @min(@as(u64, try ced.layerRowsOf(&self.cfg, self.kept, @intCast(L), @intCast(self.n))), self.n);
            const lo = self.n - rows_l;
            for (0..rows_l) |p| {
                const pos = lo + p;
                const wlo = if (pos >= self.w - 1) pos - (self.w - 1) else 0;
                var ok = true;
                for (wlo..pos + 1) |q| {
                    if (!exact[(L - 1) * self.n + q]) {
                        ok = false;
                        break;
                    }
                }
                exact[L * self.n + pos] = ok;
            }
        }
        return exact;
    }
};

test "the suffix cone: its row counts match the release numbers" {
    const cfg: Config = .{};
    try testing.expectEqual(@as(u32, 20), try ced.decoderStart(&cfg));
    try testing.expectEqual(@as(u32, 2541), try ced.suffixRows(&cfg, 1));
    try testing.expectEqual(@as(u32, 2668), try ced.suffixRows(&cfg, 128));
    // the decoder's layers shrink by a window each: the last layer keeps only the kept rows
    try testing.expectEqual(@as(u32, 2414), try ced.layerRows(&cfg, 1, 20));
    try testing.expectEqual(@as(u32, 1), try ced.layerRows(&cfg, 1, 39));
    try testing.expectEqual(@as(u32, 128), try ced.layerRows(&cfg, 128, 39));
    // an encoder layer keeps every row
    try testing.expectEqual(ced.all_rows, try ced.layerRows(&cfg, 1, 19));
    try testing.expectEqual(ced.all_rows, try ced.layerRows(&cfg, 128, 19));
    try testing.expectEqual(@as(u32, 4096), try ced.layerRowsOf(&cfg, 1, 19, 4096));
    try testing.expectEqual(@as(u32, 2414), try ced.layerRowsOf(&cfg, 1, 20, 4096));
    // a prompt shorter than the cone: the cone's rows are the prompt's
    try testing.expectEqual(@as(u32, 1), try ced.layerRowsOf(&cfg, 1, 39, 4096));
    try testing.expectEqual(@as(u32, 128), try ced.layerRowsOf(&cfg, 128, 39, 4096));
    // the rule at the cone's rows is exact
    const rule = try ced.rule(&cfg, 1);
    try testing.expectEqual(@as(u32, 2541), rule.rows);
    try testing.expect(rule.exact);
}

test "the cone oracle: the cone's rows are exact, one row fewer is not" {
    const a = testing.allocator;
    const d: u32 = 2;
    const n: u64 = 8192;
    for ([_]u32{ 4, 128 }) |w| {
        for ([_]u32{ 8, 40 }) |layers| {
            for ([_]u32{ 1, 128 }) |kept| {
                const toy = Toy.init(d, layers, w, kept, n);
                try testing.expectEqual(d, try ced.decoderStart(&toy.cfg));
                const r = try ced.suffixRows(&toy.cfg, kept);
                // the oracle under the cone's rows
                const exact = try toy.oracle(a, r);
                defer a.free(exact);
                // the kept rows of the last layer are exact
                const last_rows = @min(@as(u64, kept), n);
                for (0..last_rows) |p| try testing.expect(exact[(layers - 1) * n + (n - last_rows + p)]);
                // every decoder layer's last window rows are exact
                for (d..layers) |L| {
                    const rows_l = @min(@as(u64, try ced.layerRowsOf(&toy.cfg, kept, @intCast(L), @intCast(n))), n);
                    const lo = n - rows_l;
                    const win = @min(@as(u64, w), rows_l);
                    for (0..win) |p| try testing.expect(exact[L * n + (lo + rows_l - win + p)]);
                    // and so is every row the layer computes
                    for (0..rows_l) |p| try testing.expect(exact[L * n + (lo + p)]);
                }
                // one row fewer: the decoder's first layer's first row loses its input
                const short = try toy.oracle(a, r - 1);
                defer a.free(short);
                var any_wrong = false;
                for (0..last_rows) |p| {
                    if (!short[(layers - 1) * n + (n - last_rows + p)]) any_wrong = true;
                }
                try testing.expect(any_wrong);
                // layerRows never drops a row the oracle needs: each layer's cone holds the next one's window
                for (d..layers - 1) |L| {
                    const here = try ced.layerRowsOf(&toy.cfg, kept, @intCast(L), @intCast(n));
                    const next = try ced.layerRowsOf(&toy.cfg, kept, @intCast(L + 1), @intCast(n));
                    try testing.expect(here >= next);
                    try testing.expect(next + (w - 1) <= here + (w - 1));
                    // the window the next layer reads lies inside this layer's computed rows
                    const lo = n - next;
                    const wlo = if (lo >= w - 1) lo - (w - 1) else 0;
                    try testing.expect(wlo >= n - here);
                }
            }
        }
    }
}

test "the snapshot replay: the cone's rows before the save, or none" {
    const cfg: Config = .{};
    const r = try ced.suffixRows(&cfg, 1);
    // a prompt of 4,096 tokens: S = 4,080, the cone's rows [S - R, S)
    const rep = (try ced.snapshotReplay(4096, 0, 1)).?;
    try testing.expectEqual(@as(u64, 4080), rep.s);
    try testing.expectEqual(@as(u64, 4080 - r), rep.r0);
    try testing.expectEqual(r, rep.s - rep.r0);
    // a resumed prompt: the segment since the resume is shorter than the cone, so no replay
    try testing.expectEqual(@as(?ced.Replay, null), try ced.snapshotReplay(4096, 4000, 1));
    // a segment at least the cone long: the replay reaches back before the resume point
    const long = (try ced.snapshotReplay(8192, 0, 1)).?;
    try testing.expectEqual(@as(u64, 8176), long.s);
    try testing.expectEqual(@as(u64, 8176 - r), long.r0);
    // a prompt too short to snapshot: no save, no replay
    try testing.expectEqual(@as(?ced.Replay, null), try ced.snapshotReplay(60, 0, 1));
}

//! The real DeepSeek lane adapter: first/verify rows survive shared rounds, draft rejection and tree remapping.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("draft/iface.zig");
const twin = @import("draft/twin.zig");
const branches = @import("draft/branches.zig");
const lp = @import("logprob_pipe.zig");
const oracle = @import("logprob_test.zig");
const a = std.testing.allocator;

const ScoredTwin = struct {
    t: *twin.Twin,
    resolve: bool,
    resolvers: [4]branches.Resolver = @splat(.{}),
    scored: usize = 0,

    fn self(p: *anyopaque) *ScoredTwin {
        return @ptrCast(@alignCast(p));
    }
    fn target(s: *ScoredTwin) iface.Target {
        return .{ .ptr = s, .vtable = &.{ .prefill = prefill, .scored_prompt = scoredPrompt, .window = window, .keep = keep, .taps = taps, .release = release } };
    }
    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        return self(p).t.target().prefill(slot, ids, smp, draw);
    }
    fn scoredPrompt(p: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64, k: u8, row: *lanes.logprob.Row, finish: bool) anyerror!u32 {
        if (finish) return error.NoPieces;
        const pick = try prefill(p, slot, ids, smp, draw);
        row.* = try self(p).score(slot, ids.len - 1, ids[ids.len - 1], pick, k);
        return pick;
    }
    fn score(s: *ScoredTwin, slot: u32, r: usize, prev: u32, token: u32, k: u8) !lanes.logprob.Row {
        const d = s.t.dims;
        const taps_ = s.t.target().taps(slot);
        const h = @as([*]const f32, @ptrFromInt(taps_.address))[r * d.dim ..][0..d.dim];
        var values: [48]f32 = undefined;
        s.t.logits(h, prev, values[0..d.vocab]);
        s.scored += 1;
        return lp.score(values[0..d.vocab], token, k);
    }
    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const s = self(p);
        for (segments, choices) |g, out| {
            if (s.resolve and g.parents != null) {
                try s.resolvers[g.slot].window(.{ .ptr = s, .vtable = &.{ .run = chain, .drop = drop, .scored = chainScored } }, g, out);
                continue;
            }
            s.resolvers[g.slot].clear();
            var one = [_][]u32{out};
            try s.t.target().window(&.{g}, &one);
            if (g.logprobs) |k| for (g.tokens, out, g.logrows, 0..) |prev, token, *row, r| {
                row.* = try s.score(g.slot, r, prev, token, k);
            };
        }
    }
    fn chain(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, smp: ?lanes.Sampling, out: []u32) anyerror!void {
        return chainScored(p, .{ .slot = slot, .start = start, .tokens = tokens, .draws = draws, .sampling = smp, .parents = null }, out);
    }
    fn chainScored(p: *anyopaque, seg: iface.Segment, out: []u32) anyerror!void {
        var one = [_][]u32{out};
        // The resolver owns its map while this chain runs, so do not clear it through window().
        const s = self(p);
        try s.t.target().window(&.{seg}, &one);
        if (seg.logprobs) |k| for (seg.tokens, out, seg.logrows, 0..) |prev, token, *row, r| {
            row.* = try s.score(seg.slot, r, prev, token, k);
        };
    }
    fn drop(p: *anyopaque, slot: u32) anyerror!void {
        self(p).t.drop(slot);
    }
    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const s = self(p);
        if (!s.resolvers[slot].live) return s.t.target().keep(slot, path);
        const n = try s.resolvers[slot].keep(path);
        var rows: [64]u32 = undefined;
        for (rows[0 .. n + 1], 0..) |*r, i| r.* = @intCast(i);
        return s.t.target().keep(slot, rows[0 .. n + 1]);
    }
    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const s = self(p);
        var out = s.t.target().taps(slot);
        out.map = s.resolvers[slot].map();
        return out;
    }
    fn release(p: *anyopaque, slot: u32) void {
        self(p).t.target().release(slot);
        self(p).resolvers[slot].clear();
    }
};

test "DeepSeek logprobs: mixed requested k, first rows and drafted/shared/tree commits match plain float64 rows" {
    for ([_]bool{ false, true }) |resolve| for ([_]bool{ false, true }) |drafts| {
        const t = try twin.Twin.init(a, .{}, 4);
        defer t.deinit();
        const d = try twin.Drafter.init(a, t, 4);
        defer d.deinit();
        var scored: ScoredTwin = .{ .t = t, .resolve = resolve };
        var costs = try @import("draft/costs.zig").defaults(a, 64);
        defer costs.deinit(a);
        const x = try @import("draft/lanes.zig").Lanes.init(a, scored.target(), d.pass(), costs, .{ .shape = d.shape, .slots = 4, .siblings = if (resolve) 2 else 0, .depth = .{ .skip = false } });
        defer x.deinit();
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        var cfg = try lanes.Config.init(a, try x.model(arena.allocator(), 64), 16, 15);
        defer cfg.deinit(a);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, x.backend(), clock.clock());
        defer engine.deinit();
        x.attach(&engine);
        var streams: [4]lanes.Stream = undefined;
        const counts = [_]?u8{ null, 0, 3, 20 };
        for (&streams, counts) |*s, k| s.* = try lanes.Stream.init(a, .{ .id = "logprobs", .prompt = &.{ 3, 2, 1, 4 }, .max_new = 40, .drafts = drafts, .logprobs = k });
        defer for (&streams) |*s| s.deinit(a);
        for (&streams) |*s| try engine.addStream(s);
        while (engine.activeCount() > 0) try engine.step();
        for (&streams) |*s| {
            try std.testing.expectEqualSlices(u32, streams[0].emitted(), s.emitted());
            if (s.logprobs == null) {
                try std.testing.expectEqual(@as(usize, 0), s.rows.items.len);
                continue;
            }
            try std.testing.expectEqual(s.emitted().len, s.rows.items.len);
            const ref = try twin.Twin.init(a, .{}, 1);
            defer ref.deinit();
            var token = try ref.target().prefill(0, s.prompt(), null, s.prompt().len);
            for (s.rows.items, s.emitted(), 0..) |row, emitted, j| {
                try std.testing.expectEqual(token, emitted);
                const taps_ = ref.target().taps(0);
                const at = if (j == 0) s.prompt().len - 1 else 0;
                const h = @as([*]const f32, @ptrFromInt(taps_.address))[at * ref.dims.dim ..][0..ref.dims.dim];
                var values: [48]f32 = undefined;
                ref.logits(h, if (j == 0) s.prompt()[s.prompt().len - 1] else s.emitted()[j - 1], &values);
                try oracle.equal(try oracle.reference(&values, emitted, s.logprobs.?), row);
                var picks: [1]u32 = undefined;
                var out = [_][]u32{&picks};
                const start = s.prompt().len + j;
                try ref.target().window(&.{.{ .slot = 0, .start = start, .tokens = &.{emitted}, .draws = &.{start + 1}, .sampling = null, .parents = null }}, &out);
                token = picks[0];
            }
        }
        try std.testing.expect(scored.scored > 0);
        if (drafts) try std.testing.expect(engine.drafted > 0);
    };
}

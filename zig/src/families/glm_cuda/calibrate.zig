//! GLM-5.3-Flash's costs for the lane core, timed at load on rank 0 (rank 1 replays every call): a lone stream's
//! verify window by width, shared steps by their total rows (up to 16 streams' windows), and a chained head step.

const std = @import("std");
const lanes = @import("lanes");
const forward = @import("forward.zig");
const dflash = @import("dflash.zig");
const decode = @import("decode.zig");
const proto = @import("proto.zig");

const max_rows: usize = @intCast(forward.MAX_ROWS);
const streams = forward.MAX_STREAMS;

fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

/// ``self``: native.zig's Owned.
pub fn run(self: anytype, drafts: bool) !void {
    var w: proto.Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    var seqs: [streams]*forward.Seq = undefined;
    var toks: [streams]u32 = undefined;
    var prompt: [64]u32 = undefined;
    for (0..streams) |i| {
        for (&prompt, 0..) |*t, j| t.* = @intCast(1000 + (j * 37 + i * 101) % 5000);
        const uid = self.next_uid;
        self.next_uid += 1;
        seqs[i] = try self.newSeq(uid, 4096);
        try w.int(uid);
        try w.int(@as(i64, 4096));
        try w.tokens(&prompt);
        try proto.writeSampling(&w, null);
        try w.int(0);
        try w.int(0);
        try self.send(.prefill, &w);
        seqs[i].sampling = null;
        toks[i] = try decode.prefill(self.e, seqs[i], &prompt, 0, 0);
    }
    var picks: [forward.STEP_ROWS]decode.Pick = undefined;
    var ids: [streams][max_rows]u32 = undefined;
    // one step of ``n`` streams' windows of ``width`` rows (distinct tokens: rows routed to their own experts, as a
    // real window's are), each kept at its first row; its ms
    const Step = struct {
        fn go(o: anytype, wr: *proto.Writer, ss: []*forward.Seq, tk: []u32, idv: *[streams][max_rows]u32, pk: []decode.Pick, n: usize, width: usize, rep: usize) !f64 {
            var wins: [streams]decode.Win = undefined;
            try wr.int(n);
            for (0..n) |i| {
                for (idv[i][0..width], 0..) |*t, r| t.* = if (r == 0) tk[i] else @intCast(1000 + ((r * 7919 + rep * 104729 + width * 31 + i * 613) % 60000));
                wins[i] = .{ .s = ss[i], .ids = idv[i][0..width] };
                try wr.int(ss[i].uid);
                try wr.tokens(idv[i][0..width]);
            }
            try o.send(.verify, wr);
            const t0 = std.Io.Timestamp.now(o.io, .awake);
            try decode.verify(o.e, wins[0..n], pk);
            const ms = msSince(o.io, t0);
            try wr.int(n);
            for (0..n) |i| {
                try wr.int(ss[i].uid);
                try wr.int(width);
                try wr.int(1);
            }
            try o.send(.keep, wr);
            for (0..n) |i| {
                try decode.keep(o.e, ss[i], @intCast(width), 1);
                tk[i] = pk[i * width].tok;
            }
            return ms;
        }
    };
    // warm-up (the GPU's clocks rise over the first steps; every shape's graph captured once)
    for (0..16) |rep| _ = try Step.go(self, &w, &seqs, &toks, &ids, &picks, 1, 1, rep);
    for (1..max_rows + 1) |width| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            const ms = try Step.go(self, &w, &seqs, &toks, &ids, &picks, 1, width, rep);
            if (rep > 0) best = @min(best, ms);
        }
        self.costs[width - 1] = .{ .width = @intCast(width), .ms = best };
    }
    self.cost_count = max_rows;
    // shared steps: n streams of w rows (total rows n * w)
    const shapes = [_][2]usize{ .{ 2, 1 }, .{ 4, 1 }, .{ 8, 1 }, .{ 16, 1 }, .{ 16, 2 }, .{ 16, 4 }, .{ 16, 8 } };
    for (shapes, 0..) |sh, k| {
        var best: f64 = std.math.inf(f64);
        for (0..4) |rep| {
            const ms = try Step.go(self, &w, &seqs, &toks, &ids, &picks, sh[0], sh[1], rep);
            if (rep > 0) best = @min(best, ms);
        }
        self.shared[k] = .{ .width = @intCast(sh[0] * sh[1]), .ms = best };
    }
    self.shared_count = shapes.len;
    if (drafts) {
        var held: [max_rows]u32 = undefined;
        var probs: [max_rows]f64 = undefined;
        var at: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
        for ([_]usize{ 1, 7 }, 0..) |depth, k| {
            for (0..3) |rep| {
                const follow = [_]u32{toks[0]};
                try w.int(1);
                try w.int(seqs[0].uid);
                try w.tokens(&follow);
                try w.int(depth);
                try w.int(0); // the hidden row: the last verify's
                try w.int(0); // no confidence cut: every level timed
                try w.int(@intFromBool(self.e.drafter != null)); // DFlash2 drafts when loaded
                try self.send(.draft, &w);
                var reqs = [_]decode.DraftReq{.{ .s = seqs[0], .hidden = seqs[0].hidden, .next = &follow, .count = depth, .confidence = 0, .out = held[0..depth], .probs = probs[0..depth] }};
                const t0 = std.Io.Timestamp.now(self.io, .awake);
                // rank 1 replays the frame with the drafter it serves with: DFlash2's round when one is loaded
                if (self.e.drafter) |dr| { // as rank 1 replays it: the MTP head kept current (while the mix may pick it), then DFlash2's block
                    reqs[0].count = 1;
                    if (@import("native.zig").feedsMtp(self)) try decode.draft(self.e, &reqs);
                    try dflash.draft(dr, self.e, &.{.{ .s = seqs[0], .n = 1, .anchor = toks[0], .out = held[0 .. max_rows - 1], .probs = probs[0 .. max_rows - 1] }});
                } else try decode.draft(self.e, &reqs);
                if (rep > 0) at[k] = @min(at[k], msSince(self.io, t0));
            }
        }
        // a DFlash2 round drafts the whole block whatever the depth: its cost spread over the 7 drafts
        self.mtp_ms = if (self.e.drafter != null) at[1] / 7 else @max(0, (at[1] - at[0]) / 6);
    }
    for (seqs) |s| {
        try w.int(s.uid);
        try w.int(0);
        try self.send(.release, &w);
        self.dropSeq(s.uid);
    }
    std.log.info("glm costs: window 1/2/4/8 rows {d:.1}/{d:.1}/{d:.1}/{d:.1} ms; 16 streams x 1/2/4/8 rows {d:.1}/{d:.1}/{d:.1}/{d:.1} ms; head step {d:.2} ms", .{ self.costs[0].ms, self.costs[1].ms, self.costs[3].ms, self.costs[7].ms, self.shared[3].ms, self.shared[4].ms, self.shared[5].ms, self.shared[6].ms, self.mtp_ms });
}

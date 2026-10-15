//! Rank 1's loop: each frame rank 0 sends replayed on its half of the model until rank 0 stops.
const cuda = @import("cuda");
const forward = @import("forward.zig");
const decode = @import("decode.zig");
const proto = @import("proto.zig");
const dflash = @import("dflash.zig");
const Op = proto.Op;
const Reader = proto.Reader;
const native = @import("native.zig");
const max_depth = native.max_depth;
const max_streams = native.max_streams;
const of = native.of;
const feedsMtp = native.feedsMtp;

/// Rank 1: rank 0's calls, replayed in order on this rank's half of the model, until rank 0 stops.
pub fn followLoop(p: *anyopaque) anyerror!void {
    const self = of(p);
    var picks: [forward.STEP_ROWS]decode.Pick = undefined;
    var held: [max_streams][max_depth]u32 = undefined;
    var probs: [max_streams][max_depth]f64 = undefined;
    while (true) {
        const m = self.link.recv(self.gpa) catch |e| switch (e) {
            error.PeerClosed => return,
            else => return e,
        };
        defer self.gpa.free(m.bytes);
        var r: Reader = .{ .bytes = m.bytes };
        switch (@as(Op, @fromBackingInt(@intCast(m.tag)))) {
            .prefill => {
                const uid = try r.int();
                const fresh: i64 = @intCast(try r.int());
                const toks = try r.tokens();
                const sampling = try proto.readSampling(&r);
                const start: usize = @intCast(try r.int());
                const keep_at: usize = @intCast(try r.int());
                const seq = if (fresh > 0) try self.newSeq(uid, fresh) else self.seqs.get(uid) orelse return error.NoSuchSequence;
                seq.sampling = sampling;
                defer { // the image frame's features served this prefill alone
                    self.media_spans.clearRetainingCapacity();
                    if (self.media_feats) |*f| f.free();
                    self.media_feats = null;
                }
                if (self.media_feats) |f| forward.attach(seq, .{ .spans = self.media_spans.items, .feats = f.ptr });
                _ = try decode.prefill(self.e, seq, toks, start, keep_at);
            },
            .image => {
                _ = try r.int();
                const n: usize = @intCast(try r.int());
                self.media_spans.clearRetainingCapacity();
                for (0..n) |_| try self.media_spans.append(self.gpa, .{ @intCast(try r.int()), @intCast(try r.int()) });
                const rows: usize = @intCast(try r.int());
                const bytes = try r.raw(rows * 4096 * 2);
                if (self.media_feats) |*f| f.free();
                self.media_feats = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 4096 * 2);
                try self.media_feats.?.upload(0, bytes);
            },
            .verify => {
                const n: usize = @intCast(try r.int());
                var wins: [max_streams]decode.Win = undefined;
                for (0..n) |i| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    wins[i] = .{ .s = s, .ids = try r.tokens() };
                }
                try decode.verify(self.e, wins[0..n], &picks);
            },
            .keep => {
                const n: usize = @intCast(try r.int());
                for (0..n) |_| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    const rows: i64 = @intCast(try r.int());
                    try decode.keep(self.e, s, rows, @intCast(try r.int()));
                }
            },
            .draft => {
                const n: usize = @intCast(try r.int());
                var reqs: [max_streams]decode.DraftReq = undefined;
                var dq: [max_streams]dflash.Req = undefined;
                for (0..n) |i| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    const follow = try r.tokens();
                    const depth: usize = @intCast(try r.int());
                    const first = try r.int() == 1;
                    const cut: f64 = @bitCast(try r.int());
                    const df = try r.int() == 1;
                    reqs[i] = .{ .s = s, .hidden = if (first) s.last_hidden else s.hidden, .next = follow, .count = if (df) 1 else depth, .confidence = cut, .out = held[i][0..depth], .probs = probs[i][0..depth] };
                    dq[i] = .{ .s = s, .n = if (first) 0 else follow.len, .anchor = follow[follow.len - 1], .drafts = df };
                }
                if (feedsMtp(self)) try decode.draft(self.e, reqs[0..n]); // as rank 0: the MTP head first, then DFlash2
                if (self.e.drafter) |dr| try dflash.draft(dr, self.e, dq[0..n]);
            },
            .release => {
                const uid = try r.int();
                if (try r.int() == 0) self.dropSeq(uid);
            },
            .evict => self.dropSeq(try r.int()),
            .stop => { // rank 0 closes the link next: no lost peer
                if (self.guard) |g| g.stop(self.gpa);
                self.guard = null;
                return;
            },
            .ready, .room => {},
        }
    }
}

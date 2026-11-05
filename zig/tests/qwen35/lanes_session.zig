//! Prompts through the lane core on one engine's HIP backend, as often as asked, each run's replies kept in memory.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

const Hip = qwen35.hip_lanes.Hip;
const Draft = qwen35.draft.Draft;

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *qwen35.engine.Engine,
    h: *Hip,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    /// The DFlash2 drafter `--drafter` loaded and the target wrapped behind it (null: the target drafts alone).
    draft: ?*Draft = null,
    drafted: ?lanes.drafted.Drafted = null,
    /// The engine's taps before the drafter set its own, set again when the session ends.
    taps: qwen35.view.Taps = .{},

    /// Rank 0 of a group of `link`'s ranks (null: a lone rank) runs the core; the others follow it.
    pub fn init(s: *Session, gpa: std.mem.Allocator, io: std.Io, e: *qwen35.engine.Engine, link: ?hip.collective.Link) !void {
        return s.initDrafted(gpa, io, e, link, null, 1);
    }

    /// As `init`, with the DFlash2 drafter in `drafter` (null: none) in front, frames of up to `streams`.
    pub fn initDrafted(
        s: *Session,
        gpa: std.mem.Allocator,
        io: std.Io,
        e: *qwen35.engine.Engine,
        link: ?hip.collective.Link,
        drafter: ?[]const u8,
        streams: usize,
    ) !void {
        // as the native server opens it: taps set before the lanes make the streams' caches and rings
        const taps = e.model().taps;
        errdefer if (drafter != null) e.setTaps(taps.layers[0..taps.n]) catch {};
        const d: ?*Draft = if (drafter) |dir| try openDraft(gpa, io, e, dir, streams) else null;
        errdefer if (d) |x| x.deinit();
        const h = try Hip.init(gpa, e);
        errdefer h.deinit();
        if (link) |l| h.withLink(l);
        if (link == null) h.measure();
        const rows = Hip.max_window;
        var drafted: ?lanes.drafted.Drafted = null;
        if (d) |x| {
            for (h.costs.window[0..h.costs.windows]) |c| x.setCost(c.width, c.ms);
            drafted = try lanes.drafted.Drafted.init(gpa, h.backend(), x.drafter());
        }
        errdefer if (drafted) |*x| x.deinit();
        const facts = if (drafted) |*x| x.facts(h.facts()) else h.facts();
        s.* = .{ .gpa = gpa, .io = io, .e = e, .h = h, .cfg = try lanes.Config.init(gpa, facts, rows, rows - 1), .clock = .{ .io = io }, .draft = d, .drafted = drafted };
        s.taps = taps;
    }

    pub fn deinit(s: *Session) void {
        s.cfg.deinit(s.gpa);
        if (s.drafted) |*x| x.deinit();
        s.h.deinit();
        if (s.draft) |d| {
            d.deinit();
            s.e.setTaps(s.taps.layers[0..s.taps.n]) catch {};
        }
    }

    /// What the lane core drives: the drafter's wrapper when there is one, else the target.
    fn backend(s: *Session) lanes.backend.Backend {
        return if (s.drafted) |*x| x.backend() else s.h.backend();
    }

    pub const Job = struct {
        prompts: []const ids_file.Prompt,
        max_new: u32,
        sampling: ?lanes.Sampling = null,
        drafts: bool = true,
        /// One stream at a time instead of all at once.
        solo: bool = false,
    };

    pub const Reply = struct { name: []const u8, tokens: []const u32, rounds: u64, accepted: u64, cached: u32 };

    pub const Done = struct {
        replies: []const Reply,
        seconds: f64,
        /// Rounds run, rounds replayed from graphs and the host's nanoseconds submitting them, during the run.
        rounds: u64,
        replayed: u64,
        submit_ns: u64,
        /// Draft head batches run on the graphed chain and replayed from graphs, during the run.
        head_batches: u64 = 0,
        head_replayed: u64 = 0,
        /// Nanoseconds and calls of the backend's verifies, keeps and drafts.
        spent: [3]u64,
        calls: [3]u64,

        pub fn tokens(d: Done) usize {
            var n: usize = 0;
            for (d.replies) |r| n += r.tokens.len;
            return n;
        }

        pub fn accepted(d: Done) u64 {
            var n: u64 = 0;
            for (d.replies) |r| n += r.accepted;
            return n;
        }
    };

    /// The job's replies, in the arena, with no end token (replies run out at `max_new`).
    pub fn run(s: *Session, arena: std.mem.Allocator, job: Job) !Done {
        const streams = try s.gpa.alloc(lanes.Stream, job.prompts.len);
        defer s.gpa.free(streams);
        var made: usize = 0;
        defer for (streams[0..made]) |*st| st.deinit(s.gpa);
        for (streams, job.prompts) |*st, p| {
            st.* = try lanes.Stream.init(s.gpa, .{ .id = p.name, .prompt = p.ids, .max_new = job.max_new, .sampling = job.sampling, .drafts = job.drafts, .history_len = @intCast(p.ids.len / 2), .shared_prefixes = p.shared });
            made += 1;
        }
        const replayed = s.e.graphs.replayed;
        const rounds = s.e.graphs.rounds;
        const submit = s.e.graphs.submit_ns;
        const head_batches = s.headBatches();
        const head_replayed = s.headReplayed();
        const spent = s.h.streams.spent;
        const calls = s.h.streams.calls;
        const t0 = std.Io.Clock.awake.now(s.io);
        if (job.solo) {
            for (streams) |*st| try s.finish(&.{st});
        } else {
            const all = try s.gpa.alloc(*lanes.Stream, streams.len);
            defer s.gpa.free(all);
            for (all, streams) |*a, *st| a.* = st;
            try s.finish(all);
        }
        const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(s.io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
        const out = try arena.alloc(Reply, streams.len);
        for (out, streams, job.prompts) |*r, *st, p| r.* = .{ .name = p.name, .tokens = try arena.dupe(u32, st.emitted()), .rounds = st.rounds, .accepted = st.accepted, .cached = st.cached };
        return .{
            .replies = out,
            .seconds = seconds,
            .rounds = s.e.graphs.rounds - rounds,
            .replayed = s.e.graphs.replayed - replayed,
            .submit_ns = s.e.graphs.submit_ns - submit,
            .head_batches = s.headBatches() - head_batches,
            .head_replayed = s.headReplayed() - head_replayed,
            .spent = delta(s.h.streams.spent, spent),
            .calls = delta(s.h.streams.calls, calls),
        };
    }

    fn finish(s: *Session, streams: []const *lanes.Stream) !void {
        var engine = lanes.Engine.init(s.gpa, &s.cfg, s.backend(), s.clock.clock());
        defer engine.deinit();
        for (streams) |st| try engine.addStream(st);
        while (engine.live.items.len > 0) try engine.step();
    }

    /// The draft head's batches on the graphed chain so far (0 without a head).
    fn headBatches(s: *const Session) u64 {
        return if (s.h.head) |hd| hd.graphs.rounds else 0;
    }

    fn headReplayed(s: *const Session) u64 {
        return if (s.h.head) |hd| hd.graphs.replayed else 0;
    }
};

/// The DFlash2 drafter in `dir` on `e`, its taps set first and a context ready for each of `streams`.
fn openDraft(gpa: std.mem.Allocator, io: std.Io, e: *qwen35.engine.Engine, dir: []const u8, streams: usize) !*Draft {
    const c = try qwen35.draft_config.Config.read(gpa, io, dir, qwen35.draft_config.Target.of(e.model().spec));
    try e.setTaps(c.tapLayers());
    const d = try Draft.init(gpa, io, e, dir, @min(@max(streams, 1), qwen35.draft.max_streams));
    errdefer d.deinit();
    try d.reserve(streams);
    return d;
}

fn delta(now: [3]u64, before: [3]u64) [3]u64 {
    var out: [3]u64 = undefined;
    for (&out, now, before) |*o, a, b| o.* = a - b;
    return out;
}

/// Each prompt extended by its reply and its own first tokens: what a next turn sends.
pub fn extend(arena: std.mem.Allocator, prompts: []const ids_file.Prompt, replies: []const Session.Reply) ![]const ids_file.Prompt {
    const out = try arena.alloc(ids_file.Prompt, prompts.len);
    for (out, prompts, replies) |*o, p, r| o.* = .{ .name = p.name, .ids = try std.mem.concat(arena, u32, &.{ p.ids, r.tokens, p.ids[0..@min(p.ids.len, 4)] }), .shared = p.shared };
    return out;
}

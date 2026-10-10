//! Prompts through the lane core on one engine's HIP backend, as often as asked, each run's replies kept in memory.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

const Hip = qwen35.hip_lanes.Hip;

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *qwen35.engine.Engine,
    h: *Hip,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,

    /// Rank 0 of a group of `link`'s ranks (null: a lone rank) runs the core; the others follow it.
    pub fn init(s: *Session, gpa: std.mem.Allocator, io: std.Io, e: *qwen35.engine.Engine, link: ?*const hip.link.Link) !void {
        const h = try Hip.init(gpa, e);
        errdefer h.deinit();
        if (link) |l| h.withLink(l);
        if (link == null) h.measure();
        const rows = Hip.max_window;
        s.* = .{ .gpa = gpa, .io = io, .e = e, .h = h, .cfg = try lanes.Config.init(gpa, h.facts(), rows, rows - 1), .clock = .{ .io = io } };
    }

    pub fn deinit(s: *Session) void {
        s.cfg.deinit(s.gpa);
        s.h.deinit();
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
        const spent = s.h.spent;
        const calls = s.h.calls;
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
        return .{ .replies = out, .seconds = seconds, .rounds = s.e.graphs.rounds - rounds, .replayed = s.e.graphs.replayed - replayed, .submit_ns = s.e.graphs.submit_ns - submit, .spent = delta(s.h.spent, spent), .calls = delta(s.h.calls, calls) };
    }

    fn finish(s: *Session, streams: []const *lanes.Stream) !void {
        var engine = lanes.Engine.init(s.gpa, &s.cfg, s.h.backend(), s.clock.clock());
        defer engine.deinit();
        for (streams) |st| try engine.addStream(st);
        while (engine.live.items.len > 0) try engine.step();
    }
};

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

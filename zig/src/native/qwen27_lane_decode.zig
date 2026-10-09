//! The serial host's drafted decode on the shared lane core: its prompt pass prefills, the core's rounds decode.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");

pub const Decode = struct {
    gpa: std.mem.Allocator,
    backend: *lb.Metal,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    stream: ?lanes.Stream = null,
    lookup: ?lanes.SuffixLookup = null, // copies of the prompt and the reply (TF_QWEN27_COPIES=1)
    started: bool = false,
    sent: usize = 0, // emitted tokens already handed to the host
    counted: [3]u64 = .{ 0, 0, 0 }, // the stream's rounds, drafted and accepted at the last batch

    pub fn init(gpa: std.mem.Allocator, io: std.Io, runner: *q.decode_round.Runner, draft: *lb.Draft) !*Decode {
        const d = try gpa.create(Decode);
        errdefer gpa.destroy(d);
        const backend = try lb.Metal.init(gpa, runner, draft);
        errdefer backend.deinit();
        try backend.measure(io);
        d.* = .{ .gpa = gpa, .backend = backend, .cfg = try lanes.Config.init(gpa, backend.facts(), lb.max_rows, lb.max_rows - 1), .clock = .{ .io = io }, .core = undefined };
        d.core = lanes.Engine.init(gpa, &d.cfg, backend.backend(), d.clock.clock());
        return d;
    }

    pub fn deinit(d: *Decode) void {
        d.end();
        d.core.deinit();
        d.cfg.deinit(d.gpa);
        d.backend.deinit();
        d.gpa.destroy(d);
    }

    /// Forget the current stream, finished or stopped by the host (a stopped one leaves the core first).
    pub fn end(d: *Decode) void {
        if (d.stream) |*s| {
            if (d.started and !s.finished) d.core.discard(s);
            s.deinit(d.gpa);
        }
        d.stream = null;
        if (d.lookup) |*l| l.deinit();
        d.lookup = null;
        d.started = false;
    }

    /// A drafted request's stream, its prompt already in the runner: copies when TF_QWEN27_COPIES=1.
    pub fn begin(d: *Decode, prompt: []const u32, max_tokens: u32, eos: []const u32) !void {
        d.end();
        const copies = if (std.c.getenv("TF_QWEN27_COPIES")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (copies) d.lookup = try lanes.SuffixLookup.init(d.gpa, .{ .min_match = 4 });
        d.stream = try lanes.Stream.init(d.gpa, .{ .id = "q27", .prompt = prompt, .max_new = max_tokens, .eos = eos, .proposer = if (d.lookup) |*l| l.proposer() else null });
        d.sent = 0;
        d.counted = .{ 0, 0, 0 };
    }

    /// The prompt's first token alone, then each core round's committed tokens.
    pub fn batch(d: *Decode, runner: *q.decode_round.Runner) !api.serial_host.Batch {
        const s = if (d.stream) |*x| x else return error.NoDecodeStream;
        if (!d.started) {
            try d.backend.attach(s);
            d.started = true;
            try d.core.adoptPrefilled(s, try (q.session.Session{ .runner = runner }).greedy());
        } else {
            if (s.finished) return error.StreamFinished;
            try d.core.step();
        }
        const out = s.emitted()[d.sent..];
        d.sent = s.emitted().len;
        const stats = api.Stats{ .rounds = s.rounds - d.counted[0], .drafted = s.drafted - d.counted[1], .accepted = s.accepted - d.counted[2], .min_rows = s.min_rows };
        d.counted = .{ s.rounds, s.drafted, s.accepted };
        return .{ .tokens = out, .stats = stats };
    }
};

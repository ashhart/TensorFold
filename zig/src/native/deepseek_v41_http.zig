//! DeepSeek V4.1 without ConnectX: `--prefill-from URL --mcdma-link http` triggers a remote prefill and streams it in.
const std = @import("std");
const tf = @import("tensorfold");
const api = @import("engine_api");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const fabric = tf.fabric;
const ps = fabric.prefill_source;
const trigger = @import("remote_trigger.zig");
const remote = @import("deepseek_v41_remote.zig");
const q27 = @import("qwen27_remote.zig");
const Filled = api.LaneHost.Remote.Filled;
const Flags = api.LaneHost.Remote.Flags;
const Route = api.LaneHost.Remote.Route;
const Ask = api.LaneHost.Remote.Ask;
const Verdict = api.LaneHost.Remote.Verdict;
const Measured = api.LaneHost.Remote.Measured;
const Outcome = api.LaneHost.Remote.Outcome;

/// The lane host's remote source over HTTP: one pull at a time, each into a blank Snapshot of this model's shape.
pub const Pull = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *ds.Model,
    policy: ps.Policy,
    /// the checkpoint both ends load: the manifest's model must equal it
    checkpoint: []const u8,
    trigger: trigger.Http,
    from: ho.http.Source,
    reason: fabric.decoder.Reason = .{},
    route: Route = .{},
    next_id: u64 = 1,
    pulls: u64 = 0,
    bytes: u64 = 0,
    ns: u64 = 0,
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    off: std.atomic.Value(bool) = .init(false),
    /// the gate wait of the pull under way, for its phases
    waited_ns: u64 = 0,

    pub fn source(p: *Pull) api.LaneHost.Remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running, .measured = measured };
    }

    /// A prefix of `prompt` beyond `cached` on a whole chunk, asked for, fetched and landed whole in a blank Snapshot.
    pub fn pull(p: *Pull, prompt: []const u32, cached: u64, gone: *const std.atomic.Value(bool)) !?Filled {
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return null;
        const upto = p.policy.upto(prompt.len, cached) orelse return null;
        p.waited_ns = p.route.admit(p.io, p.route.estimate(upto, prompt.len - cached), &p.closing, gone) orelse return null;
        defer p.route.leave();
        if (p.closing.load(.acquire) or p.off.load(.acquire) or gone.load(.acquire)) return null;
        return api.LaneHost.Remote.guarded(p, prompt, cached, upto, gone);
    }

    /// The trial pull's liveness check: the GB10's server answers.
    pub fn alive(p: *Pull) bool {
        if (p.trigger.alive()) return true;
        std.log.warn("remote prefill: the GB10's server does not answer /health", .{});
        return false;
    }

    /// The pull itself, the gate held: trigger, fetch, landing; each bound by the pull's deadline.
    pub fn run(p: *Pull, prompt: []const u32, _: u64, upto: u64, gone: *const std.atomic.Value(bool), outcome: *Outcome) !?Filled {
        p.reason.len = 0;
        const at: u32 = std.math.cast(u32, upto) orelse return error.PrefixTooLong;
        if (upto >= prompt.len) return error.PrefixTooLong; // a token past the prefix goes too
        const t0 = fabric.words.nowNs();
        var id: [16]u8 = undefined;
        p.io.randomSecure(id[0..8]) catch return error.NoEntropy;
        std.mem.writeInt(u64, id[8..16], p.next_id, .little);
        p.next_id += 1;
        const limit_s = p.route.deadlineS(upto);
        p.trigger.limit_s = limit_s;
        p.trigger.gone = gone;
        defer p.trigger.gone = null; // the ticket's flag goes with its fill
        try p.trigger.trigger().start_fn(p.trigger.trigger().ptr, id, prompt[0 .. upto + 1]);
        const triggered = fabric.words.nowNs();

        const snap = try p.model.snapshotBlank();
        errdefer p.model.snapshotDrop(snap);
        var landing: ho.landing.Landing = undefined;
        try landing.init(p.gpa, at, .{ .@"packed" = remote.packedOf(p.model, snap) });
        defer landing.deinit();
        var arena: std.heap.ArenaAllocator = .init(p.gpa);
        defer arena.deinit();
        p.from.budget_ns = limit_s * std.time.ns_per_s;
        const expect: fabric.decoder.Expect = .{ .model = p.checkpoint, .tokens = prompt[0..upto], .protocol = fabric.wire.recurrent_version, .abort = gone };
        const got = p.from.pull(arena.allocator(), id, expect, landing.decoderSink(), &p.reason) catch |err| {
            if (err == error.Refused and p.reason.len > 0) std.log.warn("remote prefill: the GB10 refused ({s})", .{p.reason.text()});
            if (err != error.NotHeld and err != error.Unauthorized) p.from.release(id); // what is held there is of no use now
            if (err == error.Unauthorized) std.log.warn("remote prefill: the GB10 answered HTTP 401 on the export: {s}", .{if (p.from.key == null) "it needs an API key; name a file holding it in TF_PREFILL_API_KEY_FILE" else "it refused TF_PREFILL_API_KEY_FILE's key"});
            p.settle(err);
            return err;
        };
        p.from.release(id);
        if (got.manifest.first_token != 0) return error.PartialPrefix;
        _ = landing.complete() catch return error.PartialState;
        try p.model.snapshotSeal(snap, prompt, at);
        const took = fabric.words.nowNs() - t0;
        p.pulls += 1;
        p.bytes += got.bytes;
        p.ns += took;
        p.route.observePull(at, p.trigger.cached, took);
        outcome.* = .ok;
        return .{ .saved = snap, .at = at, .phases = .{ .queue_ns = p.waited_ns, .trigger_ns = triggered - t0, .pull_ns = took -| (triggered - t0), .land_ns = landing.land_ns, .bytes = got.bytes } };
    }

    /// A mismatch no retry mends (the checkpoint, the prompt's tokens, the protocol) turns remote prefill off for good.
    fn settle(p: *Pull, err: anyerror) void {
        if (err != error.Mismatch and err != error.Protocol) return;
        p.off.store(true, .release);
        std.log.warn("remote prefill off for this server: the GB10's export does not match ({s})", .{@errorName(err)});
    }

    fn wants(ptr: *anyopaque, ask: Ask) Verdict {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return .off;
        const n = p.policy.upto(ask.len, ask.cached) orelse return .small;
        if (n - ask.cached < p.policy.min_prefix) return .small;
        return p.route.weigh(Route.nowNs(p.io), ask, n);
    }

    fn measured(ptr: *anyopaque, m: Measured) void {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        p.route.observe(m);
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64, gone: *const std.atomic.Value(bool)) anyerror!?Filled {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        return p.pull(prompt, cached, gone);
    }

    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        p.model.snapshotDrop(@ptrCast(@alignCast(saved)));
    }
};

/// A served HTTP link: the trigger, the fetch and the pull, as long as the server runs.
pub const Link = struct {
    pull: Pull,
    url: []u8,
    base: []u8,

    /// `f.url` is the GB10 server (POST /v1/completions triggers, GET /v1/handoff/<id> serves); `id` is the checkpoint.
    pub fn start(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, f: Flags, id: []const u8) !*Link {
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        const rules = try remote.policy();
        const gate = try Route.fromEnv();
        const key = try q27.apiKey(gpa, io);
        errdefer if (key) |k| gpa.free(k);
        l.base = try gpa.dupe(u8, std.mem.trimEnd(u8, f.url.?, "/"));
        errdefer gpa.free(l.base);
        l.url = try std.fmt.allocPrint(gpa, "{s}/v1/completions", .{l.base});
        errdefer gpa.free(l.url);
        l.pull = .{
            .gpa = gpa,
            .io = io,
            .model = m,
            .policy = rules,
            .route = gate,
            .checkpoint = id,
            .trigger = .{ .gpa = gpa, .io = io, .url = l.url, .model = "", .key = key },
            .from = .{ .gpa = gpa, .io = io, .base = l.base, .key = key },
        };
        l.pull.trigger.closing = &l.pull.closing;
        l.pull.from.closing = &l.pull.closing;
        return l;
    }

    /// After the lane host stopped: fills still under way end first.
    pub fn stop(l: *Link, gpa: std.mem.Allocator) void {
        l.pull.closing.store(true, .release);
        while (l.pull.running.load(.acquire) > 0) std.Io.sleep(l.pull.io, .fromMilliseconds(1), .awake) catch {};
        if (l.pull.pulls > 0) {
            const n: f64 = @floatFromInt(l.pull.pulls);
            std.log.info("remote prefill over HTTP: {d} prefixes pulled, {d:.1} MiB and {d:.1} ms each on average", .{ l.pull.pulls, @as(f64, @floatFromInt(l.pull.bytes)) / n / (1 << 20), @as(f64, @floatFromInt(l.pull.ns)) / n / 1e6 });
        }
        if (l.pull.trigger.key) |k| {
            std.crypto.secureZero(u8, @constCast(k));
            gpa.free(k);
        }
        gpa.free(l.url);
        gpa.free(l.base);
        gpa.destroy(l);
    }
};

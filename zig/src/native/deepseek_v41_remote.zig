//! DeepSeek V4.1 on the Mac: a prefix prefilled elsewhere is pulled over MKVH protocol 2 into a blank Snapshot.
const std = @import("std");
const tf = @import("tensorfold");
const api = @import("engine_api");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const fabric = tf.fabric;
const ps = fabric.prefill_source;
const q27 = @import("qwen27_remote.zig");
const trigger = @import("remote_trigger.zig");
const Filled = api.LaneHost.Remote.Filled;
const http = @import("deepseek_v41_http.zig");
const Flags = api.LaneHost.Remote.Flags;
const Route = api.LaneHost.Remote.Route;
const Ask = api.LaneHost.Remote.Ask;
const Verdict = api.LaneHost.Remote.Verdict;
const Measured = api.LaneHost.Remote.Measured;
const Outcome = api.LaneHost.Remote.Outcome;

/// The Qwen pull's prefix policy with TF_PREFILL_TAIL: the Mac prefills only the last token by default, no 512 tail.
pub fn policy() !ps.Policy {
    var p = try q27.policy();
    p.tail = try @import("sys").environment.read(u64, "TF_PREFILL_TAIL", 1);
    if (p.tail > 8 and p.tail < 64) std.log.warn("TF_PREFILL_TAIL={d}: a 9 to 63 row pass runs the slow shared-body kernels; use 1 to 8 or 64 and up", .{p.tail});
    return p;
}
pub const keeps = q27.keeps;
pub const checkpointId = api.LaneHost.Remote.checkpointId;

/// A slot's Packed layout over a Snapshot's own buffers (host-visible: nothing is staged).
pub fn packedOf(m: *const ds.Model, snap: *ds.Model.Snapshot) ho.metal.Packed {
    const plan = m.caches[0].plan;
    return .{ .live = snap.live, .storage = snap.storage, .layers = plan.layers, .capacity = plan.capacity };
}

/// The lane host's remote source: one pull at a time over its link, each into a blank Snapshot of this model's shape.
pub const Pull = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *ds.Model,
    remote: ps.Remote,
    /// the connect daemon's mailbox and its client, mapped again when the daemon restarts (`remote.client` is its own)
    link: *fabric.relink.Relink,
    /// a KV_SPLIT pair's other producers' links, mapped again the same way (`remote.peers` are their clients)
    more: []fabric.relink.Relink = &.{},
    http: *trigger.Http,
    route: Route = .{},
    pulls: u64 = 0,
    ns: u64 = 0,
    running: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    off: std.atomic.Value(bool) = .init(false),
    linked: bool = true,
    /// the gate wait of the pull under way, for its phases
    waited_ns: u64 = 0,

    pub fn source(p: *Pull) api.LaneHost.Remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running, .measured = measured };
    }

    /// A prefix of `prompt` beyond `cached` on a whole chunk, filled remotely and landed whole in a blank Snapshot.
    pub fn pull(p: *Pull, prompt: []const u32, cached: u64, gone: *const std.atomic.Value(bool)) !?Filled {
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return null;
        const upto = p.remote.policy.upto(prompt.len, cached) orelse return null;
        p.waited_ns = p.route.admit(p.io, p.route.estimate(upto, prompt.len - cached), &p.closing, gone) orelse return null;
        defer p.route.leave();
        if (p.closing.load(.acquire) or p.off.load(.acquire) or gone.load(.acquire)) return null;
        return api.LaneHost.Remote.guarded(p, prompt, cached, upto, gone);
    }

    /// The trial pull's liveness check: the GB10's server answers.
    pub fn alive(p: *Pull) bool {
        if (p.http.alive()) return true;
        std.log.warn("remote prefill: the GB10's server does not answer /health", .{});
        return false;
    }

    /// The pull, the gate held: the link (mapped again if the daemon restarted), the trigger, the stream, the landing.
    pub fn run(p: *Pull, prompt: []const u32, cached: u64, upto: u64, gone: *const std.atomic.Value(bool), outcome: *Outcome) !?Filled {
        const event = p.renewLinks() catch |err| {
            if (p.linked) std.log.warn("remote prefill: the link is not usable ({s}); prompts are prefilled here", .{@errorName(err)});
            p.linked = false;
            outcome.* = .down;
            return null;
        };
        if (event == .reopened) std.log.info("remote prefill: the connect daemon's mailbox was replaced; the link was re-opened", .{});
        if (!p.linked) std.log.info("remote prefill: the link is back", .{});
        p.linked = true;
        p.remote.reason.len = 0;
        const src = p.remote.source();
        const offer = src.offer(prompt, cached) orelse return null;
        const at: u32 = std.math.cast(u32, offer.upto) orelse return error.PrefixTooLong;
        const t0 = fabric.words.nowNs();
        const snap = try p.model.snapshotBlank();
        errdefer p.model.snapshotDrop(snap);
        var landing: ho.landing.Landing = undefined;
        try landing.initShards(p.gpa, at, @intCast(1 + p.remote.peers.len), .{ .@"packed" = packedOf(p.model, snap) });
        defer landing.deinit();
        p.http.limit_s = p.route.deadlineS(upto);
        p.http.gone = gone;
        p.remote.abort = gone;
        defer p.http.gone = null; // the ticket's flag goes with its fill
        defer p.remote.abort = null;
        src.fill(prompt, offer, landing.state()) catch |err| {
            if (err == error.Refused and p.remote.reason.len > 0) std.log.warn("remote prefill: the GB10 refused ({s})", .{p.remote.reason.text()});
            p.settle(err);
            return err;
        };
        const pulled_ns = fabric.words.nowNs() - t0;
        _ = landing.complete() catch return error.PartialState;
        try p.model.snapshotSeal(snap, prompt, at);
        const took = fabric.words.nowNs() - t0;
        p.pulls += 1;
        p.ns += took;
        p.route.observePull(at, p.http.cached, took);
        outcome.* = .ok;
        if (p.remote.peers.len > 0) std.log.info("remote prefill: {d} tokens pulled over {d} links in {d:.0} ms", .{ at, 1 + p.remote.peers.len, @as(f64, @floatFromInt(pulled_ns)) / 1e6 });
        return .{ .saved = snap, .at = at, .phases = .{ .queue_ns = p.waited_ns, .trigger_ns = p.remote.trigger_ns, .pull_ns = took -| p.remote.trigger_ns, .land_ns = landing.land_ns, .bytes = p.remote.bytes } };
    }

    /// Every link of the pull usable again: a restarted daemon's mailbox mapped, a producer's new generation made.
    fn renewLinks(p: *Pull) !fabric.relink.Event {
        const now = fabric.words.nowNs();
        var event = try p.link.renew(now);
        for (p.more) |*r| if ((try r.renew(now)) == .reopened) {
            event = .reopened;
        };
        return event;
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
        const n = p.remote.policy.upto(ask.len, ask.cached) orelse return .small;
        if (n - ask.cached < p.remote.policy.min_prefix) return .small;
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

/// Decimal token ids, white space or commas (a JSON array reads too).
pub fn parseIds(a: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
    while (it.next()) |w| try out.append(a, try std.fmt.parseInt(u32, w, 10));
    return out.toOwnedSlice(a);
}

/// A file in one allocation, read in 1 GiB steps (one read of 2 GiB or more fails on macOS).
fn readWhole(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const n: usize = @intCast(try f.length(io));
    const buf = try a.alloc(u8, n);
    errdefer a.free(buf);
    var r = f.reader(io, &.{});
    var at: usize = 0;
    while (at < n) : (at += 1 << 30) try r.interface.readSliceAll(buf[at..@min(n, at + (1 << 30))]);
    return buf;
}

/// A hand-off file: the library's blob or a producer stream (`<id>.mkvh2`, `<id>.tokens.json`) for `checkpoint`.
pub fn loadFile(a: std.mem.Allocator, io: std.Io, path: []const u8, checkpoint: []const u8) !ho.blob.Blob {
    const bytes = try readWhole(a, io, path);
    if (!std.mem.endsWith(u8, path, ho.file.extension)) return ho.blob.Blob.parse(a, bytes);
    const ids = try std.fmt.allocPrint(a, "{s}.tokens.json", .{path[0 .. path.len - ho.file.extension.len]});
    const tokens = try parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, ids, a, .limited(64 << 20)));
    var reason: fabric.decoder.Reason = .{};
    return ho.blob.fromStream(a, bytes, .{ .model = checkpoint, .tokens = tokens, .protocol = fabric.wire.recurrent_version }, &reason) catch |err| {
        if (reason.len > 0) std.log.err("hand-off stream {s}: the producer said: {s}", .{ path, reason.text() });
        return err;
    };
}

/// TF_DSV41_HANDOFF_FILE: a hand-off file as the remote source with no network; a prompt extending its tokens uses it.
pub const FileSource = struct {
    gpa: std.mem.Allocator,
    model: *ds.Model,
    arena: std.heap.ArenaAllocator,
    blob: ho.blob.Blob,
    running: std.atomic.Value(u32) = .init(0),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, path: []const u8, checkpoint: []const u8) !*FileSource {
        const f = try gpa.create(FileSource);
        errdefer gpa.destroy(f);
        f.* = .{ .gpa = gpa, .model = m, .arena = .init(gpa), .blob = undefined };
        errdefer f.arena.deinit();
        const a = f.arena.allocator();
        f.blob = try loadFile(a, io, path, checkpoint);
        return f;
    }

    pub fn close(f: *FileSource) void {
        f.arena.deinit();
        f.gpa.destroy(f);
    }

    pub fn source(f: *FileSource) api.LaneHost.Remote.Source {
        return .{ .ptr = f, .wants = fileWants, .fill = fileFill, .drop = fileDrop, .running = &f.running };
    }

    fn fileWants(ptr: *anyopaque, ask: api.LaneHost.Remote.Ask) Verdict {
        const f: *FileSource = @ptrCast(@alignCast(ptr));
        return if (ask.len > f.blob.tokens.len and ask.cached < f.blob.tokens.len) .remote else .small;
    }

    fn fileFill(ptr: *anyopaque, prompt: []const u32, _: u64, _: *const std.atomic.Value(bool)) anyerror!?Filled {
        const f: *FileSource = @ptrCast(@alignCast(ptr));
        const tokens = f.blob.tokens;
        if (prompt.len <= tokens.len or !std.mem.eql(u32, prompt[0..tokens.len], tokens)) return null;
        const at: u32 = @intCast(tokens.len);
        const snap = try f.model.snapshotBlank();
        errdefer f.model.snapshotDrop(snap);
        var landing: ho.landing.Landing = undefined;
        try landing.init(f.gpa, at, .{ .@"packed" = packedOf(f.model, snap) });
        defer landing.deinit();
        try f.blob.feed(landing.decoderSink());
        _ = try landing.complete();
        try f.model.snapshotSeal(snap, prompt, at);
        return .{ .saved = snap, .at = at };
    }

    fn fileDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const f: *FileSource = @ptrCast(@alignCast(ptr));
        f.model.snapshotDrop(@ptrCast(@alignCast(saved)));
    }
};

/// What serve refuses before loading: one flag without the other, or a separate drafter model.
pub fn refusal(f: Flags, drafter: bool) ?[]const u8 {
    if (!f.on()) return null;
    if (f.url == null or f.link == null) return "--prefill-from and --mcdma-link go together: the GB10 server's URL and the mcdma-rpcd link it answers on";
    if (drafter) return "--prefill-from serves DeepSeek V4.1 with its own MTP drafter only: serve without --drafter";
    return null;
}

/// A served link: the connect daemon's mailbox, its client, the HTTP trigger and the pull, as long as the server runs.
pub const Link = struct {
    relink: fabric.relink.Relink,
    /// a KV_SPLIT pair's other producers' links (`--mcdma-link a,b`): shard k is `more[k - 1]`, client `peers[k - 1]`
    more: []fabric.relink.Relink = &.{},
    peers: []*fabric.mailbox.Client = &.{},
    http: trigger.Http,
    pull: Pull,

    /// `id` names the checkpoint both ends load (the manifest's model must equal it).
    pub fn start(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, f: Flags, id: []const u8) !*Link {
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        const rules = try policy();
        const gate = try Route.fromEnv();
        var names = std.mem.splitScalar(u8, f.link.?, ',');
        try l.relink.open(names.first(), fabric.words.native, 30 * std.time.ns_per_s);
        errdefer l.relink.close();
        l.more = &.{};
        l.peers = &.{};
        var opened: usize = 0;
        errdefer {
            for (l.more[0..opened]) |*x| x.close();
            gpa.free(l.more);
            gpa.free(l.peers);
        }
        const extra = std.mem.count(u8, f.link.?, ",");
        if (extra > 0) {
            l.more = try gpa.alloc(fabric.relink.Relink, extra);
            l.peers = try gpa.alloc(*fabric.mailbox.Client, extra);
            while (names.next()) |name| : (opened += 1) {
                try l.more[opened].open(name, fabric.words.native, 30 * std.time.ns_per_s);
                l.peers[opened] = &l.more[opened].client;
            }
        }
        const base = std.mem.trimEnd(u8, f.url.?, "/");
        const key = try q27.apiKey(gpa, io);
        errdefer if (key) |k| gpa.free(k);
        l.http = .{ .gpa = gpa, .io = io, .url = try std.fmt.allocPrint(gpa, "{s}/v1/completions", .{base}), .model = "", .closing = &l.pull.closing, .key = key };
        l.pull = .{ .gpa = gpa, .io = io, .model = m, .route = gate, .link = &l.relink, .more = l.more, .http = &l.http, .remote = .{ .gpa = gpa, .policy = rules, .client = &l.relink.client, .peers = l.peers, .trigger = l.http.trigger(), .model = id, .protocol = fabric.wire.recurrent_version } };
        return l;
    }

    /// After the lane host stopped: fills still under way end first.
    pub fn stop(l: *Link, gpa: std.mem.Allocator) void {
        l.pull.closing.store(true, .release);
        while (l.pull.running.load(.acquire) > 0) std.Io.sleep(l.pull.io, .fromMilliseconds(1), .awake) catch {};
        if (l.pull.pulls > 0) std.log.info("remote prefill: {d} prefixes pulled, {d:.1} ms each on average", .{ l.pull.pulls, @as(f64, @floatFromInt(l.pull.ns)) / @as(f64, @floatFromInt(l.pull.pulls)) / 1e6 });
        gpa.free(l.http.url);
        if (l.http.key) |k| {
            std.crypto.secureZero(u8, @constCast(k));
            gpa.free(k);
        }
        l.relink.close();
        for (l.more) |*x| x.close();
        gpa.free(l.more);
        gpa.free(l.peers);
        gpa.destroy(l);
    }
};

/// The link a served host holds: an MCDMA one (`--mcdma-link NAME`) or the HTTP transport (`--mcdma-link http`).
pub const Attached = union(enum) {
    mcdma: *Link,
    http: *http.Link,

    fn start(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, f: Flags, id: []const u8) !Attached {
        if (f.overHttp()) return .{ .http = try http.Link.start(gpa, io, m, f, id) };
        return .{ .mcdma = try Link.start(gpa, io, m, f, id) };
    }

    fn source(a: Attached) api.LaneHost.Remote.Source {
        return switch (a) {
            .mcdma => |l| l.pull.source(),
            .http => |l| l.pull.source(),
        };
    }

    pub fn stop(a: Attached, gpa: std.mem.Allocator) void {
        switch (a) {
            .mcdma => |l| l.stop(gpa),
            .http => |l| l.stop(gpa),
        }
    }
};

/// --prefill-from on a served host `h`: starts its link and gives the lane host the pull and the keep rule.
pub fn attach(h: anytype, io: std.Io, a: std.mem.Allocator, f: Flags, id: []const u8, why: *[]const u8) !void {
    if (!f.on()) return;
    h.link = Attached.start(h.gpa, io, h.model, f, id) catch |err| {
        why.* = if (err == error.InvalidEnvironment) "a TF_PREFILL_* setting is malformed: the line above names it, its value and its accepted forms" else if (f.overHttp()) try std.fmt.allocPrint(a, "--prefill-from {s} with --mcdma-link http: the HTTP link could not start ({s})", .{ f.url.?, @errorName(err) }) else try std.fmt.allocPrint(a, "--mcdma-link {s}: the link could not open ({s}); is mcdma-rpcd's connect daemon serving it?", .{ f.link.?, @errorName(err) });
        return err;
    };
    h.host.remote = h.link.?.source();
    h.host.keep_remote = keeps() catch |err| {
        why.* = "TF_PREFILL_KEEP takes 1 (keep states built on a pulled prefix, marked remote) or 0";
        h.link.?.stop(h.gpa);
        h.link = null;
        return err;
    };
    if (!h.host.keep_remote) std.log.info("remote prefill: states built on a pulled prefix are not kept (TF_PREFILL_KEEP=0)", .{});
}

test "remote prefill for DeepSeek V4.1 refuses half its flags and a separate drafter, by name" {
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{}, true));
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000" }, false) != null);
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, true) != null);
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, false));
}

test "the pull wants a prefix only 4096 tokens or more past what the Mac resumes, and stops for good on a mismatch" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null or std.c.getenv("TF_PREFILL_TAIL") != null) return error.SkipZigTest;
    var p: Pull = undefined;
    p.io = std.testing.io;
    p.route = .{};
    p.closing = .init(false);
    p.off = .init(false);
    p.remote.policy = try policy();
    const t = std.testing;
    try t.expectEqual(Verdict.remote, Pull.wants(&p, .{ .len = 5600, .cached = 1000, .decoding = 1 }));
    try t.expectEqual(Verdict.small, Pull.wants(&p, .{ .len = 5600, .cached = 2000, .decoding = 1 }));
    p.settle(error.Refused);
    try t.expectEqual(Verdict.remote, Pull.wants(&p, .{ .len = 8192, .cached = 0, .decoding = 1 }));
    p.settle(error.Mismatch);
    try t.expectEqual(Verdict.off, Pull.wants(&p, .{ .len = 8192, .cached = 0, .decoding = 1 }));
}

test "the pull leaves the last prompt token to the Mac by default, and the producer is asked for all the others" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null or std.c.getenv("TF_PREFILL_TAIL") != null) return error.SkipZigTest;
    const rules = try policy();
    try std.testing.expectEqual(@as(u64, 1), rules.tail);
    try std.testing.expectEqual(@as(?u64, 8191), rules.upto(8192, 0)); // the trigger sends 8192 tokens, the cut is 8191
    try std.testing.expectEqual(@as(?u64, 8641 - 1), rules.upto(8641, 0)); // no 512 boundary under it
    try std.testing.expectEqual(@as(?u64, null), rules.upto(8192, 8191));
}

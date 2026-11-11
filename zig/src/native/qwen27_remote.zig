//! #1594 step 2b on the Mac: a prompt prefix the GB10 prefilled, pulled over an MKVH link into a blank Qwen27 state.
const std = @import("std");
const tf = @import("tensorfold");
const api = @import("engine_api");
const q = tf.qwen27;
const fabric = tf.fabric;
const ps = fabric.prefill_source;
const hs = fabric.hybrid_state;
const hx = fabric.hybrid_export;
const Filled = api.LaneHost.Remote.Filled;
const Flags = api.LaneHost.Remote.Flags;
const trigger = @import("remote_trigger.zig");

/// Tokens to an attention layer's page row in the hand-off; the GB10's export uses the same.
pub const block: u32 = 16;
/// The checkpoint's name both ends give a hand-off (the import tool's real-link mode names it as the server does).
pub const checkpointId = api.LaneHost.Remote.checkpointId;

/// The lane host's remote source: one pull at a time over its link, each into a blank state of this runner's shape.
pub const Pull = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: *const q.decode_round.Runner,
    remote: ps.Remote, // protocol 2, its policy, the link's client and the trigger that asks the GB10
    mutex: std.Io.Mutex = .init,
    pulls: u64 = 0,
    ns: u64 = 0, // wall time of every pull, the trigger's prefill included
    running: std.atomic.Value(u32) = .init(0), // the lane host's fill threads under way
    closing: std.atomic.Value(bool) = .init(false), // the server stops: waiting fills give up
    off: std.atomic.Value(bool) = .init(false), // a mismatch no retry mends (checkpoint, protocol): remote prefill stops
    linked: bool = true, // the link answered last time (logged when it goes and when it comes back)

    pub fn source(p: *Pull) api.LaneHost.Remote.Source {
        return .{ .ptr = p, .wants = wants, .fill = fill, .drop = drop, .running = &p.running };
    }

    /// A prefix of `prompt` beyond `cached` on a page-row boundary, filled remotely and landed whole.
    pub fn pull(p: *Pull, prompt: []const u32, cached: u64) !?Filled {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return null;
        p.remote.client.renew() catch |err| { // the link is down or still owes a reply: nothing asks the GB10
            if (p.linked) std.log.warn("remote prefill: the link is not usable ({s}); prompts are prefilled here", .{@errorName(err)});
            p.linked = false;
            return null;
        };
        if (!p.linked) std.log.info("remote prefill: the link is back", .{});
        p.linked = true;
        p.remote.reason.len = 0;
        const src = p.remote.source();
        const offer = src.offer(prompt, cached) orelse return null;
        const at: u32 = std.math.cast(u32, offer.upto) orelse return error.PrefixTooLong;
        const plan: hx.Plan = .{ .g = hs.qwen27, .s = hx.qwen27, .at = at, .block = block };
        try plan.check();
        if (q.snapshot.slotBytes(p.runner, null, at) != hs.qwen27.bytes(at)) return error.HybridGeometryDiffers;
        const t0 = fabric.words.nowNs();
        const st = try q.snapshot.blank(p.gpa, p.runner, null, at);
        errdefer q.snapshot.drop(p.gpa, st);
        var landing: hx.Landing = .{ .plan = plan, .image = st.buf.contents()[0..@intCast(hs.qwen27.bytes(at))] };
        src.fill(prompt, offer, .{ .ptr = &landing, .put_fn = hx.Landing.putAny, .commit_fn = committed, .abort_fn = aborted }) catch |err| {
            if (err == error.Refused and p.remote.reason.len > 0) std.log.warn("remote prefill: the GB10 refused ({s})", .{p.remote.reason.text()});
            p.settle(err);
            return err;
        };
        if (!landing.whole()) return error.PartialState;
        p.pulls += 1;
        p.ns += fabric.words.nowNs() - t0;
        return .{ .saved = st, .at = at };
    }

    /// A mismatch no retry mends (the checkpoint, the prompt's tokens, the protocol) turns remote prefill off for good.
    fn settle(p: *Pull, err: anyerror) void {
        if (err != error.Mismatch and err != error.Protocol) return; // a refusal, a timeout, a down link: ask again later
        p.off.store(true, .release);
        std.log.warn("remote prefill off for this server: the GB10's export does not match ({s})", .{@errorName(err)});
    }

    fn wants(ptr: *anyopaque, ask: api.LaneHost.Remote.Ask) api.LaneHost.Remote.Verdict {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        if (p.closing.load(.acquire) or p.off.load(.acquire)) return .off;
        const n = p.remote.policy.upto(ask.len, ask.cached) orelse return .small;
        return if (n - ask.cached >= p.remote.policy.min_prefix) .remote else .small; // a gain under TF_PREFILL_MIN stays local
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, cached: u64, _: *const std.atomic.Value(bool)) anyerror!?Filled {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        return p.pull(prompt, cached);
    }

    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        const p: *Pull = @ptrCast(@alignCast(ptr));
        q.snapshot.drop(p.gpa, @ptrCast(@alignCast(saved)));
    }

    fn committed(_: *anyopaque, _: u64, _: ps.Provenance) void {} // landing.whole() is the check, after fill
    fn aborted(_: *anyopaque) void {} // the caller drops the blank state
};

/// What serve refuses before loading: one flag without the other, another model, or a drafter (no taps come across).
pub fn refusal(f: Flags, qwen27: bool, drafter: bool) ?[]const u8 {
    if (!f.on()) return null;
    if (f.url == null or f.link == null) return "--prefill-from and --mcdma-link go together: the GB10 server's URL and the mcdma-rpcd link it answers on";
    if (f.overHttp()) return "--mcdma-link http (hand-offs over HTTP) serves DeepSeek V4.1 only: give Qwen3.8-27B an mcdma-rpcd link";
    if (!qwen27) return "--prefill-from serves Qwen3.8-27B only (#1594 step 2b): serve this model without it";
    if (drafter) return "--prefill-from runs without --drafter for now: a remote prefix carries no drafter taps";
    return null;
}

/// The prefix policy: TF_PREFILL_MIN prompt tokens at least (4096), in TF_PREFILL_CHUNK pieces (512, whole page rows).
pub fn policy() !ps.Policy {
    const env = @import("sys").environment;
    var p: ps.Policy = .{ .enabled = true, .min_prefix = try env.read(u64, "TF_PREFILL_MIN", 4096) }; // a pull moves it all
    p.chunk = try env.read(u64, "TF_PREFILL_CHUNK", p.chunk); // the registry takes only whole page rows
    _ = try env.read(u64, "TF_PREFILL_AHEAD", 4); // read where used; a malformed one refuses to start here
    _ = try env.read(u64, "TF_PREFILL_TIMEOUT_S", 60); // a day at most, far inside u64 nanoseconds
    return p;
}

/// TF_PREFILL_KEEP (default 1): states built on a pulled prefix are kept, marked remote; the next turn resumes here.
pub fn keeps() !bool {
    return @import("sys").environment.read(bool, "TF_PREFILL_KEEP", true);
}

/// TF_PREFILL_API_KEY_FILE's key for the GB10 (null unset); a refusal logs why, naming the setting, never the bytes.
pub fn apiKey(gpa: std.mem.Allocator, io: std.Io) !?[]u8 {
    const env = @import("sys").environment;
    const path = (try env.path("TF_PREFILL_API_KEY_FILE", (env.Process{}).get("TF_PREFILL_API_KEY_FILE"))) orelse return null;
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |e| return refuse("names a file that cannot be opened ({s})", .{@errorName(e)});
    defer file.close(io);
    const st = file.stat(io) catch |e| return refuse("names a file whose mode cannot be read ({s})", .{@errorName(e)});
    if (st.kind != .file) return refuse("names something that is not a regular file", .{});
    if (st.permissions.toMode() & 0o077 != 0) return refuse("names a file other users can read or write; use chmod 600", .{});
    var buf: [4097]u8 = undefined;
    defer std.crypto.secureZero(u8, &buf);
    const n = file.readPositionalAll(io, &buf, 0) catch |e| return refuse("names a file that cannot be read ({s})", .{@errorName(e)});
    if (n > 4096) return refuse("names a file of more than 4096 bytes", .{});
    const key = std.mem.trim(u8, buf[0..n], " \t\r\n");
    if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null) return refuse("names a file that does not hold one key on one line", .{});
    return try gpa.dupe(u8, key);
}

fn refuse(comptime why: []const u8, args: anytype) error{InvalidEnvironment} {
    std.log.warn("TF_PREFILL_API_KEY_FILE " ++ why, args); // the served refusal points at this line
    return error.InvalidEnvironment;
}

/// Why serve stops when Link.start fails with `err`.
pub fn startRefusal(a: std.mem.Allocator, err: anyerror, link: []const u8) ![]const u8 {
    return switch (err) {
        error.InvalidEnvironment => "a TF_PREFILL_* setting is malformed: the line above names it, its value and its accepted forms",
        error.HybridGeometryDiffers => "--prefill-from pulls Qwen3.8-27B states only, and this checkpoint's state has another shape (Qwen3.5-9B?): serve it without --prefill-from",
        else => try std.fmt.allocPrint(a, "--mcdma-link {s}: the link could not open ({s}); is mcdma-rpcd's connect daemon serving it?", .{ link, @errorName(err) }),
    };
}

/// --prefill-from on a served host `h`: starts its link and gives the lane host the pull, the keep rule and `peek`.
pub fn attach(h: anytype, io: std.Io, a: std.mem.Allocator, f: Flags, model: []const u8, peek: api.LaneHost.Remote.ResumePeek, why: *[]const u8) !void {
    if (!f.on()) return;
    h.link = Link.start(h.gpa, io, &h.runner, f, model) catch |err| {
        why.* = try startRefusal(a, err, f.link.?);
        return err;
    };
    errdefer {
        h.link.?.stop(h.gpa);
        h.link = null;
    }
    h.host.remote = h.link.?.pull.source();
    h.host.keep_remote = keeps() catch |err| {
        why.* = "TF_PREFILL_KEEP takes 1 (keep states built on a pulled prefix, marked remote) or 0"; // the registry logs it
        return err;
    };
    if (!h.host.keep_remote) std.log.info("remote prefill: states built on a pulled prefix are not kept (TF_PREFILL_KEEP=0)", .{});
    h.host.resume_peek = peek;
}

/// Whether the loaded checkpoint's state is Qwen3.8-27B's hybrid geometry, the one the GB10 exports.
pub fn fits(runner: *const q.decode_round.Runner) bool {
    return q.snapshot.slotBytes(runner, null, block) == hs.qwen27.bytes(block);
}

/// A served link: the connect daemon's mailbox, its client, the HTTP trigger and the pull, as long as the server runs.
pub const Link = struct {
    box: fabric.mailbox.Mailbox,
    client: fabric.mailbox.Client,
    http: trigger.Http,
    pull: Pull,

    /// `model` names the checkpoint both ends load (the manifest's model must equal it).
    pub fn start(gpa: std.mem.Allocator, io: std.Io, runner: *const q.decode_round.Runner, f: Flags, model: []const u8) !*Link {
        if (!fits(runner)) return error.HybridGeometryDiffers; // Qwen3.5-9B shares the family, not the state
        const l = try gpa.create(Link);
        errdefer gpa.destroy(l);
        const rules = try policy();
        l.box = try fabric.mailbox.Mailbox.openConnect(f.link.?);
        errdefer l.box.close();
        l.client = try fabric.mailbox.Client.init(&l.box, fabric.words.native, 30 * std.time.ns_per_s);
        const base = std.mem.trimEnd(u8, f.url.?, "/");
        const key = try apiKey(gpa, io);
        errdefer if (key) |k| gpa.free(k);
        l.http = .{ .gpa = gpa, .io = io, .url = try std.fmt.allocPrint(gpa, "{s}/v1/completions", .{base}), .model = "", .closing = &l.pull.closing, .key = key };
        l.pull = .{ .gpa = gpa, .io = io, .runner = runner, .remote = .{ .gpa = gpa, .policy = rules, .client = &l.client, .trigger = l.http.trigger(), .model = model, .protocol = fabric.wire.recurrent_version } };
        return l;
    }

    /// After the lane host stopped: fills still under way end first (one mid-pull finishes it).
    pub fn stop(l: *Link, gpa: std.mem.Allocator) void {
        l.pull.closing.store(true, .release);
        while (l.pull.running.load(.acquire) > 0) std.Io.sleep(l.pull.io, .fromMilliseconds(1), .awake) catch {};
        if (l.pull.pulls > 0) std.log.info("remote prefill: {d} prefixes pulled, {d:.1} ms each on average", .{ l.pull.pulls, @as(f64, @floatFromInt(l.pull.ns)) / @as(f64, @floatFromInt(l.pull.pulls)) / 1e6 });
        gpa.free(l.http.url);
        if (l.http.key) |k| {
            std.crypto.secureZero(u8, @constCast(k));
            gpa.free(k);
        }
        l.box.close();
        gpa.destroy(l);
    }
};

test "remote prefill refuses half its flags, other models and a drafter, by name" {
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{}, false, true));
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000" }, true, false) != null);
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, false, false) != null);
    try std.testing.expect(refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, true, true) != null);
    try std.testing.expectEqual(@as(?[]const u8, null), refusal(.{ .url = "http://gb10:8000", .link = "tf2b" }, true, false));
}

test "remote prefill's default policy: 4096 prompt tokens at least, in 512-token chunks" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null) return error.SkipZigTest;
    const p = try policy();
    try std.testing.expectEqual(@as(u64, 4096), p.min_prefix);
    try std.testing.expectEqual(@as(u64, 512), p.chunk);
    try std.testing.expectEqual(@as(?u64, null), p.upto(4095, 0)); // a shorter prompt stays local
    try std.testing.expectEqual(@as(?u64, 4096), p.upto(4097, 0));
}

test "remote prefill takes a TF_PREFILL_TIMEOUT_S of a day at most, and any TF_PREFILL_AHEAD" {
    const env = @import("sys").env;
    if (std.c.getenv("TF_PREFILL_TIMEOUT_S") != null or std.c.getenv("TF_PREFILL_AHEAD") != null) return error.SkipZigTest;
    _ = env.setenv("TF_PREFILL_TIMEOUT_S", "86401", 1);
    defer _ = env.unsetenv("TF_PREFILL_TIMEOUT_S");
    try std.testing.expectError(error.InvalidEnvironment, policy());
    _ = env.setenv("TF_PREFILL_TIMEOUT_S", "86400", 1);
    _ = try policy();
    _ = env.setenv("TF_PREFILL_AHEAD", "86401", 1); // any number of fills ahead
    defer _ = env.unsetenv("TF_PREFILL_AHEAD");
    _ = try policy();
}

test "remote prefill pulls by default only 4096 tokens or more past the prefix the Mac resumes" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null) return error.SkipZigTest;
    var p: Pull = undefined;
    p.closing = .init(false);
    p.off = .init(false);
    p.remote.policy = try policy();
    try std.testing.expectEqual(.remote, Pull.wants(&p, .{ .len = 5600, .cached = 1000, .decoding = 0 })); // the prefix ends at 5120, 4120 past
    try std.testing.expectEqual(.small, Pull.wants(&p, .{ .len = 5600, .cached = 2000, .decoding = 0 })); // 3120 past it: the Mac prefills it
}

test "remote prefill refuses a TF_PREFILL_AHEAD or TF_PREFILL_TIMEOUT_S that is not a whole number from 1" {
    const env = @import("sys").env;
    for ([_][*:0]const u8{ "TF_PREFILL_AHEAD", "TF_PREFILL_TIMEOUT_S" }) |name| {
        if (std.c.getenv(name) != null) return error.SkipZigTest;
        _ = env.setenv(name, "4x", 1);
        defer _ = env.unsetenv(name);
        try std.testing.expectError(error.InvalidEnvironment, policy());
        _ = env.setenv(name, "0", 1);
        try std.testing.expectError(error.InvalidEnvironment, policy());
        _ = env.setenv(name, "8", 1);
        _ = try policy();
    }
}

test "remote prefill keeps states built on a pulled prefix unless TF_PREFILL_KEEP is 0" {
    const env = @import("sys").env;
    if (std.c.getenv("TF_PREFILL_KEEP") != null) return error.SkipZigTest;
    try std.testing.expect(try keeps());
    _ = env.setenv("TF_PREFILL_KEEP", "0", 1);
    defer _ = env.unsetenv("TF_PREFILL_KEEP");
    try std.testing.expect(!try keeps());
    _ = env.setenv("TF_PREFILL_KEEP", "true", 1);
    try std.testing.expectError(error.InvalidEnvironment, keeps());
}

test "remote prefill reads TF_PREFILL_API_KEY_FILE's one key from an owner-only file, and refuses any other" {
    const env = @import("sys").env;
    if (std.c.getenv("TF_PREFILL_API_KEY_FILE") != null) return error.SkipZigTest;
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(?[]u8, null), try apiKey(gpa, io));
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var path: [256]u8 = undefined;
    const Case = struct { data: []const u8, mode: u32, ok: bool };
    for ([_]Case{
        .{ .data = "sk-gb10\n", .mode = 0o600, .ok = true },
        .{ .data = "sk-gb10\n", .mode = 0o640, .ok = false },
        .{ .data = "\n", .mode = 0o600, .ok = false },
        .{ .data = "sk-one\nsk-two\n", .mode = 0o600, .ok = false },
    }) |c| {
        try dir.dir.writeFile(io, .{ .sub_path = "gb10.key", .data = c.data, .flags = .{ .permissions = .fromMode(@intCast(c.mode)) } });
        try dir.dir.setFilePermissions(io, "gb10.key", .fromMode(@intCast(c.mode)), .{});
        _ = env.setenv("TF_PREFILL_API_KEY_FILE", try std.fmt.bufPrintSentinel(&path, ".zig-cache/tmp/{s}/gb10.key", .{dir.sub_path}, 0), 1);
        defer _ = env.unsetenv("TF_PREFILL_API_KEY_FILE");
        if (c.ok) {
            const key = (try apiKey(gpa, io)).?;
            defer gpa.free(key);
            try std.testing.expectEqualStrings("sk-gb10", key);
        } else try std.testing.expectError(error.InvalidEnvironment, apiKey(gpa, io));
    }
}

test "remote prefill stops for good after a mismatch (another checkpoint, tokens or protocol), and not after a refusal or a timeout" {
    if (std.c.getenv("TF_PREFILL_MIN") != null or std.c.getenv("TF_PREFILL_CHUNK") != null) return error.SkipZigTest;
    var p: Pull = undefined;
    p.io = std.testing.io;
    p.mutex = .init;
    p.closing = .init(false);
    p.off = .init(false);
    p.remote.policy = try policy();
    for ([_]anyerror{ error.Refused, error.TriggerTimedOut, error.TriggerUnauthorized, error.LinkDown }) |err| {
        p.settle(err);
        try std.testing.expectEqual(.remote, Pull.wants(&p, .{ .len = 8192, .cached = 0, .decoding = 0 })); // still asks the GB10
    }
    p.settle(error.Mismatch); // the decoder found another model, prompt or token hash in the manifest
    try std.testing.expectEqual(.off, Pull.wants(&p, .{ .len = 8192, .cached = 0, .decoding = 0 }));
    try std.testing.expectEqual(@as(?Filled, null), try p.pull(&.{ 1, 2, 3 }, 0)); // nothing asks the GB10 any more
}

test "serve's refusal names another checkpoint's state shape, a malformed setting, or the link" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(std.mem.indexOf(u8, try startRefusal(a, error.HybridGeometryDiffers, "tf2b"), "Qwen3.8-27B states only") != null);
    try std.testing.expect(std.mem.startsWith(u8, try startRefusal(a, error.InvalidEnvironment, "tf2b"), "a TF_PREFILL_* setting"));
    try std.testing.expect(std.mem.startsWith(u8, try startRefusal(a, error.FileNotFound, "tf2b"), "--mcdma-link tf2b: the link could not open (FileNotFound)"));
}

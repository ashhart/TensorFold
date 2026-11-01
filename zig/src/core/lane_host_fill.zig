//! Prefer foreground and least remaining prefill work, with bounded waiting for every admitted fill.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// After eight skipped choices, oldest wait wins; each fill runs within eight plus the admitted lane count.
pub fn next(h: *const LaneHost) usize {
    return took(h, pick(h, .all).?);
}

/// Fill `at` taken: its wait starts over, every other one's grows.
fn took(h: *const LaneHost, at: usize) usize {
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

/// The fill `next` takes among `among`, its waits untouched; null when none waits.
fn pick(h: *const LaneHost, among: enum { all, short, long }) ?usize {
    var found: ?usize = null;
    for (h.filling.items, 0..) |j, i| {
        if (among != .all and short(j) != (among == .short)) continue;
        const at = found orelse {
            found = i;
            continue;
        };
        const best = h.filling.items[at];
        const aged = j.fill_wait >= preference_rounds;
        const best_aged = best.fill_wait >= preference_rounds;
        if (aged != best_aged) {
            if (aged) found = i;
        } else if (aged) {
            if (j.fill_wait > best.fill_wait) found = i;
        } else if (j.request.background != best.request.background) {
            if (!j.request.background) found = i;
        } else if (remaining(h, j) < remaining(h, best)) found = i;
    }
    return found;
}

fn remaining(h: *const LaneHost, j: *const LaneHost.Job) u64 {
    const completed = if (j.fill_began and h.core.backend.vtable.reports_fill)
        @max(j.stream.cached, j.stream.fill_at.load(.monotonic))
    else
        j.stream.cached;
    return j.stream.prompt_len -| completed;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}

/// Prompt steps' share p of wall time beside decoders (TENSORFOLD_PREFILL_SHARE): a step of t owes them t (1 - p) / p.
pub const Share = struct {
    p: f64 = 1,
    debt_ns: i64 = 0, // decode time owed before the next step of a long prompt
    rows: u64 = 0, // prompt rows a visit fills beside decoders before their round (TENSORFOLD_PREFILL_BUDGET; 0: none)
    // long prompts' order and chunk rows (TENSORFOLD_PREFILL_POLICY, policyRun); .stock: the default order and cuts
    policy: Policy = .stock,
    tokens: u32 = default_tokens, // TENSORFOLD_PREFILL_TOKENS: a policy's rows a visit, clamped to [min_tokens, max_tokens]
    call_rows: u32 = 0, // the row hint policyRun set for the call in flight (capRows); 0 outside its calls

    pub const short_rows = 512;
    pub const Policy = enum { stock, fcfs };
    pub const default_tokens: u32 = 2048; // fcfs at 2048 stalls decoders ~1.7 s a chunk (8192: ~5.9 s) for ~+13% last TTFT at N=8 x 14336
    pub const min_tokens: u32 = 64;
    pub const max_tokens: u32 = 16384; // 2 x flashnext Metal's prompt step (PMAX 8192), the most one staggered call takes (a solo 14336-row prompt is one 2 x 7168 call)
    pub const min_share = 0.01; // below it a step's debt is minutes of decoding, and is no share

    /// --decode-share's share x: the prompt's p is its complement, floored at min_share; 0 keeps whole prompts.
    pub fn fromDecode(x: f64) Share {
        if (!(x > 0 and x < 1)) return .{};
        return .{ .p = @max(1 - x, min_share) };
    }

    /// The env's share, with --decode-share's p in its place when given; the row budget always comes from the env.
    pub fn withDecode(decode_share: ?f64) Share {
        var s = fromEnv();
        if (decode_share) |x| s.p = fromDecode(x).p;
        return s;
    }

    pub fn fromEnv() Share {
        var s: Share = if (std.c.getenv("TENSORFOLD_PREFILL_SHARE")) |text| fromText(std.mem.span(text)) else .{};
        if (std.c.getenv("TENSORFOLD_PREFILL_BUDGET")) |text| s.rows = budget(std.mem.span(text));
        if (std.c.getenv("TENSORFOLD_PREFILL_POLICY")) |text| s.policy = policyOf(std.mem.span(text));
        if (std.c.getenv("TENSORFOLD_PREFILL_TOKENS")) |text| s.tokens = tokensOf(std.mem.span(text));
        return s;
    }

    /// `fcfs`; anything else is the default order.
    pub fn policyOf(text: []const u8) Policy {
        const t = std.mem.trim(u8, text, " \t");
        if (std.mem.eql(u8, t, "fcfs")) return .fcfs;
        return .stock;
    }

    /// Whole rows clamped to [min_tokens, max_tokens]; no number is the default.
    pub fn tokensOf(text: []const u8) u32 {
        const n = budget(text);
        if (n == 0) return default_tokens;
        return @intCast(std.math.clamp(n, min_tokens, max_tokens));
    }

    /// A share from its text: anything but a number in [min_share, 1] is the default, 1.
    pub fn fromText(text: []const u8) Share {
        const p = std.fmt.parseFloat(f64, text) catch return .{};
        return .{ .p = if (p >= min_share and p <= 1) p else 1 };
    }

    /// A row budget from its text: whole rows; 0 or anything else is none.
    pub fn budget(text: []const u8) u64 {
        const t = std.mem.trim(u8, text, " \t");
        if (t.len == 0 or t[0] < '0' or t[0] > '9') return 0;
        return std.fmt.parseInt(u32, t, 10) catch 0;
    }

    /// A step that took `ns` while `decoding` streams were live: the decode time it owes them.
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        s.steppedAt(ns, decoding, s.p);
    }

    /// `stepped` with this step's own share `p` (TENSORFOLD_PREFILL_ORDER=srpt varies it a step at a time).
    pub fn steppedAt(s: *Share, ns: i64, decoding: bool, p: f64) void {
        if (p >= 1 or !decoding) return;
        const owed = @as(f64, @floatFromInt(ns)) * (1 - p) / p;
        s.debt_ns +|= @intFromFloat(@min(owed, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
    }

    /// A decode round of `ns` pays the debt down.
    pub fn rounded(s: *Share, ns: i64) void {
        s.debt_ns = @max(0, s.debt_ns - ns);
    }
};

/// --decode-share's chunk hint: the cap fires only where a family declares chunk_invariant; not `Share.short_rows`.
pub const decode_cap_rows: u32 = 512;

/// Prompts before the next round: none waiting is main's round; short ones go whole; then a long one's step when due.
pub fn run(h: *LaneHost) void {
    if (h.filling.items.len == 0 or runnable(h) == 0) h.share.debt_ns = 0; // owed to streams that ended or paused: no one's
    if (h.filling.items.len == 0) return;
    if (h.share.policy != .stock) return policyRun(h); // opt-in only; the default path below
    const budget = h.share.rows;
    var used: u64 = 0; // rows filled while a stream could decode: the budget counts no others
    // an aged long prompt's step first, when due: under a budget, short arrivals could spend every visit's rows
    const first = if (budget > 0 and runnable(h) > 0 and due(h)) agedLong(h) else null;
    if (first) |at| {
        used = longStep(h, at);
        between(h);
    }
    // short prompts back to back, as a whole pass at admission went: a burst of them waits on no decode round
    var burst: usize = 0; // a lane count of short prompts begun, then a round: replies ending with their prompt free lanes without end
    // picked among the short ones alone: an aged long prompt's step waits on its debt, not they on it
    while (pick(h, .short)) |at| {
        if (budget > 0 and used >= budget) return; // the budget's rows are in: the decoders' round comes first
        if (!h.filling.items[at].fill_began) {
            if (burst == h.info_.lanes and (runnable(h) > 0 or pick(h, .long) != null)) break;
            burst += 1;
        }
        const counted = budget > 0 and runnable(h) > 0; // a stream that opened earlier in this visit waits from here
        const rows = step(h, at);
        if (counted) used +|= rows;
        between(h);
    }
    if (first != null or (budget > 0 and used >= budget)) return; // one long step a visit at most, as before
    const at = pick(h, .long) orelse return;
    if (!due(h)) return;
    _ = longStep(h, at);
}

/// TENSORFOLD_PREFILL_POLICY=fcfs: short prompts as by default (no budget), then one chunk of the oldest long prompt.
fn policyRun(h: *LaneHost) void {
    var burst: usize = 0;
    while (pick(h, .short)) |at| {
        if (!h.filling.items[at].fill_began) {
            if (burst == h.info_.lanes and (runnable(h) > 0 or pick(h, .long) != null)) break;
            burst += 1;
        }
        _ = step(h, at);
        between(h);
    }
    if (!due(h)) return; // TENSORFOLD_PREFILL_SHARE's debt, per visit
    const at = oldestLong(h) orelse return;
    // `tokens` rows while a stream decodes; alone, the engine's own cut, as without the policy
    h.share.call_rows = if (runnable(h) > 0) h.share.tokens else 0;
    defer h.share.call_rows = 0;
    _ = longStep(h, at);
}

/// The first long prompt in `filling`, which is in admission order: admitOne appends, removal keeps the order.
fn oldestLong(h: *const LaneHost) ?usize {
    for (h.filling.items, 0..) |j, i| if (!short(j)) return i;
    return null;
}

/// The long prompt `pick` takes, once it has waited `preference_rounds` choices; null otherwise.
fn agedLong(h: *const LaneHost) ?usize {
    const at = pick(h, .long) orelse return null;
    return if (h.filling.items[at].fill_wait >= preference_rounds) at else null;
}

/// Between two prompt calls: cancels, then queued requests into free lanes.
fn between(h: *LaneHost) void {
    @import("lane_host_end.zig").takeCancels(h);
    while (h.admitOne()) {}
}

/// Fill `at`'s next call: the rows it advanced (all it had, if it left the list); no progress shown spends any budget.
fn step(h: *LaneHost, at: usize) u64 {
    const job = h.filling.items[at];
    const before = remaining(h, job);
    h.fillAt(took(h, at));
    // the call can finish, cancel or requeue the job: it is read again only while it is still filling
    const after = for (h.filling.items) |j| {
        if (j == job) break remaining(h, j);
    } else 0;
    const advanced = before -| after;
    return if (advanced > 0) advanced else std.math.maxInt(u64);
}

/// A long prompt's step, timed while its share owes decoders a part of it; its rows as `step` counts them.
fn longStep(h: *LaneHost, at: usize) u64 {
    const decoding = runnable(h) > 0;
    if (h.srpt.on) return srptStep(h, at, decoding);
    // timed on the host, so a step owed to decoders ends on the GPU; at p = 1 none is owed, and no earlier mark stays
    h.filling.items[at].stream.settle_step = decoding and h.share.p < 1;
    if (h.share.p >= 1) return step(h, at);
    const t0 = std.Io.Clock.awake.now(h.io);
    const rows = step(h, at);
    h.share.stepped(@intCast(t0.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds()), decoding);
    return rows;
}

/// TENSORFOLD_PREFILL_ORDER=srpt: a step timed always, its share the floor when a reply has less work left.
fn srptStep(h: *LaneHost, at: usize, decoding: bool) u64 {
    const job = h.filling.items[at];
    var buf: [64]i64 = undefined;
    const left = @import("lane_host_srpt.zig").replyTokens(h.core.live.items, &buf);
    const first = h.srpt.decodeFirst(remaining(h, job), left);
    const p = h.srpt.share(first);
    h.filling.items[at].stream.settle_step = decoding and p < 1;
    const t0 = std.Io.Clock.awake.now(h.io);
    const rows = step(h, at);
    const ns: i64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds());
    h.srpt.filled(if (rows == std.math.maxInt(u64)) 0 else rows, ns); // no progress shown is no sample
    h.share.steppedAt(ns, decoding, p);
    return rows;
}

/// A prompt whose whole pass is at most `short_rows` rows (a long prompt's last rows are not).
fn short(j: *const LaneHost.Job) bool {
    return j.stream.prompt_len -| j.stream.cached <= Share.short_rows;
}

/// Fill chunks' cap: only the flag, a chunk-invariant family and live decoders together; else 0 (the engine's own cut).
pub fn capRows(h: *const LaneHost) u32 {
    if (h.share.policy != .stock and h.share.call_rows > 0) return h.share.call_rows; // policyRun's hint for the call in flight
    if (!h.decode_cap or h.share.p >= 1 or !h.core.backend.chunk_invariant) return 0;
    return if (runnable(h) > 0) decode_cap_rows else 0;
}

/// A long prompt's step runs now: none owed, or nothing decoding.
pub fn due(h: *LaneHost) bool {
    return h.share.debt_ns <= 0 or runnable(h) == 0;
}

/// Streams the next decode round runs: unfinished and not paused.
fn runnable(h: *const LaneHost) usize {
    var n: usize = 0;
    for (h.core.live.items) |s| n += @intFromBool(!s.finished and !s.paused);
    return n;
}

test "a share of 0.5 owes the decoders a step's time; 1 owes nothing; rounds pay it down" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.rounded(400);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    var one: Share = .{};
    one.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 0), one.debt_ns);
    var q: Share = .{ .p = 0.25 };
    q.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 3000), q.debt_ns);
}

test "a share below the floor, NaN and inf fall back to 1; the floor and 1 stand" {
    for ([_][]const u8{ "1e-12", "0.0099", "0", "-0.5", "nan", "inf", "-inf", "2", "x", "" }) |text| {
        try std.testing.expectEqual(@as(f64, 1), Share.fromText(text).p);
    }
    try std.testing.expectEqual(@as(f64, 0.01), Share.fromText("0.01").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("1").p);
}

test "prefill service: TENSORFOLD_PREFILL_BUDGET is whole rows; unset, 0 or anything else is none, main's order" {
    for ([_][]const u8{ "", "0", "x", "-512", "+512", "5e2", "4294967296" }) |text| {
        try std.testing.expectEqual(@as(u64, 0), Share.budget(text));
    }
    try std.testing.expectEqual(@as(u64, 512), Share.budget("512"));
    try std.testing.expectEqual(@as(u64, 1), Share.budget(" 1\t"));
    try std.testing.expectEqual(@as(u64, 0), (Share{}).rows);
}

test "a step's debt saturates instead of overflowing" {
    var s: Share = .{ .p = Share.min_share };
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expect(s.debt_ns > 0);
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), s.debt_ns);
    s.rounded(1000);
    try std.testing.expect(s.debt_ns < std.math.maxInt(i64));
}

test "prefill service: charge respects restored floors, stale progress and terminal removal" {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    const Hooks = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
        fn restored(_: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            if (chunk == 0) s.cached = 300;
        }
    };
    target.prefill_hook = Hooks.restored;
    target.prefill_hook_ctx = &host;
    const prompt: [512]u32 = @splat(3);
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 1, .drafts = false };
    try host.engine().submit(1, &request, .{ .ctx = &host, .event = Hooks.event });
    try std.testing.expect(host.admitOne());
    host.filling.items[0].stream.fill_at.store(400, .monotonic);
    // The pre-begin cursor is ignored; restoration is conservatively charged.
    try std.testing.expectEqual(@as(u64, 300), step(&host, 0));
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), step(&host, 0)); // cursor 256: under the floor
    // Newly reported rows 300..384, not the full remaining prompt.
    try std.testing.expectEqual(@as(u64, 84), step(&host, 0));
    try std.testing.expectEqual(@as(u64, 128), step(&host, 0)); // completion destroys the job after its last rows
    try std.testing.expectEqual(@as(usize, 0), host.filling.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "prefill service: aged long priority keeps the existing stable winner over foreground preference" {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    host.share = .{ .rows = Share.short_rows };
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    const Sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
    };
    const decoder: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 16, .drafts = false };
    try host.engine().submit(1, &decoder, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    for (0..4) |_| _ = step(&host, 0);
    try std.testing.expectEqual(@as(usize, 1), runnable(&host));
    const prompt: [2048]u32 = @splat(3);
    const first: api.Request = .{ .prompt = prompt[0..1024], .max_tokens = 8, .background = true };
    const second: api.Request = .{ .prompt = &prompt, .max_tokens = 8 };
    const short_request: api.Request = .{ .prompt = prompt[0..512], .max_tokens = 8 };
    try host.engine().submit(2, &first, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    try host.engine().submit(3, &second, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    try host.engine().submit(4, &short_request, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    const a = host.filling.items[0];
    const b = host.filling.items[1];
    a.fill_wait = preference_rounds;
    b.fill_wait = preference_rounds;
    run(&host);
    try std.testing.expectEqual(@as(u64, 256), a.stream.fill_at.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), b.stream.fill_at.load(.monotonic));
    run(&host);
    try std.testing.expectEqual(@as(u64, 512), b.stream.fill_at.load(.monotonic));
}

// TENSORFOLD_PREFILL_POLICY tests (the env is parsed in Share.fromEnv; these set host.share).
const PolicyCalls = struct {
    host: *LaneHost,
    streams: [16]*const @import("lanes").Stream = undefined,
    caps: [16]u32 = undefined,
    n: usize = 0,
    fn hook(ctx: *anyopaque, s: *@import("lanes").Stream, _: usize) void {
        const c: *PolicyCalls = @ptrCast(@alignCast(ctx));
        c.streams[c.n] = s;
        c.caps[c.n] = capRows(c.host); // the hint fillAt passed for this call
        c.n += 1;
    }
};

fn policyEvent(_: *anyopaque, _: @import("engine_api.zig").Id, _: *const @import("engine_api.zig").Event) void {}

/// A host with one live decoder (when `decoder`) and long prompts of `lens` rows admitted in order, then `share`.
fn policyHost(comptime body: fn (*LaneHost, *PolicyCalls) anyerror!void, decoder: bool, lens: []const usize, share: Share) !void {
    const lanes = @import("lanes");
    const api = @import("engine_api.zig");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    if (decoder) {
        const d: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 64, .drafts = false };
        try host.engine().submit(1, &d, .{ .ctx = &host, .event = policyEvent });
        try std.testing.expect(host.admitOne());
        while (host.filling.items.len > 0) _ = step(&host, 0);
        try std.testing.expectEqual(@as(usize, 1), runnable(&host));
    }
    const prompt: [4096]u32 = @splat(3);
    var requests: [3]api.Request = undefined;
    for (lens, 0..) |len, i| {
        requests[i] = .{ .prompt = prompt[0..len], .max_tokens = 8, .drafts = false };
        try host.engine().submit(@intCast(10 + i), &requests[i], .{ .ctx = &host, .event = policyEvent });
        try std.testing.expect(host.admitOne());
    }
    host.share = share;
    var calls: PolicyCalls = .{ .host = &host };
    target.prefill_hook = PolicyCalls.hook;
    target.prefill_hook_ctx = &calls;
    try body(&host, &calls);
}

test "prefill service: policy text and rows; unset or unknown is the default order, TOKENS clamped" {
    for ([_][]const u8{ "", "stock", "FCFS", "lifo", "batched", "x" }) |text| try std.testing.expectEqual(Share.Policy.stock, Share.policyOf(text));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf("fcfs"));
    try std.testing.expectEqual(Share.Policy.fcfs, Share.policyOf(" fcfs\t"));
    try std.testing.expectEqual(Share.Policy.stock, (Share{}).policy);
    try std.testing.expectEqual(@as(u32, 2048), (Share{}).tokens);
    for ([_][]const u8{ "", "0", "x", "-1", "4294967296" }) |text| try std.testing.expectEqual(@as(u32, 2048), Share.tokensOf(text));
    for ([_]u32{ 2048, 4096, 8192, 16384 }) |n| { // the default and the measured options pass through unchanged
        var buf: [8]u8 = undefined;
        try std.testing.expectEqual(n, Share.tokensOf(try std.fmt.bufPrint(&buf, "{d}", .{n})));
    }
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("1"));
    try std.testing.expectEqual(@as(u32, 64), Share.tokensOf("63"));
    try std.testing.expectEqual(@as(u32, 1000), Share.tokensOf("1000"));
    try std.testing.expectEqual(@as(u32, 8193), Share.tokensOf("8193"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16384"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("16385"));
    try std.testing.expectEqual(@as(u32, 16384), Share.tokensOf("65536"));
}

test "prefill service: stock policy keeps the default hint and its one long step, call_rows or not" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            host.share.call_rows = 777; // read only under a policy
            try std.testing.expectEqual(@as(u32, 0), capRows(host));
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n); // one long step a visit
            try std.testing.expectEqual(@as(u32, 0), calls.caps[0]);
        }
    }.body, true, &.{ 2048, 2048, 2048 }, .{});
}

test "prefill service: fcfs fills only the oldest long prompt a visit, at TOKENS rows, beside a decoder" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            const first = &host.filling.items[0].stream;
            const second = &host.filling.items[1].stream;
            const third = &host.filling.items[2].stream;
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n);
            try std.testing.expectEqual(first, calls.streams[0]);
            try std.testing.expectEqual(@as(u32, 1024), calls.caps[0]);
            try std.testing.expectEqual(@as(u64, 1024), first.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u64, 0), second.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u64, 0), third.fill_at.load(.monotonic));
            try std.testing.expectEqual(@as(u32, 0), host.share.call_rows); // no hint outlives its call
            run(host); // the oldest again, to its end: it leaves the list and decodes
            try std.testing.expectEqual(first, calls.streams[1]);
            try std.testing.expectEqual(@as(usize, 2), host.filling.items.len);
            run(host);
            try std.testing.expectEqual(second, calls.streams[2]);
            try std.testing.expectEqual(@as(u32, 1024), calls.caps[2]);
        }
    }.body, true, &.{ 2048, 2048, 2048 }, .{ .policy = .fcfs, .tokens = 1024 });
}

test "prefill service: fcfs alone with nothing decoding takes the engine's own cut" {
    try policyHost(struct {
        fn body(host: *LaneHost, calls: *PolicyCalls) !void {
            run(host);
            try std.testing.expectEqual(@as(usize, 1), calls.n);
            try std.testing.expectEqual(@as(u32, 0), calls.caps[0]);
            try std.testing.expectEqual(@as(usize, 1), host.filling.items.len); // the whole prompt went in
        }
    }.body, false, &.{ 2048, 2048 }, .{ .policy = .fcfs, .tokens = 1024 });
}

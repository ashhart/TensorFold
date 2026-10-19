//! Request-owned completion logs with a controlled awake clock, including real lane admission behind another stream.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const server = @import("server.zig");
const chat = @import("chat.zig");
const log = @import("log.zig");
const json = @import("json");
const Text = @import("model_text.zig").Text;
const helpers = @import("stream_preflight_test.zig").TestText;
const Cx = @import("errors.zig").Cx;

const Clock = struct {
    threadlocal var ns: i96 = 0;
    threadlocal var table: std.Io.VTable = undefined;
    threadlocal var next: u64 = 0;
    threadlocal var draws: usize = 0;
    threadlocal var no_entropy: bool = false;
    fn randomSecure(_: ?*anyopaque, bytes: []u8) std.Io.RandomSecureError!void {
        draws += 1;
        if (no_entropy) return error.EntropyUnavailable;
        const bits = (next + 1) | (@as(u64, 1) << 63);
        next += 1;
        @memcpy(bytes, std.mem.asBytes(&bits));
    }
    fn now(ctx: ?*anyopaque, kind: std.Io.Clock) std.Io.Timestamp {
        if (kind == .awake) return .{ .nanoseconds = ns };
        return std.testing.io.vtable.now(ctx, kind);
    }
    fn io() std.Io {
        table = std.testing.io.vtable.*;
        table.now = now;
        table.randomSecure = randomSecure;
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
};

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    failed: bool = false,
    fn line(ctx: *anyopaque, message: []const u8) void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.bytes.appendSlice(std.testing.allocator, message) catch {
            c.failed = true;
        };
    }
    fn start(c: *Capture) void {
        log.Test.sink = .{ .ctx = c, .call = line };
    }
    fn deinit(c: *Capture) void {
        log.Test.sink = null;
        c.bytes.deinit(std.testing.allocator);
    }
};

const Output = struct {
    end: i96,
    left: bool = false,
    fn emit(ctx: *anyopaque, _: json.Value) error{Closed}!void {
        const o: *Output = @ptrCast(@alignCast(ctx));
        Clock.ns = o.end;
        if (o.left) return error.Closed;
    }
    fn sink(o: *Output) chat.Sink {
        return .{ .ctx = o, .call = emit };
    }
};
const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

fn text(t: *helpers) Text {
    return .{ .ctx = t, .vtable = &.{ .encode = helpers.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
}
fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) ![]u8 {
    const out = try a.alloc(u8, ids.len);
    for (out, ids) |*v, id| v.* = @intCast(id);
    return out;
}
fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
    return null;
}
fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) ![]u8 {
    return std.fmt.allocPrint(a, "{d}", .{id});
}
fn vocab(_: *anyopaque) u32 {
    return 256;
}
fn eos(_: *anyopaque) []const u32 {
    return &.{};
}
fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: @import("model_text.zig").RenderOptions, _: *[]const u8) @import("model_text.zig").Error![]u8 {
    return a.dupe(u8, "prompt");
}
fn template(_: *anyopaque) []const u8 {
    return "";
}

const Script = struct {
    first: i96,
    cached: u32 = 0,
    tokens: []const u32 = &.{ 'h', 'i' },
    seed: ?u64 = null,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seed = if (req.sampling) |v| v.seed else null;
        Clock.ns = s.first;
        sink.event(sink.ctx, id, &.{ .prefilled = s.cached });
        sink.event(sink.ctx, id, &.{ .tokens = s.tokens });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .rounds = 19, .accepted = 70, .drafted = 94, .loop_period = 3 } } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn request(srv: *server.Server, a: std.mem.Allocator, id: []const u8, out: *Output) !chat.Reply {
    var cx: Cx = .{ .a = a };
    return chat.run(srv, &cx, .{ .id = id, .prompt = .{ .text = "prompt" }, .max_tokens = 2, .fields = .{ .object = try json.newObject(a) } }, out.sink(), Gone{});
}

fn scripted(cached: u32, left: bool) !void {
    var capture: Capture = .{};
    capture.start();
    defer capture.deinit();
    Clock.ns = 10_000_000_000;
    var script: Script = .{ .first = if (cached > 0) 10_123_000_000 else 10_384_000_000, .cached = cached };
    var tt: helpers = .{};
    const srv = try server.Server.init(std.testing.allocator, Clock.io(), script.engine(), text(&tt), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    srv.metrics.note(srv.io, 999, 998, 997, 996, 995, 994, 999, 998, 997, 996, 995);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var output: Output = .{ .end = if (left) 12_345_000_000 else if (cached > 0) 10_456_000_000 else 17_590_000_000, .left = left };
    if (left) {
        try std.testing.expectError(error.Cancelled, request(srv, arena.allocator(), "chatcmpl-test", &output));
        try std.testing.expectEqualStrings("[tensorfold] ended chatcmpl-test reason=client-left prompt=6 tokens=2 after=2.35s\n", capture.bytes.items);
    } else {
        const reply = try request(srv, arena.allocator(), "chatcmpl-test", &output);
        try std.testing.expectEqual(cached, reply.cached_tokens);
        const sha = @import("tokens.zig").tokenSha(&.{ 'h', 'i' });
        const expected = try std.fmt.allocPrint(arena.allocator(), "[tensorfold] done chatcmpl-test prompt=6 cached={d} thinking=False effort=none tokens=2 sha={s} finish=length loop=period:3 rounds=19 accepted=70/94{s}\n", .{ cached, sha, if (cached > 0) " ttft=0.123 decode=0.333 elapsed=0.456 queued=0.000 tok/s=3.0" else " ttft=0.384 decode=7.206 elapsed=7.590 queued=0.000 tok/s=0.1" });
        try std.testing.expectEqualStrings(expected, capture.bytes.items);
    }
    try std.testing.expect(!capture.failed);
    try std.testing.expectEqual(@as(u64, 2), srv.metrics.latency.n);
}

test "done times: normal request keeps every existing field and appends its own seconds" {
    try scripted(0, false);
}
test "done times: a resumed prompt logs its own cache count and times rather than global sums" {
    try scripted(5, false);
}
test "done times: a client-left request logs only its existing ended line" {
    try scripted(0, true);
}

fn seedSample(srv: *server.Server, a: std.mem.Allocator, body: []const u8) !?api.Sampling {
    var cx: Cx = .{ .a = a };
    return srv.resolveSampling(&cx, (try json.parse(a, body)).ok, 0, &.{ 1, 2, 3 });
}

fn seedControls(mode: enum { fresh, keyed, explicit, defaults, default_fresh, failed, log_drawn, log_given }) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    Clock.next = 42;
    Clock.draws = 0;
    Clock.no_entropy = false;
    defer Clock.no_entropy = false;
    var script: Script = .{ .first = 10_384_000_000 };
    var tt: helpers = .{};
    const srv = try server.Server.init(std.testing.allocator, Clock.io(), script.engine(), text(&tt), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false, .seed_salt = -17 }, null);
    defer srv.deinit();
    if (mode == .keyed) {
        var usage: @import("cli.zig").Usage = .{};
        const args = try @import("cli.zig").parse(a, &.{ "model", "--prompt-seed" }, &usage);
        if (@hasField(server.Config, "prompt_seed")) srv.config.prompt_seed = args.prompt_seed;
        const one = (try seedSample(srv, a, "{\"temperature\":0.7}")).?;
        const two = (try seedSample(srv, a, "{\"temperature\":0.7}")).?;
        try std.testing.expectEqual(api.seedFor(&.{ 1, 2, 3 }, -17), one.seed);
        try std.testing.expectEqual(one.seed, two.seed);
        srv.config.default_sampling = (try json.parse(a, "{\"temperature\":0.7,\"seed\":1234}")).ok;
        try std.testing.expectEqual(@as(u64, 1234), (try seedSample(srv, a, "{}")).?.seed);
        try std.testing.expectEqual(@as(u64, 7), (try seedSample(srv, a, "{\"seed\":7}")).?.seed);
        try std.testing.expectEqual(@as(usize, 0), Clock.draws);
    } else if (mode == .explicit) {
        Clock.no_entropy = true;
        for ([_][]const u8{ "0", "-3", "18446744073709551625", "\"7\"", "7.0" }, [_]u64{ 0, 18446744073709551613, 9, 7, 7 }) |value, bits| {
            const body = try std.fmt.allocPrint(a, "{{\"temperature\":0.7,\"seed\":{s}}}", .{value});
            try std.testing.expectEqual(bits, (try seedSample(srv, a, body)).?.seed);
        }
        try std.testing.expectEqual(@as(usize, 0), Clock.draws);
    } else if (mode == .failed) {
        Clock.no_entropy = true;
        try std.testing.expectEqual(@as(?api.Sampling, null), try seedSample(srv, a, "{\"temperature\":0}"));
        var cx: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, srv.resolveSampling(&cx, (try json.parse(a, "{\"temperature\":0.7}")).ok, 0, &.{1}));
        try std.testing.expectEqual(@import("errors.zig").Kind.server, cx.kind);
        try std.testing.expectEqual(@as(usize, 1), Clock.draws);
    } else if (mode == .log_drawn or mode == .log_given) {
        var capture: Capture = .{};
        capture.start();
        defer capture.deinit();
        Clock.ns = 10_000_000_000;
        var cx: Cx = .{ .a = a };
        const body = if (mode == .log_given) "{\"temperature\":0.7,\"seed\":-3}" else "{\"temperature\":0.7}";
        const prepared = try chat.prepare(srv, &cx, .{ .id = "seed-test", .prompt = .{ .text = "prompt" }, .max_tokens = 2, .fields = (try json.parse(a, body)).ok }, Gone{});
        var output: Output = .{ .end = 17_590_000_000 };
        _ = try chat.generate(srv, &cx, prepared, output.sink(), Gone{});
        const expected: u64 = if (mode == .log_given) 18446744073709551613 else 43;
        try std.testing.expectEqual(@as(?u64, expected), script.seed);
        const suffix = try std.fmt.allocPrint(a, " queued=0.000 tok/s=0.1 seed={d}\n", .{expected});
        try std.testing.expect(std.mem.endsWith(u8, capture.bytes.items, suffix));
        try std.testing.expect(!capture.failed);
    } else if (mode == .defaults) {
        srv.config.default_sampling = (try json.parse(a, "{\"temperature\":0.7,\"seed\":1234}")).ok;
        const one = (try seedSample(srv, a, "{}")).?;
        const two = (try seedSample(srv, a, "{\"seed\":null}")).?;
        try std.testing.expectEqual(@as(u64, 1234), one.seed);
        try std.testing.expectEqual(one.seed, two.seed);
        Clock.no_entropy = true;
        try std.testing.expectEqual(@as(u64, 1234), (try seedSample(srv, a, "{}")).?.seed);
        try std.testing.expectEqual(@as(u64, 7), (try seedSample(srv, a, "{\"seed\":7}")).?.seed);
        try std.testing.expectEqual(@as(usize, 0), Clock.draws);
    } else {
        if (mode == .default_fresh) srv.config.default_sampling = (try json.parse(a, "{\"temperature\":0.7}")).ok;
        const body = if (mode == .default_fresh) "{}" else "{\"temperature\":0.7}";
        const one = (try seedSample(srv, a, body)).?;
        const two = (try seedSample(srv, a, "{\"temperature\":0.7,\"seed\":null}")).?;
        try std.testing.expectEqual(@as(u64, 43), one.seed);
        try std.testing.expectEqual(@as(u64, 44), two.seed);
        try std.testing.expect(one.seed != two.seed);
        try std.testing.expectEqual(@as(usize, 2), Clock.draws);
    }
}

test "request seed: omitted and null keys draw different 63-bit OS seeds for one prompt" {
    try seedControls(.fresh);
}
test "request seed: the serve flag preserves prompt keys and salt without drawing" {
    try seedControls(.keyed);
}
test "request seed: explicit signed, wide, string and float keys preserve their old bits" {
    try seedControls(.explicit);
}
test "request seed: injected configured keys precede entropy while explicit requests still win" {
    try seedControls(.defaults);
}
test "request seed: sampled defaults without a key still draw fresh seeds" {
    try seedControls(.default_fresh);
}
test "request seed: entropy failure refuses instead of falling back, and greedy never draws" {
    try seedControls(.failed);
}
test "request seed: drawn sampling key reaches the engine and follows times on the done line" {
    try seedControls(.log_drawn);
}
test "request seed: an explicit unsigned key is logged without truncating its engine bits" {
    try seedControls(.log_given);
}

const Queued = struct {
    host: *api.LaneHost,
    first: *api.LaneHost.Job,
    failed: bool = false,
    planner: bool = false,
    key: usize = 0,
    prompt_len: usize = 0,
    fn engine(q: *Queued) api.Engine {
        return .{ .ctx = q, .vtable = &.{ .info = Script.info, .submit = submit, .cancel = Script.cancel, .status = Script.status, .memory = Script.memory } };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const q: *Queued = @ptrCast(@alignCast(ctx));
        q.run(id, r, sink) catch {
            q.failed = true;
            return error.Busy;
        };
    }
    fn run(q: *Queued, id: api.Id, r: *const api.Request, sink: api.Sink) !void {
        try q.host.engine().submit(id, r, sink);
        try std.testing.expect(q.host.takeJob() == .none);
        Clock.ns = 13_000_000_000;
        _ = q.host.cancel(q.first);
        if (q.planner) {
            var turns: usize = 0;
            while (q.host.queued.items.len > 0 or q.host.admitted.items.len > 0) : (turns += 1) {
                try std.testing.expect(turns < 16);
                _ = api.Rounds.Test.step(q.host);
            }
            return;
        }
        const job = q.host.takeJob().job;
        const began = q.host.setup(job).?;
        _ = q.host.launch(job, began);
        Clock.ns = 13_384_000_000;
        var turns: usize = 0;
        while (!q.host.deliver(job)) : (turns += 1) {
            try std.testing.expect(turns < 16);
            try q.host.core.step();
        }
    }
    fn begin(_: *anyopaque) void {}
    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        const q: *Queued = @ptrCast(@alignCast(ctx));
        q.key = ask.key;
        q.prompt_len = ask.prompt.len;
        return .admit;
    }
    fn admitted(ctx: *anyopaque) anyerror!void {
        const q: *Queued = @ptrCast(@alignCast(ctx));
        if (q.key != 0) {
            const job: *api.LaneHost.Job = @ptrFromInt(q.key);
            try std.testing.expectEqual(@as(?i96, 13_000_000_000), job.admitted_ns);
            Clock.ns = 13_384_000_000;
            job.noteAdmission();
            try std.testing.expectEqual(@as(?i96, 13_000_000_000), job.admitted_ns);
        }
    }
    fn arm(_: *anyopaque, _: usize) void {}
    fn started(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        const q: *Queued = @ptrCast(@alignCast(ctx));
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        if (q.key != 0) {
            try pieces.append(std.testing.allocator, .{ .key = q.key, .start = 0, .end = q.prompt_len - 1, .save = false });
            try finals.append(std.testing.allocator, q.key);
            q.key = 0;
        }
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {}
};
fn discard(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

fn queued(planner: bool) !void {
    var capture: Capture = .{};
    capture.start();
    defer capture.deinit();
    Clock.ns = 10_000_000_000;
    const io = Clock.io();
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .gpu_tokens = true }, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var timer: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.pieceBackend(), timer.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .lanes = 1 });
    defer host.stop();
    var unused: u8 = 0;
    const other: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 100 };
    try host.engine().submit(900, &other, .{ .ctx = &unused, .event = discard });
    const first = host.takeJob().job;
    _ = host.setup(first).?;
    var queue: Queued = .{ .host = &host, .first = first, .planner = planner };
    if (planner) host.rounds = .{ .ctx = &queue, .vtable = &.{ .begin_admit = Queued.begin, .admit = Queued.admit, .admitted = Queued.admitted, .arm = Queued.arm, .began = Queued.started, .plan = Queued.plan, .left = Queued.arm, .after = Queued.after } };
    var tt: helpers = .{};
    const srv = try server.Server.init(a, io, queue.engine(), text(&tt), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    Clock.ns = 11_000_000_000;
    var output: Output = .{ .end = 14_000_000_000 };
    _ = try request(srv, arena.allocator(), "queued", &output);
    try std.testing.expect(!queue.failed and !capture.failed);
    try std.testing.expect(std.mem.endsWith(u8, capture.bytes.items, " ttft=2.384 decode=0.616 elapsed=3.000 queued=2.000 tok/s=1.6\n"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, capture.bytes.items, "[tensorfold] done "));
}

test "done times: a request queued behind another stream gets its own admission wait" {
    try queued(false);
}
test "done times: family-planned admission logs the queued request's two-second wait" {
    try queued(true);
}

fn rateLine(count: usize, elapsed_ns: i96, suffix: []const u8) !void {
    var capture: Capture = .{};
    capture.start();
    defer capture.deinit();
    Clock.ns = 10_000_000_000;
    const ids: [11]u32 = @splat('h');
    var script: Script = .{ .first = 10_384_000_000, .tokens = ids[0..count] };
    var tt: helpers = .{};
    const srv = try server.Server.init(std.testing.allocator, Clock.io(), script.engine(), text(&tt), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var output: Output = .{ .end = script.first + elapsed_ns };
    _ = try chat.run(srv, &cx, .{ .id = "rate", .prompt = .{ .text = "prompt" }, .max_tokens = 11, .fields = .{ .object = try json.newObject(a) } }, output.sink(), Gone{});
    try std.testing.expect(std.mem.endsWith(u8, capture.bytes.items, suffix));
    try std.testing.expect(!capture.failed);
}

test "done times: tok/s uses this request's tokens minus one and its fixed-clock decode span" {
    try rateLine(11, 500_000_000, " tok/s=20.0\n");
    try rateLine(2, 0, " tok/s=0.0\n");
    try rateLine(1, 500_000_000, " tok/s=0.0\n");
    try rateLine(0, 500_000_000, " tok/s=0.0\n");
}

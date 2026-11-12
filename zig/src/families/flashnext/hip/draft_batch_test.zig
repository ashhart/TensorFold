//! Two streams drafting in one head pass a step against the same two drafting one at a time: caches, drafts, pooled.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const fx = @import("flashnext");

/// No per-thread alternate signal stack: its thread-local leaves HIP's threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };

const t = std.testing;
const be = lanes.backend;

/// Two short prompts of ordinary English ids: both are in any vocabulary, and they differ row for row.
const text = "372 13369 198 220 17 13 13258 19 3266 310 264 1283 12068 1794\n" ++
    "290 16147 11 11358 22986\n";
/// A prompt token must be in the head's draft vocabulary (`draft_ids` here is every id below 248320).
const draft_vocab = 248320;

fn env(a: std.mem.Allocator, key: []const u8, max: usize) !?[]u8 {
    const v = t.environ.getAlloc(a, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => return null,
        else => return err,
    };
    if (v.len > max) return error.EnvTooLong;
    return v;
}

/// Two prompts, each line blank-space separated ids (blank lines skipped); past the draft vocabulary is refused.
fn prompts(a: std.mem.Allocator, source: []const u8) ![2][]const u32 {
    var out: [2][]const u32 = undefined;
    var line = std.mem.splitScalar(u8, source, '\n');
    for (&out) |*p| {
        const one = while (line.next()) |l| if (l.len > 0) break l else continue else return error.TwoPrompts;
        var tokens: std.ArrayList(u32) = .empty;
        var words = std.mem.tokenizeAny(u8, one, " \t\r");
        while (words.next()) |w| {
            const tok = try std.fmt.parseInt(u32, w, 10);
            if (tok >= draft_vocab) return error.PromptPastDraftVocab;
            try tokens.append(a, tok);
        }
        if (tokens.items.len == 0) return error.EmptyPrompt;
        p.* = try tokens.toOwnedSlice(a);
    }
    return out;
}

/// The engine of TF_RV2696_DIR and its lane backend.
const Fixture = struct {
    e: *fx.engine.Engine,
    f: *fx.lanes.Fx,
    b: be.Backend,

    /// `matrix=off`: the head layer's keys stay in a panel's cache that a 4 GiB aperture can hold.
    fn init(gpa: std.mem.Allocator, dir: []const u8) !Fixture {
        var notes: hip.Policy.Notes = .{};
        var policy = try hip.Policy.resolve("graphs=off,prefill_step=64,fill_rows=64", .none, &notes);
        policy.matrix = .off;
        // no confidence cut: both paths chain their three steps, so the step's rows are compared too
        policy.mtp.confidence = 0;
        const main = try fx.engine.Engine.open(gpa, t.io, dir, .{ .policy = policy, .capacity = 1024, .chunk = 64, .want = 4 });
        errdefer main.deinit();
        if (main.weights.model.mtp == null) return error.NoMtpHead;
        const f = try fx.lanes.Fx.init(gpa, main, 4);
        errdefer f.deinit();
        if (f.drafts == 0) return error.NoDrafts; // a draft vocabulary the head step draws from
        return .{ .e = main, .f = f, .b = fx.lanes.Fx.backend(f) };
    }

    fn deinit(x: *Fixture) void {
        x.f.deinit();
        x.e.deinit();
    }

    fn lane(x: *Fixture, s: *lanes.stream.Stream) *fx.lanes.Lane {
        return x.f.lanes.get(s).?;
    }

    /// The prompt pass and the first token after it.
    fn start(x: *Fixture, s: *lanes.stream.Stream) !u32 {
        try x.b.prefill(s);
        return x.b.read(try x.b.first(s, s.prompt_len));
    }

    /// A window of the prompt's first token, two host drafts and a keep of two rows: the rows a draft absorbs.
    fn window(x: *Fixture, s: *lanes.stream.Stream, first: u32, drafts: [2]u32) ![3]u32 {
        const positions = [3]u64{ s.prompt_len + 1, s.prompt_len + 2, s.prompt_len + 3 };
        const w = [_]be.Window{.{ .stream = s, .pending = first, .held = 0, .tokens = &drafts, .parents = null, .positions = &positions }};
        var sampled: [3]u32 = undefined;
        var echo: [3]u32 = undefined;
        var out = [_]be.Verified{.{ .sampled = &sampled, .drafts = &echo }};
        try x.b.verify(&w, &out);
        try x.b.keep(&w, &.{&.{ 0, 1 }});
        return sampled;
    }
};

/// The kept rows a stream's draft absorbs, in the window's order.
const kept_rows = [_]u32{ 0, 1 };

/// A stream's draft request after its window: the two kept rows' tokens (a named slice), `depth` drafts.
fn request(s: *lanes.stream.Stream, follow: []const u32, depth: u32) be.DraftRequest {
    return .{
        .stream = s,
        .follow = follow,
        .rows = &kept_rows,
        .start = s.prompt_len,
        .position = s.prompt_len + 3,
        .depth = @min(depth, fx.lanes.max_depth),
    };
}

/// A draft's outcome for one stream: its held drafts, its head cache's rows and its pooled count.
const State = struct {
    held: []u32,
    pooled: usize,
    mem: []u8,

    fn capture(a: std.mem.Allocator, x: *Fixture, s: *lanes.stream.Stream) !State {
        const lane = x.lane(s);
        const head = &lane.head.?;
        const held = try a.alloc(u32, lane.held_n);
        try x.b.vtable.held.?(x.b.ptr, s, held);
        try x.e.stream.synchronize();
        const mem = try a.alloc(u8, head.layout.stateBytes(head.len));
        try head.mem.download(0, mem);
        return .{ .held = held, .pooled = head.pooled, .mem = mem };
    }
};

test "two streams draft in one head pass a step as they do one stream at a time" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_RV2696_DIR", 4096) orelse return error.SkipZigTest;
    const ps = try prompts(a, (try env(a, "TF_RV2696_PROMPTS", 1 << 16)) orelse text);
    var x = try Fixture.init(a, dir);
    defer x.deinit();
    // four streams: two draft in one call, and the same two prompts again, each drafted alone
    var streams: [4]lanes.stream.Stream = undefined;
    defer for (&streams) |*s| s.deinit(a);
    for (&streams, [_][]const u32{ ps[0], ps[1], ps[0], ps[1] }) |*s, p| s.* = try lanes.stream.Stream.init(a, .{ .id = "draft-batch", .prompt = p, .max_new = 16 });
    var firsts: [4]u32 = undefined;
    var sampled: [4][3]u32 = undefined;
    for (&streams, &firsts, &sampled, [_][2]u32{ .{ 11, 12 }, .{ 21, 22 }, .{ 11, 12 }, .{ 21, 22 } }) |*s, *first, *sam, drafts| {
        first.* = try x.start(s);
        sam.* = try x.window(s, first.*, drafts);
    }
    const depth = @min(3, x.f.drafts);
    try t.expect(depth > 0);
    var follows: [4][2]u32 = undefined;
    for (&follows, firsts, sampled) |*f, first, sam| f.* = .{ first, sam[1] };
    try x.b.draft(&[2]be.DraftRequest{ request(&streams[0], &follows[0], depth), request(&streams[1], &follows[1], depth) });
    var alone: [1]be.DraftRequest = undefined;
    for (streams[2..], follows[2..]) |*s, *follow| {
        alone[0] = request(s, follow, depth);
        try x.b.draft(&alone);
    }
    // each head cache's kept rows and pooled count, and the drafts: the batch's as the alone drafts'
    for (streams[0..2], streams[2..]) |*batched, *lone| {
        const one = try State.capture(a, &x, batched);
        const two = try State.capture(a, &x, lone);
        try t.expectEqual(depth, one.held.len); // the confidence is 0: both paths chain every step
        try t.expect(one.pooled > 0); // the pooled block both paths' last rows fill is inside the compared rows
        try t.expectEqual(one.pooled, two.pooled);
        try t.expectEqualSlices(u32, one.held, two.held);
        try t.expectEqualSlices(u8, one.mem, two.mem);
    }
}

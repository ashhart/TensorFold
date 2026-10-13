//! Generated native draft and plain requests retain CLI tokens and final state through the served host.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const api = @import("engine_api");
const host = @import("qwen27_served");
const target_graph = @import("synthetic_graph.zig");
const draft_graph = @import("dflash_graph.zig");
const Oracle = struct {
    a: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    tokens: []const u32,
    prompt: usize,
    branch: bool = false,
    fn propose(ptr: *anyopaque, pending: u32, limit: u32) !q.dflash.selector.Tree {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        const at = o.runner.offsets[0] - o.prompt;
        if (at >= o.tokens.len or pending != o.tokens[at]) return error.OraclePosition;
        const children: usize = if (o.branch) 2 else 1;
        const count = @min(@as(usize, limit) / children, @min(4, o.tokens.len - at - 1));
        const ids = try o.a.alloc(u32, count * children);
        const parents = try o.a.alloc(i32, ids.len);
        const scores = try o.a.alloc(f64, ids.len);
        for (0..count) |depth| for (0..children) |child| {
            const i = depth * children + child;
            const token = o.tokens[at + 1 + depth];
            ids[i] = if (child + 1 == children) token else @intCast((token + 1) % o.runner.model.config.vocab);
            parents[i] = if (depth == 0) -1 else @intCast(depth * children - 1);
            scores[i] = 0;
        };
        return .{ .gpa = o.a, .tokens = ids, .parents = parents, .scores = scores };
    }
};
const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    stats: api.Stats = .{},
    message: []const u8 = "",
    engine: ?api.Engine = null,
    id: api.Id = 0,
    cancel_after: ?usize = null,
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (value.*) {
            .prefilled => {},
            .logprobs, .prompt_logprobs => {},
            .tokens => |ids| {
                b.tokens.appendSlice(b.a, ids) catch {};
                if (b.cancel_after) |count| if (b.tokens.items.len >= count) if (b.engine) |engine| engine.cancel(b.id);
            },
            .finished => |f| {
                b.done = f.reason;
                b.stats = f.stats;
                b.message = b.a.dupe(u8, f.message) catch "allocation failed";
            },
        }
    }
    fn wait(b: *Box) !void {
        for (0..6000) |_| {
            b.mutex.lockUncancelable(b.io);
            const done = b.done != null;
            b.mutex.unlock(b.io);
            if (done) return;
            try std.Io.sleep(b.io, .fromMilliseconds(10), .awake);
        }
        return error.NoCompletion;
    }
};
/// Drafted requests on the lane core (its own round counts), by the host's rule: TF_QWEN27_LANES, else no tensor units.
fn lanesMode(h: *host.Host) bool {
    if (std.c.getenv("TF_QWEN27_LANES")) |v| return std.mem.eql(u8, std.mem.span(v), "1");
    return !h.model.device.tensorUnits();
}
fn stop(_: *anyopaque, ids: []const u32) bool {
    return ids.len == 3;
}
fn run(a: std.mem.Allocator, io: std.Io, h: *host.Host, id: u64, request: *const api.Request, reason: api.Reason, cancel_after: ?usize) !Box {
    var box = Box{ .a = a, .io = io, .engine = h.engine(), .id = id, .cancel_after = cancel_after };
    try h.engine().submit(id, request, .{ .ctx = &box, .event = Box.event });
    try box.wait();
    if (box.done != reason) {
        std.debug.print("draft served failed:{s}\n", .{box.message});
        return error.ServedReason;
    }
    return box;
}
fn check(a: std.mem.Allocator, io: std.Io, h: *host.Host, id: u64, request: *const api.Request, expected: []const u32, state: [32]u8, reason: api.Reason) !api.Stats {
    var box = try run(a, io, h, id, request, reason, null);
    defer box.tokens.deinit(a);
    if (!std.mem.eql(u32, expected, box.tokens.items) or !std.mem.eql(u8, &state, &try (q.session.Session{ .runner = &h.runner }).fingerprint())) return error.ServedDraftDiffers;
    return box.stats;
}
/// A cancel sent after `after` tokens lands at a round's end: a prefix of the reply, and the state at its length.
fn checkCancel(a: std.mem.Allocator, io: std.Io, h: *host.Host, id: u64, request: *const api.Request, expected: []const u32, states: []const [32]u8, after: usize) !void {
    var box = try run(a, io, h, id, request, .cancelled, after);
    defer box.tokens.deinit(a);
    const got = box.tokens.items;
    if (got.len < after or got.len >= expected.len or !std.mem.eql(u32, expected[0..got.len], got)) return error.CancelledTokensDiffer;
    if (!std.mem.eql(u8, &states[got.len], &try (q.session.Session{ .runner = &h.runner }).fingerprint())) return error.CancelledStateDiffers;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try target_graph.createLayers(a, 5);
    defer model.deinit();
    const draft = try draft_graph.create(a, model);
    defer draft.deinit();
    var runner = try q.decode_round.Runner.init(a, model, 1, 384);
    defer runner.deinit();
    const direct = q.session.Session{ .runner = &runner };
    var prompt: [130]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    try direct.prefill(&prompt, 16);
    var expected: [33]u32 = undefined;
    var states: [34][32]u8 = undefined;
    states[0] = try direct.fingerprint();
    for (&expected, 0..) |*id, i| {
        id.* = try direct.greedy();
        states[i + 1] = try direct.fingerprint();
        if (i + 1 < expected.len) try direct.step(id.*);
    }
    var why: []const u8 = "";
    // a context the window timing outgrows: a new build's first start refills the timing's slot
    const served = host.attach(a, init.io, model, draft, .{ .context = 256, .streams = 1, .cache_gib = 0 }, a, &why) catch |err| {
        std.debug.print("served host: {s}\n", .{why});
        return err;
    };
    defer host.close(served);
    var cases: usize = 0;
    for ([_]usize{ 1, 2, 7, 16, 17, 33 }) |count| { // a prompt-only request is the cache's warm pass (tf-qwen27-reuse)
        const request = api.Request{ .prompt = &prompt, .max_tokens = @intCast(count), .chunks = &.{ 16, 32, 48, 64, 80, 96, 112, 128 } };
        const stats = try check(a, init.io, served, @intCast(cases + 1), &request, expected[0..count], states[count], .length);
        if (count >= 2 and (stats.drafted == 0 or stats.min_rows == 0 or stats.min_rows > 16)) return error.DraftStatsMissing;
        cases += 1;
    }
    const stop_at = (std.mem.indexOfScalar(u32, &expected, expected[2]) orelse return error.NoEos) + 1;
    const eos_request = api.Request{ .prompt = &prompt, .max_tokens = 33, .eos = &.{expected[2]} };
    _ = try check(a, init.io, served, 20, &eos_request, expected[0..stop_at], states[stop_at], .stop);
    for (0..3) |mode| {
        const count: usize = if (mode == 2) 3 else 17;
        const request = api.Request{ .prompt = &prompt, .max_tokens = 17, .drafts = mode != 0, .sampling = if (mode == 1) .{ .seed = 7, .temperature = 0 } else null, .stop = if (mode == 2) .{ .ctx = served, .check = stop } else null };
        const stats = try check(a, init.io, served, @intCast(30 + mode), &request, expected[0..count], states[count], if (mode == 2) .stop else .length);
        // sampled and custom-stop requests draft on the served host; only "draft": false reports none
        if (mode == 0 and (stats.drafted != 0 or stats.accepted != 0)) return error.PlainReportedDrafts;
    }
    try draft.backend.prepare();
    draft.preparation = .{ .mode = .prepared_q4_reference, .quantization_verified = true, .source_sha256 = @import("tensorfold").affine4.signature() };
    const quantized = api.Request{ .prompt = &prompt, .max_tokens = 33 };
    _ = try check(a, init.io, served, 40, &quantized, &expected, states[33], .length);
    var oracle = Oracle{ .a = a, .runner = &served.runner, .tokens = &expected, .prompt = prompt.len };
    served.draft.?.generation.proposer = .{ .ptr = &oracle, .call = Oracle.propose };
    for ([_]bool{ false, true }) |branch| {
        oracle.branch = branch;
        try served.draft.?.reset();
        try served.draft.?.generation.prefill(&prompt, 16);
        const cli = try served.draft.?.generation.run(expected.len, true, 15);
        defer a.free(cli.tokens);
        if (cli.rounds != 7 or !std.mem.eql(u32, &expected, cli.tokens)) return error.OracleCli;
        const request = api.Request{ .prompt = &prompt, .max_tokens = expected.len };
        const stats = try check(a, init.io, served, if (branch) 51 else 50, &request, &expected, states[33], .length);
        std.debug.print("round delivery branch={}: CLI{d} served{d} matched{d}/{d}\n", .{ branch, cli.rounds, stats.rounds, cli.matched, stats.accepted });
        if (!lanesMode(served) and (stats.rounds != cli.rounds or stats.accepted != cli.matched)) return error.DeliveryClipsRound;
    }
    for ([_]usize{ 1, 6 }) |cancel_at| {
        const request = api.Request{ .prompt = &prompt, .max_tokens = expected.len };
        try checkCancel(a, init.io, served, @intCast(60 + cancel_at), &request, &expected, &states, cancel_at);
    }
    const reset_request = api.Request{ .prompt = &prompt, .max_tokens = expected.len };
    const reset_stats = try check(a, init.io, served, 80, &reset_request, &expected, states[33], .length);
    if (!lanesMode(served) and reset_stats.rounds != 7) return error.CancelResetRetainsDraft;
    std.debug.print("drafted served: length/EOS/reset/plain/sampling/custom-stop/BF16/prepared-q4 and cancellation at a round's end; tokens and final state equal CLI\n", .{});
}

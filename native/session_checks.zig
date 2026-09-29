const std = @import("std");
const mx = @import("mlx.zig");
const session = @import("session.zig");

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    chunks: std.ArrayList(usize) = .empty,
    cancelled: bool = false,
    fn deinit(c: *Capture) void {
        c.bytes.deinit(mx.allocator);
        c.chunks.deinit(mx.allocator);
    }
    fn emit(raw: ?*anyopaque, value: []const u8) !void {
        const c: *Capture = @ptrCast(@alignCast(raw.?));
        try c.bytes.appendSlice(mx.allocator, value);
        try c.chunks.append(mx.allocator, value.len);
    }
    fn cancellation(raw: ?*anyopaque) !void {
        const c: *Capture = @ptrCast(@alignCast(raw.?));
        if (c.cancelled) return error.RequestCancelled;
    }
    fn sink(c: *Capture) session.Sink {
        return .{ .context = c, .emit = emit, .cancellation = .{ .context = c, .callback = cancellation } };
    }
};

fn same(expected: session.Reply, actual: session.Reply, before: Capture, after: Capture) !void {
    try std.testing.expectEqualSlices(u32, expected.tokens.items, actual.tokens.items);
    try std.testing.expectEqualSlices(u8, expected.content, actual.content);
    try std.testing.expectEqual(expected.finish_reason, actual.finish_reason);
    try std.testing.expectEqual(expected.prompt_tokens, actual.prompt_tokens);
    try std.testing.expectEqualSlices(u8, before.bytes.items, after.bytes.items);
    try std.testing.expectEqualSlices(usize, before.chunks.items, after.chunks.items);
}

fn verifiedCopies(m: anytype, tok: *@import("vendor/tokenizer.zig").Tokenizer, prompt: []const i32, options: session.Options, expected: session.Reply, baseline: Capture) !void {
    const a = mx.allocator;
    for ([_]bool{ false, true }) |reject| {
        var capture = Capture{};
        defer capture.deinit();
        var g = try session.Generation(@TypeOf(m.*)).init(m, tok, a, prompt, options, capture.sink(), null);
        defer g.deinit();
        errdefer std.debug.print("Copy verification {s}: reject={any}, proposed={d}, accepted={d}, expected={any}, actual={any}\n", .{ @typeName(@TypeOf(m.*)), reject, g.proposed, g.accepted, expected.tokens.items, g.reply.tokens.items });
        // Proposals may be arbitrary: seed the lookup with a known continuation to
        // exercise full acceptance and a rejected suffix regardless of model prose.
        g.context.clearRetainingCapacity();
        try g.context.appendSlice(a, prompt);
        for (expected.tokens.items, 0..) |token, i| try g.context.append(a, @intCast(if (reject and i == 7) token ^ 1 else token));
        try g.context.appendSlice(a, prompt);
        g.proposer.?.prompt_len = g.context.items.len;
        while (!try g.step(m)) {}
        try std.testing.expect(g.proposed > 0);
        try std.testing.expect(g.accepted > 0);
        if (reject) try std.testing.expect(g.proposed > g.accepted);
        try std.testing.expectEqual(prompt.len + expected.tokens.items.len - 1, @as(usize, @intCast(g.state.position)));
        try std.testing.expectEqual(@as(i32, 0), m.position);
        var reply = try g.takeReply();
        defer reply.deinit(a);
        try same(expected, reply, baseline, capture);
    }
}

fn prefixReuse(s: *session.Session, prompt: []const i32, options: session.Options, expected: session.Reply, baseline: Capture) !void {
    const a = mx.allocator;
    const Store = @import("prompt_cache.zig").Store(session.Snapshot);
    var store = try Store.init(a, 2, null);
    defer store.deinit();
    const chunks = try (try s.prefillPlan()).chunks(a, prompt);
    defer chunks.deinit(a);
    const count = chunks.next(0);
    const boundary = @import("prompt_cache.zig").Boundary{ .starts = chunks.starts };
    {
        var donor = try session.RequestGeneration.init(s, a, prompt, options, .{}, null);
        defer donor.deinit();
        try std.testing.expectEqual(null, try donor.snapshot());
        try std.testing.expect(!try donor.step(s));
        const saved = (try donor.snapshot()) orelse return error.MissingPrefixSnapshot;
        try std.testing.expectEqual(count, saved.position());
        try std.testing.expect(saved.nbytes() > 0);
        try store.insertOwned(prompt[0..count], saved, prompt, false);
    }
    try std.testing.expectEqual(@as(usize, 0), store.longest(prompt[0..count], boundary));
    var captures: [2]Capture = @splat(.{});
    defer for (&captures) |*capture| capture.deinit();
    var requests: [2]session.RequestGeneration = undefined;
    var initialized: usize = 0;
    defer for (requests[0..initialized]) |*request| request.deinit();
    for (&requests, &captures) |*request, *capture| {
        request.* = try session.RequestGeneration.init(s, a, prompt, options, capture.sink(), null);
        initialized += 1;
        var hit = (try store.match(prompt, boundary, false)) orelse return error.MissingPrefixHit;
        defer hit.deinit(a);
        var invalid = try hit.cache.clone();
        defer invalid.deinit();
        switch (invalid) {
            inline else => |*state| state.position += 1,
        }
        try std.testing.expectError(error.IncompatibleSnapshotBoundary, request.restorePrefix(&invalid));
        try request.restorePrefix(&hit.cache);
        try std.testing.expectEqual(count, request.memoryLengths().now);
        try std.testing.expectError(error.InvalidSnapshotState, request.restorePrefix(&hit.cache));
    }
    // Both requests must retain independent state after the stored owner is evicted.
    try std.testing.expect(store.evictOne(null));
    try std.testing.expectEqual(@as(u64, 0), store.nbytes());
    var finished = [_]bool{ false, false };
    while (!std.mem.allEqual(bool, &finished, true)) {
        for (&requests, &finished) |*request, *done| if (!done.*) {
            done.* = try request.step(s);
        };
    }
    for (&requests, captures) |*request, capture| {
        var actual = try request.takeReply();
        defer actual.deinit(a);
        try same(expected, actual, baseline, capture);
        try std.testing.expectEqual(null, try request.snapshot());
    }
}

fn interleaved(s: *session.Session, m: anytype, tok: *@import("vendor/tokenizer.zig").Tokenizer) !void {
    const M = @TypeOf(m.*);
    const G = session.Generation(M);
    const a = mx.allocator;
    var storage: [3][2051]i32 = undefined;
    const prompts = [_][]const i32{ storage[0][0..37], storage[1][0..if (@hasDecl(M, "prefill")) @as(usize, 2051) else 71], storage[2][0..21] };
    for (prompts, 0..) |prompt, j| for (@constCast(prompt), 0..) |*id, i| {
        id.* = @intCast(10 + (i * 7 + j * 37) % 93);
    };
    var options = [_]session.Options{
        .{ .max_tokens = 12, .ignore_eos = true, .seed = 17 },
        .{ .max_tokens = 16, .ignore_eos = true, .seed = 123, .sampling = .{ .temperature = 0.7, .top_k = 12, .top_p = 0.8, .metal = true } },
        .{ .max_tokens = 18, .ignore_eos = true, .seed = 991, .thinking_budget = 3, .sampling = .{ .temperature = 1, .top_k = 5, .top_p = 0.9, .metal = true } },
    };
    var expected: [3]session.Reply = undefined;
    var baseline: [3]Capture = @splat(.{});
    defer for (&baseline) |*capture| capture.deinit();
    var completed: usize = 0;
    defer for (expected[0..completed]) |*reply| reply.deinit(a);
    for (prompts, options, &baseline, &expected) |prompt, opt, *capture, *reply| {
        var serial = opt;
        serial.draft = false;
        var g = try G.init(m, tok, a, prompt, serial, capture.sink(), null);
        defer g.deinit();
        while (!try g.step(m)) {}
        reply.* = try g.takeReply();
        completed += 1;
    }
    try prefixReuse(s, prompts[1], options[1], expected[1], baseline[1]);
    for (prompts[0..2], options[0..2], expected[0..2], baseline[0..2]) |prompt, opt, reply, capture| try verifiedCopies(m, tok, prompt, opt, reply, capture);
    if (s.prefillStep() > 256) {
        const plan = try s.prefillPlan();
        if (plan.assistant.len == 0) return error.MissingAssistantPrefillMarker;
        const adaptive_prompt = try a.dupe(i32, prompts[1]);
        defer a.free(adaptive_prompt);
        for ([_]usize{ 320, 640 }) |at| @memcpy(adaptive_prompt[at..][0..plan.assistant.len], plan.assistant);
        var adaptive_capture = Capture{};
        defer adaptive_capture.deinit();
        var cold = try G.init(m, tok, a, adaptive_prompt, options[1], adaptive_capture.sink(), null);
        defer cold.deinit();
        try cold.setPlan(plan);
        try std.testing.expectEqual(@as(usize, 320), cold.chunks.next(0));
        while (!try cold.step(m)) {}
        var reference = try cold.takeReply();
        defer reference.deinit(a);
        try prefixReuse(s, adaptive_prompt, options[1], reference, adaptive_capture);
    }
    var captured: [3]Capture = @splat(.{});
    defer for (&captured) |*capture| capture.deinit();
    var active: [3]G = undefined;
    var initialized: usize = 0;
    defer for (active[0..initialized]) |*g| g.deinit();
    for (prompts, options, &captured, &active) |prompt, opt, *capture, *g| {
        g.* = try G.init(m, tok, a, prompt, opt, capture.sink(), null);
        initialized += 1;
    }
    var cancelled_capture = Capture{};
    defer cancelled_capture.deinit();
    var cancelled = try G.init(m, tok, a, prompts[0], options[0], cancelled_capture.sink(), null);
    defer cancelled.deinit();
    _ = try cancelled.step(m);
    cancelled_capture.cancelled = true;
    try std.testing.expectError(error.RequestCancelled, cancelled.step(m));
    try std.testing.expectError(error.FailedGeneration, cancelled.step(m));
    var finished = [_]bool{ false, false, false };
    var round: usize = 0;
    while (!std.mem.allEqual(bool, &finished, true)) : (round += 1) {
        if (round > 256) return error.GenerationDidNotFinish;
        for (0..active.len) |j| {
            const index = (j + round) % active.len;
            if (!finished[index]) finished[index] = try active[index].step(m);
            try std.testing.expectEqual(@as(i32, 0), m.position);
        }
    }
    for (&active, expected, baseline, captured) |*g, before, before_sink, after_sink| {
        var actual = try g.takeReply();
        defer actual.deinit(a);
        try same(before, actual, before_sink, after_sink);
        try std.testing.expect(try g.step(m));
    }
    // A stop string ending inside a token must preserve the serial streaming boundary.
    if (expected[0].content.len >= 3) {
        options[0].stops = &.{expected[0].content[0..3]};
        var stopped_capture = Capture{};
        defer stopped_capture.deinit();
        var stopped = try G.init(m, tok, a, prompts[0], options[0], stopped_capture.sink(), null);
        defer stopped.deinit();
        while (!try stopped.step(m)) {}
        var reply = try stopped.takeReply();
        defer reply.deinit(a);
        try std.testing.expectEqual(.stop, reply.finish_reason);
        try std.testing.expectEqual(@as(usize, 0), reply.content.len);
        try std.testing.expectEqual(@as(usize, 0), stopped_capture.bytes.items.len);
    }
    var zero = try G.init(m, tok, a, prompts[0], .{ .max_tokens = 0 }, .{}, null);
    defer zero.deinit();
    try std.testing.expect(try zero.step(m));
    var empty = try zero.takeReply();
    defer empty.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), empty.tokens.items.len);
    std.debug.print("PASS: {s} isolated/interleaved prompts, reusable prefix snapshots, sampling seeds, streaming chunks, thinking budget, cancellation, stop strings and zero-token requests\n", .{@typeName(M)});
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    {
        var s = try session.Session.init(io, dir);
        defer s.deinit();
        switch (s.backend) {
            inline else => |*m| try interleaved(&s, m, &s.tokenizer),
        }
    }
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
}

pub fn checkImages(io: std.Io, dir: []const u8, path: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    {
        const a = mx.allocator;
        var s = try session.Session.init(io, dir);
        defer s.deinit();
        if (s.backend != .qwen) return error.ExpectedQwen;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(20 * 1024 * 1024));
        defer a.free(bytes);
        var tokens: std.ArrayList(i32) = .empty;
        defer tokens.deinit(a);
        try tokens.appendSlice(a, &.{ 10, 20, 30, 40 });
        const memory_before = try @import("memory_runtime.zig").activeBytes();
        var cpu = try @import("vision.zig").Prepared.init(io, dir, &.{.{ .bytes = bytes }}, tokens.items);
        defer cpu.deinit();
        try std.testing.expectEqual(memory_before, try @import("memory_runtime.zig").activeBytes());
        tokens.clearRetainingCapacity();
        try tokens.appendSlice(a, cpu.tokens.items);
        try mx.check(mx.c.mlx_reset_peak_memory());
        var prepared = try cpu.encode(io, dir, &s.backend.qwen.weights);
        defer prepared.deinit();
        var peak: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        try std.testing.expect(peak -| memory_before <= cpu.workspaceBytes());
        try std.testing.expect(prepared.positions.delta != 0);
        const G = session.Generation(@import("model.zig").Model);
        const prompts = [_][]const i32{ tokens.items, &.{ 50, 60, 70, 80 } };
        const images = [_]?*@import("vision.zig").Prompt{ &prepared, null };
        const options = session.Options{ .max_tokens = 12, .ignore_eos = true, .seed = 317, .sampling = .{ .temperature = 0.7, .top_k = 12, .top_p = 0.8, .metal = true } };
        var reference: [2]Capture = @splat(.{});
        var capture: [2]Capture = @splat(.{});
        defer for (&reference) |*v| v.deinit();
        defer for (&capture) |*v| v.deinit();
        var expected: [2]session.Reply = undefined;
        var finished: usize = 0;
        defer for (expected[0..finished]) |*reply| reply.deinit(a);
        for (prompts, images, &reference, &expected) |prompt, image, *output, *reply| {
            var serial = options;
            serial.draft = false;
            var g = try G.init(&s.backend.qwen, &s.tokenizer, a, prompt, serial, output.sink(), image);
            defer g.deinit();
            while (!try g.step(&s.backend.qwen)) {}
            reply.* = try g.takeReply();
            finished += 1;
        }
        var active: [2]G = undefined;
        var initialized: usize = 0;
        defer for (active[0..initialized]) |*g| g.deinit();
        for (prompts, images, &capture, &active) |prompt, image, *output, *g| {
            g.* = try G.init(&s.backend.qwen, &s.tokenizer, a, prompt, options, output.sink(), image);
            initialized += 1;
        }
        var done = [_]bool{ false, false };
        while (!std.mem.allEqual(bool, &done, true)) {
            for (&active, &done) |*g, *ended| if (!ended.*) {
                ended.* = try g.step(&s.backend.qwen);
                try std.testing.expectEqual(@as(i32, 0), s.backend.qwen.rope_delta);
            };
        }
        for (&active, expected, reference, capture) |*g, before, before_sink, after_sink| {
            var actual = try g.takeReply();
            defer actual.deinit(a);
            try same(before, actual, before_sink, after_sink);
        }
    }
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: interleaved image/text requests preserve tokens, streaming and independent multimodal positions\n", .{});
}

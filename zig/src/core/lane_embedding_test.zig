//! Real prompt scheduling must return embedding vectors without drawing tokens or retaining request storage.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const Encoder = struct {
    base: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 },
    table: lanes.backend.Backend.VTable = undefined,
    firsts: usize = 0,
    embeds: usize = 0,
    releases: usize = 0,
    mode: enum { normal, fail, partial, blocked } = .normal,
    step_gate: std.atomic.Value(bool) = .init(true),
    embed_gate: std.atomic.Value(bool) = .init(true),
    embedding_entered: std.atomic.Value(bool) = .init(false),
    steps: [256]u32 = undefined,
    step_count: usize = 0,
    first_release_step: usize = 0,
    fn backend(e: *Encoder) lanes.backend.Backend {
        e.table = e.base.stepped().vtable.*;
        e.table.first = first;
        e.table.prefill_step = step;
        e.table.embedding = embedding;
        e.table.release = release;
        return .{ .ptr = &e.base, .vtable = &e.table };
    }
    fn cast(ptr: *anyopaque) *Encoder {
        const base: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("base", base);
    }
    fn first(ptr: *anyopaque, _: *lanes.Stream, _: u64) !u64 {
        cast(ptr).firsts += 1;
        return error.EmbeddingDrewToken;
    }
    fn step(ptr: *anyopaque, s: *lanes.Stream) !bool {
        const e = cast(ptr);
        while (!e.step_gate.load(.acquire)) std.Thread.yield() catch {};
        std.debug.assert(e.step_count < e.steps.len);
        e.steps[e.step_count] = s.prompt()[0];
        e.step_count += 1;
        return e.base.stepped().vtable.prefill_step.?(ptr, s);
    }
    fn embedding(ptr: *anyopaque, s: *lanes.Stream, out: []f32) !void {
        const e = cast(ptr);
        e.embeds += 1;
        if (e.mode == .fail) return error.InjectedEmbeddingFailure;
        if (e.mode == .partial) {
            out[0] = 0;
            return;
        }
        if (e.mode == .blocked) {
            e.embedding_entered.store(true, .release);
            while (!e.embed_gate.load(.acquire)) std.Thread.yield() catch {};
        }
        const batch = s.embedding orelse return error.EmbeddingMetadataMissing;
        const history = e.base.lanes.get(s).?.history.items;
        if (!std.mem.eql(u32, history, s.prompt())) return error.IncompleteEmbeddingPrompt;
        var at: usize = 0;
        for (batch.ends, 0..) |end, i| {
            var sum: u32 = 0;
            for (history[at..end]) |token| sum += token;
            out[i * 2] = @floatFromInt(sum);
            out[i * 2 + 1] = @floatFromInt(end - at);
            at = end;
        }
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const e = cast(ptr);
        if (e.releases == 0) e.first_release_step = e.step_count;
        e.releases += 1;
        e.base.backend().vtable.release(ptr, s);
    }
};
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    tokens: usize = 0,
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (value.*) {
            .tokens => |tokens| b.tokens += tokens.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *Box) !api.Reason {
        for (0..10000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) |r| return r;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        return error.EmbeddingWorkerTimeout;
    }
};

test "embedding lane: stepped prompt batches return vectors with no generation and exactly one release" {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var encoder: Encoder = .{};
    defer encoder.base.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .lanes = 2, .context_window = 16, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 128, .max_inputs = 4 } });
    try host.start();
    defer host.stop();
    var values: [4]f32 = @splat(std.math.nan(f32));
    var box: Box = .{};
    const request: api.Request = .{ .prompt = &.{ 2, 7, 1, 8, 3, 5 }, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{ 2, 6 }, .dimensions = 2, .output = &values } };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    const reason = try box.wait();
    try std.testing.expectEqual(@as(usize, 0), encoder.firsts);
    try std.testing.expectEqual(api.Reason.length, reason);
    try std.testing.expectEqual(@as(usize, 1), encoder.embeds);
    try std.testing.expectEqual(@as(usize, 1), encoder.releases);
    try std.testing.expectEqual(@as(usize, 0), box.tokens);
    try std.testing.expectEqualSlices(f32, &.{ 9, 2, 17, 4 }, &values);
}

test "embedding lane: prompt chunks interleave before either embedding job finishes" {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var encoder: Encoder = .{};
    encoder.base.prefill_chunks = 32;
    encoder.step_gate.store(false, .release);
    defer encoder.base.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .lanes = 2, .context_window = 64, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 128 } });
    try host.start();
    defer host.stop();
    defer encoder.step_gate.store(true, .release);
    const one: [32]u32 = @splat(1);
    const two: [32]u32 = @splat(2);
    var output: [2][2]f32 = undefined;
    var boxes: [2]Box = .{ .{}, .{} };
    const requests = [_]api.Request{
        .{ .prompt = &one, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{32}, .dimensions = 2, .output = &output[0] } },
        .{ .prompt = &two, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{32}, .dimensions = 2, .output = &output[1] } },
    };
    for (&requests, &boxes, 1..) |*ask, *box, id| try host.engine().submit(id, ask, .{ .ctx = box, .event = Box.event });
    encoder.step_gate.store(true, .release);
    for (&boxes) |*box| try std.testing.expectEqual(api.Reason.length, try box.wait());
    const prefix = encoder.steps[0..encoder.first_release_step];
    try std.testing.expect(std.mem.indexOfScalar(u32, prefix, 1) != null);
    try std.testing.expect(std.mem.indexOfScalar(u32, prefix, 2) != null);
    try std.testing.expectEqualSlices(f32, &.{ 32, 32 }, &output[0]);
    try std.testing.expectEqualSlices(f32, &.{ 64, 32 }, &output[1]);
    try std.testing.expectEqual(@as(usize, 0), encoder.firsts);
    try std.testing.expectEqual(@as(usize, 2), encoder.releases);
}

test "embedding lane: failed or incomplete output ends failed and releases exactly once" {
    for ([_]@TypeOf(@as(Encoder, .{}).mode){ .fail, .partial }) |mode| {
        var cfg = try lanes.Config.init(a, .{}, 1, 0);
        defer cfg.deinit(a);
        var encoder: Encoder = .{ .mode = mode };
        defer encoder.base.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .context_window = 16, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 16 } });
        try host.start();
        defer host.stop();
        var output: [2]f32 = undefined;
        var box: Box = .{};
        const ask: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{2}, .dimensions = 2, .output = &output } };
        try host.engine().submit(1, &ask, .{ .ctx = &box, .event = Box.event });
        try std.testing.expectEqual(api.Reason.failed, try box.wait());
        try std.testing.expectEqual(@as(usize, 0), encoder.firsts);
        try std.testing.expectEqual(@as(usize, 1), encoder.releases);
        try std.testing.expectEqual(@as(usize, 0), encoder.base.lanes.count());
    }
}

test "embedding lane: cancellation during vector transfer waits for completion before release" {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var encoder: Encoder = .{ .mode = .blocked };
    encoder.embed_gate.store(false, .release);
    defer encoder.base.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .context_window = 16, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 16 } });
    try host.start();
    defer host.stop();
    defer encoder.embed_gate.store(true, .release);
    var output: [2]f32 = undefined;
    var box: Box = .{};
    const ask: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{2}, .dimensions = 2, .output = &output } };
    try host.engine().submit(1, &ask, .{ .ctx = &box, .event = Box.event });
    var waited: usize = 0;
    while (!encoder.embedding_entered.load(.acquire)) : (waited += 1) {
        if (waited > 10000) return error.EmbeddingWorkerTimeout;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
    host.engine().cancel(1);
    encoder.embed_gate.store(true, .release);
    try std.testing.expectEqual(api.Reason.cancelled, try box.wait());
    try std.testing.expectEqual(@as(usize, 1), encoder.releases);
    try std.testing.expectEqual(@as(usize, 0), encoder.base.lanes.count());
}

test "embedding lane: invalid boundaries vocabulary and output aliases refuse before execution" {
    const cap: lanes.embedding.Info = .{ .dimensions = 2, .vocabulary = 10, .max_batch_tokens = 8, .max_inputs = 2 };
    var out: [4]f32 = undefined;
    const batch: lanes.embedding.Batch = .{ .ends = &.{ 1, 3 }, .dimensions = 2, .output = &out };
    try batch.validate(cap, &.{ 1, 2, 3 }, 2);
    try std.testing.expectError(error.InvalidEmbeddingBatch, batch.validate(cap, &.{ 1, 2 }, 2));
    try std.testing.expectError(error.InvalidEmbeddingToken, batch.validate(cap, &.{ 1, 10, 3 }, 2));
    var bad = batch;
    bad.ends = &.{ 1, 1 };
    try std.testing.expectError(error.InvalidEmbeddingBatch, bad.validate(cap, &.{ 1, 2, 3 }, 2));
    var tokens = [_]u32{ 1, 2, 3, 4 };
    const alias: lanes.embedding.Batch = .{ .ends = &.{ 1, 4 }, .dimensions = 2, .output = @ptrCast(&tokens) };
    try std.testing.expectError(error.EmbeddingOutputAliasesInput, alias.validate(cap, &tokens, 4));
}

test "embedding lane: missing native hooks and generation requests refuse before admission" {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var encoder: Encoder = .{};
    defer encoder.base.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .context_window = 16, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 16 } });
    var output: [2]f32 = undefined;
    var box: Box = .{};
    const sink: api.Sink = .{ .ctx = &box, .event = Box.event };
    const generation: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectError(error.InvalidRequest, host.engine().submit(1, &generation, sink));
    const ask: api.Request = .{ .prompt = &.{1}, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = &.{1}, .dimensions = 2, .output = &output } };
    encoder.table.embedding = null;
    try std.testing.expectError(error.InvalidRequest, host.engine().submit(2, &ask, sink));
    encoder.table.embedding = Encoder.embedding;
    encoder.table.prefill_step = null;
    try std.testing.expectError(error.InvalidRequest, host.engine().submit(3, &ask, sink));
    host.info_.embeddings = null;
    host.info_.text_generation = true;
    try std.testing.expectError(error.InvalidRequest, host.engine().submit(4, &ask, sink));
    try std.testing.expectEqual(@as(usize, 0), encoder.embeds);
    try std.testing.expectEqual(@as(usize, 0), encoder.releases);
}

fn encodeFixture(chunks: u32, tokens: []const u32, ends: []const u32, output: []f32) !void {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var encoder: Encoder = .{};
    encoder.base.prefill_chunks = chunks;
    defer encoder.base.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, encoder.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .text_generation = false, .context_window = 16, .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 32 } });
    try host.start();
    defer host.stop();
    var box: Box = .{};
    const ask: api.Request = .{ .prompt = tokens, .max_tokens = 0, .drafts = false, .embedding = .{ .ends = ends, .dimensions = 2, .output = output } };
    try host.engine().submit(1, &ask, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try box.wait());
}

test "embedding lane: fake encoder rows are bit-identical alone and batched across prompt step sizes" {
    const tokens = [_]u32{ 2, 7, 1, 8, 3, 5 };
    var solo: [2][2]f32 = undefined;
    try encodeFixture(1, tokens[0..2], &.{2}, &solo[0]);
    try encodeFixture(1, tokens[2..], &.{4}, &solo[1]);
    for ([_]u32{ 1, 2, 3, 6 }) |chunks| {
        var batch: [4]f32 = undefined;
        try encodeFixture(chunks, &tokens, &.{ 2, 6 }, &batch);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&solo), std.mem.asBytes(&batch));
    }
}

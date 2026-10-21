//! Virtual stream clocks test event ownership, overlap, reuse and refusal without executing a GPU operation.
const std = @import("std");
const cuda = @import("cuda");
const prof = @import("cuda_prof.zig");
const trace = @import("cuda_prompt_trace.zig");
const t = std.testing;
const Fake = struct {
    var count: usize = 0;
    var live: usize = 0;
    var fail: usize = std.math.maxInt(usize);
    var bad_elapsed = false;
    var times: [2048]f32 = undefined;
    var streams: [2048]usize = undefined;
    var clocks: [3]f32 = .{ 0, 0, 0 };
    var records: usize = 0;
    var syncs: usize = 0;
    var elapsed_calls: usize = 0;
    fn reset() void {
        count = 0;
        live = 0;
        fail = std.math.maxInt(usize);
        clocks = .{ 0, 0, 0 };
        records = 0;
        syncs = 0;
        elapsed_calls = 0;
        bad_elapsed = false;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = sync;
        d.api.cuEventElapsedTime = elapsed;
        d.api.cuStreamWaitEvent = wait;
        d.api.cuGetErrorName = name;
        d.api.cuGetErrorString = name;
        return d;
    }
    fn createEvent(d: *const cuda.Driver, timing: bool) !cuda.Event {
        if (count == fail) return error.OutOfDeviceMemory;
        return cuda.Event.init(d, timing);
    }
    fn create(out: *cuda.abi.Event, flags: c_uint) callconv(.c) c_int {
        if (count == fail) return 2;
        if (flags != 0 or count + 1 == times.len) return 1;
        count += 1;
        live += 1;
        out.* = @ptrFromInt(count);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
    fn record(event: cuda.abi.Event, stream: cuda.abi.Stream) callconv(.c) c_int {
        const id = @intFromPtr(event);
        const owner = @intFromPtr(stream);
        records += 1;
        times[id] = clocks[owner];
        streams[id] = owner;
        return 0;
    }
    fn wait(stream: cuda.abi.Stream, event: cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        const owner = @intFromPtr(stream);
        clocks[owner] = @max(clocks[owner], times[@intFromPtr(event)]);
        return 0;
    }
    fn sync(_: cuda.abi.Event) callconv(.c) c_int {
        syncs += 1;
        return 0;
    }
    fn elapsed(out: *f32, a: cuda.abi.Event, b: cuda.abi.Event) callconv(.c) c_int {
        elapsed_calls += 1;
        if (streams[@intFromPtr(a)] != streams[@intFromPtr(b)]) return 1;
        out.* = if (bad_elapsed) std.math.nan(f32) else times[@intFromPtr(b)] - times[@intFromPtr(a)];
        return 0;
    }
    fn name(_: c_int, out: *?[*:0]const u8) callconv(.c) c_int {
        out.* = "fixture";
        return 0;
    }
};
const meta: trace.Meta = .{ .rank = 1, .slot = 7, .start = 2048, .rows = 2048 };

test "GLM prompt profile: paired owning-stream events separate overlapped work and exposed waits" {
    Fake.reset();
    var d = Fake.driver();
    const p = try trace.Trace.init(t.allocator, t.io, &d, 16);
    defer p.deinit();
    const main: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const side: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    p.begin(meta);
    p.layer = 3;
    p.piece = 2;
    p.site = .mixer;
    Fake.clocks[1] = 10;
    const projection = p.span(main, .replicated_qakva);
    Fake.clocks[1] = 15;
    projection.end();
    Fake.clocks[2] = 2;
    const filled = p.span(side, .wait_filled);
    try side.wait(p.events[0].last);
    filled.end();
    const network = p.span(side, .partials);
    Fake.clocks[2] = 22;
    network.end();
    const gathered = p.span(main, .wait_gathered);
    try main.wait(p.events[2].last);
    gathered.end();
    const kda = p.span(main, .kda);
    Fake.clocks[1] = 31;
    kda.end();
    const ffn = p.span(main, .ffn);
    const router = p.span(main, .router);
    Fake.clocks[1] = 34;
    router.end();
    Fake.clocks[1] = 41;
    ffn.end();
    try p.collect();
    const expected = [_]f32{ 5, 13, 7, 7, 9, 10, 3 };
    for (p.events[0..p.n], expected) |pair, ms| {
        try t.expectEqual(ms, pair.ms);
        try t.expectEqual(@as(?usize, 3), pair.layer);
        try t.expectEqual(@as(?usize, 2), pair.piece);
        try t.expectEqual(trace.Site.mixer, pair.site);
    }
    try t.expectEqual(meta, p.meta);
    try t.expectEqual(@as(usize, 7), Fake.syncs);
    try t.expectEqual(@as(usize, 14), Fake.records);
    p.active = false;
    const created = Fake.count;
    p.begin(.{ .rank = 0, .slot = 9, .start = 4096, .rows = 1 });
    const next = p.span(main, .head);
    next.end();
    try p.collect();
    try t.expectEqual(@as(usize, 1), p.n);
    try t.expectEqual(@as(?usize, null), p.events[0].layer);
    try t.expectEqual(@as(?usize, null), p.events[0].piece);
    try t.expectEqual(created, Fake.count);
}

test "GLM prompt profile: disabled scope performs no allocation event record sync or timing query" {
    Fake.reset();
    var d = Fake.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    prof.context(null, 3, 2, .mixer);
    const stamp = prof.span(null, s, .index_score);
    stamp.end();
    try t.expectEqual(@as(usize, 0), Fake.count + Fake.records + Fake.syncs + Fake.elapsed_calls);
}

test "GLM prompt profile: event allocation failure unwinds every successfully created event" {
    for (0..8) |fail| {
        Fake.reset();
        Fake.fail = fail;
        var d = Fake.driver();
        try t.expectError(error.OutOfDeviceMemory, trace.Trace.initWith(t.allocator, t.io, &d, 4, Fake.createEvent));
        try t.expectEqual(@as(usize, 0), Fake.live);
    }
}

test "GLM prompt profile: incomplete overflow failed chunks and nonfinite elapsed values refuse" {
    Fake.reset();
    var d = Fake.driver();
    const p = try trace.Trace.init(t.allocator, t.io, &d, 1);
    defer p.deinit();
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    p.begin(meta);
    const stamp = p.span(s, .index_score);
    try t.expectError(error.IncompletePromptProfile, p.collect());
    stamp.end();
    Fake.bad_elapsed = true;
    try t.expectError(error.InvalidPromptProfile, p.collect());
    Fake.bad_elapsed = false;
    p.invalid = true;
    try t.expectError(error.InvalidPromptProfile, p.collect());
    p.begin(meta);
    const one = p.span(s, .head);
    one.end();
    const extra = p.span(s, .head);
    extra.end();
    try t.expectError(error.InvalidPromptProfile, p.collect());
}

test "GLM prompt profile: the real profiler opens prompt-only trace storage and reuses it on the next chunk" {
    Fake.reset();
    var d = Fake.driver();
    const p = try prof.Prof.init(&d, t.io);
    const main: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    try p.beginPrompt(main, "fixture", meta, 0, 0);
    const created = Fake.count;
    prof.context(p, 4, 1, .ffn);
    const stamp = prof.span(p, main, .experts);
    Fake.clocks[1] = 5;
    stamp.end();
    p.end();
    try t.expect(!p.prompt.?.active);
    try t.expectEqual(@as(usize, 1), p.prompt.?.n);
    try t.expectEqual(@as(?usize, 4), p.prompt.?.events[0].layer);
    try t.expectEqual(@as(?usize, 1), p.prompt.?.events[0].piece);
    try p.beginPrompt(main, "fixture", .{ .rank = 0, .slot = 2, .start = 4096, .rows = 2048 }, 0, 0);
    const next = prof.span(p, main, .head);
    next.end();
    p.end();
    try t.expectEqual(created, Fake.count);
    try t.expectEqual(@as(?usize, null), p.prompt.?.events[0].layer);
    p.deinit();
    try t.expectEqual(@as(usize, 0), Fake.live);
}

test "GLM prompt profile: a failed chunk drains unfinished spans before its reusable events can be recorded again" {
    Fake.reset();
    var d = Fake.driver();
    const p = try trace.Trace.init(t.allocator, t.io, &d, 2);
    defer p.deinit();
    const side: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    p.begin(meta);
    _ = p.span(side, .partials);
    Fake.clocks[2] = 19;
    try p.drainIncomplete();
    try t.expectEqual(@as(usize, 1), Fake.syncs);
    try t.expect(p.events[0].complete and p.invalid);
    try t.expectError(error.InvalidPromptProfile, p.collect());
}

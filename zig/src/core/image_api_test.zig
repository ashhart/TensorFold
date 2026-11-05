//! The engine API's image capability, against a fake host that records what it was handed and hands back one PNG.
const std = @import("std");
const api = @import("engine_api.zig");
const t = std.testing;
const a = t.allocator;

/// The bytes a fake host appends: a PNG signature, so the caller can tell its own list from an untouched one.
const png = "\x89PNG\r\n\x1a\nIDAT";

/// What a fake host's entry does when it is reached.
const Mode = enum { ok, busy, invalid };

/// One fake image host: `serve` says whether it has the entry at all, `mode` what that entry answers.
const Stub = struct {
    serve: bool = true,
    mode: Mode = .ok,
    reason: api.Reason = .stop,
    limits: ?api.ImageInfo = null,
    /// Every field the entry was handed, as the caller passed it.
    seen: Seen = .{},
    /// The table `engine` hands out: `serve` decides whether it carries the image entry.
    vtable: api.Engine.VTable = undefined,

    pub const Seen = struct {
        calls: usize = 0,
        id: api.Id = 0,
        prompt: []const u32 = &.{},
        negative: []const u32 = &.{},
        width: u16 = 0,
        height: u16 = 0,
        steps: u32 = 0,
        guidance: f32 = 0,
        seed: u64 = 0,
    };

    fn engine(s: *Stub) api.Engine {
        s.vtable = .{
            .info = info,
            .submit = submit,
            .cancel = cancel,
            .status = status,
            .memory = memory,
            .submit_image = if (s.serve) submitImage else null,
        };
        return .{ .ctx = s, .vtable = &s.vtable };
    }

    fn info(ctx: *anyopaque) api.Info {
        const s: *Stub = @ptrCast(@alignCast(ctx));
        return .{ .text_generation = false, .image = s.limits };
    }

    fn submitImage(ctx: *anyopaque, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.seen = .{
            .calls = s.seen.calls + 1,
            .id = id,
            .prompt = request.prompt,
            .negative = request.negative,
            .width = request.width,
            .height = request.height,
            .steps = request.steps,
            .guidance = request.guidance,
            .seed = request.seed,
        };
        switch (s.mode) {
            .ok => {},
            .busy => return error.Busy,
            .invalid => return error.InvalidRequest,
        }
        request.png.appendSlice(request.alloc, png) catch return error.Busy; // no room: the caller may come back
        request.result.* = .{ .bytes = png.len, .reason = s.reason };
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = s.reason } });
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, _: *api.Status, _: []u32) void {}
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The caller's end of the sink: one `finished` a picture, with the reason the engine stated.
const Record = struct {
    events: usize = 0,
    id: api.Id = 0,
    reason: api.Reason = .stop,

    fn sink(r: *Record) api.Sink {
        return .{ .ctx = r, .event = event };
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const r: *Record = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .finished => |f| {
                r.events += 1;
                r.id = id;
                r.reason = f.reason;
            },
            else => {},
        }
    }
};

/// One caller-owned request: its prompt ids, its result, and the list its PNG lands in.
const Case = struct {
    prompt: [3]u32 = .{ 11, 12, 13 },
    negative: [2]u32 = .{ 7, 8 },
    result: api.ImageResult = .{},
    list: std.ArrayList(u8) = .empty,

    fn request(c: *Case) api.ImageRequest {
        return .{
            .prompt = &c.prompt,
            .negative = &c.negative,
            .width = 1024,
            .height = 768,
            .steps = 8,
            .guidance = 3.5,
            .seed = 42,
            .result = &c.result,
            .png = &c.list,
            .alloc = a,
        };
    }
};

test "the image entry is reached with exactly the caller's fields" {
    var stub = Stub{};
    var case = Case{};
    var record = Record{};
    const engine = stub.engine();
    var request = case.request();
    try engine.submitImage(9, &request, record.sink());
    defer case.list.deinit(a);
    try t.expectEqual(@as(usize, 1), stub.seen.calls);
    try t.expectEqual(@as(api.Id, 9), stub.seen.id);
    try t.expectEqualSlices(u32, &.{ 11, 12, 13 }, stub.seen.prompt);
    try t.expectEqualSlices(u32, &.{ 7, 8 }, stub.seen.negative);
    try t.expectEqual(@as(u16, 1024), stub.seen.width);
    try t.expectEqual(@as(u16, 768), stub.seen.height);
    try t.expectEqual(@as(u32, 8), stub.seen.steps);
    try t.expectEqual(@as(f32, 3.5), stub.seen.guidance);
    try t.expectEqual(@as(u64, 42), stub.seen.seed);
}

test "the PNG the engine appends is the caller's to free with its own allocator" {
    var stub = Stub{};
    var case = Case{};
    var record = Record{};
    const engine = stub.engine();
    var request = case.request();
    try engine.submitImage(1, &request, record.sink());
    defer case.list.deinit(a); // the allocator the engine appended with: a mismatch or a leak fails here
    try t.expectEqualSlices(u8, png, case.list.items);
    try t.expectEqual(@as(usize, png.len), case.result.bytes);
    try t.expectEqual(api.Reason.stop, case.result.reason);
}

test "an engine without the image entry is unsupported" {
    var stub = Stub{ .serve = false };
    var case = Case{};
    var record = Record{};
    const engine = stub.engine();
    var request = case.request();
    try t.expectError(error.Unsupported, engine.submitImage(1, &request, record.sink()));
    try t.expectEqual(@as(usize, 0), stub.seen.calls);
    try t.expectEqual(@as(usize, 0), case.list.items.len);
    try t.expectEqual(@as(usize, 0), record.events);
    try t.expect(engine.info().image == null);
}

test "an engine error comes back to the caller unchanged" {
    inline for (.{ Mode.busy, Mode.invalid }) |mode| {
        var stub = Stub{ .mode = mode };
        var case = Case{};
        var record = Record{};
        const engine = stub.engine();
        var request = case.request();
        const want: api.ImageSubmitError = switch (mode) {
            .busy => error.Busy,
            .invalid => error.InvalidRequest,
            .ok => unreachable,
        };
        try t.expectError(want, engine.submitImage(3, &request, record.sink()));
        try t.expectEqual(@as(usize, 0), case.list.items.len);
        try t.expectEqual(@as(usize, 0), record.events);
    }
}

test "Info carries the engine's image limits exactly, and none by default" {
    const limits = api.ImageInfo{ .max_side = 2048, .max_pixels = 4_194_304, .max_steps = 50, .max_prompt_tokens = 512 };
    var stub = Stub{ .limits = limits };
    const engine = stub.engine();
    try t.expectEqual(limits, engine.info().image.?);
    try t.expectEqual(@as(u16, 2048), engine.info().image.?.max_side);
    try t.expectEqual(@as(u32, 4_194_304), engine.info().image.?.max_pixels);
    try t.expectEqual(@as(u32, 50), engine.info().image.?.max_steps);
    try t.expectEqual(@as(u32, 512), engine.info().image.?.max_prompt_tokens);
    try t.expect((api.Info{}).image == null);
}

test "the finished event the engine sends arrives at the caller's sink" {
    var stub = Stub{ .reason = .length };
    var case = Case{};
    var record = Record{};
    const engine = stub.engine();
    var request = case.request();
    try engine.submitImage(12, &request, record.sink());
    defer case.list.deinit(a);
    try t.expectEqual(@as(usize, 1), record.events);
    try t.expectEqual(@as(api.Id, 12), record.id);
    try t.expectEqual(api.Reason.length, record.reason);
    try t.expectEqual(api.Reason.length, case.result.reason);
}

//! The actual stepped prompt pass on the lanes, stopped before any GPU work: images whole, a failed begin, a cancel.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const radio = @import("radio");
const vision = @import("cuda_vision.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Cuda = @import("cuda_lanes.zig").Cuda;
const a = std.testing.allocator;

const Count = struct {
    passes: usize = 0,
    images: usize = 0,

    fn encode(ctx: *anyopaque, _: cuda.Stream, _: []const lanes.stream.Image, spans: []const radio.Span, _: *const radio.Rows) anyerror!void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.passes += 1;
        c.images += spans.len;
    }
};

const Flag = struct {
    on: bool = false,
    fn check(ptr: *anyopaque) bool {
        return @as(*Flag, @ptrCast(@alignCast(ptr))).on;
    }
};

/// The vision lanes test's engine: its own sequence (no device memory) too short for any prompt (cap 1).
const Rig = struct {
    count: Count = .{},
    v: vision.Vision = undefined,
    e: Engine = undefined,
    b: Cuda = undefined,

    fn init(r: *Rig) void {
        r.v = .{ .gpa = a, .io = undefined, .cfg = undefined, .tower = undefined, .rows = .{ .d = undefined, .buf = undefined, .hidden = 1, .limit = 64 }, .marks = undefined, .limits = undefined, .mark_ids = undefined, .encoder = .{ .ctx = &r.count, .encode = Count.encode } };
        r.e = .{ .gpa = a, .io = undefined, .ctx = undefined, .stream = undefined, .k = undefined, .w = undefined, .b = undefined, .c = undefined, .max_len = 4096, .pinned = undefined, .history = undefined };
        r.e.serial = .{ .d = undefined, .handle = null };
        r.e.own = .{ .arena = null, .ptr = @splat(0), .cap = 1 };
        r.e.bound = &r.e.own;
        r.e.vision = &r.v;
        r.b = .{ .gpa = a, .e = &r.e, .head = null, .pinned = undefined, .rows_out = undefined, .rows_dev = 0, .prompt_out = undefined, .prompt_dev = 0 };
    }

    fn deinit(r: *Rig) void {
        r.b.lanes.deinit(a);
        r.b.fills.deinit(a);
    }

    fn step(r: *Rig, s: *lanes.Stream) !bool {
        const be = r.b.backend();
        return be.vtable.prefill_step.?(be.ptr, s, 0);
    }
};

test "nemotron cuda steps: an image prompt fills whole in its first step, each image through the tower once" {
    var r: Rig = .{};
    r.init();
    defer r.deinit();
    var images: [2]lanes.stream.Image = undefined;
    for (&images, 0..) |*img, i| img.* = .{ .pixels = &.{}, .gh = 4, .gw = 4, .tokens = 4, .at = @intCast(1 + i * 6) };
    const prompt: [14]u32 = @splat(1);
    var s = try lanes.Stream.init(a, .{ .id = "images", .prompt = &prompt, .max_new = 4, .images = &images });
    defer s.deinit(a);
    for (1..3) |request| {
        // the whole pass's own error (the prompt does not fit), after its images went through the tower once
        try std.testing.expectError(error.ContextFull, r.step(&s));
        try std.testing.expectEqual(request, r.count.passes);
        try std.testing.expectEqual(request * 2, r.count.images);
        try std.testing.expectEqual(@as(usize, 0), r.b.fills.count()); // never a pass held between an image's rows
    }
}

test "nemotron cuda steps: a text pass that cannot begin holds no pass and fails as the whole pass does" {
    var r: Rig = .{};
    r.init();
    r.e.vision = null;
    defer r.deinit();
    const prompt: [12]u32 = @splat(3);
    var s = try lanes.Stream.init(a, .{ .id = "text", .prompt = &prompt, .max_new = 4 });
    defer s.deinit(a);
    try std.testing.expectError(error.ContextFull, r.step(&s));
    try std.testing.expectEqual(@as(usize, 0), r.b.fills.count());
    try std.testing.expectError(error.ContextFull, r.b.backend().prefill(&s));
}

test "nemotron cuda steps: a cancel between chunks ends the pass at its next step and drops it" {
    var r: Rig = .{};
    r.init();
    r.e.vision = null;
    defer r.deinit();
    var flag: Flag = .{};
    const prompt: [12]u32 = @splat(3);
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &prompt, .max_new = 4, .cancel_check = .{ .ptr = &flag, .check = Flag.check } });
    defer s.deinit(a);
    try std.testing.expectError(error.ContextFull, r.step(&s)); // opens the lane on the own sequence
    // a pass between its chunks, as a first step leaves it: the next step finds the stream cancelled
    const cancel: engine.Cancel = .{ .ptr = &s, .check = Cuda.cancelled, .yields = false };
    try r.b.fills.put(a, &s, .{ .prompt = &prompt, .from = 0, .s = 4, .dump = null, .head = null, .cancel = cancel, .keep = null, .chunks = &.{}, .lessons = false, .wide = true, .background = false });
    flag.on = true;
    try std.testing.expectError(error.Cancelled, r.step(&s));
    try std.testing.expectEqual(@as(usize, 0), r.b.fills.count());
    try std.testing.expect(r.b.lanes.contains(&s)); // the lane host releases the stream itself
}

test "nemotron cuda steps: the lanes report each chunk's fill, so the host accounts it and Drafted steps them (REV59 1)" {
    var r: Rig = .{};
    r.init();
    defer r.deinit();
    const vt = r.b.backend().vtable;
    // drafted.zig steps a target only with all three: else --drafter falls back to the whole pass
    try std.testing.expect(vt.prefill_step != null and vt.reports_fill and vt.features != null);
}

const Kept = struct {
    calls: usize = 0,
    at: u32 = 0,
    fn kept(ctx: *anyopaque, _: *lanes.Stream, at: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ctx));
        k.calls += 1;
        k.at = at;
    }
};

/// A pass standing between its chunks at row `at`, on the engine's own sequence, as a first step leaves it.
fn between(r: *Rig, s: *lanes.Stream, prompt: []const u32, at: usize) !void {
    try r.b.lanes.put(a, s, .{ .seq = &r.e.own, .own = true });
    r.b.own_free = false;
    const cancel: engine.Cancel = .{ .ptr = s, .check = Cuda.cancelled, .yields = false };
    try r.b.fills.put(a, s, .{ .prompt = prompt, .from = 0, .s = at, .dump = null, .head = null, .cancel = cancel, .keep = null, .chunks = &.{}, .lessons = false, .wide = true, .background = false });
}

test "nemotron cuda steps: a lane-host cancel between chunks keeps the prefix its last chunk reached (REV59 2)" {
    var r: Rig = .{};
    r.init();
    r.e.vision = null;
    defer r.deinit();
    var kept: Kept = .{};
    const prompt: [12]u32 = @splat(3);
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &prompt, .max_new = 4, .reuse = .{ .hook = .{ .ptr = &kept, .at = Kept.kept } } });
    defer s.deinit(a);
    const keep = r.b.backend().vtable.cancel_prefill orelse return error.MissingCancelHook;
    try keep(&r.b, &s); // no pass between chunks: nothing to keep
    try std.testing.expectEqual(@as(usize, 0), kept.calls);
    try between(&r, &s, &prompt, 8);
    try keep(&r.b, &s);
    try std.testing.expectEqual(@as(usize, 1), kept.calls);
    try std.testing.expectEqual(@as(u32, 8), kept.at);
    try std.testing.expectEqual(&r.e.own, r.e.bound); // saved from the stream's own sequence
}

//! The real worker with production GLM admission capabilities and fake prompt/decode forwards.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
const api = @import("engine_api");
const a = std.testing.allocator;
const Stream = lanes.Stream;
const Backend = lanes.backend.Backend;
const Choice = union(enum) { prompt: usize, decode };

const Fake = struct {
    target: lanes.fake.Fake = .{ .gpa = a },
    table: Backend.VTable = undefined,
    choices: std.ArrayList(Choice) = .empty,
    ready: std.atomic.Value(bool) = .init(false),
    go: std.atomic.Value(bool) = .init(false),
    batches: usize = 0,

    fn backend(f: *Fake) Backend {
        var real: glm.lanes.Lanes = undefined;
        real.whole = false;
        const caps = real.backend().vtable;
        f.table = f.target.backend().vtable.*;
        f.table.prefill = whole;
        f.table.prefill_step = if (caps.prefill_step != null) step else null;
        f.table.prefill_many = if (caps.prefill_many != null) many else null;
        f.table.many_rows = caps.many_rows;
        f.table.verify = verify;
        return .{ .ptr = &f.target, .vtable = &f.table };
    }

    fn from(ptr: *anyopaque) *Fake {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }

    fn whole(ptr: *anyopaque, s: *Stream) anyerror!void {
        const f = from(ptr);
        if (s.prompt_len != 4) {
            if (!glm.lanes.canBatch(s)) return error.IneligibleFreshPrompt;
            try f.choices.append(a, .{ .prompt = s.prompt_len });
        }
        try f.target.backend().vtable.prefill(ptr, s);
    }

    fn step(ptr: *anyopaque, s: *Stream) anyerror!bool {
        if (s.prompt_len > glm.engine.prompt_rows) return error.UnboundedPromptChoice;
        try whole(ptr, s);
        return true;
    }

    fn many(ptr: *anyopaque, streams: []const *Stream) anyerror!bool {
        from(ptr).batches += 1;
        for (streams) |s| try whole(ptr, s);
        return true;
    }

    fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const f = from(ptr);
        for (windows) |w| if (w.stream.prompt_len == 4) {
            if (!f.ready.load(.acquire)) {
                f.ready.store(true, .release);
                while (!f.go.load(.acquire)) std.Thread.yield() catch {};
            }
            try f.choices.append(a, .decode);
        };
        try f.target.backend().vtable.verify(ptr, windows, out);
    }

    fn deinit(f: *Fake) void {
        f.target.deinit();
        f.choices.deinit(a);
    }
};

const Box = struct {
    done: std.atomic.Value(u32) = .init(0),
    bad: bool = false,

    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        if (e.* == .finished) {
            if (e.finished.reason != .length) b.bad = true;
            _ = b.done.fetchAdd(1, .release);
        }
    }
};

fn wait(value: anytype, wanted: anytype) !void {
    const start = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    while (value.load(.acquire) != wanted) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - start > 5 * std.time.ns_per_s)
            return error.WorkerTimedOut;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "GLM CUDA admission: a live decoder runs between three eligible fresh prompt choices" {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 1 }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 4 });
    var box: Box = .{};
    const engine = host.engine();
    const first: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 32, .drafts = false };
    try engine.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer {
        f.go.store(true, .release);
        if (!stopped) host.stop();
    }
    try wait(&f.ready, true);
    var prompts: [3][1536]u32 = undefined;
    for (&prompts, 0..) |*prompt, i| {
        @memset(prompt, @intCast(i + 1));
        const request: api.Request = .{ .prompt = prompt, .max_tokens = 8, .drafts = false };
        try engine.submit(@intCast(i + 2), &request, .{ .ctx = &box, .event = Box.event });
    }
    f.go.store(true, .release);
    try wait(&box.done, @as(u32, 4));
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    var prompts_seen: usize = 0;
    var interleaved = false;
    for (f.choices.items) |choice| switch (choice) {
        .prompt => |rows| {
            try std.testing.expect(rows <= glm.engine.prompt_rows);
            prompts_seen += 1;
        },
        .decode => if (prompts_seen == 1) {
            interleaved = true;
        },
    };
    try std.testing.expectEqual(@as(usize, 3), prompts_seen);
    try std.testing.expect(interleaved);
    try std.testing.expectEqual(@as(usize, 0), f.batches);
}

test "GLM CUDA admission: whole-prompt mode retains its batch callback" {
    var real: glm.lanes.Lanes = undefined;
    real.whole = true;
    const caps = real.backend().vtable;
    try std.testing.expect(caps.prefill_many != null);
    try std.testing.expect(caps.prefill_step == null);
}

//! A single constrained request uses the same compiler, mask walk and target rounds as a two-slot lane host.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const run = @import("flashnext_constraint_run.zig");
const a = std.testing.allocator;
const io = std.testing.io;

test {
    _ = @import("flashnext_constraint_state.zig");
}

const Box = struct {
    tokens: std.ArrayList(u32) = .empty,
    reason: api.Reason = .failed,
    stats: api.Stats = .{},
    done: std.atomic.Value(bool) = .init(false),
    cancel: std.atomic.Value(bool) = .init(false),
    cancel_tokens: bool = false,
    prefilled: usize = 0,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .tokens => |tokens| {
                b.tokens.appendSlice(a, tokens) catch unreachable;
                if (b.cancel_tokens) b.cancel.store(true, .release);
            },
            .prefilled => b.prefilled += 1,
            .finished => |f| {
                b.reason = f.reason;
                b.stats = f.stats;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
    fn cancelled(ptr: *anyopaque) bool {
        return @as(*Box, @ptrCast(@alignCast(ptr))).cancel.load(.acquire);
    }
    fn hooks(b: *Box) run.Hooks {
        return .{ .sink = .{ .ctx = b, .event = event }, .cancelled = cancelled };
    }
    fn wait(b: *Box) !void {
        const start = std.Io.Clock.awake.now(io).toNanoseconds();
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(io).toNanoseconds() - start > 5 * std.time.ns_per_s) return error.TimedOut;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

const facts: lanes.Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 8 };

fn two(target: *lanes.fake.Fake, request: *const api.Request, box: *Box) !void {
    var cfg = try lanes.Config.init(a, facts, 16, 15);
    defer cfg.deinit(a);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .lanes = 2, .structures = true });
    try host.start();
    defer host.stop();
    try host.engine().submit(1, request, box.hooks().sink);
    try box.wait();
}

test "Flash Next constraint: one and two slots return the same masked tokens with drafts and seeds" {
    const old = lanes.grammar_fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var compiler = lanes.grammar_fake.compiler();
    var err: [512]u8 = undefined;
    for ([_]api.Structure{ .{ .kind = .json }, .{ .kind = .json_schema, .text = "object" }, .{ .kind = .structural_tag, .text = "required" }, .{ .kind = .structural_tag, .text = "named" }, .{ .kind = .structural_tag, .text = "auto strict" } }) |st| {
        const compiled = try compiler.compile(io, @fromBackingInt(@intCast(@backingInt(st.kind))), st.text, &err);
        defer compiled.free();
        for ([_]?api.Sampling{ null, .{ .seed = 7, .temperature = 0.7 }, .{ .seed = 1904, .temperature = 0.9 } }) |sampling| {
            for ([_]bool{ false, true }) |drafts| {
                var structure = st;
                structure.compiler = &compiler;
                structure.compiled = compiled;
                const request: api.Request = .{ .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_tokens = 48, .sampling = sampling, .drafts = drafts, .structure = structure };
                var target: lanes.fake.Fake = .{ .gpa = a };
                defer target.deinit();
                var first: Box = .{};
                defer first.tokens.deinit(a);
                const before = lanes.grammar_fake.compiles;
                const f = try run.run(a, io, target.backend(), facts, .{ .lanes = 1, .structures = true }, null, 1, &request, first.hooks());
                try std.testing.expectEqual(api.Reason.length, f.reason);
                try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
                try std.testing.expectEqual(before, lanes.grammar_fake.compiles);
                try std.testing.expect(!first.done.load(.acquire));
                var second: Box = .{};
                defer second.tokens.deinit(a);
                try two(&target, &request, &second);
                try std.testing.expectEqualSlices(u32, second.tokens.items, first.tokens.items);
                try std.testing.expectEqual(@as(usize, 48), first.tokens.items.len);
                try std.testing.expectEqual(@as(usize, 1), first.prefilled);
                // The fake library's independent allowed-token rule, including the first token and every kept draft.
                const g = 1 + @as(usize, @intCast(@backingInt(st.kind))) * 1000 + st.text.len;
                for (first.tokens.items, 0..) |token, step| try std.testing.expectEqual(@as(usize, 0), (token + g + step) % 3);
                if (!drafts) try std.testing.expectEqual(@as(u64, 0), f.stats.drafted);
            }
        }
    }
}

test "Flash Next constraint: plain requests allocate no lane and compilation errors release the target" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var box: Box = .{};
    defer box.tokens.deinit(a);
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 4 };
    try std.testing.expectError(error.UnconstrainedRequest, run.run(a, io, target.backend(), facts, .{ .lanes = 1 }, null, 1, &request, box.hooks()));
    try std.testing.expectEqual(@as(usize, 0), target.prefill_count);
    const old = lanes.grammar_fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var compiler = lanes.grammar_fake.compiler();
    var bad = request;
    bad.structure = .{ .kind = .json_schema, .text = "bad", .compiler = &compiler };
    const f = try run.run(a, io, target.backend(), facts, .{ .lanes = 1 }, null, 2, &bad, box.hooks());
    try std.testing.expectEqual(api.Reason.failed, f.reason);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), box.tokens.items.len);
}

test "Flash Next constraint: cancellation and GPU failures end the job before the next reply" {
    const old = lanes.grammar_fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var compiler = lanes.grammar_fake.compiler();
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 100000, .structure = .{ .kind = .json, .compiler = &compiler } };
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var box: Box = .{ .cancel_tokens = true };
    defer box.tokens.deinit(a);
    const cancelled = try run.run(a, io, target.backend(), facts, .{ .lanes = 1 }, null, 1, &request, box.hooks());
    try std.testing.expectEqual(api.Reason.cancelled, cancelled.reason);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    var next = request;
    next.max_tokens = 12;
    box.cancel_tokens = false;
    box.cancel.store(false, .release);
    box.tokens.clearRetainingCapacity();
    target.fail_round = target.rounds + 1;
    const failed = try run.run(a, io, target.backend(), facts, .{ .lanes = 1 }, null, 2, &next, box.hooks());
    try std.testing.expectEqual(api.Reason.failed, failed.reason);
    try std.testing.expectEqualStrings("GpuFailed", failed.text());
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    target.fail_round = null;
    box.tokens.clearRetainingCapacity();
    const final = try run.run(a, io, target.backend(), facts, .{ .lanes = 1 }, null, 3, &next, box.hooks());
    try std.testing.expectEqual(api.Reason.length, final.reason);
    try std.testing.expectEqual(@as(usize, 12), box.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

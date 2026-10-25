//! A long prompt's step is owed no decode time by streams that cannot decode, nor by later ones (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const api = @import("engine_api.zig");

fn noop(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

test "paused streams are not decoders: a long prompt's step is due when every unfinished stream is paused" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.share = .{ .p = 0.5, .debt_ns = 1000 };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    try core.addStream(&s);
    defer core.backend.release(&s);
    try std.testing.expect(core.activeCount() == 1 and !fill.due(&host)); // a runnable decoder is owed the debt
    s.paused = true;
    try std.testing.expect(core.activeCount() == 1 and fill.due(&host)); // paused: counted active, nothing to pay with
    s.paused = false;
    s.finished = true;
    try std.testing.expect(fill.due(&host));
}

test "debt owed to decoders that all ended is not inherited by the next one" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    host.share = .{ .p = 0.01 }; // the long prompt's first step owes the first decoder 99 times its time
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const rows: [1000]u32 = @splat(5);
    const brief: api.Request = .{ .prompt = &prompt, .max_tokens = 3 };
    const long: api.Request = .{ .prompt = &rows, .max_tokens = 1 };
    const later: api.Request = .{ .prompt = &prompt, .max_tokens = 1 << 16 };
    // the long prompt's step after the first decoder ended sends the second: its rounds before the next step
    const Seen = struct {
        mutex: std.Io.Mutex = .init,
        engine: api.Engine,
        later: *const api.Request,
        first_done: bool = false,
        sent: ?usize = null, // the long prompt's step that sent the second decoder
        rounds: usize = 0, // the second decoder's rounds so far
        before: ?usize = null, // and before the long prompt's next step
        done: usize = 0,
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => if (id == 3) {
                    o.rounds += 1;
                },
                .finished => {
                    if (id == 1) o.first_done = true;
                    o.done += 1;
                },
                else => {},
            }
        }
        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != rows.len) return;
            o.mutex.lockUncancelable(std.testing.io);
            const send = o.first_done and o.sent == null;
            if (send) o.sent = chunk;
            if (o.sent) |at| if (chunk > at and o.before == null) {
                o.before = o.rounds;
            };
            o.mutex.unlock(std.testing.io);
            if (send) o.engine.submit(3, o.later, .{ .ctx = o, .event = event }) catch {};
        }
    };
    var seen: Seen = .{ .engine = host.engine(), .later = &later };
    target.prefill_hook = Seen.hook;
    target.prefill_hook_ctx = &seen;
    try seen.engine.submit(1, &brief, .{ .ctx = &seen, .event = Seen.event });
    try seen.engine.submit(2, &long, .{ .ctx = &seen, .event = Seen.event });
    try host.start();
    defer host.stop();
    while (true) {
        seen.mutex.lockUncancelable(std.testing.io);
        const d = seen.done;
        seen.mutex.unlock(std.testing.io);
        if (d >= 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seen.sent.? < 3); // the second decoder came while the long prompt still had steps left
    try std.testing.expect(seen.before.? <= 1);
}

test "a long prompt's step owed to decoders is marked to end on its device; p = 1 or a paused decoder marks none" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Seen = struct {
        marks: [4]bool = undefined,
        n: usize = 0,
        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            x.marks[x.n] = s.settle_step;
            x.n += 1;
        }
    };
    var seen: Seen = .{};
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.filling.deinit(gpa);
    var d = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer d.deinit(gpa);
    try core.addStream(&d);
    defer core.backend.release(&d);
    target.prefill_hook = Seen.hook; // the decoder's whole pass went in before: only the long prompt's steps count
    target.prefill_hook_ctx = &seen;
    const tokens: [600]u32 = @splat(7);
    const long: api.Request = .{ .prompt = &tokens, .max_tokens = 1 };
    var job: LaneHost.Job = .{ .host = &host, .id = 1, .request = &long, .sink = .{ .ctx = &host, .event = noop } };
    job.stream = try lanes.Stream.init(gpa, .{ .id = "p", .prompt = &tokens, .max_new = 1, .eos = &.{96} });
    defer job.stream.deinit(gpa);
    defer core.backend.release(&job.stream);
    try host.filling.append(gpa, &job);
    for ([_]f64{ 1, 0.5, 0.5 }, [_]bool{ false, false, true }) |p, paused| {
        host.share.p = p;
        d.paused = paused;
        fill.run(&host);
    }
    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, seen.marks[0..seen.n]);
}

//! --decode-share: the flag's share reaches the host, and a chunk-invariant family's chunks cap beside live decoders.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const env = @import("sys").env;

const share_env = "TENSORFOLD_PREFILL_SHARE";

/// The env is a process global; tests that read a host's default share clear it first so an ambient value cannot leak.
fn clearShareEnv() void {
    _ = env.unsetenv(share_env);
}

test "the flag's decode share is the prompt's complement, 0 keeps whole prompts" {
    try std.testing.expectEqual(@as(f64, 0.9), fill.Share.fromDecode(0.1).p);
    try std.testing.expectEqual(@as(f64, 1), fill.Share.fromDecode(0).p);
    try std.testing.expectEqual(@as(f64, 1), fill.Share.fromDecode(-0.5).p);
    try std.testing.expectEqual(@as(f64, 1), fill.Share.fromDecode(1.5).p);
    try std.testing.expectEqual(@as(f64, 1), fill.Share.fromDecode(std.math.nan(f64)).p);
    try std.testing.expectEqual(@as(f64, fill.Share.min_share), fill.Share.fromDecode(0.995).p);
}

test "the flag sets the host's share; absence keeps the default until the env says otherwise" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const flagged = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.75), flagged.share.p);
    try std.testing.expect(flagged.decode_cap); // the flag both sets the share and offers the cap
    const plain = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expectEqual(@as(f64, 1), plain.share.p);
    try std.testing.expect(!plain.decode_cap);
    _ = env.setenv(share_env, "0.25", 1);
    defer clearShareEnv();
    const prompted = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expectEqual(@as(f64, 0.25), prompted.share.p); // the env keeps its old meaning: the prompt's time share
    try std.testing.expect(!prompted.decode_cap); // and it never opts a family into the chunk cap
}

test "the chunk cap is 512 rows beside a live decoder under the flag, on a chunk-invariant family only" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.1 });
    var s = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    try core.addStream(&s);
    defer core.backend.release(&s);
    try std.testing.expectEqual(@as(u32, 512), fill.capRows(&host)); // the literal is the pin: the cap rides decode_cap_rows alone
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host));
    s.paused = true;
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host)); // nothing decoding: the engine's own cut
    s.paused = false;
    host.share = .{};
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host)); // no share: whole chunks
    host.share = .{ .p = 0.9 };
    core.backend.chunk_invariant = false; // a gated-off family ignores the hint entirely
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host));
    core.backend.chunk_invariant = true;
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host));
}

test "an env share keeps its time-only meaning: it never caps chunks" {
    const gpa = std.testing.allocator;
    _ = env.setenv(share_env, "0.1", 1);
    defer clearShareEnv();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock()); // the fake opts in
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expectEqual(@as(f64, 0.1), host.share.p); // the env still buys the prompt its time share
    var s = try lanes.Stream.init(gpa, .{ .id = "e", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    try core.addStream(&s);
    defer core.backend.release(&s);
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host)); // but no cap: only the flag offers it
}

test "a capped fill step advances at most the cap's rows, the rest whole" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 }; // one call would fill the whole prompt
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    const prompt = @as([2048]u32, @splat(3));
    var s = try lanes.Stream.init(gpa, .{ .id = "f", .prompt = &prompt, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    _ = try core.fillStreamCommitted(&s, true, null, fill.decode_cap_rows);
    try std.testing.expectEqual(@as(u64, fill.decode_cap_rows), s.fill_at.load(.monotonic));
    _ = try core.fillStreamCommitted(&s, false, null, fill.decode_cap_rows);
    try std.testing.expectEqual(@as(u64, 2 * fill.decode_cap_rows), s.fill_at.load(.monotonic));
    _ = try core.fillStreamCommitted(&s, false, null, 0); // uncapped: the rest in one step
    try std.testing.expectEqual(@as(u64, prompt.len), s.fill_at.load(.monotonic));
}

test "the cap and the short-prompt burst are independent constants" {
    try std.testing.expectEqual(@as(u32, 512), fill.decode_cap_rows); // move Share.short_rows: this literal must not follow it
}

test "the flag keeps the env's row budget and replaces only the env's share" {
    const gpa = std.testing.allocator;
    _ = env.setenv(share_env, "0.5", 1);
    defer clearShareEnv();
    _ = env.setenv("TENSORFOLD_PREFILL_BUDGET", "256", 1);
    defer _ = env.unsetenv("TENSORFOLD_PREFILL_BUDGET");
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const flagged = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.75), flagged.share.p); // the flag's share, not the env's 0.5
    try std.testing.expectEqual(@as(u64, 256), flagged.share.rows); // the budget survives the flag
    const plain = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expectEqual(@as(f64, 0.5), plain.share.p);
    try std.testing.expectEqual(@as(u64, 256), plain.share.rows);
}

/// Each prompt call's cap as the host passed it (capRows at the call), and the calls the prompt took.
const Calls = struct {
    host: *LaneHost,
    caps: [16]u32 = undefined,
    n: usize = 0,
    fn hook(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const c: *Calls = @ptrCast(@alignCast(ctx));
        c.caps[c.n] = fill.capRows(c.host);
        c.n += 1;
    }
};

/// A 2,048-row prompt filled through the host under --decode-share 0.5 beside a live decoder; the prompt calls it took.
fn flagFill(chunk_invariant: bool) !Calls {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 }; // uncapped, one call fills the whole prompt
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    core.backend.chunk_invariant = chunk_invariant; // the family's declaration, as its backend() gives it
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .decode_share = 0.5 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    const sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
    };
    const d: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 64, .drafts = false };
    try host.engine().submit(1, &d, .{ .ctx = &host, .event = sink.event });
    try std.testing.expect(host.admitOne());
    while (host.filling.items.len > 0) fill.run(&host);
    const prompt: [2048]u32 = @splat(3);
    const long: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .drafts = false };
    try host.engine().submit(2, &long, .{ .ctx = &host, .event = sink.event });
    try std.testing.expect(host.admitOne());
    var calls: Calls = .{ .host = &host };
    target.prefill_hook = Calls.hook;
    target.prefill_hook_ctx = &calls;
    var visits: usize = 0;
    while (host.filling.items.len > 0) : (visits += 1) {
        if (visits == 64) return error.FillNeverEnded;
        host.share.debt_ns = 0; // the decoders' round between visits paid the last step's debt
        fill.run(&host);
    }
    return calls;
}

test "a family that does not declare chunk_invariant ignores the flag's cap: whole steps beside a live decoder" {
    const gated = try flagFill(false); // Flash Next and Gemma 4 on CUDA (their backend() pins, cuda_cap_test.zig)
    try std.testing.expectEqual(@as(usize, 1), gated.n); // the family's own cut: the whole prompt in one call
    try std.testing.expectEqual(@as(u32, 0), gated.caps[0]);
    const capped = try flagFill(true); // the control: a declaring family gets the cap through the same host
    try std.testing.expectEqual(@as(usize, 2048 / fill.decode_cap_rows), capped.n);
    for (capped.caps[0..capped.n]) |cap| try std.testing.expectEqual(fill.decode_cap_rows, cap);
}

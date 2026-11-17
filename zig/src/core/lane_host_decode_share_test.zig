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
    const n = s.prompt_len -| s.cached; // the rows fillAt would pass: this stream's whole prompt
    try std.testing.expectEqual(@as(u32, 512), fill.capRows(&host, n)); // the literal is the pin: the cap rides decode_cap_rows alone
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host, n));
    s.paused = true;
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, n)); // nothing decoding: the engine's own cut
    s.paused = false;
    host.share = .{};
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, n)); // no share: whole chunks
    host.share = .{ .p = 0.9 };
    core.backend.chunk_invariant = false; // a gated-off family ignores the hint entirely
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, n));
    core.backend.chunk_invariant = true;
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host, n));
}

test "an engine's default decode-share arms without the flag; the flag and the env beat it" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const defaulted = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .default_decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.75), defaulted.share.p); // the engine's default, as --decode-share 0.25
    try std.testing.expect(defaulted.share.defaulted); // armed by the engine, so the threshold gates it
    try std.testing.expect(defaulted.decode_cap); // the engine opted in, so it offers the cap
    const plain = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try std.testing.expectEqual(@as(f64, 1), plain.share.p); // no flag, no default: whole prompts
    try std.testing.expect(!plain.decode_cap);
    const flagged = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .decode_share = 0.1, .default_decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.9), flagged.share.p); // the flag wins over the engine's default
    try std.testing.expect(flagged.decode_cap);
    _ = env.setenv(share_env, "0.1", 1);
    defer clearShareEnv();
    const envd = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .default_decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.1), envd.share.p); // the env beats the engine's default
    try std.testing.expect(!envd.decode_cap); // and the env still never opts a family into the cap
    _ = env.setenv(share_env, "1", 1);
    const whole = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .default_decode_share = 0.25 });
    try std.testing.expect(!whole.share.defaulted); // an env share of 1 says no sharing: the default stays off
}

test "an engine's default keeps the env's row budget, as the flag does" {
    clearShareEnv();
    _ = env.setenv("TENSORFOLD_PREFILL_BUDGET", "256", 1);
    defer _ = env.unsetenv("TENSORFOLD_PREFILL_BUDGET");
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const defaulted = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .default_decode_share = 0.25 });
    try std.testing.expectEqual(@as(f64, 0.75), defaulted.share.p);
    try std.testing.expectEqual(@as(u64, 256), defaulted.share.rows); // the budget survives the engine's default
    try std.testing.expect(defaulted.decode_cap);
}

test "the engine's default caps a long prompt's chunks beside a live decoder, never a short one's" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .default_decode_share = 0.25 }); // no flag
    const long_prompt = try gpa.alloc(u32, fill.Share.default_arm_rows);
    defer gpa.free(long_prompt);
    @memset(long_prompt, 3);
    var long = try lanes.Stream.init(gpa, .{ .id = "g", .prompt = long_prompt, .max_new = 8, .eos = &.{96} });
    defer long.deinit(gpa);
    try core.addStream(&long);
    defer core.backend.release(&long);
    const long_rows = long.prompt_len -| long.cached; // past the threshold: the default caps like the flag
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host, long_rows));
    core.backend.chunk_invariant = false; // a gated-off family ignores the hint entirely
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, long_rows));
    core.backend.chunk_invariant = true;
    long.paused = true;
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, long_rows)); // nothing decoding: the engine's own cut
    long.paused = false;
    var short = try lanes.Stream.init(gpa, .{ .id = "h", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer short.deinit(gpa);
    try core.addStream(&short);
    defer core.backend.release(&short);
    const short_rows = short.prompt_len -| short.cached; // below the threshold: main's whole chunks
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, short_rows));
    host.share.defaulted = false; // the flag's shape: the cap fires at any rows
    try std.testing.expectEqual(@as(u32, fill.decode_cap_rows), fill.capRows(&host, short_rows));
    host.share.p = 1;
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, short_rows)); // p is 1: no cap
}

test "the engine's default shares a long prompt's steps beside a decoder and leaves a short one whole" {
    clearShareEnv();
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .default_decode_share = 0.25 });
    defer {
        @import("lane_host_end.zig").closeAll(&host);
        host.stop();
    }
    const Sink = struct {
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
    };
    const decoder: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 64, .drafts = false };
    try host.engine().submit(1, &decoder, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    for (0..4) |_| _ = fill.step(&host, 0);
    try std.testing.expectEqual(@as(usize, 1), fill.runnable(&host));
    const long_prompt = try gpa.alloc(u32, fill.Share.default_arm_rows + 2048);
    defer gpa.free(long_prompt);
    @memset(long_prompt, 3);
    const long: api.Request = .{ .prompt = long_prompt, .max_tokens = 8, .drafts = false };
    try host.engine().submit(2, &long, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    fill.run(&host); // a long step beside the decoder: the default shares it and owes the round
    try std.testing.expect(host.share.debt_ns > 0);
    const after = host.filling.items[0].stream.fill_at.load(.monotonic);
    try std.testing.expectEqual(@as(u64, fill.decode_cap_rows), after); // the cap rode the call
    host.share.debt_ns = 0; // the round paid: the next visit steps again
    fill.run(&host);
    try std.testing.expect(host.filling.items[0].stream.fill_at.load(.monotonic) > after);
    // a fresh host, same decoder shape, a prompt below the threshold: whole steps, no debt, no cap
    var host2 = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .default_decode_share = 0.25 });
    defer {
        @import("lane_host_end.zig").closeAll(&host2);
        host2.stop();
    }
    const decoder2: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 64, .drafts = false };
    try host2.engine().submit(1, &decoder2, .{ .ctx = &host2, .event = Sink.event });
    try std.testing.expect(host2.admitOne());
    for (0..4) |_| _ = fill.step(&host2, 0);
    const prompt: [2048]u32 = @splat(3);
    const mid: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .drafts = false };
    try host2.engine().submit(2, &mid, .{ .ctx = &host2, .event = Sink.event });
    try std.testing.expect(host2.admitOne());
    fill.run(&host2); // below the threshold: the step goes whole and owes nothing
    try std.testing.expectEqual(@as(i64, 0), host2.share.debt_ns);
    try std.testing.expect(host2.filling.items[0].stream.fill_at.load(.monotonic) > 0); // the step ran uncapped
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
    try std.testing.expectEqual(@as(u32, 0), fill.capRows(&host, s.prompt_len -| s.cached)); // but no cap: only the flag offers it
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
    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const c: *Calls = @ptrCast(@alignCast(ctx));
        c.caps[c.n] = fill.capRows(c.host, s.prompt_len -| s.cached);
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

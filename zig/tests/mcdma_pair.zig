//! tf-mcdma-pair: the #1594 gate between two ranks over MCDMA (ConnectX or Thunderbolt), one process per rank.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const harness = @import("mcdma_pair_rank.zig");
const Rank = harness.Rank;
const Settings = harness.Settings;
const Arm = harness.Arm;
const Phase = harness.Phase;
const Layout = harness.Layout;
const Report = harness.Report;
const Status = harness.Status;
const first_test = harness.first_test;
const last_test = harness.last_test;
const expectText = harness.expectText;
const writeExit = harness.writeExit;

fn usage() noreturn {
    std.debug.print(
        \\usage: tf-mcdma-pair SETTINGS.json [overrides]                      one rank (the hardware runs)
        \\       tf-mcdma-pair --pair OUTDIR RANK0.json RANK1.json [overrides] both ranks here (MCDMA's offline stub)
        \\overrides: --arm none|exit|kill|stop|hang|external --arm-rank N --arm-test T --arm-step N --arm-delay-ms N
        \\  --arm-hold-s N --tests a,b --rounds N --duration-s N --seed N --mtu N --label S --expect Name,.. --wait-s N
        \\  --timeout-s N --watchdog-s N --connect-timeout-s N --window-mib N --bulk-mib N --hash-detail 0|1
        \\  --expect-log TEXT --refuse-log TEXT (each repeats)
        \\exit: 0 pass, or the arm's victim did its part; 1 a result differs from the host reference; 2 settings or usage
        \\  (a Mac ConnectX window past the kext's 64 regions too); 3 failed by name; 4 the watchdog fired (HANG)
        \\settings: the fields of Settings in zig/tests/mcdma_pair_rank.zig; tools/zig/mcdma_pair/README.md for the runs
        \\
    , .{});
    std.process.exit(2);
}

fn bad(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tf-mcdma-pair: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

fn num(comptime T: type, text: []const u8, flag: []const u8) T {
    return std.fmt.parseInt(T, text, 10) catch bad("{s}: not a number: {s}", .{ flag, text });
}

fn list(a: std.mem.Allocator, text: []const u8) []const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |x| out.append(a, x) catch bad("out of memory", .{});
    return out.items;
}

/// `items` with `x` after them (an override that may repeat: one --expect-log per text, which may hold commas).
fn appendOne(a: std.mem.Allocator, items: []const []const u8, x: []const u8) []const []const u8 {
    const out = a.alloc([]const u8, items.len + 1) catch bad("out of memory", .{});
    @memcpy(out[0..items.len], items);
    out[items.len] = x;
    return out;
}

fn apply(a: std.mem.Allocator, s: *Settings, mtu: *?c_int, args: []const [:0]const u8) void {
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const k = args[i];
        if (i + 1 >= args.len) bad("{s} needs a value", .{k});
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--arm")) s.arm = std.meta.stringToEnum(Arm, v) orelse bad("--arm: none, exit, kill, stop, hang or external, not {s}", .{v}) else if (std.mem.eql(u8, k, "--arm-rank")) s.arm_rank = num(u32, v, k) else if (std.mem.eql(u8, k, "--arm-test")) s.arm_test = v else if (std.mem.eql(u8, k, "--arm-step")) s.arm_step = num(u32, v, k) else if (std.mem.eql(u8, k, "--arm-delay-ms")) s.arm_delay_ms = num(u32, v, k) else if (std.mem.eql(u8, k, "--arm-hold-s")) s.arm_hold_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--tests")) s.tests = list(a, v) else if (std.mem.eql(u8, k, "--rounds")) s.rounds = num(u32, v, k) else if (std.mem.eql(u8, k, "--duration-s")) s.duration_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--seed")) s.seed = num(u64, v, k) else if (std.mem.eql(u8, k, "--mtu")) mtu.* = num(c_int, v, k) else if (std.mem.eql(u8, k, "--label")) s.label = v else if (std.mem.eql(u8, k, "--expect")) s.expect = list(a, v) else if (std.mem.eql(u8, k, "--expect-log")) s.expect_log = appendOne(a, s.expect_log, v) else if (std.mem.eql(u8, k, "--refuse-log")) s.refuse_log = appendOne(a, s.refuse_log, v) else if (std.mem.eql(u8, k, "--wait-s")) s.wait_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--timeout-s")) s.timeout_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--watchdog-s")) s.watchdog_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--connect-timeout-s")) s.connect_timeout_s = num(u32, v, k) else if (std.mem.eql(u8, k, "--window-mib")) s.window_mib = num(u32, v, k) else if (std.mem.eql(u8, k, "--bulk-mib")) s.bulk_mib = num(u32, v, k) else if (std.mem.eql(u8, k, "--hash-detail")) s.hash_detail = !std.mem.eql(u8, v, "0") else bad("unknown override {s}", .{k});
    }
}

fn load(a: std.mem.Allocator, io: std.Io, path: []const u8, overrides: []const [:0]const u8, out: Report, pair: bool) *Rank {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch |e| bad("{s}: {s}", .{ path, @errorName(e) });
    var s = std.json.parseFromSliceLeaky(Settings, a, text, .{ .allocate = .alloc_always }) catch |e| bad("{s}: {s} (fields: see Settings in zig/tests/mcdma_pair_rank.zig)", .{ path, @errorName(e) });
    var mtu: ?c_int = null;
    apply(a, &s, &mtu, overrides);
    if (s.rank > 1) bad("{s}: rank is 0 or 1", .{path});
    if (s.links.len != 1) bad("{s}: one link, to the other rank", .{path});
    if (s.links[0].peer != 1 - s.rank) bad("{s}: the link's peer is {d}", .{ path, 1 - s.rank });
    if (s.rounds == 0) bad("{s}: rounds is at least 1", .{path});
    if (s.arm_rank > 1) bad("{s}: arm_rank is 0 or 1", .{path});
    if (s.watchdog_s <= @max(s.wait_s, s.timeout_s) + 2) bad("{s}: watchdog_s ({d}) must exceed wait_s and timeout_s by more than 2 s, so their names come first", .{ path, s.watchdog_s });
    if (pair and s.arm == .kill) bad("--pair runs both ranks in one process: the kill arm would kill both (use exit or stop; kill runs on hardware)", .{});
    var tests: std.ArrayList(Phase) = .empty;
    for (s.tests) |name| {
        if (std.mem.eql(u8, name, "all")) {
            for (@backingInt(first_test)..@backingInt(last_test) + 1) |t| tests.append(a, @fromBackingInt(@intCast(t))) catch bad("out of memory", .{});
            continue;
        }
        const t = std.meta.stringToEnum(Phase, name) orelse bad("{s}: unknown test {s}", .{ path, name });
        if (@backingInt(t) < @backingInt(first_test) or @backingInt(t) > @backingInt(last_test)) bad("{s}: unknown test {s}", .{ path, name });
        tests.append(a, t) catch bad("out of memory", .{});
    }
    if (tests.items.len == 0) bad("{s}: no tests", .{path});
    if (s.arm == .exit or s.arm == .kill or s.arm == .stop or s.arm == .hang) {
        const t = std.meta.stringToEnum(Phase, s.arm_test) orelse bad("arm_test {s} is not a test", .{s.arm_test});
        if (std.mem.indexOfScalar(Phase, tests.items, t) == null) bad("arm_test {s} is not among the selected tests", .{s.arm_test});
    }
    const lay = Layout.of(&s) catch |e| bad("{s}: {s} (window_mib below the suite's need, or a zero size)", .{ path, @errorName(e) });
    const r = a.create(Rank) catch bad("out of memory", .{});
    var nonce: [8]u8 = undefined;
    io.random(&nonce);
    r.* = .{ .s = s, .rank = s.rank, .peer = 1 - s.rank, .lay = lay, .out = out, .tests = tests.items, .pair = pair, .link = .{s.links[0]}, .nonce = std.mem.readInt(u64, &nonce, .little) };
    if (mtu) |m| r.link[0].mtu = m;
    return r;
}

fn header(r: *Rank, mode: []const u8) void {
    var host: [256]u8 = @splat(0);
    _ = std.c.gethostname(&host, host.len - 1);
    const ln = r.link[0];
    r.out.line("TF-MCDMA-PAIR v=2 rank={d} ranks=2 mode={s} label={s} host={s} os={s} arch={s} pid={d} library={s}", .{ r.rank, mode, if (r.s.label.len > 0) r.s.label else "-", std.mem.sliceTo(&host, 0), @tagName(builtin.os.tag), @tagName(builtin.cpu.arch), std.c.getpid(), r.s.library });
    r.out.line("CONFIG device={s} via={s} port={d} peer_port={d} gid={d} mtu={d} kind={s} expect_kind={s} name={s} window={d} staging={d} rounds={d} duration_s={d} arm={s} arm_rank={d} arm_test={s} arm_step={d} wait_s={d} timeout_s={d} watchdog_s={d} connect_timeout_s={d} expect={s} expect_within_s={d}", .{ ln.device, ln.via, ln.port, ln.peer_port, ln.gid, ln.mtu, if (ln.kind) |k| @tagName(k) else "any", if (r.s.expect_kind) |k| @tagName(k) else "-", ln.name, r.lay.window, r.lay.staging, r.s.rounds, r.s.duration_s, @tagName(r.s.arm), r.s.arm_rank, r.s.arm_test, r.s.arm_step, r.s.wait_s, r.s.timeout_s, r.s.watchdog_s, r.s.connect_timeout_s, expectText(r), r.s.expect_within_s });
    for (r.s.expect_log) |t| r.out.line("EXPECT-LOG {s}", .{t});
    for (r.s.refuse_log) |t| r.out.line("REFUSE-LOG {s}", .{t});
}

fn pairThread(r: *Rank, code: *Status) void {
    header(r, "pair");
    code.* = r.run();
    writeExit(r.rank, code.*); // now: a watchdog that fires on the other rank exits the process
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h")) usage();
    if (std.mem.eql(u8, args[1], "--pair")) {
        if (args.len < 5) usage();
        harness.pair_dir = args[2];
        var ranks: [2]*Rank = undefined;
        for (0..2) |i| {
            var path: [1024]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&path, "{s}/rank{d}.log", .{ harness.pair_dir, i }, 0) catch bad("OUTDIR too long", .{});
            const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o644));
            if (fd < 0) bad("cannot create {s}", .{p});
            ranks[i] = load(a, init.io, args[3 + i], args[5..], .{ .fd = fd }, true);
        }
        if (ranks[0].rank != 0 or ranks[1].rank != 1) bad("--pair takes rank 0's settings, then rank 1's", .{});
        var codes: [2]Status = .{ .pass, .pass };
        const t = std.Thread.spawn(.{}, pairThread, .{ ranks[1], &codes[1] }) catch bad("cannot start rank 1's thread", .{});
        pairThread(ranks[0], &codes[0]);
        t.join();
        var worst: u8 = 0;
        for (codes, 0..) |c, i| {
            std.debug.print("rank {d}: exit {d}\n", .{ i, @backingInt(c) });
            worst = @max(worst, @backingInt(c));
        }
        std.process.exit(worst);
    }
    const r = load(a, init.io, args[1], args[2..], .{ .fd = 1 }, false);
    header(r, "process");
    std.process.exit(@backingInt(r.run()));
}

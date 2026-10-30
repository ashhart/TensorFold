//! Separate processes use the production key builder with real parsed config and unchanged checkpoint files.
const std = @import("std");
const q = @import("qwen35");
const adapter = q.native.cost_adapter;
const Driver = struct {
    release: i32 = 13000,
    pub fn version(d: Driver) !i32 {
        return d.release;
    }
};
const Context = struct {
    d: Driver = .{},
    pub fn name(_: Context, _: []u8) ![]const u8 {
        return "example GPU";
    }
    pub fn capability(_: Context) !u32 {
        return 121;
    }
    pub fn attribute(_: Context, _: enum { multiprocessor_count }) !i32 {
        return 48;
    }
};
const Draft = struct {
    c: q.draft.Config,
    pub fn drafter(d: *Draft) *Draft {
        return d;
    }
    pub fn facts(_: *Draft) struct { trees: bool, tree_nodes: u32 } {
        return .{ .trees = true, .tree_nodes = 32 };
    }
};
const Mtp = struct {
    depth: usize = 7,
    trees: bool = false,
    capacity: usize = 4096,
    pub fn drafter(d: *Mtp) *Mtp {
        return d;
    }
    pub fn facts(d: *Mtp) struct { trees: bool, tree_nodes: u32 } {
        return .{ .trees = d.trees, .tree_nodes = 0 };
    }
};
const Lanes = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *q.engine.Engine,
    cost_context: ?*Context,
    cost_identity: ?[32]u8,
    cost_kind: adapter.Kind,
};
fn probe(init: std.process.Init, root: []const u8, kind: []const u8, release: []const u8) !void {
    const a = init.arena.allocator();
    const target = try std.fs.path.join(a, &.{ root, "target" });
    const eight = std.mem.startsWith(u8, kind, "dflash8");
    const draft = try std.fs.path.join(a, &.{ root, if (eight) "draft8" else "draft" });
    var context: Context = .{ .d = .{ .release = try std.fmt.parseInt(i32, release, 10) } };
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.dir = target;
    e.learning = null;
    e.features = 0;
    e.pockets.layers = &.{};
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    e.kv8 = false;
    e.verify_rows = 128;
    e.window_rows = 64;
    e.n_taps = if (eight) 8 else 2;
    e.taps[0..8].* = .{ 10, 20, 21, 22, 23, 24, 25, 26 };
    const noise = if (@import("builtin").os.tag == .windows) std.os.windows.GetCurrentProcessId() else @as(u32, @intCast(std.c.getpid()));
    // The inactive tail has no tensor or kernel consumer, even when old stack bytes differ between processes.
    for (e.g.linear[e.g.layers..]) |*v| v.* = noise & 1 == 1;
    for (e.g.bits.proj[e.g.layers..]) |*row| @memset(row, @intCast(2 + noise % 7));
    var l: Lanes = .{ .gpa = a, .io = init.io, .e = &e, .cost_context = &context, .cost_identity = try adapter.file_identity.checkpoint(a, init.io, draft), .cost_kind = if (std.mem.eql(u8, kind, "mtp")) .mtp else .dflash };
    var cache: q.engine.Cache = undefined;
    cache.cap = 4096;
    const keys = if (l.cost_kind == .dflash) blk: {
        const config = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ draft, "config.json" }), a, .limited(1 << 20));
        var d: Draft = .{ .c = try q.draft.Config.parse(a, config, e.g) };
        @memset(d.c.taps[d.c.n_taps..], noise);
        if (std.mem.endsWith(u8, kind, "-live")) d.c.taps[0] ^= 1;
        break :blk try adapter.identity(&l, &d, &cache);
    } else blk: {
        var d: Mtp = .{};
        break :blk try adapter.identity(&l, &d, &cache);
    };
    var out: [130]u8 = undefined;
    @memcpy(out[0..64], &std.fmt.bytesToHex(keys.current, .lower));
    out[64] = '\n';
    @memcpy(out[65..129], &std.fmt.bytesToHex(keys.reference, .lower));
    out[129] = '\n';
    try std.Io.File.stdout().writeStreamingAll(init.io, &out);
}
fn child(init: std.process.Init, exe: []const u8, root: []const u8, kind: []const u8, release: []const u8, trace: ?[]const u8) !std.process.RunResult {
    var env: std.process.Environ.Map = .init(init.gpa);
    defer env.deinit();
    if (init.environ_map.get("SystemRoot")) |value| try env.put("SystemRoot", value);
    if (trace) |value| try env.put("TF_COST_KEY_TRACE", value);
    return std.process.run(init.gpa, init.io, .{ .argv = &.{ exe, "--probe", root, kind, release }, .environ_map = &env, .stdout_limit = .limited(512), .stderr_limit = .limited(4096) });
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 5 and std.mem.eql(u8, args[1], "--probe")) return probe(init, args[2], args[3], args[4]);
    if (args.len != 2) return error.FixtureDirectoryRequired;
    const exe = try std.process.executablePathAlloc(init.io, a);
    for ([_][]const u8{ "dflash", "dflash8", "mtp" }) |kind| {
        const first = try child(init, exe, args[1], kind, "13000", null);
        defer init.gpa.free(first.stdout);
        defer init.gpa.free(first.stderr);
        const second = try child(init, exe, args[1], kind, "13000", null);
        defer init.gpa.free(second.stdout);
        defer init.gpa.free(second.stderr);
        if (first.term != .exited or first.term.exited != 0 or second.term != .exited or second.term.exited != 0) return error.KeyChildFailed;
        if (first.stdout.len != 130 or !std.mem.eql(u8, first.stdout, second.stdout)) return error.CrossProcessCostKeysDiffer;
        const changed = try child(init, exe, args[1], kind, "13001", null);
        defer init.gpa.free(changed.stdout);
        defer init.gpa.free(changed.stderr);
        if (changed.term != .exited or changed.term.exited != 0) return error.ChangedKeyChildFailed;
        if (std.mem.eql(u8, first.stdout[0..64], changed.stdout[0..64]) or std.mem.eql(u8, first.stdout[65..129], changed.stdout[65..129])) return error.DriverChangeDidNotInvalidate;
        const traced = try child(init, exe, args[1], kind, "13000", "1");
        defer init.gpa.free(traced.stdout);
        defer init.gpa.free(traced.stderr);
        if (traced.term != .exited or traced.term.exited != 0 or !std.mem.eql(u8, first.stdout, traced.stdout)) return error.TraceChangedKey;
        const names = [_][]const u8{ "profile", "gpu", "driver", "geometry", "policy", "target-taps", "draft-kind", "draft-config", "draft-facts", "draft-checkpoint", "target-checkpoint", "executable" };
        var lines = std.mem.tokenizeScalar(u8, traced.stderr, '\n');
        var n: usize = 0;
        while (lines.next()) |line| : (n += 1) {
            const at = std.mem.indexOf(u8, line, "window cost key part ") orelse return error.UnexpectedTrace;
            var fields = std.mem.tokenizeScalar(u8, line[at + "window cost key part ".len ..], ' ');
            const name = fields.next() orelse return error.MissingTraceName;
            const hash = fields.next() orelse return error.MissingTraceHash;
            if (n >= names.len or !std.mem.eql(u8, name, names[n]) or hash.len != 12 or fields.next() != null) return error.BadTrace;
            for (hash) |c| if (!std.ascii.isHex(c)) return error.BadTrace;
        }
        if (n != names.len or first.stderr.len != 0 or second.stderr.len != 0) return error.IncompleteOrUnrequestedTrace;
        const again = try child(init, exe, args[1], kind, "13000", "1");
        defer init.gpa.free(again.stdout);
        defer init.gpa.free(again.stderr);
        if (again.term != .exited or again.term.exited != 0 or !std.mem.eql(u8, traced.stdout, again.stdout) or !std.mem.eql(u8, traced.stderr, again.stderr)) return error.CrossProcessPartDigestsDiffer;
        if (!std.mem.eql(u8, kind, "mtp")) {
            const live_kind = if (std.mem.eql(u8, kind, "dflash8")) "dflash8-live" else "dflash-live";
            const live = try child(init, exe, args[1], live_kind, "13000", null);
            defer init.gpa.free(live.stdout);
            defer init.gpa.free(live.stderr);
            if (live.term != .exited or live.term.exited != 0) return error.LiveTapChildFailed;
            if (std.mem.eql(u8, first.stdout[0..64], live.stdout[0..64]) or std.mem.eql(u8, first.stdout[65..129], live.stdout[65..129])) return error.LiveTapDidNotInvalidate;
        }

        const disabled = try child(init, exe, args[1], kind, "13000", "0");
        defer init.gpa.free(disabled.stdout);
        defer init.gpa.free(disabled.stderr);
        if (disabled.term != .exited or disabled.term.exited != 0 or disabled.stderr.len != 0 or !std.mem.eql(u8, first.stdout, disabled.stdout)) return error.DisabledTraceChangedKey;
    }
}

//! `check <model dir> [--tp N --rank R] [--truth T --ids IDS] [--only invariants|accuracy|speed]`: a line a check.

const std = @import("std");
const qwen35 = @import("qwen3_5");
const check_accuracy = @import("check_accuracy.zig");
const check_ctx = @import("check_ctx.zig");
const check_invariants = @import("check_invariants.zig");
const check_radix = @import("check_radix.zig");
const check_report = @import("check_report.zig");
const check_speed = @import("check_speed.zig");
const group_mod = @import("group.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");

const Phase = enum { invariants, accuracy, speed };

const Options = struct {
    model: []const u8,
    group: group_mod.Group = .{},
    prompts: ?[]const u8 = null,
    truth: ?[]const u8 = null,
    ids: ?[]const u8 = null,
    speed: bool = false,
    explain: bool = false,
    only: ?Phase = null,
    tokens: u32 = 48,
    bar: check_accuracy.Bar = .{},

    fn runs(o: Options, p: Phase) bool {
        if (o.only) |only| return only == p;
        return switch (p) {
            .invariants => true,
            .accuracy => o.truth != null,
            .speed => o.speed,
        };
    }
};

fn parse(args: []const [:0]const u8) !Options {
    if (args.len < 1) return error.MissingArgument;
    var o: Options = .{ .model = args[0] };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (try o.group.option(args, &i)) continue;
        const a = args[i];
        if (std.mem.eql(u8, a, "--speed")) {
            o.speed = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--explain-kernels")) {
            o.explain = true;
            continue;
        }
        i += 1;
        if (i >= args.len) return error.MissingArgument;
        if (std.mem.eql(u8, a, "--prompts")) o.prompts = args[i] else if (std.mem.eql(u8, a, "--truth")) o.truth = args[i] else if (std.mem.eql(u8, a, "--ids")) o.ids = args[i] else if (std.mem.eql(u8, a, "--only")) o.only = std.meta.stringToEnum(Phase, args[i]) orelse return error.UnknownPhase else if (std.mem.eql(u8, a, "--tokens")) o.tokens = try std.fmt.parseInt(u32, args[i], 10) else if (std.mem.eql(u8, a, "--kl")) o.bar.kl_mean = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, a, "--top1")) o.bar.top1 = try std.fmt.parseFloat(f64, args[i]) else return error.UnknownOption;
    }
    if (o.runs(.accuracy) and o.truth == null and !o.explain) return error.NoTruth;
    return o;
}

/// The ids file the truth was made from, next to it, when none is given.
fn idsPath(arena: std.mem.Allocator, o: Options) !?[]const u8 {
    if (o.ids) |p| return p;
    const t = o.truth orelse return null;
    const stem = if (std.mem.endsWith(u8, t, ".npy")) t[0 .. t.len - 4] else t;
    return try std.fmt.allocPrint(arena, "{s}.ids.npy", .{stem});
}

/// Pages of the largest phase (four streams at the capacity, or the radix streams and tree) and the padding slot's.
fn poolPages(capacity: usize, tokens: u32) usize {
    const pages = qwen35.pages.pagesFor;
    const radix = check_radix.most_streams * pages(check_radix.capacity(tokens)) + check_radix.tree_pages;
    return @max(4 * pages(capacity), radix) + pages(rows);
}

/// Rows a round of the checks holds at most: four streams of the widest window.
const rows = 4 * qwen35.hip_lanes.Hip.max_window;

/// Exit code: 0 when every check passed (or skipped), 1 when one failed.
pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const o = try parse(args);
    const ids = if (try idsPath(arena, o)) |p| try ids_file.load(arena, io, p) else try ids_file.synthetic(arena, 1536);
    const short = try ids_file.short(arena);
    const long = if (o.prompts) |p| try ids_file.read(arena, io, p) else try ids_file.long(arena, ids);

    var capacity = ids_file.longest(long) + 2 * o.tokens + 64;
    capacity = @max(capacity, check_radix.capacity(o.tokens));
    if (o.runs(.accuracy)) capacity = @max(capacity, ids.len + 64);
    if (o.runs(.speed)) capacity = @max(capacity, check_speed.longest_prefill + 64);
    var joined = try o.group.join(io);
    defer joined.close();
    const e = try o.group.engine(gpa, io, o.model, joined, .{ .capacity = capacity, .prompt_rows = capacity, .streams = check_radix.most_streams, .snapshots = check_radix.slots, .batch_rows = rows, .pool_pages = poolPages(capacity, o.tokens) });
    defer e.deinit();
    if (o.group.rank > 0) {
        try group_mod.follow(gpa, e, &joined);
        return 0;
    }
    if (o.explain) {
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try e.lib.affine.reg.explain(e.lib.affine.env(), .mlx, &w, e.weights.spec.bits, e.weights.spec.group);
        std.debug.print("{s}", .{w.buffered()});
        return 0;
    }
    var session: lanes_session.Session = undefined;
    try session.init(gpa, io, e, if (o.group.world > 1) &joined.link else null);
    defer session.deinit();

    var report: check_report.Report = .{};
    var c: check_ctx.Ctx = .{ .gpa = gpa, .io = io, .arena = arena, .e = e, .session = &session, .report = &report, .group = o.group, .ids = ids, .short = short, .long = long, .tokens = o.tokens, .pages_at_start = e.pool.ids.used() };
    const t0 = std.Io.Clock.awake.now(io);
    std.debug.print("check {s}, tp {d}\n", .{ o.model, o.group.world });
    if (o.runs(.invariants)) check_invariants.run(&c);
    if (o.runs(.accuracy)) check_accuracy.run(&c, o.truth.?, o.bar);
    if (o.runs(.speed)) check_speed.run(&c);
    report.summary(@as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9);
    return if (report.failed == 0) 0 else 1;
}

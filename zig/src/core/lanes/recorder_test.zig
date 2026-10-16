//! The recorder in front of the fake target: replies unchanged, and each stream's files hold its committed rows.
const std = @import("std");
const platform = @import("sys");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const rec = @import("recorder.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;
const io = std.testing.io;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    drafts: bool = true,
    sampling: ?Sampling = null,
    logprobs: ?u8 = null,
    restored: u32 = 0,
};

const Mode = struct { mtp: bool = true, pipelined: bool = false, shared: bool = true, chunks: usize = 0, planted: ?[2][]const u8 = null }; // chunks: prefill_step; planted: a link's target and name

var costs: [16]Cost = undefined;

fn model(m: Mode) Model {
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return .{ .exact_width = 16, .gpu_tokens = m.pipelined, .mtp = m.mtp, .speculate = m.mtp, .speculate_early = false, .drafts = if (m.mtp) 4 else 0, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = m.shared };
}

/// A stream's files as the recorder left them.
const Files = struct {
    tok: []u8,
    pos: []u8,
    kind: []u8,
    st: []u8,
    json: []u8,
    top: ?[]u8,

    fn read(a: std.mem.Allocator, base: []const u8) !Files {
        var f: Files = undefined;
        inline for (.{ "tok", "pos", "kind", "st", "json" }) |ext| {
            const path = try std.fmt.allocPrint(a, "{s}.{s}", .{ base, ext });
            @field(f, ext) = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
        }
        const top = try std.fmt.allocPrint(a, "{s}.top", .{base});
        f.top = std.Io.Dir.cwd().readFileAlloc(io, top, a, .limited(1 << 24)) catch null;
        return f;
    }
};

const Summary = struct { rows: u64, first_row: u64, prompt_len: u64, taps: []u32, row_bytes: u32, dtype: []const u8, top_k: u8, gap: ?[]const u8 };

/// The cases run together with or without the recorder; each one's reply, then each one's files checked.
fn run(cases: []const Case, mode: Mode, record: bool) ![][]u32 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
    var cfg = try Config.init(gpa, model(mode), 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = mode.chunks };
    defer target.deinit();
    var recorder = try rec.Recorder.init(gpa, io, dir, &.{ 30, 31 });
    recorder.floor = 0;
    defer recorder.deinit();
    if (mode.planted) |p| recorder.space.symLink(io, p[0], p[1], .{}) catch |e| return if (e == error.PermissionDenied and @import("builtin").os.tag == .windows) error.SkipZigTest else e; // symlinks need privilege on Windows
    const inner = if (mode.chunks > 0) target.stepped() else target.backend();
    var recorded = try rec.Recorded.init(inner, &recorder);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (record) recorded.backend() else inner, clock.clock());
    defer engine.deinit();
    const streams = try a.alloc(sm.Stream, cases.len);
    const proposers = try a.alloc(SuffixLookup, cases.len);
    const kept = try a.alloc(?*anyopaque, cases.len);
    defer for (kept) |k| if (k) |saved| target.drop(saved);
    for (cases, streams, proposers, kept) |c, *s, *p, *k| {
        k.* = null;
        if (c.restored > 0) {
            var early = try sm.Stream.init(gpa, .{ .id = "early", .prompt = c.prompt[0..c.restored], .max_new = 1, .eos = &.{96} });
            defer early.deinit(gpa);
            const b = target.backend();
            try b.prefill(&early);
            k.* = try target.save(&early, c.restored);
            b.release(&early);
        }
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .logprobs = c.logprobs, .reuse = .{ .saved = k.*, .at = c.restored } });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| if (mode.chunks > 0) {
        var first = true;
        while (!try engine.fillStream(s, first)) first = false;
    } else try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    if (record and mode.planted == null) try check(a, try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, recorder.name }), cases, streams, mode);
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

/// Files <n> in the recorder's folder, in admission order: rows from the first computed one to the cache length.
fn check(a: std.mem.Allocator, dir: []const u8, cases: []const Case, streams: []const sm.Stream, mode: Mode) !void {
    for (cases, streams, 0..) |c, *s, n| {
        const f = try Files.read(a, try std.fmt.allocPrint(a, "{s}/{d}", .{ dir, n }));
        const sum = try std.json.parseFromSliceLeaky(Summary, a, f.json, .{});
        try std.testing.expectEqual(@as(u64, c.restored), sum.first_row);
        try std.testing.expectEqualSlices(u32, &.{ 30, 31 }, sum.taps);
        try std.testing.expectEqualStrings("u32", sum.dtype);
        const end = if (mode.pipelined and !(mode.mtp and c.drafts) and c.logprobs == null) s.prompt_len else s.cache_len;
        if (end != s.cache_len) try std.testing.expectEqualStrings("a one-token round", sum.gap.?) else try std.testing.expect(sum.gap == null);
        try std.testing.expectEqual(end - c.restored, sum.rows);
        const want = s.context.items[c.restored..end];
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), f.tok);
        try std.testing.expectEqualSlices(u8, f.tok, f.st); // the fake's row state is its token
        for (std.mem.bytesAsSlice(u32, f.pos), f.kind, c.restored..) |p, k, row| {
            try std.testing.expectEqual(@as(u32, @intCast(row)), p);
            try std.testing.expectEqual(@intFromBool(row >= s.prompt_len), k);
        }
        if (c.logprobs) |k| {
            try std.testing.expectEqual(k, sum.top_k);
            const top = f.top.?;
            try std.testing.expectEqual(s.rows.items.len * k * 8, top.len);
            for (s.rows.items, 0..) |row, i| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(row.ids[0..k]), top[i * k * 8 ..][0 .. k * 4]);
        } else try std.testing.expect(f.top == null);
    }
}

fn expectSame(cases: []const Case, mode: Mode) !void {
    const plain = try run(cases, mode, false);
    defer free(plain);
    const recorded = try run(cases, mode, true);
    defer free(recorded);
    for (plain, recorded) |p, r| try std.testing.expectEqualSlices(u32, p, r);
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const sampled: Sampling = .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 };

test "recorded replies equal unrecorded ones, and a stream's files hold its rows: drafted, plain, greedy and sampled" {
    for ([_]?Sampling{ null, sampled }) |s| {
        try expectSame(&.{.{ .prompt = &p1, .sampling = s }}, .{});
        try expectSame(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, .{});
        try expectSame(&.{.{ .prompt = &p1, .sampling = s }}, .{ .mtp = false });
    }
}

test "shared rounds: each stream's files hold its own rows, batched or not" {
    for ([_]bool{ true, false }) |shared| {
        try expectSame(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled }, .{ .prompt = &long, .drafts = false } }, .{ .shared = shared });
    }
}

test "a restored prompt's files start at the first row the pass computed" {
    try expectSame(&.{.{ .prompt = &long, .restored = 12 }}, .{});
    try expectSame(&.{ .{ .prompt = &long, .restored = 7, .drafts = false }, .{ .prompt = &p2 } }, .{});
}

test "logprob rows land beside the rows, one a reply token" {
    try expectSame(&.{ .{ .prompt = &p1, .logprobs = 5 }, .{ .prompt = &p2, .logprobs = 3, .sampling = sampled } }, .{});
}

test "one-token rounds end a stream's rows after its prompt, and say so" {
    try expectSame(&.{ .{ .prompt = &p1, .drafts = false }, .{ .prompt = &p2, .drafts = false } }, .{ .pipelined = true });
}

test "a prompt filled a chunk a call is recorded when its last chunk completes: one chunk and several" {
    for ([_]usize{ 1, 3 }) |chunks| {
        try expectSame(&.{ .{ .prompt = &p1 }, .{ .prompt = &long, .drafts = false } }, .{ .chunks = chunks });
        try expectSame(&.{.{ .prompt = &long, .restored = 7 }}, .{ .chunks = chunks });
    }
}

test "each recorder makes its own new folder: two in one folder, and a link where one would go, never followed" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var away = std.testing.tmpDir(.{ .iterate = true });
    defer away.cleanup();
    var buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
    const first = try std.fmt.allocPrint(gpa, "{d}-0", .{platform.c.getpid()});
    defer gpa.free(first);
    var abuf: [64]u8 = undefined;
    tmp.dir.symLink(io, try std.fmt.bufPrint(&abuf, "../{s}", .{away.sub_path}), first, .{ .is_directory = true }) catch |e|
        return if (e == error.PermissionDenied and @import("builtin").os.tag == .windows) error.SkipZigTest else e; // symlinks need privilege on Windows
    var one = try rec.Recorder.init(gpa, io, dir, &.{1});
    defer one.deinit();
    var two = try rec.Recorder.init(gpa, io, dir, &.{1});
    defer two.deinit();
    try std.testing.expect(!std.mem.eql(u8, one.name, first) and !std.mem.eql(u8, two.name, first));
    try std.testing.expect(!std.mem.eql(u8, one.name, two.name));
    const f = try one.space.createFile(io, "0.tok", .{ .exclusive = true });
    f.close(io);
    try std.testing.expectError(error.PathAlreadyExists, one.space.createFile(io, "0.tok", .{ .exclusive = true }));
    _ = try two.space.statFile(io, ".", .{});
    try std.testing.expectError(error.FileNotFound, two.space.statFile(io, "0.tok", .{}));
    var it = away.dir.iterate();
    try std.testing.expect(try it.next(io) == null); // nothing went through the link
}

test "a recorder that finds every name taken gives up once, owning its name" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
    for (0..3) |k| {
        var nbuf: [32]u8 = undefined;
        try tmp.dir.createDir(io, try std.fmt.bufPrint(&nbuf, "{d}-{d}", .{ platform.c.getpid(), k }), .default_dir);
    }
    try std.testing.expectError(error.PathAlreadyExists, rec.Recorder.initTries(gpa, io, dir, &.{1}, 3));
    var r = try rec.Recorder.initTries(gpa, io, dir, &.{1}, 4);
    defer r.deinit();
    var nbuf: [32]u8 = undefined;
    try std.testing.expectEqualStrings(try std.fmt.bufPrint(&nbuf, "{d}-3", .{platform.c.getpid()}), r.name);
}

test "a stream's files are only ever new ones: links planted in the recorder's folder are not written through" {
    var away = std.testing.tmpDir(.{ .iterate = true });
    defer away.cleanup();
    var buf: [64]u8 = undefined;
    const target = try std.fmt.bufPrint(&buf, "../../{s}/x", .{away.sub_path}); // from <tmp>/<recorder's folder>
    std.testing.log_level = .err; // the refused files are logged as warnings, as they are meant to be
    defer std.testing.log_level = .warn;
    for ([_][]const u8{ "0.tok", "0.st", "0.json" }) |name| {
        try expectSame(&.{.{ .prompt = &p1 }}, .{ .planted = .{ target, name } });
        var it = away.dir.iterate();
        try std.testing.expect(try it.next(io) == null);
    }
}

test "a target that gives no rows cannot be recorded" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [64]u8 = undefined;
    var recorder = try rec.Recorder.init(gpa, io, try std.fmt.bufPrintSentinel(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0), &.{1});
    defer recorder.deinit();
    var table = target.backend().vtable.*;
    table.features = null;
    try std.testing.expectError(error.NoFeatures, rec.Recorded.init(.{ .ptr = &target, .vtable = &table }, &recorder));
}

//! Host receipts through the real trace glue and rank/window format, with zero-work off controls.
const std = @import("std");
const hash = @import("layer_hash.zig");
const hf = @import("layer_hash_forward.zig");
const arm = @import("serve_arm.zig");
const fwd = @import("forward.zig");
const calls = @import("block_calls.zig");
const mock = @import("load_mock.zig");
const run = @import("run.zig");
const t = std.testing;

const default_arm: arm.Arm = .{ .rank = 1, .world = 2, .gather = "nccl", .hcas = 0, .graphs_requested = false, .graphs_effective = false, .layer_hash = false, .pieces_requested = false, .floor_effective = false, .adaptive_rows_effective = false, .prefill_rows = 2048, .slots = 1, .drafts = true, .sessions = true, .prefill = "full" };

fn parse(a: std.mem.Allocator, bytes: []const u8) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
}

fn read(a: std.mem.Allocator, prefix: []const u8, rank: u32) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(t.io, try std.fmt.allocPrint(a, "{s}.rank{d}.jsonl", .{ prefix, rank }), a, .limited(1 << 20));
}

fn dir(a: std.mem.Allocator, tmp: anytype, stem: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ try tmp.dir.realPathFileAlloc(t.io, ".", a), stem });
}

test "layer hash: startup names sorted set knobs and actual gather, without unrelated environment values" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    defer env.deinit();
    try env.put("PRIVATE_TEST_VALUE", "must stay out");
    try t.expectEqual(@as(usize, 0), (try arm.entries(a, &env)).len);
    const empty = try arm.line(a, default_arm, try arm.entries(a, &env));
    try t.expect(std.mem.indexOf(u8, empty, "TF_DSV41_") == null);
    try env.put("TF_DSV41_PIECES_FLOOR", "1");
    try env.put("TF_DSV41_FUTURE_KNOB", "value\nquoted\"");
    try env.put("TF_ROCE", "0");
    const entries = try arm.entries(a, &env);
    try t.expectEqual(@as(usize, 3), entries.len);
    try t.expectEqualStrings("TF_DSV41_FUTURE_KNOB", entries[0].name);
    var selected = default_arm;
    selected.gather = "roce+nccl";
    selected.hcas = 2;
    selected.graphs_requested = true;
    selected.layer_hash = true;
    const text = try arm.line(a, selected, entries);
    const root = try parse(a, text);
    const got = root.object.get("arm").?.object;
    try t.expectEqualStrings("roce+nccl", got.get("gather").?.string);
    try t.expectEqual(@as(i64, 2), got.get("hcas").?.integer);
    try t.expect(got.get("graphs_requested").?.bool and !got.get("graphs_effective").?.bool);
    try t.expect(std.mem.indexOf(u8, text, "must stay out") == null);
}

test "layer hash: off neither allocates, opens files, synchronizes nor reads device bytes" {
    var fail = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    var f: fwd.Forward = undefined;
    f.gpa = fail.allocator();
    f.trace = null;
    f.layer_hash = null;
    f.opts = .{};
    try hf.attach(&f, t.io, null, 0);
    try hf.attach(&f, t.io, "", 0);
    try hf.begin(&f, 0, &.{ 1, 2 });
    var runner: run.Runner = undefined;
    var call: calls.Call = undefined;
    try hf.trace(&f, &runner, &call, .decode);
    try t.expect(f.layer_hash == null and !f.opts.trace);
    try t.expectEqual(@as(usize, 0), fail.allocated_bytes);
}

fn make(a: std.mem.Allocator, prefix: []const u8, rank: u32, flip: bool) !void {
    var d = mock.init();
    var r: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer r.deinit();
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &r, undefined);
    defer f.arena.deinit();
    var legacy: std.AutoArrayHashMapUnmanaged(u32, [64]u8) = .empty;
    defer legacy.deinit(t.allocator);
    f.trace = &legacy;
    try hf.attach(&f, t.io, prefix, rank);
    defer f.layer_hash.?.close();
    try t.expect(f.opts.trace);
    const req: hash.Request = if (rank == 0) try f.layer_hash.?.begin(0, &.{ 10, 20 }) else .{ .sequence = 1, .sha256 = hash.tokenSha(&.{ 10, 20 }) };
    if (rank != 0) try hf.follow(&f, &hash.frame(0, req));
    const words = hash.frame(0, req);
    const follower = try hash.readFrame(&words);
    try t.expectEqualDeep(req, follower.request);
    f.ids = &.{ 10, 20 };
    f.slot.pending = .{ .start = 7, .n = 2 };
    try r.external("w.trace", 256);
    const tensor: calls.Tensor = .{ .role = .{ .buf = "w.trace" }, .dt = .bf16, .shape = try a.dupe(i64, &.{ 2, 4 }), .stride = &.{ 4, 1 } };
    for (0..3) |L| {
        @memset(mock.memory[1][0..16], @intCast(L));
        if (flip and L == 1) mock.memory[1][7] ^= 1;
        const args = [_]calls.Named{ .{ .name = "0", .arg = .{ .t = tensor } }, .{ .name = "1", .arg = .{ .i = @intCast(L) } } };
        const c: calls.Call = .{ .triton = false, .glue = true, .name = "glue.trace", .args = &args };
        try @import("forward_glue.zig").glueFn(&f, &r, &c);
        var expected: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(mock.memory[1][0..16], &expected, .{});
        try t.expectEqualStrings(&std.fmt.bytesToHex(expected, .lower), &legacy.get(@intCast(L)).?);
    }
}

test "layer hash: two equal runs of the real glue give equal files; one flipped bit identifies its layer" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const first = try dir(a, tmp, "first");
    const again = try dir(a, tmp, "again");
    const changed = try dir(a, tmp, "changed");
    for (0..2) |rank| {
        try make(a, first, @intCast(rank), false);
        try make(a, again, @intCast(rank), false);
        try make(a, changed, @intCast(rank), true);
        const bytes = try read(a, first, @intCast(rank));
        try t.expectEqualStrings(bytes, try read(a, again, @intCast(rank)));
        var old = std.mem.tokenizeScalar(u8, bytes, '\n');
        var new = std.mem.tokenizeScalar(u8, try read(a, changed, @intCast(rank)), '\n');
        var mismatches: u32 = 0;
        var layers: u32 = 0;
        while (old.next()) |line| {
            const before = try parse(a, line);
            const after = try parse(a, new.next().?);
            const o = before.object;
            if (!std.mem.eql(u8, o.get("kind").?.string, "layer")) continue;
            layers += 1;
            try t.expectEqual(@as(i64, @intCast(rank)), o.get("rank").?.integer);
            try t.expectEqual(@as(i64, 1), o.get("request").?.integer);
            try t.expectEqual(@as(i64, 1), o.get("window").?.integer);
            try t.expectEqualStrings("decode", o.get("phase").?.string);
            try t.expectEqual(@as(usize, 64), o.get("request_sha256").?.string.len);
            const different = !std.mem.eql(u8, o.get("sha256").?.string, after.object.get("sha256").?.string);
            if (different) {
                mismatches += 1;
                try t.expectEqual(@as(i64, 1), o.get("layer").?.integer);
            }
        }
        try t.expectEqual(@as(u32, 3), layers);
        try t.expectEqual(@as(u32, 1), mismatches);
        try t.expect(new.next() == null);
    }
}

test "layer hash: files are exclusive, malformed frames fail and interleaved windows retain their identifiers" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const prefix = try dir(a, tmp, "trace");
    const w = (try hash.Writer.open(t.allocator, t.io, prefix, 0)).?;
    defer w.close();
    _ = try w.begin(0, &.{ 1, 2 });
    try t.expectError(error.PathAlreadyExists, hash.Writer.open(t.allocator, t.io, prefix, 0));
    try t.expectError(error.BadHashRequest, hash.readFrame(&.{ hash.op_request, 0 }));
    const req: hash.Request = .{ .sequence = 1, .sha256 = hash.tokenSha(&.{1}) };
    var frame = hash.frame(0, req);
    frame[3] = 'z';
    try t.expectError(error.BadHashRequest, hash.readFrame(&frame));
    const bytes = [_]u8{ 1, 2, 3, 4 };
    const one: hash.Span = .{ .slot = 0, .start = 0, .row0 = 0, .rows = 1, .ids = &.{1} };
    var two = one;
    two.start = 1;
    two.ids = &.{2};
    for (0..3) |L| {
        try w.layer(.prefill, @intCast(L), one, 2, &bytes);
        try w.layer(.prefill, @intCast(L), two, 2, &bytes);
    }
    var lines = std.mem.tokenizeScalar(u8, try read(a, prefix, 0), '\n');
    var count: u32 = 0;
    while (lines.next()) |line| {
        const v = (try parse(a, line)).object;
        if (!std.mem.eql(u8, v.get("kind").?.string, "layer")) continue;
        try t.expectEqual(@as(i64, 1 + count % 2), v.get("window").?.integer);
        count += 1;
    }
    try t.expectEqual(@as(u32, 6), count);
}

test "layer hash: no path leaves actual decode and prefill emissions identical" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = @import("block.zig");
    var cfg: @import("config.zig").Config = .{};
    var widths: block.Widths = .{};
    const check = @import("calls_check.zig");
    try check.parseDense(a, &widths, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &widths, @embedFile("fixtures/expert-k2.txt"));
    for (0..2) |L| try widths.gm.put(a, @intCast(L), .{ (1 << 4) | (1 << 6), (1 << 4) | (1 << 8) });
    var runner: run.Runner = undefined;
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{ .world = 2, .r1 = true }, &runner, undefined);
    defer f.arena.deinit();
    const before = try block.emit(a, &cfg, &widths, f.opts, &.{ 0, 1 }, 2, 7, false);
    const pf_before = try @import("block_prefill.zig").emitPrefill(a, &cfg, &widths, f.opts, &.{ 0, 1 }, 32, 0, false);
    try hf.attach(&f, t.io, null, 0);
    const after = try block.emit(a, &cfg, &widths, f.opts, &.{ 0, 1 }, 2, 7, false);
    const pf_after = try @import("block_prefill.zig").emitPrefill(a, &cfg, &widths, f.opts, &.{ 0, 1 }, 32, 0, false);
    try t.expectEqualDeep(before, after);
    try t.expectEqualDeep(pf_before, pf_after);
    for (after) |c| try t.expect(!std.mem.eql(u8, c.name, "glue.trace"));
    f.opts.trace = true;
    const traced = try block.emit(a, &cfg, &widths, f.opts, &.{ 0, 1 }, 2, 7, false);
    var count: u32 = 0;
    for (traced) |c| count += @intFromBool(std.mem.eql(u8, c.name, "glue.trace"));
    try t.expectEqual(@as(u32, 2), count);
}

test "layer hash: batched and prefill spans retain request IDs and exclude padded rows" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const prefix = try dir(a, tmp, "spans");
    var d = mock.init();
    var r: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer r.deinit();
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &r, undefined);
    defer f.arena.deinit();
    try hf.attach(&f, t.io, prefix, 0);
    defer f.layer_hash.?.close();
    _ = try f.layer_hash.?.begin(0, &.{10});
    _ = try f.layer_hash.?.begin(1, &.{20});
    try r.external("w.trace", 256);
    @memset(mock.memory[1][0..8], 1);
    @memset(mock.memory[1][8..16], 2);
    @memset(mock.memory[1][16..24], 255);
    const tensor: calls.Tensor = .{ .role = .{ .buf = "w.trace" }, .dt = .bf16, .shape = &.{ 3, 4 }, .stride = &.{ 4, 1 } };
    const args = [_]calls.Named{ .{ .name = "0", .arg = .{ .t = tensor } }, .{ .name = "1", .arg = .{ .i = 0 } } };
    const c: calls.Call = .{ .triton = false, .glue = true, .name = "glue.trace", .args = &args };
    const mix = [_]@import("rowtab.zig").Seg{ .{ .slot = 0, .start = 3, .rows = 1 }, .{ .slot = 1, .start = 7, .rows = 1 } };
    try hf.batch(&f, &r, &c, &mix, &.{ 10, 20 });
    const segs = [_]@import("forward_types.zig").MultiSeg{ .{ .slot = 0, .start = 3, .ids = &.{10} }, .{ .slot = 1, .start = 7, .ids = &.{20} } };
    var multi: @import("forward_types.zig").Multi = .{ .segs = &segs, .row0 = &.{ 0, 1 }, .tails = &.{}, .ids = &.{ 10, 20 } };
    f.multi = &multi;
    try hf.trace(&f, &r, &c, .prefill);
    var lines = std.mem.tokenizeScalar(u8, try read(a, prefix, 0), '\n');
    var count: usize = 0;
    while (lines.next()) |line| {
        const v = (try parse(a, line)).object;
        if (!std.mem.eql(u8, v.get("kind").?.string, "layer")) continue;
        const slot: usize = @intCast(v.get("slot").?.integer);
        try t.expectEqual(@as(i64, @intCast(slot + 1)), v.get("request").?.integer);
        try t.expectEqual(@as(i64, if (slot == 0) 3 else 7), v.get("start").?.integer);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(mock.memory[1][8 * slot ..][0..8], &sha, .{});
        try t.expectEqualStrings(&std.fmt.bytesToHex(sha, .lower), v.get("sha256").?.string);
        count += 1;
    }
    try t.expectEqual(@as(usize, 4), count);
}

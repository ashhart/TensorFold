//! The prompt cache under CUDA rules: no fork marks unless a family opts in, and where they sit when it does.
const std = @import("std");
const api = @import("engine_api");
const cacheRules = @import("cuda_snaps.zig").cacheRules;
const warmQuiet = @import("cuda_snaps.zig").warmQuiet;

/// Copies that hold nothing: a store keys its states by tokens and position, all that its plans read.
const Held = struct {
    var one: u8 = 0;
    const vtable: api.prompt_cache.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
        return &one;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
    fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
};

const Asked = struct { from: [3]u32 = .{ 0, 0, 0 }, marks: [3][]const u32 = undefined, kept: []u32 = &.{} };

/// Three questions of `q` tokens on one document under `F`'s rules: where each resumes, its marks, every state kept.
fn questions(comptime F: type, a: std.mem.Allocator, sys: u32, doc: u32, q: u32, rewind: bool) !Asked {
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &Held.one, .vtable = &Held.vtable }, cacheRules(F), 1 << 30);
    defer store.deinit();
    var asked: Asked = .{};
    for (0..3) |i| {
        const prompt = try a.alloc(u32, sys + doc + q + 2); // the question, then the template's two-token tail
        for (prompt, 0..) |*t, j| t.* = @intCast(if (j >= sys + doc and j < sys + doc + q) 1_000_000 * (i + 1) + j else j + 1);
        const p = try store.beginRewind(a, prompt, sys + doc + q, if (rewind) sys else 0, &.{}, &.{}, null, &.{});
        for (p.marks) |m| try std.testing.expect(store.keep(prompt, m, null, &.{}, &.{}));
        asked.from[i] = p.from;
        asked.marks[i] = p.marks;
    }
    asked.kept = try a.alloc(u32, store.entries.items.len);
    for (asked.kept, store.entries.items) |*at, e| at.* = e.at;
    std.mem.sort(u32, asked.kept, {}, std.sort.asc(u32));
    return asked;
}

test "snap fork: without it a family keeps the states it kept before the rule, the stable mark at the part" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Other = struct {};
    const Grid = struct {
        pub const snap_grid: u32 = 2048;
    };
    try std.testing.expect(!cacheRules(Other).fork and !cacheRules(Grid).fork);
    // A document re-sent with a new question: each pass misses and keeps only its history's mark.
    const missed = try questions(Other, a, 4, 1000, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, &missed.from);
    for (missed.marks) |m| try std.testing.expectEqualSlices(u32, &.{1006}, m);
    try std.testing.expectEqualSlices(u32, &.{ 1006, 1006, 1006 }, missed.kept);
    // Resuming the rewind state, the stable mark sits at the part, keyed with the question's first token.
    const resumed = try questions(Other, a, 300, 1000, 300, true);
    try std.testing.expectEqualSlices(u32, &.{ 0, 299, 299 }, &resumed.from);
    try std.testing.expectEqualSlices(u32, &.{ 299, 1600 }, resumed.marks[0]);
    for (resumed.marks[1..]) |m| try std.testing.expectEqualSlices(u32, &.{ 1300, 1600 }, m);
    try std.testing.expectEqualSlices(u32, &.{ 299, 1300, 1300, 1600, 1600, 1600 }, resumed.kept);
}

test "snap fork: a family that opts in keeps where the questions part, and the next question resumes there" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Fork = struct {
        pub const snap_fork = true;
    };
    const GridFork = struct {
        pub const snap_grid: u32 = 4;
        pub const snap_fork = true;
    };
    try std.testing.expect(cacheRules(Fork).fork and cacheRules(GridFork).fork);
    // A miss keeps one before the part, 3 tokens from the history's mark (under min_gap), and the next resumes it.
    const missed = try questions(Fork, a, 4, 1000, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 1003, 1006 }, missed.marks[1]);
    try std.testing.expectEqual(@as(u32, 1003), missed.from[2]);
    // Resuming the rewind state, the stable mark sits one before the part, keyed with the document alone.
    const resumed = try questions(Fork, a, 300, 1000, 300, true);
    try std.testing.expectEqualSlices(u32, &.{ 1299, 1600 }, resumed.marks[1]);
    try std.testing.expectEqual(@as(u32, 1299), resumed.from[2]);
    // A planned family keeps the fork on its grid start (16), however near the history's (20).
    const planned = try questions(GridFork, a, 4, 16, 2, false);
    try std.testing.expectEqualSlices(u32, &.{ 16, 20 }, planned.marks[1]);
    try std.testing.expectEqual(@as(u32, 16), planned.from[2]);
}

test "CUDA warm rules: a warm end stays usable only where a family declares it, and its own wait comes through" {
    const Both = struct {
        pub const warm_quiet: u64 = 7;
        pub const warm_usable = true;
    };
    const Ends = struct {
        pub const warm_quiet: u64 = 7;
    };
    try std.testing.expectEqual(@as(u64, 7), warmQuiet(Both));
    try std.testing.expectEqual(@as(u64, 7), warmQuiet(Ends));
    try std.testing.expectEqual(@as(u64, 0), warmQuiet(struct {}));
    // A family's wait and the rule that follows it come from one declaration: a nonzero wait is a waiting family.
    inline for (@import("cuda.zig").registry) |F| {
        try std.testing.expectEqual(warmQuiet(F) > 0, cacheRules(F).warm == .waits);
    }
    try std.testing.expect(cacheRules(Both).warm_usable);
    try std.testing.expect(!cacheRules(Ends).warm_usable);
}

test "snap hooks: a family's spare, trim and reuses reach the prompt cache, and the Qwen entries declare all three" {
    const Snaps = @import("cuda_snaps.zig").Snaps;
    const Plain = struct {
        pub const snap_bytes = Held.bytes;
        pub const snap_save = Held.save;
        pub const snap_restore = Held.restore;
        pub const snap_drop = Held.drop;
    };
    const Pooled = struct {
        pub const snap_bytes = Held.bytes;
        pub const snap_save = Held.save;
        pub const snap_restore = Held.restore;
        pub const snap_drop = Held.drop;
        pub fn snap_spare(_: *anyopaque) u64 {
            return 7;
        }
        pub fn snap_trim(_: *anyopaque, _: u64) void {}
        pub fn snap_reuses(_: *anyopaque, _: u32) bool {
            return true;
        }
    };
    const plain = Snaps(Plain, null).vtable;
    try std.testing.expect(plain.spare == null and plain.trim == null and plain.reuses == null);
    const pooled = Snaps(Pooled, null).vtable;
    try std.testing.expect(pooled.spare.? == &Pooled.snap_spare and pooled.trim.? == &Pooled.snap_trim and pooled.reuses.? == &Pooled.snap_reuses);
    var qwen: usize = 0;
    inline for (@import("cuda.zig").registry) |F| if (comptime std.mem.startsWith(u8, F.model_type, "qwen3_5") or std.mem.eql(u8, F.model_type, "sensenova_loop")) {
        const v = Snaps(F, null).vtable;
        try std.testing.expect(v.spare != null and v.trim != null and v.reuses != null);
        qwen += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), qwen); // dense, MoE and Loop entry points
}

test "request starts: snap_cut_invariant: message starts are on by default for invariant families, opt-in for the rest" {
    const snaps = @import("cuda_snaps.zig");
    // Nemotron/LFM-style: chunk-invariant, no switch; the step is prompt_rows or a declared snap_grid.
    const Nemotron = struct {
        pub const snap_cut_invariant = true;
        pub const prompt_rows: u32 = 2048;
    };
    const Lfm = struct {
        pub const snap_cut_invariant = true;
        pub const prompt_rows: u32 = 512;
    };
    const Planned = struct {
        pub const snap_cut_invariant = true;
        pub const snap_grid: u32 = 4096;
    };
    // Qwen/GLM-style: no invariant declaration, so the old opt-in contract holds.
    const Qwen = struct {
        pub const snap_grid: u32 = 4096;
        pub const request_starts = true;
        pub const request_starts_switch = "TF_SNAPS_TEST_QWEN_STARTS";
    };
    const Glm = struct {
        pub const snap_grid: u32 = 2048;
        pub const request_starts = true;
        pub const request_min_chunk: u32 = 256;
        pub const request_starts_switch = "TF_SNAPS_TEST_GLM_STARTS";
    };
    // Invariant families: starts on with no switch, every start, cache cuts at request starts.
    try std.testing.expectEqual(@as(u32, 2048), snaps.requestStep(Nemotron));
    try std.testing.expectEqual(@as(u32, 512), snaps.requestStep(Lfm));
    try std.testing.expectEqual(@as(u32, 4096), snaps.requestStep(Planned));
    try std.testing.expectEqual(@as(u32, 1), snaps.requestMinimum(Nemotron));
    try std.testing.expect(snaps.passRules(Nemotron, 2048).chunk_starts and snaps.passRules(Planned, 4096).chunk_starts);
    // Qwen/GLM: still off without their switch, 256 apart, cache ignores request starts.
    try std.testing.expectEqual(@as(u32, 0), snaps.requestStep(Qwen));
    try std.testing.expectEqual(@as(u32, 0), snaps.requestStep(Glm));
    try std.testing.expectEqual(@as(u32, 256), snaps.requestMinimum(Qwen));
    try std.testing.expect(!snaps.passRules(Qwen, 4096).chunk_starts and !snaps.passRules(Glm, 2048).chunk_starts);
    // A store under invariant rules keeps its state at the message start the server now sends.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &Held.one, .vtable = &Held.vtable }, snaps.passRules(Planned, 4096), 1 << 30);
    defer store.deinit();
    const prompt = try a.alloc(u32, 12000);
    for (prompt, 0..) |*t, j| t.* = @intCast(j + 1);
    const marks = try store.marks(a, prompt, 0, 10000, &.{}, &.{6000}, &.{});
    try std.testing.expectEqualSlices(u32, &.{6000}, marks);
}

test "Qwen prompt rows: changing reported chunk cannot change planned cache marks, interior lookup or warm usability" {
    const F = @import("qwen35").native;
    const snaps = @import("cuda_snaps.zig");
    const pc = api.prompt_cache;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 32768);
    for (prompt, 0..) |*value, i| value.* = @intCast(i);
    const old = snaps.passRules(F, 8192);
    const corrected = snaps.passRules(F, 4096);
    try std.testing.expect(old.planned and corrected.planned and old.grid == 4096 and old.min_prompt == 0);
    var same = corrected;
    same.chunk = old.chunk;
    try std.testing.expectEqualDeep(old, same);
    var before = pc.Store.init(a, .{ .ptr = &Held.one, .vtable = &Held.vtable }, old, 1 << 30);
    defer before.deinit();
    var after = pc.Store.init(a, .{ .ptr = &Held.one, .vtable = &Held.vtable }, corrected, 1 << 30);
    defer after.deinit();
    const named = [_]u32{ 4096, 12288, 16384, 28672 };
    for ([_][]const u32{ &.{}, &named }) |starts| {
        for ([_]u32{ 0, 4095, 4096, 15000, 32768 }) |history| {
            try std.testing.expectEqualSlices(u32, try before.marks(a, prompt, 0, history, &.{}, starts, &.{}), try after.marks(a, prompt, 0, history, &.{}, starts, &.{}));
            const left = try before.lookupUser(a, prompt, @intCast(prompt.len), history, 0, 30000, &.{}, starts, &.{});
            const right = try after.lookupUser(a, prompt, @intCast(prompt.len), history, 0, 30000, &.{}, starts, &.{});
            try std.testing.expect(left.entry == null and right.entry == null);
            try std.testing.expectEqualDeep(left.miss, right.miss);
            try std.testing.expectEqualSlices(u32, left.marks, right.marks);
        }
        for ([_]u32{ 1, 4095, 4096, 6000, 8192, 12288, 32768 }) |at|
            try std.testing.expectEqual(before.usable(at, starts), after.usable(at, starts));
    }
}

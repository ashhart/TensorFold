//! TF_KV_SHARE on Metal (families/nemotron/prompt_pages.zig) over a recording store: who resumes and who keeps.
const std = @import("std");
const lanes = @import("lanes");
const kv = @import("core/kv_pool.zig");
const hooks = @import("core/kv_prompts.zig");
const Pass = @import("families/nemotron/prompt_pages.zig").Pass;
const t = std.testing;
const gpa = t.allocator;

/// One kept state of one prompt, its pages held by the store; `begin` hands them on as the real store does.
const Store = struct {
    pool: *kv.Pool,
    backend: *Backend,
    key: []const u32 = &.{},
    at: u32 = 0,
    pages: [2]u32 = undefined,
    begins: usize = 0,
    keeps: usize = 0,
    reports: usize = 0,

    fn of(ctx: *anyopaque) *Store {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(ctx: *anyopaque, q: hooks.Query, _: u32, _: []const u32, _: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!hooks.Plan {
        const x = of(ctx);
        x.begins += 1;
        const marks = try gpa.dupe(u32, &.{256});
        if (x.at == 0 or !std.mem.startsWith(u32, q.key, x.key)) return .{ .marks = marks };
        for (x.pages) |id| x.pool.retain(id); // held once more for the stream
        try adopt.appendSlice(gpa, &x.pages);
        x.backend.pending = x; // the state the stream's pass restores
        return .{ .from = x.at, .marks = marks };
    }
    fn keep(ctx: *anyopaque, q: hooks.Query, at: u32, _: *anyopaque, mine: []const u32, path: []u32) usize {
        const x = of(ctx);
        x.keeps += 1;
        x.key = q.key[0..at];
        x.at = at;
        for (mine, 0..) |id, i| {
            x.pool.retain(id);
            x.pages[i] = id;
        }
        @memcpy(path, mine);
        return mine.len;
    }
    fn report(ctx: *anyopaque, _: usize, _: u32, _: u64) void {
        of(ctx).reports += 1;
    }
    fn free(_: *anyopaque, marks: []const u32) void {
        gpa.free(marks);
    }
    const vtable: hooks.Prompts.VTable = .{ .begin = begin, .keep = keep, .report = report, .free = free };
};

/// What Pass reads of the Metal backend: the store, its pending resume and the pool its pages go back to.
const Backend = struct {
    gpa: std.mem.Allocator = gpa,
    prompts: ?hooks.Prompts = null,
    pending: ?*const anyopaque = null,
    kv_pages: ?*Pages,
};

const Pages = struct { pool: kv.Pool };

test "TF_KV_SHARE Metal: an untaught stream neither resumes from nor keeps a taught prompt's pages and state" {
    var pages: Pages = .{ .pool = try kv.Pool.init(gpa, 8, 1) };
    defer pages.pool.deinit();
    var b: Backend = .{ .kv_pages = &pages };
    var store: Store = .{ .pool = &pages.pool, .backend = &b };
    b.prompts = .{ .ctx = &store, .vtable = &Store.vtable };
    const ids: [300]u32 = @splat(7);
    // a taught pass keeps its first 256 rows: two of its lane's pages and its state
    var taught = try lanes.Stream.init(gpa, .{ .id = "taught", .prompt = &ids, .max_new = 1 });
    defer taught.deinit(gpa);
    var p = try Pass.begin(&b, &taught);
    try t.expect(p.on);
    try t.expectEqualSlices(u32, &.{256}, p.marks);
    var lane: [2]u32 = undefined;
    try t.expect(pages.pool.hold(&lane));
    var path: [2]u32 = undefined;
    try t.expectEqual(@as(usize, 2), b.prompts.?.keep(.of(taught.prompt()), 256, &taught, &lane, &path));
    p.end(&b, &taught, true);
    for (lane) |id| pages.pool.release(id); // the taught stream ends: the store's hold remains
    try t.expectEqual(@as(usize, 1), store.reports);
    // the same prompt untaught: no resume from the taught state, no pages adopted, no marks to keep at
    var plain = try lanes.Stream.init(gpa, .{ .id = "untaught", .prompt = &ids, .max_new = 1, .untaught = true });
    defer plain.deinit(gpa);
    var u = try Pass.begin(&b, &plain);
    try t.expect(!u.on);
    try t.expectEqual(@as(u32, 0), u.from);
    try t.expectEqual(@as(usize, 0), u.held());
    try t.expectEqual(@as(usize, 0), u.marks.len);
    try t.expectEqual(@as(?*const anyopaque, null), b.pending);
    u.end(&b, &plain, true);
    try t.expectEqual(@as(usize, 1), store.begins);
    try t.expectEqual(@as(usize, 1), store.keeps);
    try t.expectEqual(@as(usize, 1), store.reports);
    for (lane) |id| try t.expectEqual(@as(u32, 1), pages.pool.holders(id)); // the store's alone
    // control: a taught stream of that prompt does resume there, its adopted pages back when its pass fails
    var again = try lanes.Stream.init(gpa, .{ .id = "again", .prompt = &ids, .max_new = 1 });
    defer again.deinit(gpa);
    var c = try Pass.begin(&b, &again);
    try t.expectEqual(@as(u32, 256), c.from);
    try t.expectEqual(@as(usize, 2), c.held());
    try t.expect(b.pending != null);
    c.end(&b, &again, false);
    for (lane) |id| try t.expectEqual(@as(u32, 1), pages.pool.holders(id));
    for (lane) |id| pages.pool.release(id);
    try t.expectEqual(@as(usize, 8), pages.pool.available());
}

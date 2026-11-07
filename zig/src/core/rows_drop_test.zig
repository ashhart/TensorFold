//! Each state the store drops as it opens is said on the log with its cause; the earlier states left still restore.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const note = @import("prompt_cache_note.zig");
const Dropped = @import("rows_dropped.zig").Dropped;
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const fixture = @import("rows_scale_fixture.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;
const gpa = t.allocator;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const ats = [_]u32{ 800, 1400, 2000 }; // one conversation's states, each sharing the blocks below the one before

fn options(dir: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
}

fn request(a: Allocator, s: *pc.Store, f: *Fake, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
    f.prompt = p;
    const plan = try s.begin(a, p, history, &.{}, f.starts, null, f.spans);
    return .{ .from = plan.from, .sum = f.pass(s, plan) };
}

/// A state's own file cut in half, deleted or its name's first byte flipped; a block only it names deleted or flipped.
const Fault = enum { truncate, lose, rename, delete, flip };
const Plant = struct { at: u32, fault: Fault };

/// Whether the state at `at` is one `plants` make the store drop at open.
fn lost(plants: []const Plant, at: u32) bool {
    for (plants) |x| if (x.at == at and x.fault != .flip) return true;
    return false;
}

/// The blocks no state left after `plants` names, less those a plant deletes: what the reopened store removes.
fn unnamed(r: *rows.Rows, plants: []const Plant) !usize {
    var kept: std.ArrayList(rows.write.Want) = .empty;
    defer kept.deinit(gpa);
    for (r.states.items) |*s| if (!lost(plants, s.at)) try r.wantsOf(gpa, s, &kept);
    var n: usize = 0;
    var it = r.blocks.keyIterator();
    while (it.next()) |b| n += @intFromBool(for (kept.items) |w| {
        if (std.mem.eql(u8, &rows.chain.stem(w.name), b)) break false;
    } else true);
    for (plants) |x| n -= @intFromBool(x.fault == .delete);
    return n;
}

fn plant(r: *rows.Rows, x: Plant) !void {
    const i = for (r.states.items, 0..) |s, i| {
        if (s.at == x.at) break i;
    } else return error.TestNoState;
    const own = x.fault == .truncate or x.fault == .lose or x.fault == .rename;
    var buf: [1200]u8 = undefined;
    const p = try rows.file.path(&buf, r.dir, &rows.chain.hex(if (own) r.states.items[i].name else try onlyBlock(r, i)), if (own) ".st" else ".blk", false);
    if (x.fault == .delete or x.fault == .lose) return t.expectEqual(@as(c_int, 0), platform.c.unlink(p));
    const fd = platform.c.open(p, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try t.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (x.fault == .truncate) return t.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, @divTrunc(end, 2)));
    const at = if (x.fault == .rename) end - @sizeOf(rows.file.Trailer) + @offsetOf(rows.file.Trailer, "name") else 100;
    var b: [1]u8 = undefined;
    try t.expectEqual(@as(isize, 1), platform.c.pread(fd, &b, 1, at));
    b[0] ^= 0xff;
    try t.expectEqual(@as(isize, 1), platform.c.pwrite(fd, &b, 1, at));
}

/// A block state `i` names and no other state does.
fn onlyBlock(r: *rows.Rows, i: usize) !rows.chain.Name {
    var mine: std.ArrayList(rows.write.Want) = .empty;
    defer mine.deinit(gpa);
    var others: std.ArrayList(rows.write.Want) = .empty;
    defer others.deinit(gpa);
    for (r.states.items, 0..) |*s, j| try r.wantsOf(gpa, s, if (j == i) &mine else &others);
    for (mine.items) |w| {
        for (others.items) |o| {
            if (std.mem.eql(u8, &o.name, &w.name)) break;
        } else return w.name;
    }
    return error.TestNoBlock;
}

/// The files under `dir` whose names end in `ext`.
fn files(dir: [:0]const u8, ext: []const u8) usize {
    const d = platform.c.opendir(dir) orelse return 0;
    defer _ = platform.c.closedir(d);
    var n: usize = 0;
    while (platform.c.readdir(d)) |ent| n += @intFromBool(std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), ext));
    return n;
}

test "rows drop: a state dropped before an open fails part way is still said" {
    var said: usize = 0;
    for (0..32) |index| {
        var tag: [40]u8 = undefined;
        var x = try fixture.Fixture.init(try std.fmt.bufPrint(&tag, "drop-{d}", .{index}));
        defer x.close();
        try x.seed(2); // two states at 64 tokens, the first found cut
        var buf: [1200]u8 = undefined;
        const dir = blk: {
            const r = try rows.Rows.open(gpa, x.f.snapshots(), x.options());
            defer r.deinit();
            try plant(r, .{ .at = 64, .fault = .truncate });
            break :blk try std.fmt.bufPrintSentinel(&buf, "{s}", .{r.dir}, 0);
        };
        var heard: note.Heard = .{};
        note.sink = &heard;
        defer note.sink = null;
        var fa = t.FailingAllocator.init(gpa, .{ .fail_index = index });
        const failed_open = if (rows.Rows.open(fa.allocator(), x.f.snapshots(), x.options())) |r| blk: {
            r.deinit();
            break :blk false;
        } else |err| blk: {
            try t.expectEqual(error.OutOfMemory, err);
            break :blk true;
        };
        const gone = files(dir, ".st") == 1;
        try t.expectEqualStrings(if (gone) failed else "", heard.text());
        said += @intFromBool(gone and failed_open);
        try t.expectEqual(@as(usize, 1), try x.files()); // the valid state stays
    }
    try t.expect(said > 0); // some opens failed after the drop, and said it
}

test "rows drop: a cause past eight is said at once, and a flush says each other cause once with its count" {
    var heard: note.Heard = .{};
    note.sink = &heard;
    defer note.sink = null;
    var d: Dropped = .{};
    for ([_]anyerror{ error.A, error.B, error.C, error.D, error.E, error.F, error.G, error.H, error.I, error.A }) |e| d.add(e);
    try t.expectEqualStrings("imprint: a disk state failed its check (I); it is forgotten\n", heard.text());
    heard.len = 0;
    d.flush();
    try t.expectEqual(@as(usize, 8), std.mem.count(u8, heard.text(), "\n"));
    try t.expect(std.mem.startsWith(u8, heard.text(), "imprint: 2 disk states failed their check (A); they are forgotten\n"));
    heard.len = 0;
    d.flush();
    try t.expectEqualStrings("", heard.text());
    for ([_]usize{ 1, 2 }) |n| {
        d.blocks += n;
        d.flush();
    }
    try t.expectEqualStrings("imprint: a disk block no kept state names is removed" ++ why ++ "imprint: 2 disk blocks no kept state names are removed" ++ why, heard.text());
}

const failed = "imprint: a disk state failed its check (RowsRead); it is forgotten\n";
const missing = "imprint: a disk state is missing one of its blocks; it is forgotten\n";
const why = " (a state's file is gone, or a write was cut short)\n";

test "rows drop: each state dropped at open is said with its cause, and the earlier state still restores" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try a.alloc(u32, 2100);
    for (p, 0..) |*x, i| x.* = @intCast(50_000 + i);
    const Case = struct { plants: []const Plant, opened: []const u8, read: []const u8 = "", from: u32 };
    const cases = [_]Case{
        .{ .plants = &.{}, .opened = "", .from = 2000 },
        .{ .plants = &.{.{ .at = 2000, .fault = .truncate }}, .opened = failed, .from = 1400 },
        // with its own file gone the state is never known: only the blocks only it named are said, as they go
        .{ .plants = &.{.{ .at = 2000, .fault = .lose }}, .opened = "", .from = 1400 },
        .{ .plants = &.{.{ .at = 2000, .fault = .rename }}, .opened = failed, .from = 1400 },
        .{ .plants = &.{.{ .at = 2000, .fault = .delete }}, .opened = missing, .from = 1400 },
        .{ .plants = &.{ .{ .at = 2000, .fault = .truncate }, .{ .at = 1400, .fault = .rename } }, .opened = "imprint: 2 disk states failed their check (RowsRead); they are forgotten\n", .from = 800 },
        .{ .plants = &.{ .{ .at = 1400, .fault = .truncate }, .{ .at = 2000, .fault = .delete } }, .opened = failed ++ missing, .from = 800 },
        // a flipped byte is found only as the state is read, by the read's own check, which says so
        .{ .plants = &.{.{ .at = 2000, .fault = .flip }}, .opened = "", .read = "imprint: a disk block failed its check (RowsHash); it and the states naming it are forgotten\n", .from = 0 },
    };
    for (cases, 0..) |c, n| {
        var buf: [128]u8 = undefined;
        const dir = try std.fmt.bufPrint(&buf, "/tmp/tf-rows-drop{d}-{d}", .{ n, platform.c.getpid() });
        defer rmTree(dir);
        const gone = blk: {
            var f: Fake = .{ .gpa = gpa };
            const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
            defer r.deinit();
            var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
            defer s.deinit();
            s.disk = r;
            for (ats) |at| {
                _ = try request(a, &s, &f, p[0 .. at + 1], at);
                while (rows.write.step(r)) {}
            }
            var i: usize = 0;
            while (i < r.states.items.len) { // the states kept at each earlier prompt's end go: these three stay
                if (std.mem.indexOfScalar(u32, &ats, r.states.items[i].at) == null) r.remove(i) else i += 1;
            }
            var kept: std.ArrayList(u32) = .empty;
            for (r.states.items) |st| try kept.append(a, st.at);
            try t.expectEqualSlices(u32, &ats, kept.items);
            const k = try unnamed(r, c.plants);
            for (c.plants) |x| try plant(r, x);
            break :blk k;
        };
        const removed: []const u8 = switch (gone) {
            0 => "",
            1 => "imprint: a disk block no kept state names is removed" ++ why,
            else => try std.fmt.allocPrint(a, "imprint: {d} disk blocks no kept state names are removed" ++ why, .{gone}),
        };
        var heard: note.Heard = .{};
        note.sink = &heard;
        defer note.sink = null;
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ c.opened, removed }), heard.text());
        var drops: usize = 0;
        for (ats) |at| {
            drops += @intFromBool(lost(c.plants, at));
            try t.expectEqual(!lost(c.plants, at), for (r.states.items) |s| {
                if (s.at == at) break true;
            } else false);
        }
        try t.expectEqual(ats.len - drops, r.states.items.len);
        try t.expectEqual(r.states.items.len, files(r.dir, ".st"));
        try t.expectEqual(@as(usize, r.blocks.count()), files(r.dir, ".blk"));
        heard.len = 0;
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        const got = try request(a, &s, &f, p[0..2050], 2049);
        try t.expectEqualStrings(c.read, heard.text());
        try t.expectEqual(c.from, got.from);
        try t.expect(f.restored_ok);
        try t.expectEqual(fresh(p[0..2050]), got.sum);
        heard.len = 0;
        const blocks = r.blocks.count();
        while (r.states.items.len > 0) r.remove(0); // collection while serving removes their blocks, quietly
        try t.expect(r.blocks.count() < blocks);
        try t.expectEqualStrings("", heard.text());
    }
}

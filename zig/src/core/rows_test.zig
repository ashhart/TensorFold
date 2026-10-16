//! The rows store's tests: states written behind and read back equal, shared blocks, checks, the cap, budget and floor.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Allocator = std.mem.Allocator;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-rows-{s}-{d}", .{ tag, std.c.getpid() }) catch unreachable;
}

fn options(dir: []const u8) rows.Options {
    return .{ .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
}

/// A conversation's prompt: a head shared by every conversation `c`, then `n` tokens of its own.
fn prompt(a: Allocator, head: u32, c: u32, n: u32) ![]u32 {
    const out = try a.alloc(u32, head + n);
    for (out, 0..) |*t, i| t.* = if (i < head) @intCast(1000 + i * 7 % 997) else @intCast(50_000 + c * 10_000 + i);
    return out;
}

/// One request through a store with `f`: its plan, then a pass from it; whether every restore matched a fresh pass.
fn request(a: Allocator, s: *pc.Store, f: *Fake, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
    f.prompt = p;
    const plan = try s.begin(a, p, history, &.{}, f.starts, null, f.spans);
    return .{ .from = plan.from, .sum = f.pass(s, plan) };
}

/// Every queued state written.
fn drain(r: *rows.Rows) void {
    while (rows.write.step(r)) {}
}

test "a conversation's states outlive the server: a restart resumes from disk, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "restart");
    defer rmTree(dir);
    const turn1 = try prompt(a, 0, 1, 1500);
    {
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        const got = try request(a, &s, &f, turn1, 1400);
        try std.testing.expectEqual(fresh(turn1), got.sum);
        try std.testing.expect(rows.write.pending(r));
        drain(r);
        try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
        // 1400 tokens: 5 blocks of 256, and the ring's 1024..1280
        try std.testing.expectEqual(@as(usize, 6), r.blocks.count());
    }
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    const turn2 = try std.mem.concat(a, u32, &.{ turn1[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    const got = try request(a, &s, &f, turn2, 1404);
    try std.testing.expectEqual(@as(u32, 1400), got.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(turn2), got.sum);
    try std.testing.expectEqual(@as(u64, 1), r.counts.reads);
}

test "a disk read takes a snapshot slot: with every slot held it frees one first, with none it reads nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "slots");
    defer rmTree(dir);
    const turn1 = try prompt(a, 0, 1, 1500);
    {
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        _ = try request(a, &s, &f, turn1, 1400);
        drain(r);
    }
    const turn2 = try std.mem.concat(a, u32, &.{ turn1[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    for ([_]usize{ 0, 1 }) |most| {
        var f: Fake = .{ .gpa = gpa, .slots = most };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        s.most = most;
        if (most == 1) { // another conversation holds the one slot
            _ = try request(a, &s, &f, try prompt(a, 0, 2, 1200), 1100);
            try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
        }
        const got = try request(a, &s, &f, turn2, 1404);
        try std.testing.expectEqual(@as(u32, if (most == 0) 0 else 1400), got.from);
        try std.testing.expectEqual(fresh(turn2), got.sum);
        try std.testing.expectEqual(@as(u64, most), r.counts.reads);
        try std.testing.expect(s.entries.items.len <= most);
    }
}

test "ten conversations with one head store its blocks once" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "head");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    for (0..10) |c| {
        const p = try prompt(a, 2048, @intCast(c), 600);
        _ = try request(a, &s, &f, p, 2600);
        drain(r);
    }
    try std.testing.expectEqual(@as(usize, 10), r.states.items.len);
    // the head's 7 once (its 8th reads token 2048), each tail's 3 and 2 of its ring
    try std.testing.expectEqual(@as(usize, 7 + 10 * 5), r.blocks.count());
    var named: u64 = 0;
    for (r.states.items) |st| named += st.rows;
    var stored: u64 = 0;
    var it = r.blocks.valueIterator();
    while (it.next()) |b| stored += b.bytes;
    try std.testing.expect(stored * 2 < named); // the head written once, named ten times
}

test "a block that fails its check is forgotten with every state naming it, and the turn reads its tokens again" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "check");
    defer rmTree(dir);
    const p = try prompt(a, 0, 3, 900);
    {
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        _ = try request(a, &s, &f, p, 800);
        drain(r);
        var it = r.blocks.keyIterator();
        var path: [1200]u8 = undefined;
        const file = try rows.file.path(&path, r.dir, &rows.chain.hex(it.next().?.*), ".blk", false);
        const fd = std.c.open(file, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, "X", 1, 100));
    }
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const p2 = try std.mem.concat(a, u32, &.{ p[0..801], &[_]u32{ 4, 4 } });
    const got = try request(a, &s, &f, p2, 802);
    try std.testing.expectEqual(@as(u32, 0), got.from);
    try std.testing.expectEqual(fresh(p2), got.sum);
    try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
    try std.testing.expectEqual(@as(u64, 1), r.counts.failed);
}

test "decoded spans and a planned family's chunk starts must match for a disk state to resume" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "spans");
    defer rmTree(dir);
    const planned: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1, .planned = true };
    var o = options(dir);
    o.planned = true;
    const p = try prompt(a, 0, 4, 1300);
    {
        var f: Fake = .{ .gpa = gpa, .planned = true, .starts = &.{ 512, 1024, 1200 }, .spans = &.{.{ 300, 340 }} };
        const r = try rows.Rows.open(gpa, f.snapshots(), o);
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), planned, 1 << 30);
        defer s.deinit();
        s.disk = r;
        _ = try request(a, &s, &f, p, 1200);
        drain(r);
        try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    }
    const cases = [_]struct { starts: []const u32, spans: []const [2]u32, from: u32 }{
        .{ .starts = &.{ 512, 1024, 1200 }, .spans = &.{.{ 300, 340 }}, .from = 1200 },
        .{ .starts = &.{ 512, 1024, 1200 }, .spans = &.{}, .from = 0 }, // rows decoded there in the kept state
        .{ .starts = &.{ 512, 1000, 1200 }, .spans = &.{.{ 300, 340 }}, .from = 0 }, // another chunk below it
        .{ .starts = &.{ 512, 1024 }, .spans = &.{.{ 300, 340 }}, .from = 0 }, // 1200 is no chunk start of this request
    };
    for (cases) |c| {
        var f: Fake = .{ .gpa = gpa, .planned = true, .starts = c.starts, .spans = c.spans };
        const r = try rows.Rows.open(gpa, f.snapshots(), o);
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), planned, 1 << 30);
        defer s.deinit();
        s.disk = r;
        const got = try request(a, &s, &f, p, 1299);
        try std.testing.expectEqual(c.from, got.from);
        try std.testing.expect(f.restored_ok);
        try std.testing.expectEqual(fresh(p), got.sum);
    }
}

test "an entry evicted before its write leaves the queue; once written, an eviction costs nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "evict");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    const one = try prompt(a, 0, 5, 700);
    const budget = Fake.snapshots(&f).vtable.bytes(&f, 600) + 1000; // room for one conversation's state
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, budget);
    defer s.deinit();
    s.disk = r;
    _ = try request(a, &s, &f, one, 600);
    _ = rows.write.step(r); // its job starts
    try std.testing.expect(r.job != null);
    const two = try prompt(a, 0, 6, 700);
    _ = try request(a, &s, &f, two, 600); // the first state goes from RAM mid-write
    try std.testing.expect(r.job == null); // given up with its entry
    drain(r);
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    _ = try request(a, &s, &f, one, 600); // the second goes from RAM, already on disk
    drain(r);
    try std.testing.expectEqual(@as(usize, 2), r.states.items.len);
    const again = try std.mem.concat(a, u32, &.{ two[0..601], &[_]u32{ 1, 2 } });
    const got = try request(a, &s, &f, again, 602);
    try std.testing.expectEqual(@as(u32, 600), got.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(again), got.sum);
}

test "a step writes one block at most, and a state's own file goes out in pieces" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "steps");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    _ = try request(a, &s, &f, try prompt(a, 0, 7, 1100), 1030);
    var steps: usize = 0;
    var blocks: usize = 0;
    while (rows.write.step(r)) {
        steps += 1;
        try std.testing.expect(r.blocks.count() <= blocks + 1);
        blocks = r.blocks.count();
    }
    try std.testing.expectEqual(@as(usize, 6), blocks); // 4 of rows from 0, 2 of the ring's last 600
    try std.testing.expectEqual(@as(usize, 1 + 6 + 1 + 1), steps); // start, 6 blocks, its file's one piece, the finish
}

test "past the cap a state goes by value, and past today's write budget a state stays in RAM" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "cap");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    for (0..4) |c| {
        _ = try request(a, &s, &f, try prompt(a, 0, @intCast(10 + c), 1100), 1000);
        drain(r);
        if (c == 0) r.cap = r.held() * 5 / 2; // two such states, not three
        try std.testing.expect(r.held() <= r.cap);
    }
    try std.testing.expect(r.states.items.len < 4 and r.states.items.len > 0);
    try std.testing.expect(r.counts.evicted > 0);
    r.daily = 0;
    const skipped = r.counts.skipped;
    const p = try prompt(a, 0, 20, 1100);
    try std.testing.expectEqual(fresh(p), (try request(a, &s, &f, p, 1000)).sum);
    drain(r);
    try std.testing.expectEqual(skipped + 1, r.counts.skipped);
}

test "a read that costs more than prefilling the tokens it saves is not made" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "worth");
    defer rmTree(dir);
    const p = try prompt(a, 0, 8, 1100);
    {
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        _ = try request(a, &s, &f, p, 1000);
        drain(r);
    }
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    r.read_rate = 1000; // a slow disk: 40 KB reads in 40 s
    r.prefill_rate = 1e6; // and a fast prefill: 1,000 tokens in 1 ms
    const x = rows.read.resumable(&s, p, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 1000), x.disk);
    try std.testing.expect(x.from == .none and x.seconds < 0.01);
    const got = try request(a, &s, &f, p, 1099);
    try std.testing.expectEqual(@as(u32, 0), got.from);
    try std.testing.expectEqual(@as(u64, 0), r.counts.reads);
}

test "states past the ring's window keep its last rows in blocks of their own, read back equal" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "ring");
    defer rmTree(dir);
    const p = try prompt(a, 0, 9, 3000);
    {
        var f: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        for ([_]u32{ 1030, 2600 }) |h| _ = try request(a, &s, &f, p[0 .. h + 5], h);
        drain(r);
        try std.testing.expectEqual(@as(usize, 2), r.states.items.len);
    }
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const got = try request(a, &s, &f, p, 2999);
    try std.testing.expectEqual(@as(u32, 2600), got.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(p), got.sum);
}

test "TF_IMPRINT_VERIFY's read-back finds every written state byte-equal to the kept state it came from" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "verify");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    r.verify = true;
    for ([_]u32{ 700, 1300 }) |h| _ = try request(a, &s, &f, try prompt(a, 300, 2, h + 5), h);
    drain(r);
    try std.testing.expect(r.states.items.len >= 2);
    try std.testing.expectEqual(@as(u64, r.states.items.len), r.counts.verified);
    try std.testing.expectEqual(@as(u64, 0), r.counts.differed);
}

test "a learn dir that is a regular file is refused, and open frees what it made once" {
    const gpa = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const path = root(&buf, "file");
    var zbuf: [129]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zbuf, "{s}", .{path}, 0);
    const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = std.c.close(fd);
    defer _ = std.c.unlink(z);
    var f: Fake = .{ .gpa = gpa };
    try std.testing.expectError(error.RowsDir, rows.Rows.open(gpa, f.snapshots(), options(path)));
}

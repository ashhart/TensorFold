//! Actual page planning, calibration publication and empty-pool ownership, without a driver.
const std = @import("std");
const sizing = @import("cuda_pool_after_warmup.zig");
const memory = @import("cuda_memory.zig");
const kv = @import("core").kv_pool;
const lanes = @import("lanes");
const startup = @import("cuda_startup_costs.zig");
const t = std.testing;

fn pool(free: u64) memory.Pool {
    return .{ .free = free, .total = 4096, .reserve = 64, .limit = null, .unified = false };
}

test "CUDA pool after warmup: measured room grows an estimated plan or keeps it, in whole pages within its window" {
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 64, .snap_bytes = 0 };
    const estimated = pool(320);
    const old = try kv.plan(l, .{ .room = estimated.room(0), .engine = 0, .lanes = 3, .fixed = true, .window = 128, .engine_paged = true });
    try t.expectEqual(@as(usize, 4), old.pages);
    const useful = 3 * l.pages(128);
    for ([_]struct { free: u64, expected: usize }{
        .{ .free = 64, .expected = 4 },   .{ .free = 127, .expected = 4 },
        .{ .free = 128, .expected = 5 },  .{ .free = 192, .expected = 6 },
        .{ .free = 4096, .expected = 6 },
    }) |row| {
        const measured = pool(row.free);
        const pages = try sizing.size(old.pages, useful, l.pageBytes(), measured, 0);
        try t.expectEqual(row.expected, pages);
        try t.expect(pages >= old.pages);
        try t.expect(row.free - (pages - old.pages) * l.pageBytes() >= measured.reserve);
    }
}

test "CUDA pool after warmup: configured limit, reserve loss and geometry never silently shrink or overspend" {
    var p = pool(4096);
    p.limit = 1064;
    try t.expectEqual(@as(usize, 3), try sizing.size(2, 8, 64, p, 1000));
    p.limit = 1000;
    try t.expectEqual(@as(usize, 2), try sizing.size(2, 8, 64, p, 1000));
    try t.expectError(error.PoolReserveUnavailable, sizing.size(2, 8, 64, p, 1001));
    try t.expectError(error.PoolReserveUnavailable, sizing.size(2, 8, 64, pool(63), 0));
    try t.expectError(error.InvalidPoolGeometry, sizing.size(2, 1, 64, pool(200), 0));
    try t.expectError(error.InvalidPoolGeometry, sizing.size(2, 8, 0, pool(200), 0));
    try t.expectEqual(std.math.maxInt(usize), try sizing.size(std.math.maxInt(usize) - 1, std.math.maxInt(usize), 1, pool(std.math.maxInt(u64)), 0));
}

const Fake = struct {
    const Lane = struct { used: bool = false };
    const Pages = struct {
        pool: kv.Pool,
        lanes: []Lane,
        closed: bool = false,
        pub fn deinit(p: *@This()) void {
            p.closed = true;
        }
    };
    const Caches = struct {
        n: usize = 0,
        pub fn count(c: @This()) usize {
            return c.n;
        }
    };
    caches: Caches = .{},
    pool: struct { items: []u8 = &.{} } = .{},
    prefix: ?*u8 = null,
    pages: ?*Pages,
    next: *Pages,
    attached: bool = false,
    pub fn attach(p: *anyopaque, more: u32, pages: usize) !void {
        const l: *Fake = @ptrCast(@alignCast(p));
        try t.expect(l.pages == null);
        try t.expectEqual(@as(u32, 1), more);
        try t.expectEqual(@as(usize, 6), pages);
        l.attached = true;
        l.pages = l.next;
    }
};

test "CUDA pool after warmup: borrowed caches, promises, live and retained pages all prevent empty arena replacement" {
    var physical = try kv.Pool.init(t.allocator, 4, 64);
    defer physical.deinit();
    var lane = [_]Fake.Lane{.{}};
    var old: Fake.Pages = .{ .pool = physical, .lanes = &lane };
    var next: Fake.Pages = .{ .pool = physical, .lanes = &lane };
    var l: Fake = .{ .pages = &old, .next = &next };
    var lease: kv.Lease = .{};
    defer old.pool.finish(&lease);
    try old.pool.promise(&lease, 1);
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    _ = try old.pool.take(&lease);
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    old.pool.finish(&lease);
    lane[0].used = true;
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    lane[0].used = false;
    l.caches.n = 1;
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    l.caches.n = 0;
    var byte: u8 = 0;
    l.prefix = &byte;
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    l.prefix = null;
    l.pool.items = @as(*[1]u8, &byte);
    try t.expectError(error.PoolInUse, sizing.replace(Fake, &l, 2, 6));
    l.pool.items = &.{};
    try t.expect(!old.closed and !l.attached and l.pages == &old);
    try sizing.replace(Fake, &l, 2, 6);
    try t.expect(old.closed and l.attached and l.pages == &next);
}

const Probe = struct {
    calls: u32 = 0,
    synchronized: bool = false,
    const costs = [_]lanes.config.Cost{ .{ .width = 1, .ms = 1 }, .{ .width = 2, .ms = 1.5 } };
    fn run(p: *anyopaque) !lanes.drafter.Facts {
        const self: *Probe = @ptrCast(@alignCast(p));
        self.calls += 1;
        try t.expect(!self.synchronized);
        return .{ .depth = 1, .step_ms = 0.5, .window_costs = &costs, .lane_costs = &costs };
    }
    pub fn synchronize(self: *Probe) !void {
        try t.expectEqual(@as(u32, 1), self.calls);
        self.synchronized = true;
    }
};
const Served = struct {
    facts: lanes.Model = .{ .exact_width = 2 },
    prepare: ?startup.Prepare,
    pub fn calibrate(self: *@This(), gpa: std.mem.Allocator, cfg: *lanes.Config) !void {
        try startup.calibrate(self, gpa, cfg);
    }
};

test "CUDA pool after warmup: actual calibration publishes once and drains before the free memory probe" {
    var probe: Probe = .{};
    var served: Served = .{ .prepare = .{ .ctx = &probe, .run = Probe.run } };
    var engine = .{ .stream = &probe };
    try sizing.prepare(&served, t.allocator, 2, &engine);
    try t.expect(served.prepare == null and probe.synchronized);
    try t.expectEqualSlices(lanes.config.Cost, &Probe.costs, served.facts.window_costs);
    try sizing.prepare(&served, t.allocator, 2, &engine);
    try t.expectEqual(@as(u32, 1), probe.calls);
}

test "CUDA pool after warmup: strict default-off switch has exactly one counted dense CUDA row" {
    const env = @import("sys").environment;
    const name = "TF_CUDA_POOL_AFTER_WARMUP";
    try t.expect(!try env.value(bool, name, null, false));
    try t.expect(!try env.value(bool, name, "0", false));
    try t.expect(try env.value(bool, name, "1", false));
    for ([_][]const u8{ "", " ", "true", "on", "01", "2", "-1" }) |raw| try t.expectError(error.InvalidEnvironment, env.value(bool, name, raw, false));
    const profile = @import("core").profile;
    var count: usize = 0;
    for (profile.entries) |row| if (std.mem.eql(u8, row.name(), name)) {
        count += 1;
        try t.expect(row.applies("qwen3_5", "cuda") and !row.applies("qwen3_5_moe", "cuda"));
        try t.expect(!row.applies("qwen3_5", "metal") and row.stable == null);
        try t.expect(row.differs("1") and !row.differs("0"));
    };
    try t.expectEqual(@as(usize, 1), count);
    try t.expectEqualStrings("", try sizing.text(t.allocator, null));
}

test "CUDA pool after warmup: real native open keeps early unsupported-mode refusals before device access" {
    try @import("slide_open_test.zig").check(@import("cuda.zig").open, "CUDA", &.{"qwen4"});
}

test "CUDA pool after warmup: startup explains both allocation phases and the selected capacity" {
    const line = try sizing.text(t.allocator, .{
        .estimated_free_bytes = 2 << 30,
        .measured_free_bytes = 1 << 30,
        .final_free_bytes = 1 << 29,
        .reserve_bytes = 1 << 29,
        .growth_room_bytes = 1 << 29,
        .estimated_pages = 2,
        .measured_pages = 4,
        .page_bytes = 64,
        .page_tokens = 128,
    });
    defer t.allocator.free(line);
    try t.expect(std.mem.indexOf(u8, line, "estimated-plan free 2.00 GiB before allocation") != null);
    try t.expect(std.mem.indexOf(u8, line, "measured free 1.00 GiB with initial pool resident") != null);
    try t.expect(std.mem.indexOf(u8, line, "reserve 0.50 GiB, pages 2 -> 4") != null);
}

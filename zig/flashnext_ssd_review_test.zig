//! Reviewed SSD pool boundaries and ownership, with actual shared formulas and a recording allocator.
const std = @import("std");
const t = std.testing;
const memory = @import("flashnext_memory");
const layout = @import("src/families/flashnext/fsr_layout.zig");
const Settings = @import("src/families/flashnext/window_config.zig").Settings;
const GiB = memory.GiB;
const Ctx = struct { limit: u64, existing: u64 = 0, extra: u64 = 0, needs_build: bool = false, paired: bool = false, rotary_growth: u64 = 0 };
const inventory: memory.weights_plan.Inventory = .{ .raw = 110 * GiB, .routed = 80 * GiB, .ple = 24 * GiB, .expert_parts = @splat(256 * 1024) };
const settings: Settings = .{ .capacity = 65568, .ssd_experts = 8 * GiB, .ple_on_ssd = true };

fn budget(s: Settings, ctx: Ctx) !memory.Budget {
    const p = try layout.of(inventory, s);
    var resident = inventory;
    resident.raw = p.raw;
    return memory.budgetLiveFor(0, s.capacity, resident, ctx.extra + ctx.existing + p.extra, ctx.existing, s.sibling(ctx.paired));
}

test "REV99 SSD budget runs in all and the Linux host shard" {
    const build = @embedFile("build/flashnext_batch_host.zig");
    const ci = @embedFile("build/host_ci.zig");
    try t.expect(std.mem.indexOf(u8, ci, "\"test-flashnext-ssd-budget-host\"") != null);
    try t.expect(std.mem.indexOf(u8, build, "all.dependOn(&bounded_run.step)") != null);
}

test "REV99 automatic cache includes unfaulted allocated pool pages" {
    const cache = @import("src/native/flashnext_cache_budget.zig");
    try t.expectEqual(@as(u64, 0), try cache.automatic(24 * GiB, 15 * GiB, 22 * GiB));
    try t.expectEqual(@as(u64, 2 * GiB), try cache.automatic(24 * GiB, 15 * GiB, 20 * GiB));
    try t.expectEqual(@as(u64, 3 * GiB), try cache.automatic(24 * GiB, 19 * GiB, 12 * GiB));
}

test "REV99 native automatic pool budget uses the allocated-page ceiling before its receipt" {
    const source = @embedFile("src/native/flashnext_cache_setup.zig");
    try t.expect(std.mem.indexOf(u8, source, "flashnext_cache_budget.zig") != null);
    try t.expect(std.mem.indexOf(u8, source, "budgetUnderMaximum") != null);
    try t.expect(std.mem.indexOf(u8, source, "gib == null") != null);
}

test "REV99 fitted placement reprices caller and existing charges before the final budget" {
    const source = @embedFile("src/families/flashnext/engine_load.zig");
    const placed = std.mem.indexOf(u8, source, "placement = try @import(\"fsr_layout.zig\").of(source_inventory,") orelse return error.TestUnexpectedResult;
    const charged = std.mem.indexOfPos(u8, source, placed, "base_extra = try std.math.add") orelse return error.TestUnexpectedResult;
    const priced = std.mem.indexOfPos(u8, source, placed, "plain = try memory.budgetLiveFor") orelse return error.TestUnexpectedResult;
    try t.expect(charged < priced);
}

test "REV99 a refused tensor-unit pool retains rotary headroom when a smaller pool permits it" {
    var exact = settings;
    const p = (try layout.of(inventory, exact)).pool.?;
    exact.ssd_experts = 768 * p.stride;
    var ctx: Ctx = .{ .limit = 0, .rotary_growth = 33028 };
    ctx.limit = (try budget(exact, ctx)).steady;
    const fitted = try layout.fit(inventory, settings, ctx);
    try t.expectEqual(@as(?u64, 767 * p.stride), fitted.ssd_experts);
    try t.expect((try budget(fitted, ctx)).steady + ctx.rotary_growth <= ctx.limit);
    try t.expectEqualDeep(exact, try layout.fit(inventory, exact, ctx)); // an admitted request stays exact
}

const fsr = @import("src/core/fsr/arena.zig");
const Source = struct {
    pub fn fileSize(_: @This(), _: usize) !u64 {
        return 4096;
    }
    pub fn readAt(_: @This(), _: usize, out: []u8, _: u64) !usize {
        @memset(out, 0);
        return out.len;
    }
};
fn measured(slots: usize) !usize {
    var peak: @import("src/core/rows_peak_allocator.zig").Peak = .{ .child = t.allocator };
    const A = fsr.Arena(Source);
    const span: @import("src/core/fsr/reader.zig").Span = .{ .file = 0, .offset = 0, .length = 4 };
    const experts: [16]A.Expert = @splat(@splat(span));
    var raw: [9][32]u8 = undefined;
    var slabs: [9][]u8 = undefined;
    for (&slabs, &raw) |*slab, *bytes| slab.* = bytes[0 .. slots * 4];
    var map: [16]i32 = undefined;
    var arena = try A.initMapped(peak.allocator(), .{}, &experts, @splat(4), slabs, &map);
    const bytes = peak.live;
    arena.deinit();
    try t.expectEqual(@as(usize, 0), peak.live);
    return bytes;
}

test "REV99 slot metadata covers actual demand and forecast job allocations" {
    const plan = layout.plan;
    const small = try plan.pool(4 * 36, @splat(4), 16, 4);
    const large = try plan.pool(8 * 36, @splat(4), 16, 4);
    const small_live = try measured(4);
    const large_live = try measured(8);
    try t.expect(large.metadata - small.metadata >= large_live - small_live);
}

const Buffer = struct {
    r: *Recording,
    pub fn deinit(b: @This()) void {
        b.r.live -= 1;
        b.r.freed += 1;
    }
};
const Recording = struct {
    live: usize = 0,
    freed: usize = 0,
    calls: usize = 0,
    kept: usize = 0,
    fail_at: usize,
    pub fn allocate(r: *@This(), _: usize) !Buffer {
        r.calls += 1;
        r.live += 1;
        return .{ .r = r };
    }
    pub fn keep(r: *@This(), _: Buffer) !void {
        if (r.calls == r.fail_at) return error.OutOfMemory;
        r.kept += 1;
    }
    fn close(r: *@This()) void {
        r.live -= r.kept;
        r.freed += r.kept;
        r.kept = 0;
    }
};

test "REV99 every failed slab registration releases only its unowned buffer" {
    const own = @import("src/families/flashnext/fsr_buffer.zig");
    for ([_]usize{ 3, 9 }) |count| for (1..count + 1) |fail_at| {
        var r: Recording = .{ .fail_at = fail_at };
        for (0..count) |_| _ = own.allocate(&r, 4096) catch break;
        const unowned = r.live - r.kept;
        r.close();
        try t.expectEqual(@as(usize, 0), unowned);
        try t.expectEqual(@as(usize, 0), r.live);
        try t.expectEqual(r.calls, r.freed);
    };
}

test "REV99 expert and PLE slabs share the checked ownership handoff" {
    const source = @embedFile("src/families/flashnext/fsr_storage.zig");
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, source, "@import(\"fsr_buffer.zig\").allocate(r, n)"));
}

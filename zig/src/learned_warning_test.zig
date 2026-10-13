const std = @import("std");
const pc = @import("core/prompt_cache.zig");
const Imprint = @import("core/prompt_imprint.zig").Imprint;
const Fake = @import("core/prompt_cache_fake.zig").Fake;

pub const std_options: std.Options = .{ .log_level = .warn, .logFn = capture };
var messages: [8][256]u8 = undefined;
var lengths: [8]usize = undefined;
var count: usize = 0;
var overflow = false;

fn capture(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (count == messages.len) {
        overflow = true;
        return;
    }
    const line = std.fmt.bufPrint(&messages[count], format, args) catch {
        overflow = true;
        return;
    };
    lengths[count] = line.len;
    count += 1;
}

const Disk = struct {
    var space: u64 = 1 << 20;
    var calls: usize = 0;
    var drop_at: usize = 0;
    fn free(_: [:0]const u8) ?u64 {
        calls += 1;
        return if (drop_at != 0 and calls == drop_at) 0 else space;
    }
    fn clock() ?u64 {
        return 100;
    }
};

fn attempt(s: *pc.Store, f: *Fake, first: u32) !void {
    var prompt = [_]u32{ first, 7, 7, 7, 7, 7, 9, 10 };
    f.at = 5;
    var arena = std.heap.ArenaAllocator.init(s.gpa);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    if (!s.keep(&prompt, 5, null, &.{}, &.{})) return error.KeepFailed;
}

fn expectedWarnings(n: usize) !void {
    if (overflow or count != n) {
        std.debug.print("learning-paused warning count: expected {d}, found {d}\n", .{ n, count });
        return error.WarningCount;
    }
    const expected = "prompt cache: learning paused: free disk cannot preserve the 100 MiB floor";
    for (0..count) |i| if (!std.mem.eql(u8, expected, messages[i][0..lengths[i]])) return error.WarningText;
}

pub fn main() !void {
    const a = std.heap.page_allocator;
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-warning-{d}", .{std.c.getpid()});
    defer @import("core/prompt_imprint_test.zig").rmTree(root);
    {
        var im = try Imprint.open(a, root, 1, 1 << 20);
        defer im.deinit();
        im.admission = .{ .floor = 100 << 20, .free = Disk.free, .clock = Disk.clock };
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        try attempt(&s, &f, 7);
        try expectedWarnings(1);
        if (f.writes != 0 or f.disk_forgets != 0 or im.admission.failures != 1) return error.RefusalChangedState;
        try attempt(&s, &f, 8);
        try expectedWarnings(1);
        if (f.writes != 0 or f.disk_forgets != 0 or im.admission.failures != 1) return error.BackoffChangedState;
        Disk.space = 1 << 30;
        try attempt(&s, &f, 9);
        try expectedWarnings(1);
        if (f.writes != 1 or im.metas.items.len != 1 or im.admission.failures != 0 or im.admission.reserved != 0) return error.LearningDidNotRecover;
    }
    {
        Disk.calls = 0;
        Disk.drop_at = 4;
        var im = try Imprint.open(a, root, 2, 1 << 20);
        defer im.deinit();
        im.admission = .{ .floor = 100 << 20, .free = Disk.free, .clock = Disk.clock };
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        try attempt(&s, &f, 10);
        try expectedWarnings(2);
        if (f.writes != 0 or f.disk_forgets != 0 or im.metas.items.len != 0 or im.admission.reserved != 0) return error.ReserveRefusalChangedState;
    }
}

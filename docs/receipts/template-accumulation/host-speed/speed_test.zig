//! Interleaved namespace concatenations must not retain quadratic scratch.
const std = @import("std");
const template = @import("template.zig");
const testing = std.testing;

const limit = 240 * 1024 * 1024;

const Peak = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    free_calls: usize = 0,

    fn allocator(self: *Peak) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn allowed(self: *const Peak, old: usize, new: usize) bool {
        return new <= limit - (self.live - old);
    }

    fn account(self: *Peak, old: usize, new: usize) void {
        self.live = self.live - old + new;
        self.peak = @max(self.peak, self.live);
    }

    fn alloc(ctx: *anyopaque, n: usize, align_: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(0, n)) {
            std.debug.print("CAP live={d} request={d}\n", .{ self.live, n });
            return null;
        }
        const ptr = self.child.rawAlloc(n, align_, ret) orelse return null;
        self.account(0, n);
        return ptr;
    }

    fn resize(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, n: usize, ret: usize) bool {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(memory.len, n)) return false;
        if (!self.child.rawResize(memory, align_, n, ret)) return false;
        self.account(memory.len, n);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(memory.len, n)) return null;
        const ptr = self.child.rawRemap(memory, align_, n, ret) orelse return null;
        self.account(memory.len, n);
        return ptr;
    }

    fn free(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, ret: usize) void {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, align_, ret);
        self.free_calls += 1;
        self.account(memory.len, 0);
    }
};

fn measure(count: usize, interleaved: bool) !void {
    const a = testing.allocator;
    var input_arena = std.heap.ArenaAllocator.init(a);
    defer input_arena.deinit();
    const data = input_arena.allocator();
    const body_len: usize = if (interleaved) 1024 else 32;
    const body = try data.alloc(u8, body_len);
    @memset(body, 'x');
    var messages = std.json.Array.init(data);
    for (0..count) |_| {
        var message: std.json.ObjectMap = .empty;
        try message.put(data, "content", .{ .string = body });
        try messages.append(.{ .object = message });
    }
    var context: std.json.ObjectMap = .empty;
    const breaker = try data.alloc(u8, 4096);
    @memset(breaker, 'z');
    try context.put(data, "breaker", .{ .string = breaker });
    const source = if (interleaved)
        "{% set ns = namespace(out='') %}{% for m in messages %}" ++
            "{% set ns.out = ns.out + m.content ~ '|' %}{% set unused = breaker ~ '!' %}" ++
            "{% endfor %}{{ ns.out }}"
    else
        "{% for m in messages %}{{ m.content }}|{% endfor %}";
    var peak: Peak = .{ .child = a };
    var compiled = try template.compile(peak.allocator(), source, null);
    defer compiled.deinit();
    const clock = std.Io.Clock.awake;
    for (0..5) |rep| {
        const start = clock.now(testing.io).toNanoseconds();
        const result = template.renderProvenance(a, &compiled, .{ .array = messages }, .null, .{ .object = context }, false, null) catch |err| {
            std.debug.print("INVALID {d} rep={d} live={d} peak={d} error={s}\n", .{ count, rep, peak.live, peak.peak, @errorName(err) });
            return err;
        };
        const finish = clock.now(testing.io).toNanoseconds();
        defer a.free(result.s);
        defer a.free(result.spans);
        try testing.expectEqual(count * (body_len + 1), result.s.len);
        for (0..count) |i| {
            const at = i * (body_len + 1);
            try testing.expectEqualSlices(u8, body, result.s[at..][0..body_len]);
            try testing.expectEqual(@as(u8, '|'), result.s[at + body_len]);
        }
        var span_index: usize = 0;
        for (result.s, 0..) |_, i| {
            while (span_index < result.spans.len and result.spans[span_index][1] <= i) : (span_index += 1) {}
            const covered = span_index < result.spans.len and result.spans[span_index][0] <= i;
            try testing.expectEqual(i % (body_len + 1) < body_len, covered);
        }
        if (rep >= 2) std.debug.print("SPEED {s} {d} {d} {d}\n", .{ if (interleaved) "interleaved" else "short", count, rep - 2, finish - start });
    }
}

test "speed measurement short" {
    try measure(8, false);
}
test "speed measurement 128" {
    try measure(128, true);
}
test "speed measurement 256" {
    try measure(256, true);
}
test "speed measurement 512" {
    try measure(512, true);
}

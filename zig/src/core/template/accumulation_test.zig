//! Interleaved namespace concatenations must not retain quadratic scratch.
const std = @import("std");
const template = @import("template.zig");
const testing = std.testing;

const limit = 240 * 1024 * 1024;
const body_len = 1024;

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
        if (!self.allowed(0, n)) return null;
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

fn run(count: usize) !usize {
    const a = testing.allocator;
    var input_arena = std.heap.ArenaAllocator.init(a);
    defer input_arena.deinit();
    const data = input_arena.allocator();
    var messages = std.json.Array.init(data);
    const body = try data.alloc(u8, body_len);
    @memset(body, 'x');
    for (0..count) |_| {
        var message: std.json.ObjectMap = .empty;
        try message.put(data, "content", .{ .string = body });
        try messages.append(.{ .object = message });
    }
    const breaker = try data.alloc(u8, 4096);
    @memset(breaker, 'z');
    var context: std.json.ObjectMap = .empty;
    try context.put(data, "breaker", .{ .string = breaker });
    const source = "{% set ns = namespace(out='') %}{% for m in messages %}" ++
        "{% set ns.out = ns.out + m.content ~ '|' %}{% set unused = breaker ~ '!' %}" ++
        "{% endfor %}{{ ns.out }}";
    var peak: Peak = .{ .child = a };
    var compiled = try template.compile(peak.allocator(), source, null);
    defer compiled.deinit();
    const base = peak.live;
    peak.peak = base;
    const result = template.renderProvenance(a, &compiled, .{ .array = messages }, .null, .{ .object = context }, false, null) catch |err| {
        std.debug.print("{d} messages: render failed at peak scratch {d} bytes: {s}\n", .{ count, peak.peak - base, @errorName(err) });
        return err;
    };
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
    return peak.peak - base;
}

test "interleaved namespace accumulation keeps render scratch bounded" {
    var measurements: [3]usize = undefined;
    for ([_]usize{ 128, 256, 512 }, 0..) |count, i| {
        measurements[i] = try run(count);
        std.debug.print("{d} messages: peak scratch {d} bytes\n", .{ count, measurements[i] });
    }
    // Includes arena growth and value/provenance bookkeeping, with a fixed startup allowance.
    try testing.expect(measurements[1] <= 3 * measurements[0] + 4 * 1024 * 1024);
    try testing.expect(measurements[2] <= 3 * measurements[1] + 4 * 1024 * 1024);
    try testing.expect(measurements[2] <= 4 * 1024 * 1024 + 32 * 512 * (body_len + 1));
}

test "long snapshots, independent branches and self concatenation preserve bytes and provenance" {
    const a = testing.allocator;
    const x = try a.alloc(u8, 4096);
    defer a.free(x);
    @memset(x, 'x');
    var context: std.json.ObjectMap = .empty;
    defer context.deinit(a);
    try context.put(a, "x", .{ .string = x });
    const source = "{% set ns = namespace(a=x ~ '|') %}{% set saved = ns.a %}" ++
        "{% set ns.a = ns.a + x %}{% set ns.b = saved ~ 'B' %}" ++
        "{% set ns.twice = ns.a + ns.a %}{{ saved }}#{{ ns.a }}#{{ ns.b }}#{{ ns.twice }}";
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const got = try template.renderProvenance(a, &compiled, .null, .null, .{ .object = context }, false, null);
    defer a.free(got.s);
    defer a.free(got.spans);
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(a);
    try expected.appendSlice(a, x);
    try expected.appendSlice(a, "|#");
    try expected.appendSlice(a, x);
    try expected.append(a, '|');
    try expected.appendSlice(a, x);
    try expected.append(a, '#');
    try expected.appendSlice(a, x);
    try expected.appendSlice(a, "|B#");
    for (0..2) |_| {
        try expected.appendSlice(a, x);
        try expected.append(a, '|');
        try expected.appendSlice(a, x);
    }
    try testing.expectEqualSlices(u8, expected.items, got.s);
    var span_index: usize = 0;
    for (got.s, 0..) |byte, i| {
        while (span_index < got.spans.len and got.spans[span_index][1] <= i) : (span_index += 1) {}
        const covered = span_index < got.spans.len and got.spans[span_index][0] <= i;
        try testing.expectEqual(byte == 'x', covered);
    }
}

test "two growing chains and a grouped long right operand stay bounded" {
    const a = testing.allocator;
    var data_arena = std.heap.ArenaAllocator.init(a);
    defer data_arena.deinit();
    const data = data_arena.allocator();
    const x = try data.alloc(u8, 4096);
    @memset(x, 'x');
    const body = try data.alloc(u8, body_len);
    @memset(body, 'c');
    const count = 128;
    var messages = std.json.Array.init(data);
    for (0..count) |_| try messages.append(.{ .string = body });
    var context: std.json.ObjectMap = .empty;
    try context.put(data, "x", .{ .string = x });
    const source = "{% set ns = namespace(a='', b='') %}{% for m in messages %}" ++
        "{% set ns.a = ns.a + (x ~ m) %}{% set ns.b = ns.b ~ m ~ '|' %}" ++
        "{% endfor %}{{ ns.a }}#{{ ns.b }}";
    var peak: Peak = .{ .child = a };
    var compiled = try template.compile(peak.allocator(), source, null);
    defer compiled.deinit();
    const base = peak.live;
    peak.peak = base;
    const got = try template.renderProvenance(a, &compiled, .{ .array = messages }, .null, .{ .object = context }, false, null);
    defer a.free(got.s);
    defer a.free(got.spans);
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(a);
    for (0..count) |_| {
        try expected.appendSlice(a, x);
        try expected.appendSlice(a, body);
    }
    try expected.append(a, '#');
    for (0..count) |_| {
        try expected.appendSlice(a, body);
        try expected.append(a, '|');
    }
    try testing.expectEqualSlices(u8, expected.items, got.s);
    var span_index: usize = 0;
    for (got.s, 0..) |byte, i| {
        while (span_index < got.spans.len and got.spans[span_index][1] <= i) : (span_index += 1) {}
        const covered = span_index < got.spans.len and got.spans[span_index][0] <= i;
        try testing.expectEqual(byte == 'x' or byte == 'c', covered);
    }
    try testing.expect(peak.peak - base <= 4 * 1024 * 1024 + 32 * expected.items.len);
}

test "retained buffer text relocation preserves a borrowed snapshot" {
    const v = @import("provenance.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var tracked: Peak = .{ .child = arena.allocator() };
    const a = tracked.allocator();
    var b: v.Buffer = .{};
    try b.appendRetained(a, try v.Str.request(a, "abc"));
    const snapshot = b.str();
    const spans_ptr = b.spans.items.ptr;
    const extra = try a.alloc(u8, b.text.capacity + 1);
    @memset(extra, 'z');
    try b.appendRetained(a, .{ .s = extra });
    try testing.expectEqual(@as(usize, 0), tracked.free_calls);
    try testing.expect(snapshot.s.ptr != b.text.items.ptr);
    try testing.expect(spans_ptr == b.spans.items.ptr);
    try testing.expectEqualStrings("abc", snapshot.s);
    try testing.expectEqualSlices(v.Span, &.{.{ 0, 3 }}, snapshot.spans);
    try testing.expectEqualStrings("abc", b.text.items[0..3]);
}

test "retained buffer span relocation preserves a borrowed snapshot" {
    const v = @import("provenance.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var tracked: Peak = .{ .child = arena.allocator() };
    const a = tracked.allocator();
    var b: v.Buffer = .{};
    const literal = try a.alloc(u8, 1024);
    @memset(literal, 'z');
    try b.appendRetained(a, .{ .s = literal });
    const request = try v.Str.request(a, "x");
    try b.appendRetained(a, request);
    const snapshot = b.str();
    const text_ptr = b.text.items.ptr;
    const before = b.spans.capacity;
    for (0..before) |_| try b.appendRetained(a, request);
    try testing.expectEqual(@as(usize, 0), tracked.free_calls);
    try testing.expect(snapshot.spans.ptr != b.spans.items.ptr);
    try testing.expect(text_ptr == b.text.items.ptr);
    try testing.expectEqual(@as(usize, 1025), snapshot.s.len);
    try testing.expectEqual(@as(u8, 'x'), snapshot.s[1024]);
    try testing.expectEqualSlices(v.Span, &.{.{ 1024, 1025 }}, snapshot.spans);
}

test "identical request and literal spelling keeps distinct provenance" {
    const a = testing.allocator;
    const length = 4096;
    const same = try a.alloc(u8, length);
    defer a.free(same);
    @memset(same, 'x');
    var context: std.json.ObjectMap = .empty;
    defer context.deinit(a);
    try context.put(a, "x", .{ .string = same });
    const source = try std.fmt.allocPrint(
        a,
        "{{% set ns = namespace(out=x ~ '{s}') %}}{{% set saved = ns.out %}}" ++
            "{{% set ns.out = ns.out ~ x ~ '{s}' %}}{{{{ saved }}}}|{{{{ ns.out }}}}",
        .{ same, same },
    );
    defer a.free(source);
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const got = try template.renderProvenance(a, &compiled, .null, .null, .{ .object = context }, false, null);
    defer a.free(got.s);
    defer a.free(got.spans);
    try testing.expectEqual(6 * length + 1, got.s.len);
    for (got.s, 0..) |byte, i| {
        try testing.expectEqual(if (i == 2 * length) @as(u8, '|') else 'x', byte);
    }
    const expected = [_]template.Span{ .{ 0, length }, .{ 2 * length + 1, 3 * length + 1 }, .{ 4 * length + 1, 5 * length + 1 } };
    for (got.s, 0..) |_, i| {
        var covered = false;
        for (got.spans) |span| if (span[0] <= i and i < span[1]) {
            covered = true;
            break;
        };
        var wanted = false;
        for (expected) |span| if (span[0] <= i and i < span[1]) {
            wanted = true;
            break;
        };
        try testing.expectEqual(wanted, covered);
    }
}

test "compact builder indices retain layout and never escape a render" {
    const v = @import("value.zig");
    if (@sizeOf(usize) == 8) {
        try testing.expectEqual(@as(usize, 40), @sizeOf(v.Str));
        try testing.expectEqual(@as(usize, 48), @sizeOf(v.Value));
    }
    const a = testing.allocator;
    const body = try a.alloc(u8, 4096);
    defer a.free(body);
    @memset(body, 'x');
    var context: std.json.ObjectMap = .empty;
    defer context.deinit(a);
    try context.put(a, "x", .{ .string = body });
    const source = "{% macro rhs() %}{% set n = namespace(v='') %}" ++
        "{% for i in range(32) %}{% set n.v = x ~ i %}{% endfor %}Z{% endmacro %}" ++
        "{% set ns = namespace(a=x ~ '!') %}{% set saved = ns.a %}" ++
        "{% set ns.a = ns.a ~ rhs() %}{{ saved }}|{{ ns.a }}";
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    for (0..2) |_| {
        const result = try template.renderProvenance(a, &compiled, .null, .null, .{ .object = context }, false, null);
        defer a.free(result.s);
        defer a.free(result.spans);
        try testing.expectEqual(@as(u32, 0), result.builder_id);
        try testing.expectEqual(2 * body.len + 4, result.s.len);
        try testing.expectEqualSlices(u8, body, result.s[0..body.len]);
        try testing.expectEqualStrings("!|", result.s[body.len..][0..2]);
        try testing.expectEqualSlices(u8, body, result.s[body.len + 2 ..][0..body.len]);
        try testing.expectEqualStrings("!Z", result.s[result.s.len - 2 ..]);
        const expected = [_]template.Span{ .{ 0, body.len }, .{ body.len + 2, 2 * body.len + 2 } };
        for (result.s, 0..) |_, i| {
            var actual = false;
            for (result.spans) |span| if (span[0] <= i and i < span[1]) {
                actual = true;
                break;
            };
            var wanted = false;
            for (expected) |span| if (span[0] <= i and i < span[1]) {
                wanted = true;
                break;
            };
            try testing.expectEqual(wanted, actual);
        }
    }
}

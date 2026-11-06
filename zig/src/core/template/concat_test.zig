//! Interleaved string growth preserves immutable values and request byte ranges.
const std = @import("std");
const template = @import("template.zig");

test "interleaved concatenations keep namespace append memory linear" {
    const a = std.testing.allocator;
    var data = std.heap.ArenaAllocator.init(a);
    defer data.deinit();
    const d = data.allocator();
    const content = try d.alloc(u8, 1000);
    @memset(content, 'x');
    const pad = try d.alloc(u8, 4096);
    @memset(pad, 'p');
    var messages = std.json.Array.init(d);
    for (0..600) |_| {
        var message: std.json.ObjectMap = .empty;
        try message.put(d, "content", .{ .string = content });
        try messages.append(.{ .object = message });
    }
    var context: std.json.ObjectMap = .empty;
    try context.put(d, "pad", .{ .string = pad });
    const source = "{% set ns=namespace(out='') %}{% for m in messages %}" ++
        "{% set unrelated=pad+pad %}{% set ns.out=ns.out+m.content %}" ++
        "{% endfor %}{{ ns.out }}";
    const budget = try a.alloc(u8, 32 << 20);
    defer a.free(budget);
    var fba = std.heap.FixedBufferAllocator.init(budget);
    const out = try template.renderChat(fba.allocator(), source, .{ .array = messages }, .null, .{ .object = context }, false, null);
    try std.testing.expectEqual(@as(usize, 600_000), out.len);
    for (out) |byte| try std.testing.expectEqual(@as(u8, 'x'), byte);
}

test "independent namespace strings retain their own append buffers" {
    const a = std.testing.allocator;
    var data = std.heap.ArenaAllocator.init(a);
    defer data.deinit();
    const d = data.allocator();
    const content = try d.alloc(u8, 1000);
    @memset(content, 'x');
    var messages = std.json.Array.init(d);
    for (0..600) |_| try messages.append(.{ .string = content });
    const source = "{% set ns=namespace(a='', b='') %}{% for m in messages %}" ++
        "{% set ns.a=ns.a+m %}{% set ns.b=ns.b~'<'~m~'>' %}" ++
        "{% endfor %}{{ ns.a }}|{{ ns.b }}";
    const budget = try a.alloc(u8, 32 << 20);
    defer a.free(budget);
    var fba = std.heap.FixedBufferAllocator.init(budget);
    const out = try template.renderChat(fba.allocator(), source, .{ .array = messages }, .null, .null, false, null);
    try std.testing.expectEqual(@as(usize, 600_000 + 1 + 601_200), out.len);
    for (out[0..600_000]) |byte| try std.testing.expectEqual(@as(u8, 'x'), byte);
    try std.testing.expectEqual(@as(u8, '|'), out[600_000]);
    for (0..600) |i| {
        const chunk = out[600_001 + i * 1002 ..][0..1002];
        try std.testing.expectEqual(@as(u8, '<'), chunk[0]);
        try std.testing.expectEqualStrings(content, chunk[1..1001]);
        try std.testing.expectEqual(@as(u8, '>'), chunk[1001]);
    }
}

test "appending forks old aliases without changing their text or provenance" {
    const a = std.testing.allocator;
    const seed: [8192]u8 = @splat('s');
    var context: std.json.ObjectMap = .empty;
    defer context.deinit(a);
    try context.put(a, "seed", .{ .string = &seed });
    const source = "{% set ns=namespace(out=seed+'a') %}{% set saved=ns.out %}" ++
        "{% set ns.out=ns.out+'b' %}{% set fork=saved+'c' %}" ++
        "{% set ns.out=ns.out+ns.out %}{{ saved }}|{{ fork }}|{{ ns.out }}";
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const out = try template.renderProvenance(a, &compiled, .null, .null, .{ .object = context }, false, null);
    defer a.free(out.s);
    defer a.free(out.spans);
    const want = seed ++ "a|" ++ seed ++ "ac|" ++ seed ++ "ab" ++ seed ++ "ab";
    try std.testing.expectEqualStrings(want, out.s);
    for (out.s, 0..) |byte, i| {
        var marked = false;
        for (out.spans) |span| if (span[0] <= i and i < span[1]) {
            marked = true;
            break;
        };
        try std.testing.expectEqual(byte == 's', marked);
    }
}

test "namespace list append and prefix replacement use linear memory" {
    const a = std.testing.allocator;
    var data = std.heap.ArenaAllocator.init(a);
    defer data.deinit();
    const d = data.allocator();
    const content = try d.alloc(u8, 256);
    @memset(content, 'x');
    var messages = std.json.Array.init(d);
    for (0..2000) |_| try messages.append(.{ .string = content });
    const sources = [_][]const u8{
        "{% set ns=namespace(xs=[]) %}{% for m in messages %}" ++
            "{% set ns.xs=ns.xs+[m] %}{% endfor %}{{ ns.xs|join('') }}",
        "{% set ns=namespace(xs=messages) %}{% for m in messages %}" ++
            "{% set ns.xs=ns.xs[:-1]+[m] %}{% endfor %}{{ ns.xs|join('') }}",
    };
    const budget = try a.alloc(u8, 16 << 20);
    defer a.free(budget);
    for (sources) |source| {
        var fba = std.heap.FixedBufferAllocator.init(budget);
        const out = try template.renderChat(fba.allocator(), source, .{ .array = messages }, .null, .null, false, null);
        try std.testing.expectEqual(@as(usize, 512_000), out.len);
        for (out) |byte| try std.testing.expectEqual(@as(u8, 'x'), byte);
    }
}

test "list concatenation and slice forks preserve saved elements and provenance" {
    const a = std.testing.allocator;
    var context: std.json.ObjectMap = .empty;
    defer context.deinit(a);
    try context.put(a, "seed", .{ .string = "s" });
    const source = "{% set ns=namespace(xs=[seed]) %}{% set saved=ns.xs %}" ++
        "{% set ns.xs=ns.xs+['a'] %}{% set prefix=ns.xs[:-1] %}" ++
        "{% set fork=prefix+['b'] %}{% set ns.xs=ns.xs[:-1]+['c'] %}" ++
        "{% set doubled=ns.xs+ns.xs %}{{ saved|join }}|{{ prefix|join }}|" ++
        "{{ fork|join }}|{{ ns.xs|join }}|{{ doubled|join }}|{{ doubled[::-1]|join }}|" ++
        "{{ doubled[1::2]|join }}|{{ doubled[9:1]|length }}|{{ (1,)+(2,) }}";
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const out = try template.renderProvenance(a, &compiled, .null, .null, .{ .object = context }, false, null);
    defer a.free(out.s);
    defer a.free(out.spans);
    try std.testing.expectEqualStrings("s|s|sb|sc|scsc|cscs|cc|0|(1, 2)", out.s);
    for (out.s, 0..) |byte, i| {
        var marked = false;
        for (out.spans) |span| if (span[0] <= i and i < span[1]) {
            marked = true;
            break;
        };
        try std.testing.expectEqual(byte == 's', marked);
    }
}

test "deep sequence concatenations flatten without using the call stack" {
    const v = @import("value.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var rt: v.Rt = .{ .a = arena.allocator() };
    var xs = try rt.list(&.{}, .list);
    for (0..20_000) |i| {
        const items = try rt.a.alloc(v.Value, 1);
        items[0] = .{ .int = @intCast(i) };
        const next = try rt.list(items, .list);
        xs = .{ .list = try xs.list.concat(rt.a, next.list) };
    }
    try std.testing.expectEqual(@as(i64, 19_999), xs.list.at(19_999).int);
    const values = try xs.list.values(rt.a);
    for (values, 0..) |item, i| try std.testing.expectEqual(@as(i64, @intCast(i)), item.int);
    try std.testing.expectEqual(@intFromPtr(values.ptr), @intFromPtr((try xs.list.values(rt.a)).ptr));
}

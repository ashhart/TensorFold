const std = @import("std");
const source = @import("tp_source.zig").source;
const testing = std.testing;

test "pair kernel source never polls host-written memory" {
    for ([_][]const u8{
        "while (",
        "atomic_load_explicit",
        "tp_post_poll",
        "polls",
        "400000000",
        "sync[2048]",
    }) |polling| try testing.expect(std.mem.indexOf(u8, source, polling) == null);
}

test "every pair kernel checks a full-width epoch before reading payload" {
    const marker = "kernel void ";
    var remaining: []const u8 = source;
    var kernels: usize = 0;
    while (std.mem.indexOf(u8, remaining, marker)) |at| {
        const declaration = remaining[at..];
        const open = std.mem.indexOfScalar(u8, declaration, '{') orelse return error.TestUnexpectedResult;
        const header = declaration[0..open];
        try testing.expect(std.mem.indexOf(u8, header, "constant ulong& expected") != null);
        const word: []const u8 = if (std.mem.indexOf(u8, header, "const device ulong* ready") != null)
            "ready[0] != expected"
        else if (std.mem.indexOf(u8, header, "const device ulong* previous") != null)
            "previous[0] != expected"
        else
            return error.TestUnexpectedResult;
        var end = open + 1;
        var depth: usize = 1;
        while (end < declaration.len and depth != 0) : (end += 1) switch (declaration[end]) {
            '{' => depth += 1,
            '}' => depth -= 1,
            else => {},
        };
        try testing.expectEqual(@as(usize, 0), depth);
        const body = declaration[open + 1 .. end - 1];
        const guard = std.mem.indexOf(u8, body, word) orelse return error.TestUnexpectedResult;
        try testing.expect(guard >= 4 and std.mem.eql(u8, body[guard - 4 .. guard], "if ("));
        const after = body[guard + word.len ..];
        try testing.expect(std.mem.startsWith(u8, after, ") return;") or std.mem.startsWith(u8, after, " || "));
        const stop = std.mem.indexOfScalar(u8, after, ';') orelse return error.TestUnexpectedResult;
        try testing.expect(std.mem.endsWith(u8, after[0 .. stop + 1], ") return;"));
        for ([_][]const u8{
            "src[", "PEER[", "THEIRS[", "MINE[", "mine[", "peer[", "pick[", "ids[", "Y[", "logits +",
        }) |read| if (std.mem.indexOf(u8, body, read)) |payload| try testing.expect(guard < payload);
        kernels += 1;
        remaining = declaration[end..];
    }
    try testing.expect(kernels > 0);
}

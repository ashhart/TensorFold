const std = @import("std");
const library = @import("windows_library.zig");
const t = std.testing;
const u = std.unicode.utf8ToUtf16LeStringLiteral;
const Fake = struct {
    pub const Handle = usize;
    var result: ?usize = null;
    var full: []const u16 = &.{};
    var fail_load = false;
    var resolves: usize = 0;
    var loads: usize = 0;
    var asked: [1024]u16 = undefined;
    var asked_len: usize = 0;
    var loaded: [1024]u16 = undefined;
    var loaded_len: usize = 0;
    fn reset() void {
        result = null;
        full = &.{};
        fail_load = false;
        resolves = 0;
        loads = 0;
    }
    pub fn fullPath(path: [:0]const u16, out: []u16) usize {
        resolves += 1;
        asked_len = path.len;
        @memcpy(asked[0..asked_len], path);
        const n = result orelse full.len;
        if (n > 0 and n < out.len) @memcpy(out[0..n], full[0..n]);
        return n;
    }
    pub fn load(path: [:0]const u16) ?Handle {
        loads += 1;
        loaded_len = path.len;
        @memcpy(loaded[0..loaded_len], path);
        std.debug.assert(path[path.len] == 0);
        return if (fail_load) null else 7;
    }
};
test "Windows DLL: relative, absolute and Unicode paths use the production resolver" {
    const cases = .{
        .{ "runtime/cublasLt.dll", u("runtime\\cublasLt.dll"), u("C:\\project\\runtime\\cublasLt.dll") },
        .{ "C:\\CUDA\\cublas.dll", u("C:\\CUDA\\cublas.dll"), u("C:\\CUDA\\cublas.dll") },
        .{ "runtime/音𝄞.dll", u("runtime\\音𝄞.dll"), u("C:\\project\\runtime\\音𝄞.dll") },
    };
    inline for (cases) |c| {
        Fake.reset();
        Fake.full = c[2];
        try t.expectEqual(@as(usize, 7), try library.open(Fake, c[0]));
        try t.expectEqual(@as(usize, 1), Fake.resolves);
        try t.expectEqual(@as(usize, 1), Fake.loads);
        try t.expectEqualSlices(u16, c[1], Fake.asked[0..Fake.asked_len]);
        try t.expectEqualSlices(u16, c[2], Fake.loaded[0..Fake.loaded_len]);
    }
}
test "Windows DLL: bare names, failed resolution, bounds and failed loads" {
    Fake.reset();
    _ = try library.open(Fake, "nvcuda.dll");
    try t.expectEqual(@as(usize, 0), Fake.resolves);
    try t.expectEqualSlices(u16, u("nvcuda.dll"), Fake.loaded[0..Fake.loaded_len]);
    for ([_]usize{ 0, 1024, 2048 }) |n| {
        Fake.reset();
        Fake.result = n;
        _ = try library.open(Fake, "runtime/cublas.dll");
        try t.expectEqual(@as(usize, 1), Fake.resolves);
        try t.expectEqualSlices(u16, u("runtime\\cublas.dll"), Fake.loaded[0..Fake.loaded_len]);
    }
    Fake.reset();
    Fake.full = u("C:\\runtime\\missing.dll");
    Fake.fail_load = true;
    try t.expectError(error.FileNotFound, library.open(Fake, "runtime/missing.dll"));
    try t.expectEqual(@as(usize, 1), Fake.loads);
    Fake.reset();
    try t.expectError(error.NameTooLong, library.open(Fake, &@as([1024]u8, @splat('a'))));
    try t.expectError(error.InvalidWtf8, library.open(Fake, &.{0xff}));
    try t.expectEqual(@as(usize, 0), Fake.loads);
    try t.expectEqual(@as(usize, 0), Fake.resolves);
}

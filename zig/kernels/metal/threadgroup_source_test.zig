//! Host source checks for complete Inkling libraries and declaration reservation edits.
const std = @import("std");
const guard = @import("threadgroup_source.zig");
const ink = @import("inkling2_sources.zig");
const attribute = "[[max_total_threads_per_threadgroup(256)]]\n";

fn check(source: []const u8, expected: []const u8) !void {
    const out = try std.testing.allocator.alloc(u8, source.len);
    defer std.testing.allocator.free(out);
    const len = try guard.write(source, out);
    try std.testing.expectEqualStrings(expected, out[0..len]);
}

test "Inkling assembled runtime has one reservation per kernel including convolution entries" {
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, ink.runtime, attribute ++ attribute));
    try std.testing.expectEqualStrings(ink.runtime, @import("sources.zig").inkling2_runtime);
}

test "equal reservations collapse without altering kernel bodies or unrelated source bytes" {
    const body = "kernel void k() { int x = 2; x += 3; }\n";
    try check(attribute ++ attribute ++ body, attribute ++ body);
    try std.testing.expectEqualStrings(attribute ++ body, guard.normalize(attribute ++ attribute ++ body));
    try check(attribute ++ attribute ++ attribute ++ body, attribute ++ body);
    const combined = "[[kernel, max_total_threads_per_threadgroup(256)]] void k() {}\n";
    try check(attribute ++ combined, combined);
    try check(combined ++ attribute ++ body, combined ++ attribute ++ body);
    try check("[[kernel, max_total_threads_per_threadgroup(256)]]\n" ++ attribute ++ "void k() {}", "[[kernel, max_total_threads_per_threadgroup(256)]]\nvoid k() {}");
}

test "conflicting and malformed reservations are refused" {
    const inputs = [_][]const u8{
        attribute ++ "[[max_total_threads_per_threadgroup(128)]] kernel void k() {}",
        "[[kernel, max_total_threads_per_threadgroup(128)]]\n" ++ attribute ++ "void k() {}",
    };
    var out: [512]u8 = undefined;
    for (inputs) |source| try std.testing.expectError(error.ConflictingThreadgroupReservation, guard.write(source, &out));
    try std.testing.expectError(error.RepeatedCombinedReservation, guard.write("[[kernel, max_total_threads_per_threadgroup(256), max_total_threads_per_threadgroup(256)]] void k() {}", &out));
    try std.testing.expectError(error.BadMetalSource, guard.write("[[max_total_threads_per_threadgroup()]] kernel void k() {}", &out));
}

test "comments strings nested expressions and declaration boundaries do not create false reservations" {
    const head = "[[max_total_threads_per_threadgroup((8 * 32))]]\n";
    const spaced = "[[max_total_threads_per_threadgroup((8*32))]]\n";
    const body = "kernel void k() { const char* s = \"[[max_total_threads_per_threadgroup(1)]]\"; }\n";
    const comment = "/" ++ "* }; max_total_threads_per_threadgroup(1) */\n";
    try check(head ++ comment ++ spaced ++ body, head ++ comment ++ body);
    try check(head ++ body ++ "// max_total_threads_per_threadgroup(1)\n" ++ head ++ body, head ++ body ++ "// max_total_threads_per_threadgroup(1)\n" ++ head ++ body);
}

test "every complete Inkling library is unchanged by a second reservation check" {
    inline for (@typeInfo(ink).@"struct".decl_names) |decl| {
        const source = @field(ink, decl);
        try check(source, source);
    }
}

test "LFM2 masked builder preserves one existing reservation on either side of the kernel marker" {
    const masked = @import("lfm2/prompt_masked.zig");
    const before = "[[max_total_threads_per_threadgroup(128)]]\n" ++ "[[kernel]]";
    const after = "[[kernel]] [[max_total_threads_per_threadgroup(128)]]";
    const with_before = comptime blk: {
        @setEvalBranchQuota(20_000_000);
        const at = std.mem.indexOf(u8, masked.original, "[[kernel]]").?;
        break :blk masked.original[0..at] ++ before ++ masked.original[at + "[[kernel]]".len ..];
    };
    const with_after = comptime blk: {
        @setEvalBranchQuota(20_000_000);
        const at = std.mem.indexOf(u8, masked.original, "[[kernel]]").?;
        break :blk masked.original[0..at] ++ after ++ masked.original[at + "[[kernel]]".len ..];
    };
    const count = "max_total_threads_per_threadgroup(128)";
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, masked.build(with_before), count));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, masked.build(with_after), count));
    try check(masked.source, masked.source);
}

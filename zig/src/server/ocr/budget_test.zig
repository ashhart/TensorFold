//! The image-encode budget (docs/api.md OCR: Pages and budgets).
const std = @import("std");
const budget = @import("budget.zig");
const types = @import("types.zig");
const t = std.testing;

test "ocr budget: the 257th encode faults before it runs" {
    var fault: types.Fault = undefined;
    var e = budget.Encodes.init(1 << 30, 1);
    for (0..256) |_| try e.before(&fault, 1, 1);
    try t.expectError(error.Faulted, e.before(&fault, 1, 1));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqualStrings("too many image regions", fault.message);
    try t.expectEqual(@as(u32, 256), e.count);
}

test "ocr budget: encoded pixels stop at 4x the request's page pixels" {
    var fault: types.Fault = undefined;
    var e = budget.Encodes.init(100 * 100, 1);
    try t.expectEqual(@as(u64, 40_000), e.limit_pixels);
    try e.before(&fault, 100, 300);
    try e.before(&fault, 100, 100); // exactly at the limit
    try t.expectError(error.Faulted, e.before(&fault, 1, 1));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqual(@as(u64, 40_000), e.pixels);
    try t.expectEqual(@as(u32, 2), e.count);
}

test "ocr budget: encoded bytes fault after the encode that passes 16 MiB and 2 MiB a page" {
    for ([_]struct { usize, u64 }{ .{ 1, 18 << 20 }, .{ 16, 48 << 20 } }) |c| {
        var fault: types.Fault = undefined;
        var e = budget.Encodes.init(1 << 30, c[0]);
        try t.expectEqual(c[1], e.max_bytes);
        try e.after(&fault, c[1] - 1);
        try e.after(&fault, 1); // exactly at the limit
        try t.expectError(error.Faulted, e.after(&fault, 1));
        try t.expectEqual(types.Class.unreadable, fault.class);
        try t.expectEqualStrings("too many image regions", fault.message);
    }
}

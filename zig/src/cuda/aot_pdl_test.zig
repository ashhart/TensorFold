const std = @import("std");
const aot = @import("aot.zig");

const disabled =
    \\{"kernels":[{"fn":"test","hash":"ab","name":"disabled_pdl","num_warps":4,
    \\ "pdl":true,"params":[],"consts":{"PDL":{"int":0}}}]}
;
const enabled =
    \\{"kernels":[{"fn":"test","hash":"ab","name":"enabled_pdl","num_warps":4,
    \\ "pdl":false,"params":[],"consts":{"PDL":{"int":1}}}]}
;

fn refused(text: []const u8) !void {
    if (aot.parseSpecs(std.testing.allocator, text)) |parsed| {
        parsed.deinit();
        return error.ExpectedPdlRefusal;
    } else |err| try std.testing.expectEqual(error.PdlMismatch, err);
}

test "captured PDL: disabled constexpr refuses an enabled launch" {
    try refused(disabled);
}

test "captured PDL: enabled constexpr refuses a disabled launch" {
    try refused(enabled);
}

test "captured PDL: every spec is checked before any cubin is opened" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const driver: @import("driver.zig").Driver = undefined;
    for ([_][]const u8{ disabled, enabled }) |text| {
        const first =
            \\{"kernels":[{"fn":"test","hash":"cd","name":"valid_first","num_warps":4,"params":[],"consts":{}},
        ;
        const both = try std.mem.concat(t.allocator, u8, &.{ first, text[12..] });
        defer t.allocator.free(both);
        try tmp.dir.writeFile(t.io, .{ .sub_path = "aot.json", .data = both });
        try t.expectError(error.PdlMismatch, aot.Set.load(t.allocator, t.io, &driver, 0, dir));
    }
}

test "captured PDL: missing or non-boolean constexpr cannot enable overlap" {
    const t = std.testing;
    const bad = [_][]const u8{ "{}", "{\"PDL\":{}}", "{\"PDL\":{\"f32\":1}}", "{\"PDL\":{\"int\":2}}", "{\"PDL\":{\"int\":-1}}", "{\"PDL\":{\"int\":1,\"f32\":0}}" };
    for (bad) |consts| {
        const text = try std.fmt.allocPrint(
            t.allocator,
            "{{\"kernels\":[{{\"fn\":\"test\",\"hash\":\"ab\",\"name\":\"invalid_pdl\",\"num_warps\":4," ++
                "\"pdl\":true,\"params\":[],\"consts\":{s}}}]}}",
            .{consts},
        );
        defer t.allocator.free(text);
        try refused(text);
    }
}

test "captured PDL: matching flags and legacy non-PDL specs keep their metadata" {
    const t = std.testing;
    const good =
        \\{"kernels":[
        \\ {"fn":"test","hash":"ab","name":"off","num_warps":4,"pdl":false,"params":[],"consts":{"PDL":{"int":0}}},
        \\ {"fn":"test","hash":"cd","name":"on","num_warps":4,"pdl":true,"params":[],"consts":{"PDL":{"int":1}}},
        \\ {"fn":"test","hash":"ef","name":"legacy","num_warps":4,"params":[],"consts":{}},
        \\ {"fn":"test","hash":"aa","name":"default_off","num_warps":4,"params":[],"consts":{"PDL":{"int":0}}}
        \\]}
    ;
    const parsed = try aot.parseSpecs(t.allocator, good);
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 4), parsed.value.kernels.len);
    for (parsed.value.kernels, [_]bool{ false, true, false, false }) |k, want| try t.expectEqual(want, k.pdl);
}

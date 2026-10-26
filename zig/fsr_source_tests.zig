const std = @import("std");
const source = @import("src/families/flashnext/fsr_source.zig");
const gu = @embedFile("kernels/metal/flashnext/qa_expert_gateup_fd30a642315017a8.metal");
const down = @embedFile("kernels/metal/flashnext/qa_expert_down_y_346a3e7c6008e1ac.metal");

test "FSR source touches only routed addresses and leaves every floating point expression intact" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ gu, down }, [_]source.Kind{ .gate, .down }) |original, kind| {
        const text = try source.patch(a, original, kind);
        try std.testing.expectError(error.FsrKernelPatch, source.patch(a, text, kind));
        const kernel = std.mem.indexOf(u8, original, "[[kernel]]").?;
        try std.testing.expectEqualStrings(original[0..kernel], text[0..kernel]);
        for ([_][]const u8{ "fma(", "simd_sum(", "bfloat(", "simd_topk<", "metal::exp(", "gateup_rows<SWB", "down_rows<SWB" }) |expr|
            try std.testing.expectEqual(std.mem.count(u8, original, expr), std.mem.count(u8, text, expr));
        const shared = if (kind == .gate)
            "if (shared) gateup_rows<SWB, SWG, K, RPS>(SGW, SUW, SGS, SGB, SUS, SUB, size_t(row0), X + r * K, lane, ag, au);"
        else
            "if (shared) down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out);";
        try std.testing.expect(std.mem.indexOf(u8, text, shared) != null);
        const wrong = try std.mem.replaceOwned(u8, a, original, "PICK", "MOVED");
        if (kind == .down) try std.testing.expectError(error.FsrKernelPatch, source.patch(a, wrong, kind));
    }
}

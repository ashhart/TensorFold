//! Complete runtime-embedded libraries, not a partial source-file fixture.
const std = @import("std");
const ks = @import("kernel_sources");
const patch = @import("src/families/flashnext/fsr_prompt_source.zig");
const contract_source = @import("src/families/flashnext/fsr_contract_source.zig");
const records = @import("src/core/contract/fsr_records.zig");

test "FSR complete prompt library keeps the appended non-expert kernels byte-identical" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const text = try patch.patch(a, ks.flashnext_qmm6, false);
    const suffix = @embedFile("kernels/metal/prefill/qmm6_nax_b.metal");
    try std.testing.expect(std.mem.endsWith(u8, text, suffix));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "FSR_MAP [[buffer(27)]]"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "max_total_threads_per_threadgroup(128)"));
    try std.testing.expectError(error.FsrPromptPatch, patch.patch(a, text, false));
    const drift = try std.mem.replaceOwned(u8, a, ks.flashnext_qmm6, "uint sg [[simdgroup_index_in_threadgroup]]", "uint group [[simdgroup_index_in_threadgroup]]");
    try std.testing.expectError(error.FsrPromptPatch, patch.patch(a, drift, false));
}

fn entry(a: std.mem.Allocator, text: []const u8, name: []const u8) !usize {
    const marker = try std.fmt.allocPrint(a, "[[kernel]] void {s}(", .{name});
    if (std.mem.count(u8, text, marker) != 1) return error.MissingKernel;
    return std.mem.indexOf(u8, text, marker).?;
}

fn body(a: std.mem.Allocator, text: []const u8, name: []const u8) ![]const u8 {
    const at = try entry(a, text, name);
    const open = std.mem.indexOfScalarPos(u8, text, at, '{') orelse return error.MissingBody;
    var level: usize = 1;
    for (text[open + 1 ..], open + 1..) |c, i| {
        if (c == '{') level += 1;
        if (c == '}') level -= 1;
        if (level == 0) return text[open + 1 .. i];
    }
    return error.MissingBody;
}

test "FSR all fifteen launch records price their actual source threadgroup arrays" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    try std.testing.expectEqual(@as(usize, 15), records.metal.len);
    for (records.metal) |record| {
        const symbol = record.name[std.mem.lastIndexOfScalar(u8, record.name, ' ').? + 1 ..];
        const generated = try contract_source.source(a, symbol);
        const wide = std.mem.indexOf(u8, symbol, "qmmw") != null or std.mem.indexOf(u8, symbol, "guw") != null;
        const name = if (!wide) symbol else if (std.mem.indexOf(u8, symbol, "guw") != null) "tf_gather_guw_nax_##BM" else "tf_gather_qmmw_nax_##BM";
        const kernel = try body(a, generated.text, name);
        var expected: u32 = 0;
        if (std.mem.eql(u8, symbol, "fz_route")) {
            try std.testing.expect(std.mem.indexOf(u8, generated.text, "FZ_UMAX = 320") != null);
            try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, kernel, "threadgroup int "));
            try std.testing.expect(std.mem.indexOf(u8, kernel, "threadgroup int tids[FZ_UMAX];") != null);
            try std.testing.expect(std.mem.indexOf(u8, kernel, "threadgroup int first[FZ_UMAX];") != null);
            expected = 2 * 320 * @sizeOf(i32);
        } else if (std.mem.startsWith(u8, symbol, "tf_gather_")) {
            const arrays: u32 = if (std.mem.indexOf(u8, symbol, "_gu") != null) 2 else 1;
            const shape = "[64 * (64 + 16 / sizeof(bfloat16_t))]";
            const declaration = if (arrays == 1)
                "threadgroup bfloat16_t tile" ++ shape ++ ";"
            else
                "threadgroup bfloat16_t tile_g" ++ shape ++ ", tile_u" ++ shape ++ ";";
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, kernel, "threadgroup "));
            try std.testing.expect(std.mem.indexOf(u8, kernel, declaration) != null);
            expected = arrays * 64 * (64 + 16 / @sizeOf(u16)) * @sizeOf(u16);
        } else try std.testing.expect(std.mem.indexOf(u8, kernel, "threadgroup ") == null);
        try std.testing.expectEqual(expected, record.tg_memory);
        try std.testing.expectEqual(@as(u32, 0), record.dynamic);
        const kernel_at = try entry(a, generated.text, name);
        const bound = try std.fmt.allocPrint(a, "[[max_total_threads_per_threadgroup({d})]] ", .{record.threads});
        try std.testing.expect(std.mem.endsWith(u8, generated.text[0..kernel_at], bound));
    }
}

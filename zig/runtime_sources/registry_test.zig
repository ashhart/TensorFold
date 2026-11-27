//! Exact production variants and compile options required in the runtime source inventory.
const std = @import("std");
const mtl = @import("metal");
const inventory = @import("root.zig");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;
const t = std.testing;

fn expectSource(set: Set, text: []const u8, options: mtl.CompileOptions) !void {
    for (set.entries.items) |entry| {
        if (std.meta.eql(entry.options, options) and std.mem.eql(u8, entry.text, text)) return;
    }
    return error.MissingExactRuntimeSource;
}

fn expectEvery(set: Set, text: []const u8) !void {
    try expectSource(set, text, mtl.mlx());
}

test "Metal runtime source registry: pre-M5 Qwen27 joined attention is exact" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const set = try inventory.generate(a, false, 13);
    const attention = @import("../src/families/qwen27/simd_attention.zig");
    for ([_]bool{ false, true }) |paged| for ([_][2]u32{ .{ 6, 4 }, .{ 4, 4 }, .{ 8, 4 } }) |g| {
        try expectEvery(set, try attention.joined(a, false, false, g[0], g[1], paged));
    };
}

test "Metal runtime source registry: H3 and Ling standalone libraries preserve options" {
    const precise = mtl.CompileOptions{ .language = mtl.mlx().language, .math_functions = .precise };
    const encoder = @import("../kernels/metal/h3/image_encoder.zig").source;
    const prompt = @import("../kernels/metal/h3/prompt_kernels.zig").source;
    const decoder = @import("../kernels/metal/h3/image_decoder.zig").source;
    const ling = @import("../kernels/metal/ling/sources.zig");
    for ([_]bool{ false, true }) |units| {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const set = try inventory.generate(arena.allocator(), units, if (units) 17 else 13);
        try expectSource(set, encoder, precise);
        try expectEvery(set, prompt);
        try expectEvery(set, decoder);
        try expectEvery(set, ling.route);
        try expectEvery(set, ling.mla);
        try expectEvery(set, ks.lfm3);
    }
}

test "Metal runtime source registry: Flash Next scale-index variants match both loaders" {
    const experts4 = @import("../src/families/flashnext/experts4.zig");
    const mb2 = @import("../src/families/flashnext/gx_mb2.zig");
    const prompt_index = @import("../src/families/flashnext/prompt_index_source.zig");
    const qmmw = @import("../src/core/qmmw_source.zig");
    const precision = @import("../src/core/attention_precision.zig");
    const fsr = @import("../src/families/flashnext/fsr_prompt_source.zig");
    for ([_]bool{ false, true }) |units| {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const set = try inventory.generate(a, units, if (units) 17 else 13);
        const grouped = try experts4.indexedGrouped(a, ks.flashnext_gx);
        try expectEvery(set, grouped);
        try expectEvery(set, try mb2.text(a, grouped, true));
        for ([_]bool{ false, true }) |ssd| {
            const body = if (ssd) try fsr.patch(a, ks.flashnext_qmmw, true) else ks.flashnext_qmmw;
            const indexed = try prompt_index.patch(a, body);
            const specialized = try qmmw.source(a, indexed, 4, 32);
            try expectEvery(set, try precision.source(a, units, specialized, ks.nax, false));
        }
    }
}

test "Metal runtime source registry: raw Hy4 elementwise is present on pre-M5 and tensor units" {
    for ([_]bool{ false, true }) |units| {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const set = try inventory.generate(arena.allocator(), units, if (units) 17 else 13);
        try expectEvery(set, ks.elementwise);
    }
}

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

test "FSR cold expert inventory excludes shared weights and prices every stored component" {
    const weights = @import("src/families/flashnext/load_weights.zig");
    const Entry = struct { dtype: enum { u32, bf16, f32 }, begin: usize = 0, end: usize };
    var inv: weights.Inventory = .{};
    const bases = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var name: [160]u8 = undefined;
    for (bases) |base| for ([_][]const u8{ "weight", "scales", "biases" }) |suffix| {
        try inv.include(try std.fmt.bufPrint(&name, "language_model.model.layers.0.mlp.switch_mlp.{s}.{s}", .{ base, suffix }), Entry{ .dtype = .u32, .end = 512 * 4 });
    };
    try inv.include("language_model.model.layers.0.mlp.shared_expert.gate_proj.weight", Entry{ .dtype = .u32, .end = 64 });
    try std.testing.expectEqual(@as(u64, 512 * 36), inv.routed);
    try std.testing.expectEqual(inv.routed + 64, inv.raw);
    try std.testing.expectEqual(@as([9]u64, @splat(4)), inv.expert_parts);
    try std.testing.expectError(error.BadExpertTensor, inv.include("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", Entry{ .dtype = .u32, .end = 513 }));
}

test "FSR prompt and six-bit variants retain resident arithmetic and reject source drift" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = @import("src/families/flashnext/fsr_prompt_source.zig");
    const six = @embedFile("kernels/metal/prefill/qmm6_nax.metal");
    const wide = @embedFile("kernels/metal/prefill/qmmw_nax.metal");
    for ([_][]const u8{ six, wide }, [_]bool{ false, true }) |text, is_wide| {
        const slotted = try prompt.patch(a, text, is_wide);
        try std.testing.expectError(error.FsrPromptPatch, prompt.patch(a, slotted, is_wide));
        for ([_][]const u8{ "k_loop", "fma(", "store_act<", "frag<float>", "expert_tile<BM>", "row + tm + SM <= M" }) |expr|
            try std.testing.expectEqual(std.mem.count(u8, text, expr), std.mem.count(u8, slotted, expr));
    }
    const xnew = @import("src/families/flashnext/replay_expert_source.zig").text;
    const xsource = @import("src/families/flashnext/fsr_xnew_source.zig");
    const slotted = try xsource.patch(a, xnew);
    try std.testing.expectError(error.FsrXnewPatch, xsource.patch(a, slotted));
    for ([_][]const u8{ "fma(", "fz_group(", "simd_shuffle_xor(", "bsilu(", "metal::exp(", "simd_topk<" }) |expr|
        try std.testing.expectEqual(std.mem.count(u8, xnew, expr), std.mem.count(u8, slotted, expr));
}

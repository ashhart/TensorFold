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

test "FSR source injectors preserve an intended reservation already on a declaration" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const attr64 = "[[max_total_threads_per_threadgroup(64)]] ";
    for ([_][]const u8{ gu, down }, [_]source.Kind{ .gate, .down }) |original, kind| {
        const attributed = try std.mem.replaceOwned(u8, a, original, "[[kernel]]", attr64 ++ "[[kernel]]");
        const text = try source.patch(a, attributed, kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "max_total_threads_per_threadgroup(64)"));
    }
    const prompt = @import("src/families/flashnext/fsr_prompt_source.zig");
    const attr128 = "[[max_total_threads_per_threadgroup(128)]] ";
    const six = @embedFile("kernels/metal/prefill/qmm6_nax.metal");
    const wide = @embedFile("kernels/metal/prefill/qmmw_nax.metal");
    for ([_][]const u8{ six, wide }, [_]bool{ false, true }) |original, is_wide| {
        const names: []const []const u8 = if (is_wide)
            &.{ "tf_gather_qmmw_nax_##BM", "tf_gather_guw_nax_##BM" }
        else
            &.{ "tf_gather_qmm6_nax_32", "tf_gather_qmm6_nax_64", "tf_gather_qmm6_nax_128", "tf_gather_gu6_nax_32", "tf_gather_gu6_nax_64" };
        var attributed = original;
        for (names) |name| {
            const marker = try std.fmt.allocPrint(a, "[[kernel]] void {s}(", .{name});
            const replacement = try std.fmt.allocPrint(a, "{s}{s}", .{ attr128, marker });
            attributed = try std.mem.replaceOwned(u8, a, attributed, marker, replacement);
        }
        const text = try prompt.patch(a, attributed, is_wide);
        try std.testing.expectEqual(names.len, std.mem.count(u8, text, "max_total_threads_per_threadgroup(128)"));
    }
    var xnew: []const u8 = @import("src/families/flashnext/replay_expert_source.zig").text;
    for ([_][]const u8{ "fz_xgu", "fz_xdown" }) |name| {
        const marker = try std.fmt.allocPrint(a, "[[kernel]] void {s}(", .{name});
        const replacement = try std.fmt.allocPrint(a, "{s}{s}", .{ attr128, marker });
        xnew = try std.mem.replaceOwned(u8, a, xnew, marker, replacement);
    }
    xnew = try std.mem.replaceOwned(u8, a, xnew, "[[kernel]] void fz_route(", "[[max_total_threads_per_threadgroup(512)]] [[kernel]] void fz_route(");
    const text = try @import("src/families/flashnext/fsr_xnew_source.zig").patch(a, xnew);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "max_total_threads_per_threadgroup(128)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "max_total_threads_per_threadgroup(512)"));
}

fn expectDeclaration(text: []const u8, name: []const u8, threads: ?usize) !void {
    var marker_storage: [160]u8 = undefined;
    const marker = std.fmt.bufPrint(&marker_storage, "[[kernel]] void {s}(", .{name}) catch unreachable;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, marker));
    const at = std.mem.indexOf(u8, text, marker).?;
    const boundary = std.mem.lastIndexOfAny(u8, text[0..at], "{};");
    const prefix = text[(if (boundary) |i| i + 1 else 0)..at];
    try std.testing.expectEqual(@as(usize, if (threads == null) 0 else 1), std.mem.count(u8, prefix, "max_total_threads_per_threadgroup"));
    if (threads) |count| {
        var storage: [80]u8 = undefined;
        const attribute = std.fmt.bufPrint(&storage, "[[max_total_threads_per_threadgroup({d})]]", .{count}) catch unreachable;
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, prefix, attribute));
    }
}

fn withAttribute(a: std.mem.Allocator, text: []const u8, name: []const u8, attribute: []const u8) ![]const u8 {
    const marker = try std.fmt.allocPrint(a, "[[kernel]] void {s}(", .{name});
    const replacement = try std.fmt.allocPrint(a, "{s} {s}", .{ attribute, marker });
    return std.mem.replaceOwned(u8, a, text, marker, replacement);
}

test "FSR final qmmw assemblies reserve each changed declaration once for every width and composition" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ks = @import("kernel_sources");
    const indexed = @import("src/families/flashnext/prompt_index_source.zig").patch;
    const slotted = @import("src/families/flashnext/fsr_prompt_source.zig").patch;
    var assemblies: usize = 0;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64, 128 }) |group| for ([_]bool{ false, true }) |ssd| {
        const body = if (ssd) try slotted(a, ks.flashnext_qmmw, true) else ks.flashnext_qmmw;
        for ([_]bool{ false, true }) |use_index| {
            const selected = if (use_index) try indexed(a, body) else body;
            const width = try @import("src/core/qmmw_source.zig").source(a, selected, bits, group);
            const final = try @import("src/core/attention_precision.zig").source(a, true, width, ks.nax, false);
            const expected: ?usize = if (use_index or ssd) 128 else null;
            try expectDeclaration(final, "tf_gather_qmmw_nax_##BM", expected);
            try expectDeclaration(final, "tf_gather_guw_nax_##BM", expected);
            assemblies += 1;
        }
    };
    try std.testing.expectEqual(@as(usize, 72), assemblies);
}

test "FSR final XNew assemblies reserve each changed declaration once for every packing and SSD state" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const expert = try @import("src/families/flashnext/fsr_contract_source.zig").recorded("qa_expert_gateup_fd30a642315017a8.metal");
    const declarations = [_]struct { name: []const u8, threads: usize }{
        .{ .name = "fz_xgu", .threads = 128 }, .{ .name = "fz_xdown", .threads = 128 }, .{ .name = "fz_route", .threads = 512 },
    };
    var assemblies: usize = 0;
    for ([_]bool{ false, true }) |packing| for ([_]bool{ false, true }) |ssd| {
        const assembled = try @import("src/families/flashnext/replay_compile.zig").xnewSource(a, expert, packing, ssd);
        for (declarations) |d| try expectDeclaration(assembled.text, d.name, if (ssd) d.threads else null);
        assemblies += 1;
    };
    try std.testing.expectEqual(@as(usize, 4), assemblies);
}

test "FSR final declaration injectors reject repeated and conflicting reservations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ks = @import("kernel_sources");
    const indexed = @import("src/families/flashnext/prompt_index_source.zig").patch;
    for ([_][]const u8{ "tf_gather_qmmw_nax_##BM", "tf_gather_guw_nax_##BM" }) |name| {
        for ([_][]const u8{ "[[max_total_threads_per_threadgroup(64)]]", "[[max_total_threads_per_threadgroup(128)]] [[max_total_threads_per_threadgroup(128)]]" }) |attribute| {
            try std.testing.expectError(error.KernelPatch, indexed(a, try withAttribute(a, ks.flashnext_qmmw, name, attribute)));
        }
    }
    const slotted = try @import("src/families/flashnext/fsr_prompt_source.zig").patch(a, ks.flashnext_qmmw, true);
    const repeated = try withAttribute(a, slotted, "tf_gather_qmmw_nax_##BM", "[[max_total_threads_per_threadgroup(128)]]");
    try std.testing.expectError(error.KernelPatch, indexed(a, repeated));
    const xnew = @import("src/families/flashnext/replay_expert_source.zig").text;
    const patch = @import("src/families/flashnext/fsr_xnew_source.zig").patch;
    const declarations = [_]struct { name: []const u8, threads: usize }{
        .{ .name = "fz_xgu", .threads = 128 }, .{ .name = "fz_xdown", .threads = 128 }, .{ .name = "fz_route", .threads = 512 },
    };
    for (declarations) |d| {
        const intended = try std.fmt.allocPrint(a, "[[max_total_threads_per_threadgroup({d})]]", .{d.threads});
        for ([_][]const u8{ "[[max_total_threads_per_threadgroup(64)]]", intended }) |attribute| {
            const pair = if (std.mem.eql(u8, attribute, intended)) try std.fmt.allocPrint(a, "{s} {s}", .{ attribute, attribute }) else attribute;
            try std.testing.expectError(error.FsrXnewPatch, patch(a, try withAttribute(a, xnew, d.name, pair)));
        }
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

test {
    _ = @import("fsr_prompt_tests.zig");
}

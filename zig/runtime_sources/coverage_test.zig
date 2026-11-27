//! Every exported Metal source string must be assembled into a runtime library or explicitly excluded.
const std = @import("std");
const sources = @import("kernel_sources");
const inventory = @import("root.zig");

const Exclusion = struct { path: []const u8, reason: []const u8 };
const exclusions = [_]Exclusion{
    .{ .path = "kernel_sources.core_contract_controls", .reason = "planted kernel-contract violations, not engine runtime libraries" },
    .{ .path = "kernel_sources.drama3", .reason = "the engine loads prebuilt Drama3 metallibs; these texts are for kernel contracts" },
    .{ .path = "kernel_sources.packed_metallib", .reason = "prebuilt binary bytes, not a runtime source export" },
    .{ .path = "kernel_sources.nemotron_paged.glue_names", .reason = "pipeline symbols, not Metal source" },
    .{ .path = "kernel_sources.nemotron_paged.partial_names", .reason = "pipeline symbols, not Metal source" },
    .{ .path = "kernel_sources.nemotron_paged.tree_name", .reason = "pipeline symbol, not Metal source" },
    .{ .path = "kernel_sources.nemotron_paged.prefill_tags", .reason = "source-selection tags, not Metal source" },
    .{ .path = "kernel_sources.nemotron_paged.prefill_names", .reason = "pipeline symbols, not Metal source" },
    .{ .path = "kernel_sources.nemotron_prompt_masked.tags", .reason = "source-selection tags, not Metal source" },
    .{ .path = "h3_kernels.gemm_whole", .reason = "pipeline symbol, not Metal source" },
    .{ .path = "h3_kernels.gemm_ragged", .reason = "pipeline symbol, not Metal source" },
    .{ .path = "h3_kernels.gemm_32x64", .reason = "pipeline symbol, not Metal source" },
    .{ .path = "h3_kernels.gemm_32x64_ragged", .reason = "pipeline symbol, not Metal source" },
    .{ .path = "h3_kernels.names", .reason = "pipeline symbols, not Metal source" },
    .{ .path = "h3_image_encoder_kernels.names", .reason = "pipeline symbols, not Metal source" },
    .{ .path = "h3_prompt_kernels.names", .reason = "pipeline symbols, not Metal source" },
};

const Checker = struct {
    sets: []const inventory.Set,
    failures: usize = 0,

    const OptionRule = enum { mlx, precise, mlx_or_precise };
    const precise_exports = [_][]const u8{
        "h3_image_encoder_kernels.source",
        "inkling2_metal.inkling2_dflash2",
        "kernel_sources.core_affine_embed",
        "kernel_sources.core_shared_attention",
        "kernel_sources.flint_glue",
        "kernel_sources.glm_draft_attention",
        "kernel_sources.llama_glue",
        "kernel_sources.minimax_m3_dflash2",
        "kernel_sources.minimax_m3_glue",
        "kernel_sources.qwen3_heads",
        "kernel_sources.ricursive_dflash2",
        "kernel_sources.ricursive_fp4",
        "kernel_sources.ricursive_glue",
        "kernel_sources.spark25_glue",
        "kernel_sources.step5_attend_nax",
        "kernel_sources.step5_glue",
        "kernel_sources.step5_sparse",
    };
    const dual_exports = [_][]const u8{
        "kernel_sources.core_moe_route",
        "kernel_sources.minimax_m3_dspark",
        "kernel_sources.nax",
    };

    fn requiredOptions(path: []const u8) OptionRule {
        for (precise_exports) |candidate| if (std.mem.eql(u8, path, candidate)) return .precise;
        for (dual_exports) |candidate| if (std.mem.eql(u8, path, candidate)) return .mlx_or_precise;
        return .mlx;
    }

    fn optionsMatch(rule: OptionRule, actual: @import("metal").CompileOptions) bool {
        const mlx = @import("metal").mlx();
        if (std.meta.eql(actual, mlx)) return rule != .precise;
        if (rule == .mlx) return false;
        var precise = mlx;
        precise.math_functions = .precise;
        return std.meta.eql(actual, precise);
    }

    fn registered(self: *const Checker, path: []const u8, text: []const u8) bool {
        const rule = requiredOptions(path);
        for (self.sets) |set| for (set.origins.items) |origin| {
            if (!std.mem.eql(u8, origin.text, text)) continue;
            if (optionsMatch(rule, origin.options)) return true;
        };
        return false;
    }

    fn excluded(path: []const u8) bool {
        for (exclusions) |entry| {
            if (std.mem.eql(u8, entry.path, path)) return true;
            if (path.len > entry.path.len and std.mem.startsWith(u8, path, entry.path) and path[entry.path.len] == '.') return true;
        }
        return false;
    }

    fn source(self: *Checker, path: []const u8, text: []const u8) void {
        if (self.registered(path, text) or excluded(path)) return;
        self.failures += 1;
        std.debug.print("unregistered Metal source export: {s} ({d} bytes)\n", .{ path, text.len });
    }

    fn structure(self: *Checker, comptime item_value: anytype, path: []const u8) !void {
        inline for (comptime @typeInfo(@TypeOf(item_value)).@"struct".field_names) |field_name| {
            if (comptime !std.mem.eql(u8, field_name, "text") and !std.mem.eql(u8, field_name, "source")) continue;
            const field_path = try std.fmt.allocPrint(std.testing.allocator, "{s}.{s}", .{ path, field_name });
            defer std.testing.allocator.free(field_path);
            try self.value(@field(item_value, field_name), field_path);
        }
    }

    fn value(self: *Checker, comptime item_value: anytype, path: []const u8) !void {
        const T = @TypeOf(item_value);
        if (comptime T == type) return self.namespace(item_value, path);
        switch (comptime @typeInfo(T)) {
            .pointer => |p| {
                if (comptime isString(T)) {
                    const text: []const u8 = item_value;
                    self.source(path, text);
                } else if (comptime p.size == .slice) {
                    if (comptime isString(p.child)) {
                        inline for (item_value) |item| self.source(path, item);
                    } else if (comptime @typeInfo(p.child) == .@"struct") {
                        inline for (item_value) |item| try self.structure(item, path);
                    }
                } else if (comptime p.size == .one and @typeInfo(p.child) == .@"struct") {
                    try self.structure(item_value.*, path);
                }
            },
            .array => |a| {
                if (comptime isString(a.child)) {
                    inline for (item_value) |item| self.source(path, item);
                } else if (comptime @typeInfo(a.child) == .@"struct") {
                    inline for (item_value) |item| try self.structure(item, path);
                }
            },
            .@"struct" => try self.structure(item_value, path),
            else => {},
        }
    }

    fn namespace(self: *Checker, comptime module: type, prefix: []const u8) !void {
        inline for (comptime std.meta.declarations(module)) |decl| {
            const path = try std.fmt.allocPrint(std.testing.allocator, "{s}.{s}", .{ prefix, decl });
            defer std.testing.allocator.free(path);
            try self.value(@field(module, decl), path);
        }
    }
};

fn isString(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| switch (p.size) {
            .one => switch (@typeInfo(p.child)) {
                .array => |a| a.child == u8,
                else => false,
            },
            .slice => p.child == u8,
            else => false,
        },
        else => false,
    };
}

test "Metal runtime source coverage: every export is registered or explicitly excluded" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sets = [_]inventory.Set{
        try inventory.generate(a, false, 13),
        try inventory.generate(a, true, 17),
    };
    var checker = Checker{ .sets = &sets };
    try checker.namespace(sources, "kernel_sources");
    try checker.namespace(@import("inkling2_metal"), "inkling2_metal");
    try checker.namespace(@import("qwen_runtime_sources"), "qwen_runtime_sources");
    try checker.namespace(@import("bf16_topk_sources"), "bf16_topk_sources");
    try checker.namespace(@import("tree_round_sources"), "tree_round_sources");
    try checker.namespace(@import("tree_commit_sources"), "tree_commit_sources");
    try checker.namespace(@import("draft_ops_sources"), "draft_ops_sources");
    try checker.namespace(@import("h3_kernels"), "h3_kernels");
    try checker.namespace(@import("../kernels/metal/h3/image_encoder.zig"), "h3_image_encoder_kernels");
    try checker.namespace(@import("../kernels/metal/h3/prompt_kernels.zig"), "h3_prompt_kernels");
    try checker.namespace(@import("../kernels/metal/h3/image_decoder.zig"), "h3_image_decoder_kernels");
    try checker.namespace(@import("../kernels/metal/ling/sources.zig"), "ling_sources");
    if (checker.failures != 0) return error.UnregisteredMetalSources;
}

test "Metal runtime source coverage: controls reject an unregistered slice export" {
    const Fixture = struct {
        pub const text: []const u8 = "kernel void unregistered_slice() {}\n";
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const set = inventory.Set{ .a = arena.allocator(), .units = false, .generation = 13 };
    var checker = Checker{ .sets = &.{set} };
    try checker.value(Fixture.text, "fixture.slice");
    try std.testing.expectEqual(@as(usize, 1), checker.failures);
}

test "Metal runtime source coverage: controls reject a new struct array entry" {
    const Fixture = struct {
        pub const files = [_]struct { text: []const u8 }{.{ .text = "kernel void unregistered_struct() {}\n" }};
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const set = inventory.Set{ .a = arena.allocator(), .units = false, .generation = 13 };
    var checker = Checker{ .sets = &.{set} };
    try checker.value(Fixture.files, "fixture.files");
    try std.testing.expectEqual(@as(usize, 1), checker.failures);
}

test "Metal runtime source coverage: controls reject an unregistered near-duplicate" {
    const base = "shared longest source line in the registered library\nshared source line two\nshared source line three\nshared source line four\nunique registered line\n";
    const candidate = "shared longest source line in the registered library\nshared source line two\nshared source line three\nshared source line four\nunique unregistered line\n";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var set = inventory.Set{ .a = arena.allocator(), .units = false, .generation = 13 };
    try set.add("fixture", "base", base, @import("metal").mlx());
    var checker = Checker{ .sets = &.{set} };
    checker.source("fixture.near_duplicate", candidate);
    try std.testing.expectEqual(@as(usize, 1), checker.failures);
}

test "Metal runtime source coverage: controls reject wrong compiler options in both directions" {
    const mtl = @import("metal");
    const ling = @import("../kernels/metal/ling/sources.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var precise = mtl.mlx();
    precise.math_functions = .precise;
    var set = inventory.Set{ .a = arena.allocator(), .units = false, .generation = 13 };
    try set.add("fixture", "mlx-only", ling.route, precise);
    try set.add("fixture", "precise-only", sources.qwen3_heads, mtl.mlx());
    var checker = Checker{ .sets = &.{set} };
    checker.source("ling_sources.route", ling.route);
    try std.testing.expectEqual(@as(usize, 1), checker.failures);
    checker.source("kernel_sources.qwen3_heads", sources.qwen3_heads);
    try std.testing.expectEqual(@as(usize, 2), checker.failures);
}

test "Metal runtime source coverage: controls reject an altered prefix" {
    const original = "#define TF_RUNTIME_PROBE 1\nkernel void registered() {}\n";
    const altered = "#define TF_RUNTIME_PROBE 0\nkernel void registered() {}\n";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var set = inventory.Set{ .a = arena.allocator(), .units = false, .generation = 13 };
    try set.add("fixture", "base", original, @import("metal").mlx());
    var checker = Checker{ .sets = &.{set} };
    checker.source("fixture.prefix", altered);
    try std.testing.expectEqual(@as(usize, 1), checker.failures);
}

//! ROCm visible devices on fictional rocminfo text: ROCr applied ROCR_VISIBLE_DEVICES, the probe applies HIP's list.
const std = @import("std");
const host = @import("benchmark_host.zig");
const receipt = @import("benchmark_receipt.zig");
const check = @import("benchmark_check.zig");
const suite = @import("benchmark_suite.zig");

const Card = struct { name: []const u8, uuid: ?[]const u8 };
const card_a: Card = .{ .name = "AMD Radeon PRO W7800", .uuid = "GPU-a1a1a1a1a1a1a1a1" };
const card_b: Card = .{ .name = "AMD Radeon PRO V620", .uuid = "GPU-b2b2b2b2b2b2b2b2" };
const card_c: Card = .{ .name = "AMD Instinct MI210", .uuid = "GPU-c3c3c3c3c3c3c3c3" };

/// rocminfo's text for a CPU agent and then these GPU agents in order; a card with no uuid prints no Uuid line.
fn rocminfo(a: std.mem.Allocator, cards: []const Card) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "*******\nAgent 1\n*******\n  Uuid:  CPU-XX\n  Marketing Name:  AMD EPYC 9004\n  Vendor Name:  CPU\n  Device Type:  CPU\n");
    for (cards, 2..) |c, n| {
        try out.print(a, "*******\nAgent {d}\n*******\n", .{n});
        if (c.uuid) |u| try out.print(a, "  Uuid:  {s}\n", .{u});
        try out.print(a, "  Marketing Name:  {s}\n  Vendor Name:  AMD\n  Device Type:  GPU\n  Compute Unit:  64\n", .{c.name});
        try out.appendSlice(a, "      Segment:  GLOBAL; FLAGS: COARSE GRAINED\n      Size:  33554432(0x2000000) KB\n");
    }
    return out.items;
}

/// What the probe records under ``vars`` on this rocminfo text, as "[GPU, ...] source", once the save rules accept it.
fn listed(a: std.mem.Allocator, vars: []const [2][]const u8, text: []const u8) ![]const u8 {
    var env: std.process.Environ.Map = .init(a);
    for (vars) |kv| try env.put(kv[0], kv[1]);
    const hw = host.rocm(a, "AMD EPYC 9004", 128 << 30, text, host.rocmVisible(&env)).hardware;
    const names = try a.alloc([]const u8, hw.gpus.len);
    for (hw.gpus, names) |g, *n| n.* = g.name;
    const runtime: receipt.Runtime = .{ .tensorfold_version = "0.0.0-test", .backend = "rocm", .python_version = "none", .platform = "linux" };
    const settings: receipt.Settings = .{ .tokens = 2, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = true, .kv_cache = "bf16" };
    const sample: suite.Sample = .{ .fixture_id = "code", .temperature = 0, .repeat = -1, .seed = 1, .status = .@"error", .error_code = .server_error };
    const v = try receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .repo_id = "example-org/test-model", .local = false }, runtime, hw, settings, &.{sample});
    var why: []const u8 = "";
    if (!check.receipt(v, .save, 1767225600000, &why)) return std.fmt.allocPrint(a, "save refused: {s}", .{why});
    return std.fmt.allocPrint(a, "[{s}] {s}", .{ try std.mem.join(a, ", ", names), hw.source });
}

const Case = struct { vars: []const [2][]const u8, text: []const u8, want: []const u8 };

/// Every case whose record differs from ``want`` is printed; the count of those is returned.
fn wrong(a: std.mem.Allocator, cases: []const Case) !usize {
    var n: usize = 0;
    for (cases) |c| {
        const got = try listed(a, c.vars, c.text);
        if (std.mem.eql(u8, got, c.want)) continue;
        for (c.vars) |kv| std.debug.print("{s}={s} ", .{ kv[0], kv[1] });
        std.debug.print("(rocminfo lists {d} GPUs): recorded {s}, want {s}\n", .{ std.mem.count(u8, c.text, "Device Type:  GPU"), got, c.want });
        n += 1;
    }
    return n;
}

test "rocminfo lists only the GPUs ROCR_VISIBLE_DEVICES leaves, renumbered; the probe applies HIP's list on top, once" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // rocminfo under ROCR_VISIBLE_DEVICES=1 on a host with cards A and B, and under =1,2 on a host with A, B and C
    const rocr_1 = try rocminfo(a, &.{card_b});
    const rocr_12 = try rocminfo(a, &.{ card_b, card_c });
    const all = try rocminfo(a, &.{ card_a, card_b, card_c });
    const cases = [_]Case{
        .{ .vars = &.{.{ "ROCR_VISIBLE_DEVICES", "1" }}, .text = rocr_1, .want = "[AMD Radeon PRO V620] detected" },
        .{ .vars = &.{.{ "ROCR_VISIBLE_DEVICES", "1,2" }}, .text = rocr_12, .want = "[AMD Radeon PRO V620, AMD Instinct MI210] detected" },
        .{ .vars = &.{ .{ "ROCR_VISIBLE_DEVICES", "1,2" }, .{ "HIP_VISIBLE_DEVICES", "1" } }, .text = rocr_12, .want = "[AMD Instinct MI210] detected" },
        .{ .vars = &.{.{ "CUDA_VISIBLE_DEVICES", "1" }}, .text = all, .want = "[AMD Radeon PRO V620] detected" },
        .{ .vars = &.{ .{ "CUDA_VISIBLE_DEVICES", "1" }, .{ "HIP_VISIBLE_DEVICES", "2" } }, .text = all, .want = "[AMD Instinct MI210] detected" },
        .{ .vars = &.{ .{ "HIP_VISIBLE_DEVICES", "" }, .{ "CUDA_VISIBLE_DEVICES", "1" } }, .text = all, .want = "[] partial" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "\"1\"" }}, .text = all, .want = "[AMD Radeon PRO V620] detected" },
    };
    try std.testing.expectEqual(@as(usize, 0), try wrong(a, &cases));
}

test "HIP's list: the first bad or out-of-range entry ends it, a repeat is skipped, and GPU- text matches rocminfo's Uuid" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const all = try rocminfo(a, &.{ card_a, card_b, card_c });
    // rocminfo text with no Uuid lines: no GPU- entry can be matched, so the cards before it stand, partial
    const no_uuid = try rocminfo(a, &.{ .{ .name = card_a.name, .uuid = null }, .{ .name = card_b.name, .uuid = null } });
    const cases = [_]Case{
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "0,3,2" }}, .text = all, .want = "[AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "0,0" }}, .text = all, .want = "[AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "2,0,2" }}, .text = all, .want = "[AMD Instinct MI210, AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "1,x,0" }}, .text = all, .want = "[AMD Radeon PRO V620] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "1,,0" }}, .text = all, .want = "[AMD Radeon PRO V620] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "0," }}, .text = all, .want = "[AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", " 1" }}, .text = all, .want = "[] partial" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "01" }}, .text = all, .want = "[] partial" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "-1" }}, .text = all, .want = "[] partial" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "GPU-c3c3c3c3c3c3c3c3" }}, .text = all, .want = "[AMD Instinct MI210] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "GPU-c3c3,0" }}, .text = all, .want = "[AMD Instinct MI210, AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "1,GPU-0000000000000000,0" }}, .text = all, .want = "[AMD Radeon PRO V620] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "GPU-C3C3C3C3C3C3C3C3" }}, .text = all, .want = "[] partial" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "1,0" }}, .text = no_uuid, .want = "[AMD Radeon PRO V620, AMD Radeon PRO W7800] detected" },
        .{ .vars = &.{.{ "HIP_VISIBLE_DEVICES", "1,GPU-b2b2" }}, .text = no_uuid, .want = "[AMD Radeon PRO V620] partial" },
    };
    try std.testing.expectEqual(@as(usize, 0), try wrong(a, &cases));
}

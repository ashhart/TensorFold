//! Check the inventory against the capability predicates used by the production loaders.
const std = @import("std");
const mtl = @import("metal");
const root = @import("root.zig");
const Entry = @import("set.zig").Entry;
const lfm2 = @import("../src/families/lfm2_moe/source.zig");
const llama = @import("../src/families/llama/contracts.zig");
const fused = @import("../src/families/gemma4/fused_lane.zig");
const gemma = @import("../src/families/gemma4/kernels.zig");
const moss = @import("../src/families/moss_image/kernels.zig");
const h3 = @import("../src/families/h3/source.zig");
const t = std.testing;

fn group(e: Entry) ?usize {
    if (std.mem.eql(u8, e.family, "lfm2") and !std.mem.eql(u8, e.name, "base")) return 0;
    if (std.mem.eql(u8, e.family, "llama")) return 1;
    if (std.mem.eql(u8, e.family, "gemma4")) {
        for (@import("../src/families/gemma4/contracts.zig").entries) |v|
            if (v.kind != 0 and std.mem.eql(u8, v.name, e.name)) return 2;
        if (std.mem.indexOf(u8, e.name, "-true-embed") != null) return 3;
    }
    if (std.mem.eql(u8, e.family, "vision-image") and std.mem.startsWith(u8, e.name, "moss_image")) return 4;
    if (std.mem.eql(u8, e.family, "h3")) return 5;
    return null;
}

test "Metal runtime source modes: every restricted entry agrees with its loader on M1 through M5" {
    for ([_]u32{ 13, 14, 15, 16, 17 }) |generation| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const device = mtl.Device{ .units = generation >= 17, .gen = generation };
        const set = try root.generate(arena.allocator(), device.units, generation);
        const allowed = [_]bool{ lfm2.optionalAvailable(device), llama.available(device), fused.available(device), gemma.draftAvailable(device), moss.available(device), h3.available(device) };
        var counts: [allowed.len]usize = @splat(0);
        var dense: usize = 0;
        for (set.entries.items) |e| {
            if (std.mem.eql(u8, e.family, "gemma4") and std.mem.eql(u8, e.name, "affine")) dense += 1;
            if (group(e)) |g| {
                counts[g] += 1;
                if (!allowed[g]) {
                    std.debug.print("unreachable runtime source: generation {d} {s}/{s}\n", .{ generation, e.family, e.name });
                    return error.UnreachableRuntimeSource;
                }
                if (g == 5) try t.expect(std.mem.eql(u8, e.text, try h3.library(arena.allocator(), 56, 18432, 288, "")));
            }
        }
        try t.expect(dense > 0);
        for (counts, allowed) |count, available| try t.expectEqual(available, count > 0);
        try t.expectEqual(@as(usize, if (device.units) 2 else 0), counts[0]);
    }
}

test "Metal runtime source modes: H3 preserves the loader's complete prefix and byte order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sources = @import("kernel_sources");
    const kernels = @import("h3_kernels");
    const gemm_text = blk: {
        for (sources.prefill) |f| if (std.mem.eql(u8, f.name, "gemm_nax")) break :blk f.text;
        return error.MissingGemmSource;
    };
    const gemm = try std.mem.replaceOwned(u8, a, gemm_text, "#include \"../nax.h\"", sources.nax);
    for ([_]usize{ 56, 16 }, [_]usize{ 288, 4 }, [_][]const u8{ "", "#define H3_TEST 1\n" }) |heads, tiles, defines| {
        const sizes = try std.fmt.allocPrint(a, "#define H3_HEADS {d}\n#define H3_SLOTS {d}\n#define H3_TILES {d}\n{s}\n{s}", .{ heads, tiles * 64, tiles, defines, kernels.source });
        const before = try std.mem.concat(a, u8, &.{ gemm, sizes });
        const after = try h3.library(a, heads, tiles * 64, tiles, defines);
        try t.expect(std.mem.eql(u8, before, after));
    }
}

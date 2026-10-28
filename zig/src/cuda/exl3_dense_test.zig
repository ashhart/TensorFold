//! Host tests of the EXL3 launch choices (planOf, wcPick, groupable) and of Dense's scratch: no device work.
const std = @import("std");
const exl3 = @import("exl3.zig");
const mock = @import("exl3_mock.zig");
const memory = @import("memory.zig");
const Stream = @import("stream.zig").Stream;
const Weight = exl3.Weight;
const expectEqual = std.testing.expectEqual;

fn weight(k: u32, n: u32, halves: u32) Weight {
    const st = exl3.strides(k, n, halves, false);
    return .{ .trellis = 1, .suh = 2, .svh = 3, .k = k, .n = n, .halves = halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

test "planOf takes plan()'s launch unless the weight fixes one" {
    var w = weight(4096, 1024, 8);
    try expectEqual(exl3.plan(4096, 1024, 192, 8).sk, exl3.planOf(w).sk);
    try expectEqual(@as(u32, 8), exl3.planOf(w).sk);
    try expectEqual(@as(u32, 4), exl3.planOf(w).wk);
    w.sk = 5;
    w.wk = 8;
    try expectEqual(@as(u32, 5), exl3.planOf(w).sk);
    try expectEqual(@as(u32, 8), exl3.planOf(w).wk);
}

test "wcPick: 4- and 6-bit mul1 rows from min_rows to 128, the tile linear.cu's dispatch_wc names" {
    const w = weight(5120, 17408, 8); // plan(): 1 split, 8 warps, 40 k tiles a warp
    const p = exl3.planOf(w);
    try expectEqual(@as(?usize, 1), exl3.wcPick(w, 17, p.sk, p.wk, 17)); // 32-row tiles, 4 k steps a stage
    try expectEqual(@as(?usize, 3), exl3.wcPick(w, 128, p.sk, p.wk, 17)); // 64-row tiles
    try expectEqual(@as(?usize, null), exl3.wcPick(w, 36, p.sk, p.wk, 17)); // 33-40 rows, 4-bit: mid-M
    try expectEqual(@as(?usize, null), exl3.wcPick(w, 16, p.sk, p.wk, 17));
    try expectEqual(@as(?usize, 1), exl3.wcPick(w, 16, p.sk, p.wk, 1)); // (the contract's runs)
    try expectEqual(@as(?usize, null), exl3.wcPick(w, 129, p.sk, p.wk, 1));
    var odd = weight(1920, 1024, 8);
    try expectEqual(@as(?usize, null), exl3.wcPick(odd, 64, 5, 8, 17)); // 3 k tiles a warp: odd
    odd.halves = 7;
    try expectEqual(@as(?usize, null), exl3.wcPick(odd, 64, 1, 4, 17));
    var mcg = w;
    mcg.codebook = .mcg;
    try expectEqual(@as(?usize, null), exl3.wcPick(mcg, 64, p.sk, p.wk, 17));
}

test "groupable: two or three 4-bit mul1 layers of one K and one warp count, 1 to 16 rows" {
    const gate = weight(5120, 17408, 8);
    try std.testing.expect(exl3.groupable(&.{ gate, gate }, 1));
    try std.testing.expect(exl3.groupable(&.{ gate, gate }, 16));
    try std.testing.expect(!exl3.groupable(&.{ gate, gate }, 17));
    try std.testing.expect(!exl3.groupable(&.{ gate, gate }, 0));
    try std.testing.expect(!exl3.groupable(&.{gate}, 8));
    try std.testing.expect(!exl3.groupable(&.{ gate, gate, gate, gate }, 8));
    try std.testing.expect(exl3.groupable(&.{ weight(5120, 12288, 8), gate, weight(5120, 10240, 8) }, 8)); // 8 warps
    try std.testing.expect(exl3.groupable(&.{ weight(5120, 6144, 8), weight(5120, 1024, 8) }, 8)); // 4 warps, K split
    try std.testing.expect(!exl3.groupable(&.{ weight(5120, 12288, 8), weight(5120, 1024, 8) }, 8)); // 8 warps and 4
    try std.testing.expect(!exl3.groupable(&.{ gate, weight(5120, 17408, 12) }, 8)); // 6-bit
    try std.testing.expect(!exl3.groupable(&.{ gate, weight(6144, 17408, 8) }, 8)); // another K
}

fn release(de: *exl3.Dense) void {
    for ([_]*?memory.DeviceBuffer{ &de.xh, &de.z, &de.counters }) |slot| {
        if (slot.* != null) slot.*.?.free();
        slot.* = null;
    }
}

test "after a dense layer's reserves, its gate/up group decodes 1-16 rows without allocating" {
    var d = mock.init();
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(0x5100) };
    const ws = [_]Weight{ weight(5120, 17408, 8), weight(5120, 17408, 8) };
    var de: exl3.Dense = .{ .d = &d, .lin = undefined };
    defer release(&de);
    for (ws) |w| try de.reserve(s, w); // what the loader does for each projection, then for the layer's groups
    try de.reserveGroup(s, &ws);
    const allocs = mock.allocations;
    const fills = mock.memsets;
    for (1..17) |rows| _ = try de.groupScratch(s, &ws, rows);
    try expectEqual(allocs, mock.allocations);
    try expectEqual(fills, mock.memsets);
    // without the group's reserve the members' scratch is too small: the group would allocate (inside a capture)
    var bare: exl3.Dense = .{ .d = &d, .lin = undefined };
    defer release(&bare);
    for (ws) |w| try bare.reserve(s, w);
    const before = mock.allocations;
    _ = try bare.groupScratch(s, &ws, 16);
    try std.testing.expect(mock.allocations > before);
}

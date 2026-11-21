//! The stream tile's item-rows gate (tiles/stream.hpp): skips dead-row lane folds where stream_pick.zig picks it.

const std = @import("std");
const affine = @import("affine.zig");
const stream_pick = @import("stream_pick.zig");
const Function = @import("../module.zig").Function;
const Caps = @import("../caps.zig").Caps;

/// A launch's columns a lane, block width, whether dead item rows skip the fold, and optional override kernel.
pub const Tile = struct { at: usize, per_block: c_int, item_rows: bool = false, f: ?Function = null };

/// Enables the item-rows gate on RDNA3.5, with optional control overrides for Gate 3 contract testing.
pub const Table = struct {
    enabled: bool = false,
    override: ?[2]Function = null,

    /// Enables the item-rows gate on this GPU's generation; none elsewhere.
    pub fn load(caps: Caps) Table {
        return .{ .enabled = caps.generation == .rdna3_5 };
    }

    /// stream_pick.zig's tile for `a` over `n` columns and `items`, setting `a.route.item_rows` when picked.
    pub fn pick(t: *const Table, caps: Caps, a: *affine.Arg, lpc_log2: u5, n: c_int, items: c_int, pair_cols: c_int, relu2: bool) Tile {
        const base = stream_pick.pick(caps.generation, a.m, lpc_log2, a.bits, n, items, pair_cols > 0);
        const plain = pair_cols == 0 and !relu2 and a.out16 == 0;
        const use = t.enabled and stream_pick.variant(caps.generation, a.m, a.k, a.bits, a.route.items != 0, plain) == .items;
        a.route.item_rows = @intFromBool(use);
        const f = if (use) if (t.override) |ov| ov[base.at] else null else null;
        return .{ .at = base.at, .per_block = base.per_block, .item_rows = use, .f = f };
    }
};

test "item_rows stays off on an empty table, >4 rows, K > single_round_k, or fused epilogues" {
    const caps = Caps.of("gfx1151").?;
    var empty: Table = .{};
    var a = std.mem.zeroInit(affine.Arg, .{ .m = 4, .n = 2560, .k = 640, .bits = 4, .group = 32, .out = 1, .route = .{ .items = 1, .members = 1 } });
    const base = stream_pick.pick(caps.generation, 4, 5, 4, 2560, 22, false);
    const tile0 = empty.pick(caps, &a, 5, 2560, 22, 0, false);
    try std.testing.expect(!tile0.item_rows and a.route.item_rows == 0 and tile0.f == null and tile0.at == base.at and tile0.per_block == base.per_block);

    const loaded = Table.load(caps);
    const tile1 = loaded.pick(caps, &a, 5, 2560, 22, 0, false);
    try std.testing.expect(tile1.item_rows and a.route.item_rows == 1 and tile1.f == null);

    var a5 = a;
    a5.m = 5;
    const tile5 = loaded.pick(caps, &a5, 5, 2560, 22, 0, false);
    try std.testing.expect(!tile5.item_rows and a5.route.item_rows == 0 and tile5.f == null);

    var ak = a;
    ak.k = 1856;
    const tilek = loaded.pick(caps, &ak, 5, 2560, 22, 0, false);
    try std.testing.expect(!tilek.item_rows and ak.route.item_rows == 0 and tilek.f == null);

    var ar = a;
    ar.out = 0;
    ar.out16 = 1;
    const tiler = loaded.pick(caps, &ar, 5, 2560, 22, 0, true);
    try std.testing.expect(!tiler.item_rows and ar.route.item_rows == 0 and tiler.f == null);
}

//! The routing width (experts a token takes) and the shared expert's width, chosen on load from config.json.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");

/// One supported shape: the routed experts a token takes and the shared expert's intermediate width.
pub const Widths = struct {
    top_k: usize,
    shared: usize,

    /// A row's slots in the expert outputs and their weights: the routed experts, then the shared expert.
    pub fn slots(w: Widths) usize {
        return w.top_k + 1;
    }

    pub fn isDefault(w: Widths) bool {
        return w.top_k == default.top_k and w.shared == default.shared;
    }
};

/// The shape the cuda_prompt_wt.zig constants hold: top-10 routing and a 640-wide shared expert.
pub const default: Widths = .{ .top_k = wt.slots - 1, .shared = wt.moe_w };

/// Every shape the CUDA path serves: the default, and top-5 routing with a 1280-wide shared expert.
pub const supported = [_]Widths{ default, .{ .top_k = 5, .shared = 2 * wt.moe_w } };

/// A selection's result: the supported shape, or the name of the config field that has no supported value.
pub const Selected = union(enum) { widths: Widths, field: []const u8 };

/// The supported shape of `num_experts_per_tok` and `shared_expert_intermediate_size` (less `learned` pocket neurons).
pub fn select(top_k: ?i64, shared: ?i64, learned: i64) Selected {
    for (supported) |w| {
        if (top_k != @as(i64, @intCast(w.top_k))) continue;
        if (shared != @as(i64, @intCast(w.shared)) + learned) return .{ .field = "shared_expert_intermediate_size" };
        return .{ .widths = w };
    }
    return .{ .field = "num_experts_per_tok" };
}

/// Whether a shared expert of `rows` rows has a supported width (its gate_proj's rows).
pub fn sharedSupported(rows: usize) bool {
    for (supported) |w| if (w.shared == rows) return true;
    return false;
}

/// The MTP layer's widths: the main layers' top-k, and its shared expert's rows if they are the main or routed width.
pub fn mtp(main: Widths, shared_rows: usize) ?Widths {
    if (shared_rows != main.shared and shared_rows != wt.moe_w) return null;
    return .{ .top_k = main.top_k, .shared = shared_rows };
}

/// Whether a captured kernel set can serve the shape: its `_topk_rows` is built for the default only.
pub fn capturedServes(w: Widths) bool {
    return w.isDefault();
}

test "the default shape is the constants' and a top-5 checkpoint selects a 1280-wide shared expert" {
    try std.testing.expectEqual(@as(usize, 11), default.slots());
    try std.testing.expectEqual(Selected{ .widths = default }, select(10, 640, 0));
    const five = select(5, 1280, 0).widths;
    try std.testing.expectEqual(Widths{ .top_k = 5, .shared = 1280 }, five);
    try std.testing.expectEqual(@as(usize, 6), five.slots());
    try std.testing.expect(!five.isDefault());
    try std.testing.expectEqual(Selected{ .widths = default }, select(10, 640 + 64, 64));
}

test "a width outside the supported list is refused by its field's name" {
    const Case = struct { top_k: ?i64, shared: ?i64, field: []const u8 };
    for ([_]Case{
        .{ .top_k = 7, .shared = 640, .field = "num_experts_per_tok" },
        .{ .top_k = null, .shared = 640, .field = "num_experts_per_tok" },
        .{ .top_k = 5, .shared = 640, .field = "shared_expert_intermediate_size" },
        .{ .top_k = 10, .shared = 1280, .field = "shared_expert_intermediate_size" },
        .{ .top_k = 5, .shared = null, .field = "shared_expert_intermediate_size" },
    }) |c| try std.testing.expectEqualStrings(c.field, select(c.top_k, c.shared, 0).field);
    try std.testing.expect(sharedSupported(640) and sharedSupported(1280));
    try std.testing.expect(!sharedSupported(960) and !sharedSupported(704));
}

test "the MTP layer routes at the main layers' top-k with its own shared width, and another width is refused" {
    const five: Widths = .{ .top_k = 5, .shared = 1280 };
    try std.testing.expectEqual(Widths{ .top_k = 5, .shared = 640 }, mtp(five, 640).?);
    try std.testing.expectEqual(five, mtp(five, 1280).?);
    try std.testing.expectEqual(default, mtp(default, 640).?);
    try std.testing.expectEqual(@as(?Widths, null), mtp(five, 960));
    try std.testing.expectEqual(@as(?Widths, null), mtp(default, 1280));
}

test "a captured kernel set serves the default shape only" {
    try std.testing.expect(capturedServes(default));
    try std.testing.expect(!capturedServes(supported[1]));
}

test "the scratch is sized by the shape: a wider shared expert's three buffers, fewer slots' pairs" {
    const Scratch = @import("cuda_batch_state.zig").Scratch;
    const cap = 64;
    const base = Scratch.bytes(cap, .{ .slots = default.slots(), .shared = default.shared });
    try std.testing.expectEqual(Scratch.bytes(cap, .{}), base);
    try std.testing.expectEqual(base + cap * (1280 - 640) * 2 * 3, Scratch.bytes(cap, .{ .slots = default.slots(), .shared = 1280 }));
    try std.testing.expect(Scratch.bytes(cap, .{ .slots = supported[1].slots(), .shared = default.shared }) < base);
}

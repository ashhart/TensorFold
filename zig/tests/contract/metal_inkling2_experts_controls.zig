//! Count-checked M3 source mutations: every shader entry has a genuine arithmetic control.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    nv_gu_up,
    nv_down_phi,
    mx_gu_up,
    mx_down_scale,
    nv_decode_nibble,
    mx_decode_scale,
    combine_slot_order,
    /// Grid faults, each in all five grid kernels; a grid cell's control runs one of them against main's bytes.
    grid_block_drop,
    grid_pair_index,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .nv_gu_up => "control: metal inkling2 experts tf_ink2_nvfp4_gu up row",
            .nv_down_phi => "control: metal inkling2 experts tf_ink2_nvfp4_down phi",
            .mx_gu_up => "control: metal inkling2 experts tf_ink2_mxfp8_gu up row",
            .mx_down_scale => "control: metal inkling2 experts tf_ink2_mxfp8_down scale",
            .nv_decode_nibble => "control: metal inkling2 experts tf_ink2_nvfp4_quantize nibble",
            .mx_decode_scale => "control: metal inkling2 experts tf_ink2_mxfp8_quantize scale",
            .combine_slot_order => "control: metal inkling2 experts tf_ink2_combine route-slot order",
            .grid_block_drop => "control: metal inkling2 experts output block dropped",
            .grid_pair_index => "control: metal inkling2 experts pair index halved",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .nv_gu_up => .{ .from = "const uint gr = 2u * n, ur = gr + 1u;", .to = "const uint gr = 2u * n, ur = gr;" },
            .mx_gu_up => .{ .from = "const uint gr = 2u * n, ur = gr + 1u;", .to = "const uint gr = 2u * n, ur = gr;" },
            .nv_down_phi => .{ .from = "const float phi = ink2_phi(amax, denominator, expert);", .to = "const float phi = precise::divide(2.0f * amax[expert], denominator[0]);" },
            .nv_decode_nibble => .{ .from = "W[row * (a.k / 2u) + col / 2u], col,", .to = "W[row * (a.k / 2u) + col / 2u], col + 1u," },
            .mx_down_scale => .{ .from = "S + expert_s + ulong(n) * (a.k / 32u), X + ulong(group) * a.k, a.k));", .to = "S + expert_s + ulong(n) * (a.k / 16u), X + ulong(group) * a.k, a.k));" },
            .mx_decode_scale => .{ .from = "S[row * (a.k / 32u) + col / 32u]", .to = "S[row * (a.k / 32u) + col / 16u]" },
            .combine_slot_order => .{
                .from = "while (at > 0u && uint(ids[group * a.topk + slots[at - 1u]]) > id) {",
                .to = "while (false) {",
            },
            // every output block past the first returns before writing: columns 32 on stay unwritten
            .grid_block_drop => .{ .from = "if (n >= a.n) return;", .to = "if (n >= a.n || tg.x > 0u) return;" },
            // two route pairs (tokens) write one row, and the other stays unwritten
            .grid_pair_index => .{ .from = "const uint n = tg.x * 32u + lane, group = tg.y;", .to = "const uint n = tg.x * 32u + lane, group = tg.y / 2u;" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.experts);
    const change = m.edit();
    const expected: usize = switch (m) {
        .nv_gu_up, .mx_gu_up, .nv_down_phi => 2,
        .grid_block_drop, .grid_pair_index => 5,
        else => 1,
    };
    if (std.mem.count(u8, src.experts, change.from) != expected) return error.ControlEditCount;
    // The same textual mutation serves both formats; a control compiles only its selected entry.
    return std.mem.replaceOwned(u8, a, src.experts, change.from, change.to);
}

test "every expert control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 experts "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.experts, plain);
}

test "expert source exports all seven entries and has no activation quantizer" {
    for ([_][]const u8{ "tf_ink2_nvfp4_gu", "tf_ink2_nvfp4_down", "tf_ink2_mxfp8_gu", "tf_ink2_mxfp8_down", "tf_ink2_nvfp4_quantize", "tf_ink2_mxfp8_quantize", "tf_ink2_combine" }) |name|
        try std.testing.expect(std.mem.indexOf(u8, src.experts, name) != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "never quantize an activation") != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "e2m1(float") == null);
}

test "inkling2 experts: one thread an output, and each dot stays the serial chain in column order" {
    const text = src.experts;
    // no lane loops over outputs; four products and the combine each take one output a thread
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, "for (uint n = lane; n < a.n; n += 32u)"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "const uint n = tg.x * 32u + lane"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "if (n >= a.n) return;"));
    // the two dots: one FP32 chain from column 0 up, products rounded before each add, no lane reduction
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "for (uint c = 0; c < k; ++c)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "#pragma METAL fp contract(off)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, "simd_sum"));
}

test "inkling2 experts: every output is one thread's, exactly once, and threads past the row write nothing" {
    const plan = @import("inkling2_experts_plan");
    for ([_]u32{ 1152, 2304, 33, 1 }) |n| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = n, .format = .mxfp8 };
        const grid = plan.geometry(c);
        const seen = try std.testing.allocator.alloc(u8, n);
        defer std.testing.allocator.free(seen);
        @memset(seen, 0);
        var idle: u32 = 0;
        for (0..grid.blocks) |block| for (0..grid.threads) |lane| {
            if (grid.output(n, @intCast(block), @intCast(lane))) |at| seen[at] += 1 else idle += 1;
        };
        for (seen) |times| try std.testing.expectEqual(@as(u8, 1), times);
        try std.testing.expect(idle < grid.threads); // only the last block has threads past the row
    }
}

test "inkling2 experts: block counts round up without overflow through the largest u32 width" {
    const plan = @import("inkling2_experts_plan");
    const cases = [_][2]u32{ .{ 1, 1 }, .{ 32, 1 }, .{ 33, 2 }, .{ 4294967264, 134217727 }, .{ 4294967265, 134217728 }, .{ 4294967280, 134217728 }, .{ 4294967295, 134217728 } };
    for (cases) |case| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = case[0], .format = .mxfp8 };
        try std.testing.expectEqual(case[1], plan.geometry(c).blocks);
        try std.testing.expectEqual(case[1], plan.combineGeometry(c).blocks);
        // the last block reaches the last output, and none past it
        const grid = plan.geometry(c);
        const tail = (case[0] - 1) % 32;
        try std.testing.expectEqual(@as(?u32, case[0] - 1), grid.output(case[0], grid.blocks - 1, tail));
        if (tail < 31) try std.testing.expectEqual(@as(?u32, null), grid.output(case[0], grid.blocks - 1, tail + 1));
    }
}

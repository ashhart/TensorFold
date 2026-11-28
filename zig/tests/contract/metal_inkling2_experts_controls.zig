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
    expert_group_drop_last,
    grouped_block_drop,
    grouped_tile_drop,
    grouped_nv_gu_up,
    grouped_mx_gu_up,
    grouped_mx_down_scale,

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
            .expert_group_drop_last => "control: metal inkling2 experts tf_ink2_expert_group tile dropped",
            .grouped_block_drop => "control: metal inkling2 experts grouped output block dropped",
            .grouped_tile_drop => "control: metal inkling2 experts grouped tile pair dropped",
            .grouped_nv_gu_up => "control: metal inkling2 experts tf_ink2_nvfp4_gu_grouped up row",
            .grouped_mx_gu_up => "control: metal inkling2 experts tf_ink2_mxfp8_gu_grouped up row",
            .grouped_mx_down_scale => "control: metal inkling2 experts tf_ink2_mxfp8_down_grouped scale",
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
                .from = "(other < my_id || (other == my_id && s < tid)) ? 1u : 0u;",
                .to = "((other < my_id && false) || s < tid) ? 1u : 0u;",
            },
            // every output block past the first returns before writing: solo + combine drop tail blocks
            .grid_block_drop => .{ .from = "if (n >= a.n) return;", .to = "if (n >= a.n || tg.x > 0u) return;" },
            // two route pairs (tokens) write one row, and the other stays unwritten
            .grid_pair_index => .{ .from = "group = tg.y;", .to = "group = tg.y / 2u;" },
            .expert_group_drop_last => .{ .from = "if (tid == 0u) group_buf[0] = base_offsets[1];", .to = "if (tid == 0u) group_buf[0] = base_offsets[1] - 1u;" },
            .grouped_block_drop => .{ .from = "if (tile_id >= group_buf[0]) return;", .to = "if (tile_id >= group_buf[0] || tg.x > 0u) return;" },
            .grouped_tile_drop => .{ .from = ", len = ((item >> 28u) & 0xfu) + 1u;", .to = ", len = 1u;" },
            .grouped_nv_gu_up => .{ .from = "const uint nv_up = GU ? PLANE_STRIDE : 0u,", .to = "const uint nv_up = 0u," },
            .grouped_mx_gu_up => .{ .from = "const uint mx_up = GU ? PLANE_STRIDE : 0u,", .to = "const uint mx_up = 0u," },
            .grouped_mx_down_scale => .{ .from = "mx_s_bytes = a.k >> 5u;", .to = "mx_s_bytes = a.k >> 4u;" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.experts);
    const change = m.edit();
    const expected: usize = switch (m) {
        .nv_down_phi, .grouped_tile_drop, .grouped_block_drop => 4,
        .grid_block_drop, .grid_pair_index => 5,
        .nv_gu_up, .mx_gu_up => 2,
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

test "expert source exports all twelve entries and has no activation quantizer" {
    for ([_][]const u8{
        "tf_ink2_nvfp4_gu",
        "tf_ink2_nvfp4_down",
        "tf_ink2_mxfp8_gu",
        "tf_ink2_mxfp8_down",
        "tf_ink2_nvfp4_quantize",
        "tf_ink2_mxfp8_quantize",
        "tf_ink2_combine",
        "tf_ink2_expert_group",
        "tf_ink2_nvfp4_gu_grouped",
        "tf_ink2_nvfp4_down_grouped",
        "tf_ink2_mxfp8_gu_grouped",
        "tf_ink2_mxfp8_down_grouped",
    }) |name|
        try std.testing.expect(std.mem.indexOf(u8, src.experts, name) != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "never quantize an activation") != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "e2m1(float") == null);
}

test "inkling2 experts: one thread an output, and each dot stays the serial chain in column order" {
    const text = src.experts;
    // no lane loops over outputs; solo (32), grouped (64), and combine (128) each take one output a thread
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, "for (uint n = lane; n < a.n; n += 32u)"));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, text, "const uint n = tg.x * 32u + lane"));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, text, "tg.x * 64u"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "const uint n = tg.x * 128u + tid;"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "if (n >= a.n) return;"));
    // the two dots: one FP32 chain from column 0 up, products rounded before each add, no lane reduction
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "for (uint b = 0; b < k / 16u; ++b)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "for (uint b = 0; b < k / 32u; ++b)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "#pragma METAL fp contract(off)"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, "simd_sum"));
}

test "inkling2 experts: every output is one thread's, exactly once, and threads past the row write nothing" {
    const plan = @import("inkling2_experts_plan");
    for ([_]u32{ 1152, 2304, 33, 65, 129, 1 }) |n| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = n, .format = .mxfp8 };
        for ([_]plan.Geometry{ plan.geometry(c), plan.groupedGeometry(c), plan.combineGeometry(c) }) |grid| {
            const seen = try std.testing.allocator.alloc(u8, n);
            defer std.testing.allocator.free(seen);
            @memset(seen, 0);
            var idle: u32 = 0;
            for (0..grid.blocks) |block| for (0..grid.threads) |lane| {
                if (grid.output(n, @intCast(block), @intCast(lane))) |at| seen[at] += 1 else idle += 1;
            };
            for (seen) |times| try std.testing.expectEqual(@as(u8, 1), times);
            try std.testing.expect(idle < grid.threads);
        }
    }
}

test "inkling2 experts: block counts round up without overflow through the largest u32 width" {
    const plan = @import("inkling2_experts_plan");
    const cases_32 = [_][2]u32{ .{ 1, 1 }, .{ 32, 1 }, .{ 33, 2 }, .{ 4294967264, 134217727 }, .{ 4294967265, 134217728 }, .{ 4294967280, 134217728 }, .{ 4294967295, 134217728 } };
    for (cases_32) |case| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = case[0], .format = .mxfp8 };
        const grid = plan.geometry(c);
        try std.testing.expectEqual(case[1], grid.blocks);
        const tail = (case[0] - 1) % 32;
        try std.testing.expectEqual(@as(?u32, case[0] - 1), grid.output(case[0], grid.blocks - 1, tail));
        if (tail < 31) try std.testing.expectEqual(@as(?u32, null), grid.output(case[0], grid.blocks - 1, tail + 1));
    }
    const cases_64 = [_][2]u32{ .{ 1, 1 }, .{ 64, 1 }, .{ 65, 2 }, .{ 4294967232, 67108863 }, .{ 4294967233, 67108864 }, .{ 4294967295, 67108864 } };
    for (cases_64) |case| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = case[0], .format = .mxfp8 };
        const grid = plan.groupedGeometry(c);
        try std.testing.expectEqual(case[1], grid.blocks);
        const tail = (case[0] - 1) % 64;
        try std.testing.expectEqual(@as(?u32, case[0] - 1), grid.output(case[0], grid.blocks - 1, tail));
        if (tail < 63) try std.testing.expectEqual(@as(?u32, null), grid.output(case[0], grid.blocks - 1, tail + 1));
    }
    const cases_128 = [_][2]u32{ .{ 1, 1 }, .{ 128, 1 }, .{ 129, 2 }, .{ 4294967168, 33554431 }, .{ 4294967169, 33554432 }, .{ 4294967295, 33554432 } };
    for (cases_128) |case| {
        const c = plan.Call{ .rows = 1, .k = 32, .n = case[0], .format = .mxfp8 };
        const grid = plan.combineGeometry(c);
        try std.testing.expectEqual(case[1], grid.blocks);
        const tail = (case[0] - 1) % 128;
        try std.testing.expectEqual(@as(?u32, case[0] - 1), grid.output(case[0], grid.blocks - 1, tail));
        if (tail < 127) try std.testing.expectEqual(@as(?u32, null), grid.output(case[0], grid.blocks - 1, tail + 1));
    }
}

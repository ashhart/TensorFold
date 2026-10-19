//! Actual allocation owners and the shared set meet at a recording Metal boundary.
const std = @import("std");
const mtl = @import("metal");
const glm = @import("src/families/glm/glm.zig");
const kept = @import("src/families/glm/kept.zig");
pub const checkpoint_metal = @import("src/core/checkpoint_metal.zig");
pub const checkpoint_host = @import("src/core/checkpoint.zig");
pub const Checkpoint = checkpoint_host.Checkpoint;
pub const gpu_profile = @import("src/core/gpu_profile.zig");
pub const tree_slots = @import("src/core/tree_slots.zig");
pub const draft_ops = @import("draft_ops");
pub const row_projection = @import("src/core/row_projection.zig");
pub const lane_projection = @import("src/core/lane_projection.zig");
pub const affine4 = @import("src/core/affine4.zig");
pub const affine4_lane = @import("src/core/affine4_lane.zig");
pub const shared_attention = @import("src/core/shared_attention.zig");
pub const segments = @import("src/core/segments.zig");
pub const bf16_topk_gpu = @import("bf16_topk_gpu");
pub const qwen_vl = @import("src/core/qwen_vl/qwen_vl.zig");
pub const tree_round = @import("tree_round");
pub const tree_commit_gpu = @import("tree_commit_gpu");
pub const bf16_topk = @import("bf16_topk");
const a = std.testing.allocator;

test "residency owner GLM: actual arena slots tower and snapshots match live unique allocations" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    for ([_]bool{ false, true }) |images| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            const weight = try device.buffer(64, mtl.ResourceOptions.shared);
            defer weight.deinit();
            var weights: glm.weights.Weights = undefined;
            weights.buffers = .empty;
            defer weights.buffers.deinit(a);
            try weights.buffers.append(a, weight);
            var e: glm.engine.Engine = undefined;
            e.gpa = a;
            e.device = device;
            e.queue = record.queue();
            e.w = &weights;
            e.arena = .{ .device = device, .gpa = a };
            defer e.arena.deinit();
            _ = try e.arena.buffer(128);
            e.vision = null;
            e.residency = null;
            e.residency_enabled = true;
            e.limit = std.mem.zeroes(@TypeOf(e.limit));
            e.c = .{ .hidden = 4, .layers = 1, .run = 1, .kda_heads = 1, .kda_dim = 2, .conv = 3, .mtp = 0, .mla = .empty };
            var kernels: glm.kernels.Kernels = undefined;
            kernels.copy_u32 = .{ .id = @ptrFromInt(1) };
            e.k = &kernels;
            e.fused_route = true;
            e.draft_vocab = 8;
            e.ep = null;
            e.ev = 0;
            e.event = .{};
            e.s = try glm.state.initState(&e.arena, &e.c, 2048, null);
            var slots = try glm.slots.Slots.init(a, &e, 2);
            defer slots.deinit(a);
            var vision: glm.vision.Vision = undefined;
            var tower: [2]?mtl.Buffer = @splat(null);
            defer for (tower) |maybe| if (maybe) |b| b.deinit();
            if (images) {
                tower[0] = try device.buffer(64, mtl.ResourceOptions.shared);
                tower[1] = try device.buffer(128, mtl.ResourceOptions.shared);
                vision.w.buf = tower[0].?;
                vision.scratch = tower[1].?;
                e.vision = &vision;
                e.vision_rows = try e.arena.buffer(64);
            }
            glm.residency.wire(&e, std.testing.io);
            defer glm.residency.drop(&e);
            try std.testing.expect(e.residency != null);
            try record.check();
            _ = try e.arena.buffer(256);
            try std.testing.expectEqual(@as(usize, 1), e.residency.?.held.joining.items.len);
            mtl.keepalive.Target.tick(&e.keepalive_target);
            try record.check();
            try std.testing.expectEqual(@as(usize, 1), record.uses);
            try std.testing.expectEqual(@as(usize, 1), record.copies);
            try std.testing.expectEqual(@as(usize, 1), record.commits);
            const snap = try kept.blank(&slots, 3);
            const id = snap.id;
            defer kept.drop(&slots, id);
            e.residency.?.held.flush();
            try record.check();
            kept.drop(&slots, id);
            try record.check();
            slots.slots[0].used = true;
            slots.slots[0].s.pos = 3;
            const saved = try kept.save(&slots, 0, 3, 0, true);
            const saved_id = saved.id;
            defer kept.drop(&slots, saved_id);
            e.residency.?.held.flush();
            try record.check();
            const old_buffer = saved.buf;
            try kept.copyIds(&slots, 0, &.{saved_id});
            try std.testing.expect(saved.buf.id != old_buffer.id);
            e.residency.?.held.flush();
            try record.check();
            var temp = std.testing.tmpDir(.{});
            defer temp.cleanup();
            var path: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const n = try temp.dir.realPath(std.testing.io, &path);
            const file = try std.fmt.allocPrintSentinel(a, "{s}/state.bin", .{path[0..n]}, 0);
            defer a.free(file);
            try kept.writeSnap(&slots, saved_id, file);
            const read = try kept.readSnap(&slots, 100, 3, file);
            const read_id = read.id;
            defer kept.drop(&slots, read_id);
            e.residency.?.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}

test "residency owner GLM: zero and a failed bound never publish a set or keepalive scratch" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    for ([_]bool{ false, true }) |enabled| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            const weight = try device.buffer(64, mtl.ResourceOptions.shared);
            defer weight.deinit();
            var weights: glm.weights.Weights = undefined;
            weights.buffers = .empty;
            defer weights.buffers.deinit(a);
            try weights.buffers.append(a, weight);
            var e: glm.engine.Engine = undefined;
            e.gpa = a;
            e.device = device;
            e.queue = record.queue();
            e.w = &weights;
            e.arena = .{ .device = device, .gpa = a };
            defer e.arena.deinit();
            e.vision = null;
            e.residency = null;
            e.residency_enabled = try glm.residency.policy.flag(if (enabled) "1" else "0");
            e.limit = std.mem.zeroes(@TypeOf(e.limit));
            e.limit.asked = true;
            e.limit.bytes = mtl.ResidencySet.margin;
            glm.residency.wire(&e, std.testing.io);
            defer glm.residency.drop(&e);
            try std.testing.expect(e.residency == null);
            try std.testing.expect(e.arena.follow == null);
            try std.testing.expect(e.keepalive_target.scratch == null);
            try std.testing.expect(e.keepalive_target.flush == null);
            try std.testing.expectEqual(@as(usize, 0), e.keepalive_target.sets.len);
            try std.testing.expectEqual(@as(usize, if (enabled) 2 else 1), record.allocations.items.len);
            try record.check();
        }
        try record.allFreed();
    }
}

test {
    _ = @import("residency_gemma_tests.zig");
    _ = @import("residency_step5_tests.zig");
    _ = @import("residency_nemotron_tests.zig");
    _ = @import("residency_qwen27_tests.zig");
}

test {
    _ = @import("residency_small_tests.zig");
}

test {
    _ = @import("residency_dsa_tests.zig");
}

test {
    _ = @import("residency_concurrency_tests.zig");
}

test {
    _ = @import("residency_flashnext_tests.zig");
}

test {
    _ = @import("residency_drama_tests.zig");
}

test {
    _ = @import("residency_qwen27_share_tests.zig");
}

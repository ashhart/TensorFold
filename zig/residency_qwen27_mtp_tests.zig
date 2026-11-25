//! The lane drafter's direct buffers and its nested scratch share the normal residency inventory.
const std = @import("std");
const mtl = @import("metal");
const residency = @import("src/families/qwen27/residency.zig");
const Config = @import("src/families/qwen27/config.zig").Config;
const Model = @import("src/families/qwen27/model.zig").Model;
const sn_mtp = @import("src/families/qwen27/sn_mtp.zig");
const mtp_weights = @import("core").sensenova_loop.mtp_weights;
const a = std.testing.allocator;

fn snMtpConfig() Config {
    return .{
        .hidden = 128,
        .intermediate = 256,
        .layers = 1,
        .vocab = 64,
        .heads = 16,
        .kv_heads = 4,
        .head_dim = 256,
        .k_heads = 1,
        .v_heads = 1,
        .dk = 128,
        .dv = 128,
        .conv_kernel = 4,
        .eps = 1e-5,
        .rope_dims = 64,
        .rope_theta = 10000,
        .looped = true,
    };
}

fn initMtpFixture(dev: mtl.Device, q: mtl.Queue, cfg: Config, embed_buf: mtl.Buffer, weight_buf: mtl.Buffer) !struct { model: Model, weights: mtp_weights.Weights } {
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer owned.deinit(a);
    try owned.append(a, weight_buf);
    const mat: mtp_weights.MtpMat = .{ .bf16 = .{ .buf = weight_buf, .n = 128, .k = 128 } };
    const norm: @import("core").checkpoint_metal.Tensor = .{ .buffer = weight_buf, .offset = 0, .bytes = 256, .dtype = .bf16 };
    return .{
        .model = .{
            .allocator = a,
            .device = dev,
            .queue = q,
            .shard = undefined,
            .config = cfg,
            .checkpoint = undefined,
            .weights = .{
                .allocator = a,
                .layers = &.{},
                .embed = undefined,
                .embed_bf16 = .{ .buffer = embed_buf, .offset = 0, .bytes = 256, .dtype = .bf16 },
                .norm = undefined,
                .head = .{ .words = .{ .buffer = embed_buf }, .pairs = .{ .buffer = embed_buf }, .n = 64, .k = 128, .tile = 32, .slices = 1, .bits = 4 },
            },
            .kernels = .{ .allocator = a, .device = dev, .sn = .{ .allocator = a, .device = dev, .tensor_units = false } },
            .glue = undefined,
            .frame = undefined,
        },
        .weights = .{
            .allocator = a,
            .owned = owned,
            .bits = 16,
            .norm_e = norm,
            .norm_h = norm,
            .input_norm = norm,
            .post_norm = norm,
            .q_norm = norm,
            .k_norm = norm,
            .norm = norm,
            .fc = mat,
            .qkv = mat,
            .o = mat,
            .gate_up = mat,
            .down = mat,
        },
    };
}

test "residency owner Qwen27 MTP: staging and every scratch plane join and leave the same set" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var buffers: [5]mtl.Buffer = undefined;
        var made: usize = 0;
        defer for (buffers[0..made]) |b| b.deinit();
        for (&buffers, 0..) |*b, i| {
            b.* = try record.device().buffer(64 * (i + 1), mtl.ResourceOptions.shared);
            made += 1;
        }
        var head = .{ .stage = buffers[0], .index = buffers[1], .scratch = .{
            .planes = buffers[2],
            .rows = buffers[3],
            .logits = buffers[4],
        } };
        var list: std.ArrayList(mtl.Buffer) = .empty;
        defer list.deinit(a);
        try residency.appendMtp(&head, a, &list);
        try std.testing.expectEqualSlices(mtl.Buffer, &buffers, list.items);
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), list.items, "qwen27")).?;
        defer owner.deinit();
        residency.mtpBuffers(&head, owner.follow(), true);
        defer residency.mtpBuffers(&head, owner.follow(), false);
        try std.testing.expectEqual(@as(u32, 6), owner.held.members.count());
        try std.testing.expectEqual(@as(u64, 1024), owner.held.bytes);
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Qwen27 MTP: failed initFromWeights leaves caller weights owned once and frees internal buffers" {
    const cfg = snMtpConfig();
    var internal_allocs: usize = 0;
    {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        const dev = record.device();
        const embed_buf = try dev.buffer(256, mtl.ResourceOptions.shared);
        defer embed_buf.deinit();
        const weight_buf = try dev.buffer(256, mtl.ResourceOptions.shared);
        var fix = try initMtpFixture(dev, record.queue(), cfg, embed_buf, weight_buf);
        const base = record.allocations.items.len;
        const m = try sn_mtp.Mtp.initFromWeights(a, &fix.model, fix.weights, 2, 64);
        internal_allocs = record.allocations.items.len - base;
        m.deinit();
    }
    try std.testing.expect(internal_allocs > 4);

    for ([_]usize{ 0, internal_allocs / 2, internal_allocs - 1 }) |fail_offset| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const dev = record.device();
            const embed_buf = try dev.buffer(256, mtl.ResourceOptions.shared);
            defer embed_buf.deinit();
            const weight_buf = try dev.buffer(256, mtl.ResourceOptions.shared);
            var fix = try initMtpFixture(dev, record.queue(), cfg, embed_buf, weight_buf);
            defer fix.weights.deinit();
            record.fail_at = record.allocations.items.len + fail_offset;
            try std.testing.expectError(error.InjectedBufferFailure, sn_mtp.Mtp.initFromWeights(a, &fix.model, fix.weights, 2, 64));
            try std.testing.expectEqual(@as(usize, 0), record.bad);
            try std.testing.expectEqual(@as(usize, 256), weight_buf.length());
        }
        try record.allFreed();
    }
}

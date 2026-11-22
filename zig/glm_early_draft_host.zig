//! Real MTP draft scheduling recorded on the host; no shaders or Metal device execute.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("src/families/glm/config.zig");
const wts = @import("src/families/glm/weights.zig");
const st = @import("src/families/glm/state.zig");
const Slots = @import("src/families/glm/slots.zig").Slots;
const Engine = @import("src/families/glm/engine.zig").Engine;
const Kernels = @import("src/families/glm/kernels.zig").Kernels;

const a = std.testing.allocator;

fn prepare(value: anytype, dummy: wts.Ref, next_id: *usize) void {
    const P = @TypeOf(value);
    const T = @typeInfo(P).pointer.child;
    if (T == wts.Ref) {
        value.* = dummy;
        return;
    }
    if (T == wts.Affine) {
        value.* = .{ .w = dummy, .s = dummy, .b = dummy, .n = 64, .k = 64, .bits = 4, .group = 64 };
        return;
    }
    if (T == mtl.Pipeline) {
        value.* = .{ .id = @ptrFromInt(next_id.*) };
        next_id.* += 16;
        return;
    }
    switch (@typeInfo(T)) {
        .pointer => {},
        .@"struct" => |info| inline for (info.fields) |field| {
            prepare(&@field(value.*, field.name), dummy, next_id);
        },
        .array => for (value.*) |*item| prepare(item, dummy, next_id),
        .optional => if (value.*) |*item| prepare(item, dummy, next_id),
        .@"union" => switch (value.*) {
            inline else => |*item| prepare(item, dummy, next_id),
        },
        else => {},
    }
}

fn count(record: *const mtl.Record, id: mtl.objc.Id) usize {
    var n: usize = 0;
    for (record.passes[0..record.pass_count]) |pass| {
        if (pass.pipeline == id) n += 1;
    }
    return n;
}

fn first(record: *const mtl.Record, id: mtl.objc.Id) !*const mtl.Pass {
    for (record.passes[0..record.pass_count]) |*pass| {
        if (pass.pipeline == id) return pass;
    }
    return error.MissingPass;
}

fn nth(record: *const mtl.Record, id: mtl.objc.Id, wanted: usize) !*const mtl.Pass {
    var n: usize = 0;
    for (record.passes[0..record.pass_count]) |*pass| {
        if (pass.pipeline == id) {
            if (n == wanted) return pass;
            n += 1;
        }
    }
    return error.MissingPass;
}

fn bound(pass: *const mtl.Pass, expected: mtl.Buffer, offset: usize) !void {
    const got = pass.buffers[0];
    try std.testing.expect(got.buffer != null);
    try std.testing.expect(got.buffer.?.id == expected.id);
    try std.testing.expectEqual(offset, got.offset);
}

fn externalEncode(_: *anyopaque, _: *Slots, _: mtl.CommandBuffer, _: mtl.ComputeEncoder, _: []const Slots.Win) void {}
fn externalLand(_: *anyopaque, _: *Slots, _: bool) void {}

fn checkCase(early: bool, external: bool) !void {
    var record: mtl.Record = .{ .gpa = a, .record_passes = true };
    defer record.deinit();
    var e: Engine = undefined;
    e.gpa = a;
    e.device = record.device();
    e.queue = record.queue();
    e.c = .{
        .hidden = 64,
        .vocab = 64,
        .dense_inter = 64,
        .experts = 8,
        .own = .{ 0, 8 },
        .inter = .{ 0, 8 },
        .topk = 2,
        .moe_inter = 8,
        .shared_inter = 8,
        .kda_heads = 2,
        .kda_dim = 1,
        .conv = 3,
        .mla_heads = 2,
        .nope = 4,
        .v_dim = 4,
        .q_lora = 8,
        .kv_lora = 8,
        .i_heads = 2,
        .i_dim = 4,
        .i_topk = 16,
        .kpool = 4,
    };
    e.arena = .{ .device = e.device, .gpa = a };
    defer e.arena.deinit();
    const initialized = try st.init(&e.arena, &e.c, 256);
    e.s = initialized.state;
    e.sc = initialized.scratch;

    var weights = std.mem.zeroes(wts.Weights);
    weights.buffers = .empty;
    weights.gpa = a;
    weights.layers[e.c.layers].hc = null;
    weights.layers[e.c.layers].attn = .{ .mla = std.mem.zeroes(wts.Mla) };
    weights.layers[e.c.layers].mlp = .{ .moe = std.mem.zeroes(wts.Moe) };
    weights.mtp = std.mem.zeroes(wts.Mtp);
    var kernels = std.mem.zeroes(Kernels);
    var next_id: usize = 0x1000;
    prepare(&weights, e.sc.normed, &next_id);
    prepare(&kernels, e.sc.normed, &next_id);
    e.w = &weights;
    e.k = &kernels;
    e.ev = 0;
    e.event = .{};
    e.committed = 0;
    e.dflash = null;
    e.draft_vocab = e.c.vocab;
    e.ep = null;
    e.fused_route = true;
    e.early_draft = early;
    var slots = try Slots.init(a, &e, 1);
    defer slots.deinit(a);
    var external_ctx: u8 = 0;
    if (external) slots.follow = .{ .ctx = &external_ctx, .encode = externalEncode, .land = externalLand };
    const slot = &slots.slots[0];
    slot.used = true;
    slot.mtp = true;
    slot.seen = 0;
    slot.width = 3;
    slot.mtp_pos = 7;
    slot.row0 = 2;
    slot.s.pos = 0;

    try slots.draftAll(&.{.{ .slot = 0, .prompt = false, .follow = &.{ 11, 12, 13 }, .depth = 2 }});
    try std.testing.expectEqual(@as(@TypeOf(slot.held_n), 2), slot.held_n);
    try std.testing.expectEqual(@as(@TypeOf(slot.mtp_pos), 10), slot.mtp_pos);
    try std.testing.expectEqual(if (early) @as(usize, 2) else @as(usize, 5), count(&record, e.k.copy_u32.id));
    try std.testing.expectEqual(if (early) @as(usize, 2) else @as(usize, 3), count(&record, e.k.qmv_x.id));
    const qr = try first(&record, e.k.qmv_qr.id);
    const qr_offset = if (early) 2 * @as(usize, e.c.q_lora) * 2 else 0;
    try bound(qr, e.sc.qr.buf, qr_offset);
    const add = try first(&record, e.k.add.id);
    if (early) {
        try bound(add, e.sc.m_x.buf, 2 * @as(usize, e.c.hidden) * 2);
    } else {
        try bound(add, slots.lasts.buf, 0);
    }
    const hnorm = try nth(&record, e.k.rms.id, 1);
    if (early) {
        try bound(hnorm, e.sc.hidden.buf, 2 * @as(usize, e.c.hidden) * 2);
    } else {
        try bound(hnorm, slots.rows.buf, 0);
    }
    try std.testing.expectEqual(@as(usize, 0), record.waits);
    if (early and !external) {
        try std.testing.expect(slots.open == null);
        try std.testing.expect(slots.launched != null);
        try std.testing.expectEqual(@as(usize, 1), record.commits);
        try std.testing.expectEqual(@as(usize, 1), record.fence_updates);
    } else {
        try std.testing.expect(slots.open != null);
        try std.testing.expect(slots.launched == null);
        try std.testing.expectEqual(@as(usize, 0), record.commits);
        try std.testing.expectEqual(@as(usize, 0), record.fence_updates);
    }
    record.record_passes = false;
    try slots.flush();
    try std.testing.expect(slots.open == null and slots.launched == null);
    try std.testing.expectEqual(@as(usize, 1), record.commits);
    try std.testing.expectEqual(@as(usize, 1), record.waits);
}

test "GLM Metal early draft: exact single-request bindings and early commit are opt-in" {
    try checkCase(false, false);
    try checkCase(true, false);
    try checkCase(true, true);
}

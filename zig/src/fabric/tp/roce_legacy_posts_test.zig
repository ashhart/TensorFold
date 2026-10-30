//! Compatibility fixture captured before any multirank transport edit. No device is opened.
const std = @import("std");
const fake = @import("roce_fake.zig");
const Roce = @import("roce.zig").Roce;
const abi = @import("verbs").abi;
const Call = struct { qp: u32, id: u64, opcode: u32, flags: u32, local: u64, remote: u64, length: u32, lkey: u32, rkey: u32, value: u32, next: bool };
const Capture = struct {
    var calls: [128]Call = undefined;
    var count: usize = 0;
    fn post(qp: *abi.Qp, first: *abi.SendWr, bad: *?*abi.SendWr) callconv(.c) c_int {
        var at: ?*abi.SendWr = first;
        while (at) |wr| : (at = wr.next) {
            const sge = wr.sg_list[0];
            const inline_data = wr.send_flags & abi.send_inline != 0;
            calls[count] = .{ .qp = qp.qp_num, .id = wr.wr_id, .opcode = @intCast(wr.opcode), .flags = wr.send_flags, .local = if (inline_data) 0 else sge.addr - @intFromPtr(fake.Fake.region.ptr), .remote = wr.remote_addr - 0x10000000, .length = sge.length, .lkey = sge.lkey, .rkey = wr.rkey, .value = if (inline_data) @as(*const u32, @ptrFromInt(sge.addr)).* else 0, .next = wr.next != null };
            count += 1;
        }
        return fake.Fake.postSend(qp, first, bad);
    }
};

test "legacy world two exact ordered WR chains and completion pressure on one and two HCAs" {
    const slot: u64 = fake.slot;
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(slot));
    defer std.testing.allocator.free(region);
    fake.Fake.region = region;
    fake.Fake.slot_bytes = slot;
    const sizes = [_]u32{ 16, 48, 1024, 1 << 20, 1 << 20, 1 << 20 };
    for ([_]u32{ 1, 2 }) |n_hca| {
        fake.Fake.reset(3);
        Capture.count = 0;
        var hw: fake.Hw = .{};
        var wire = hw.wire(region);
        wire.n_hca = n_hca;
        wire.peer_region = 0x10000000;
        wire.peer_rkey = .{ 11, 13 };
        for (&hw.ctx) |*ctx| ctx.ops.post_send = Capture.post;
        var expected: [128]Call = undefined;
        var count: usize = 0;
        for (sizes, 0..) |size, index| {
            const seq: u32 = @intCast(index + 1);
            const s: u32 = seq & 1;
            try wire.post(s, seq, size);
            var offset: u64 = 0;
            for (0..n_hca) |h| {
                const packs = size / 16;
                const bytes: u32 = (packs / n_hca + @intFromBool(h < packs % n_hca)) * 16;
                if (bytes != 0) {
                    expected[count] = .{ .qp = @intCast(h), .id = seq, .opcode = @intCast(abi.wr_rdma_write), .flags = 0, .local = 8192 + s * slot + offset, .remote = 8192 + 2 * slot + s * slot + offset, .length = bytes, .lkey = 7, .rkey = if (h == 0) 11 else 13, .value = 0, .next = true };
                    count += 1;
                }
                expected[count] = .{ .qp = @intCast(h), .id = seq, .opcode = @intCast(abi.wr_rdma_write), .flags = abi.send_signaled | abi.send_inline, .local = 0, .remote = 4096 + (s * 2 + h) * 128, .length = 4, .lkey = 0, .rkey = if (h == 0) 11 else 13, .value = seq, .next = false };
                count += 1;
                offset += bytes;
            }
        }
        try std.testing.expectEqual(count, Capture.count);
        for (expected[0..count], Capture.calls[0..count]) |want, got| try std.testing.expectEqualDeep(want, got);
        while (fake.Fake.inflight > 0) try wire.reap();
        try std.testing.expectEqual(@as(u64, 0), fake.Fake.outside);
        try std.testing.expect(fake.Fake.max_inflight <= 2 << 20);
        try std.testing.expect(fake.Fake.polls > 0);
    }
}

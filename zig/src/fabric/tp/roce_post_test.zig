//! VerbsWire.post on fake verbs: a post that would leave its slot or region is refused before any verb.
const std = @import("std");
const fake = @import("roce_fake.zig");
const Roce = @import("roce.zig").Roce;
const Config = @import("config.zig").Config;
const Fake = fake.Fake;

test "a post past its slot is refused before any verb; a full slot posts inside the region" {
    const small: u64 = 64 << 10;
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(small));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.slot_bytes = small;
    defer Fake.slot_bytes = fake.slot;
    Fake.reset(1);
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    try std.testing.expectError(error.OutOfRegion, w.post(1, 1, small + 16));
    try std.testing.expectError(error.OutOfRegion, w.post(2, 1, 16));
    try std.testing.expectEqual(@as(u64, 0), Fake.posts);
    try w.post(1, 1, small);
    try std.testing.expectEqual(@as(u64, 2), Fake.posts);
    try std.testing.expectEqual(@as(u64, 0), Fake.outside);
}

test "no RDMA write past 1 MiB, whatever the slot: a 9.57 MB and a 1 MiB + 16 post refused in a 10 MiB slot" {
    const big: u64 = 10 << 20;
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(big));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.slot_bytes = big;
    defer Fake.slot_bytes = fake.slot;
    Fake.reset(1);
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    try std.testing.expectError(error.WriteTooLarge, w.post(0, 1, 9_568_256));
    try std.testing.expectError(error.WriteTooLarge, w.post(1, 1, (1 << 20) + 16));
    try std.testing.expectEqual(@as(u64, 0), Fake.posts);
    try w.post(0, 1, 1 << 20);
    try std.testing.expectEqual(@as(u64, 1 << 20), Fake.max_write * 2);
}

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};

    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "boot refuses a RoCE slot past 1 MiB (TF_TP_MAILBOX_MAX_KB), never clamping it" {
    for ([_][]const u8{ "1028", "2048", "16384" }) |kb| {
        Env.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_MAILBOX_MAX_KB", kb } };
        try std.testing.expectError(error.BadSetting, Config.parse(&Env.get));
    }
    Env.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_MAILBOX_MAX_KB", "1024" } };
    try std.testing.expectEqual(@as(usize, 1 << 20), (try Config.parse(&Env.get)).mailbox_max_bytes);
    // the cap is RoCE's: the same-host mailbox keeps its larger slots
    Env.pairs = &.{ .{ "TF_COMM_BACKEND", "mailbox" }, .{ "TF_TP_MAILBOX_MAX_KB", "4096" } };
    try std.testing.expectEqual(@as(usize, 4 << 20), (try Config.parse(&Env.get)).mailbox_max_bytes);
}

test "chunked strides posted back to back: at most 2 MiB in flight over both HCAs and no write past 1 MiB" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(fake.slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.slot_bytes = fake.slot;
    // gathers of R x 512 x 584 B strides and 584 B rows up to a 128 MiB union, each in <= 1 MiB ops
    var strides: [16]u64 = undefined;
    var n: usize = 0;
    for ([_]u64{ 1, 2, 3, 4, 6, 16, 32, 64 }) |r| {
        strides[n] = r * 512 * 584;
        n += 1;
    }
    for ([_]u64{ 1795, 1796, (128 << 20) / (3 * 584) }) |m| {
        strides[n] = m * 584;
        n += 1;
    }
    for ([_]u32{ 1, 3, 17 }) |lag| {
        Fake.reset(lag);
        var hw: fake.Hw = .{};
        var w = hw.wire(region);
        var seq: u32 = 0;
        var want: u64 = 0;
        for (strides[0..n]) |stride| {
            var off: u64 = 0;
            while (off < stride) : (off += fake.slot) {
                seq += 1;
                const padded: u32 = @intCast(std.mem.alignForward(u64, @min(fake.slot, stride - off), 16));
                try w.post(seq & 1, seq, padded);
                want += padded;
            }
        }
        while (Fake.inflight > 0) try w.reap();
        try std.testing.expectEqual(want, Fake.bytes);
        try std.testing.expectEqual(@as(u64, 0), Fake.outside);
        // the cap's numbers, not roce_verbs.zig's constants: 1 MiB a write, 2 MiB in flight
        try std.testing.expect(Fake.max_write <= 1 << 20);
        try std.testing.expect(Fake.max_inflight <= 2 << 20);
        // the bound is reached, not avoided: posts outran completions
        try std.testing.expect(Fake.max_inflight > 1 << 20);
    }
}

test "Roce.abort (the fail-fast hook) moves both QPs to error: no send goes out and no peer write lands" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(fake.slot));
    defer std.testing.allocator.free(region);
    @memset(region, 0);
    Fake.region = region;
    Fake.slot_bytes = fake.slot;
    Fake.reset(1);
    fake.Halts.reset();
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    fake.Halts.install(&hw, &w);
    var r: Roce = .{ .me = 0, .slot_bytes = fake.slot, .timeout_ns = 1, .region = region, .mapping = region, .wire = .{ .verbs = w }, .cpu = null };
    r.abort();
    try std.testing.expect(fake.Halts.halted[0] and fake.Halts.halted[1]);
}

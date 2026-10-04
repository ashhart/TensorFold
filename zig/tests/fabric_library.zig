//! The fabric bound to MCDMA's own libmcdma-rpc (built from MCDMA's rpc/ by tools/zig/test_fabric.sh, MCDMA_RPC_LIBRARY).
const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const layout = fabric.layout;
const words = fabric.words;
const mailbox = fabric.mailbox;

const gpa = std.testing.allocator;

fn open() !words.Library {
    if (std.c.getenv("MCDMA_RPC_LIBRARY") == null) return error.SkipZigTest;
    return words.Library.open(null);
}

test "ABI 1, and waits that follow MCDMA's own helper test" {
    var lib = try open();
    defer lib.close();
    const ws = lib.words();
    var w: u64 = 0;
    try std.testing.expectEqual(@as(u64, 0), ws.wait(&w, 7, true, 1000, 2_000_000));
    try std.testing.expectEqual(@as(u64, 0), ws.wait(&w, 7, false, 1000, 2_000_000));
    ws.store(&w, layout.word(7, 12));
    try std.testing.expectEqual(layout.word(7, 12), ws.wait(&w, 7, true, 1000, 2_000_000));
    try std.testing.expectEqual(@as(u64, 0), ws.wait(&w, 7, false, 1000, 2_000_000));
    try std.testing.expectEqual(layout.word(7, 12), ws.wait(&w, 6, false, 1000, 2_000_000));
    const began = words.nowNs();
    try std.testing.expectEqual(@as(u64, 0), ws.wait(&w, 9, true, 0, 30_000_000));
    try std.testing.expect(words.nowNs() - began >= 30_000_000);
}

const Publish = struct {
    word: u64 = 0,
    payload: [4096]u8 = undefined,

    fn run(p: *Publish, ws: words.Words, seq: u32) void {
        words.pause(2 * std.time.ns_per_ms);
        for (&p.payload, 0..) |*b, i| b.* = @truncate(seq + i);
        ws.store(&p.word, layout.word(seq, p.payload.len));
    }
};

test "a store from another thread publishes its payload first, through the library" {
    var lib = try open();
    defer lib.close();
    const ws = lib.words();
    var p: Publish = .{};
    for (100..164) |s| {
        const seq: u32 = @intCast(s);
        const t = try std.Thread.spawn(.{}, Publish.run, .{ &p, ws, seq });
        const got = ws.wait(&p.word, seq, true, 1_000_000, 2 * std.time.ns_per_s);
        try std.testing.expectEqual(layout.word(seq, 4096), got);
        for (p.payload, 0..) |b, i| try std.testing.expectEqual(@as(u8, @truncate(seq + i)), b);
        t.join();
    }
}

test "a Metal buffer over a mailbox's reply half aliases it, without a copy" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    var lib = try open();
    defer lib.close();
    if (lib.metal_wrap == null) return error.SkipZigTest;
    const mem = try mailbox.allocFake(layout.segment, layout.segment);
    defer std.posix.munmap(mem);
    const box = try mailbox.Mailbox.fromMemory(mem);
    const buffer = lib.metalWrap(box.replyHalf()) orelse return error.WrapRefused;
    lib.metalRelease(buffer);
}

test "a client and a service talk through the fake link with the library's words on both ends" {
    var lib = try open();
    defer lib.close();
    const ws = lib.words();
    var link = try fabric.link_fake.Link.init(gpa, layout.segment, layout.segment, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const Echo = struct {
        fn run(s: *mailbox.Service, flag: *std.atomic.Value(bool)) void {
            while (!flag.load(.acquire)) {
                const req = (s.next(std.time.ns_per_ms) catch return) orelse continue;
                for (req.payload, s.replyArea()[0..req.payload.len]) |b, *o| o.* = b ^ 0x5a;
                s.publish(req.seq, req.payload.len) catch return;
            }
        }
    };
    var svc_stop: std.atomic.Value(bool) = .init(false);
    var svc = mailbox.Service.init(&peer, ws);
    const service = try std.Thread.spawn(.{}, Echo.run, .{ &svc, &svc_stop });
    defer {
        svc_stop.store(true, .release);
        stop.store(true, .release);
        service.join();
        daemons.join();
    }
    var client = try mailbox.Client.init(&mac, ws, 2 * std.time.ns_per_s);
    var msg: [70_000]u8 = undefined;
    for (0..100) |k| {
        const len = 1 + k * 691 % msg.len;
        for (msg[0..len], 0..) |*b, i| b.* = @truncate(i * 3 + k);
        const reply = try client.call(msg[0..len]);
        try std.testing.expectEqual(len, reply.len);
        for (reply, 0..) |b, i| try std.testing.expectEqual(@as(u8, @truncate(i * 3 + k)) ^ 0x5a, b);
    }
    try std.testing.expectEqual(@as(u32, 100), client.seq);
}

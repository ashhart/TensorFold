//! CPU-only qualification of TCP window exchanges. No model, Metal or RDMA device is opened.
const std = @import("std");
const fabric = @import("fabric");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6 and args.len != 7) {
        std.debug.print("usage: tf-tcp-probe RANK HOST PORT SESSION ROUNDS [IDLE_MS]\n", .{});
        return error.Arguments;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const port = try std.fmt.parseInt(u16, args[3], 10);
    const rounds = try std.fmt.parseInt(usize, args[5], 10);
    const idle_ms = if (args.len == 7) try std.fmt.parseInt(u64, args[6], 10) else 0;
    if (rounds == 0 or rounds > 10000) return error.Arguments;
    if (idle_ms > 1000) return error.Arguments;
    const sizes = [_]usize{ 64, 4096, 7 * 5120, 256 * 7 * 5120 };
    const maximum = sizes[sizes.len - 1];
    const ep = try fabric.tcp.Endpoint.create(init.gpa, .{ .rank = rank, .host = args[2], .port = port, .session = args[4] }, fabric.Channel.bytes(2, maximum), null);
    defer ep.deinit();
    const send = try init.gpa.alloc(u8, maximum);
    defer init.gpa.free(send);
    var ch = fabric.Channel.init(ep.rdma(), 0, maximum);
    for (sizes) |size| {
        var elapsed: u64 = 0;
        for (0..rounds + 4) |step| {
            const value: u8 = @truncate(step * 2 + rank);
            @memset(send[0..size], value);
            var received: [2][]const u8 = undefined;
            if (idle_ms != 0) fabric.words.pause(idle_ms * std.time.ns_per_ms);
            const started = try fabric.words.checkedNowNs();
            try ch.exchange(&.{ send[0..size], send[0..size] }, &received);
            const ns = (try fabric.words.checkedNowNs()) - started;
            if (step >= 4) elapsed += ns;
            const expected: u8 = @truncate(step * 2 + 1 - rank);
            if (received[1 - rank].len != size or !std.mem.allEqual(u8, received[1 - rank], expected)) return error.PayloadMismatch;
        }
        std.debug.print("{{\"transport\":\"tcp\",\"rank\":{d},\"payload_bytes\":{d},\"rounds\":{d},\"idle_ms\":{d},\"exchange_ns_total\":{d},\"payload_verified\":true}}\n", .{ rank, size, rounds, idle_ms, elapsed });
    }
    try ep.finish();
}

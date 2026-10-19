//! CPU-only checks of the same framed exchanges used by DeepSeek MoE; no model, Metal device or RDMA is opened.
const std = @import("std");
const tf = @import("tensorfold");
const clock = @import("metal").clock;
const Transport = tf.deepseek_v41.tp_transport.Transport;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: tf-dsv41-transport-probe TCP_SETTINGS ROUNDS\n", .{});
        return error.Arguments;
    }
    const rounds = try std.fmt.parseInt(usize, args[2], 10);
    if (rounds == 0 or rounds > 10000) return error.Arguments;
    const json = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.arena.allocator(), .limited(65536));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, init.arena.allocator(), json, .{});
    if (parsed != .object) return error.TcpRequired;
    const kind = parsed.object.get("transport") orelse return error.TcpRequired;
    if (kind != .string or !std.mem.eql(u8, kind.string, "tcp")) return error.TcpRequired;
    var transport = try Transport.init(init.gpa, json, 256);
    defer transport.deinit();
    const rank = transport.channel.myRank();
    const exchange = transport.exchange();
    const sizes = [_]usize{ 48, 4096, 7 * 5120, 256 * 7 * 5120 };
    const send = try init.gpa.alloc(u8, sizes[sizes.len - 1]);
    defer init.gpa.free(send);
    const receive = try init.gpa.alloc(u8, send.len);
    defer init.gpa.free(receive);
    var tag: u64 = 0;
    for (sizes) |size| {
        var elapsed: f64 = 0;
        for (0..rounds + 4) |step| {
            @memset(send[0..size], @truncate(step * 2 + rank));
            const began = clock.seconds();
            try exchange.call(exchange.ptr, tag, send[0..size], receive[0..size]);
            const duration = clock.seconds() - began;
            tag += 1;
            if (step >= 4) elapsed += duration;
            const expected: u8 = @truncate(step * 2 + 1 - rank);
            if (!std.mem.allEqual(u8, receive[0..size], expected)) return error.PayloadMismatch;
        }
        std.debug.print("{{\"mode\":\"deepseek_transport_probe\",\"rank\":{d},\"bytes\":{d},\"rounds\":{d},\"exchange_seconds_total\":{d},\"payload_verified\":true}}\n", .{ rank, size, rounds, elapsed });
    }
    try transport.endpoint.tcp.finish();
}

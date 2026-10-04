//! Runs a tokenizer over tools/zig/tokenizer_parity.py's hex-encoded cases: tokenizer-check TOKENIZER CASES OUT.
const std = @import("std");
const tokenizer = @import("tokenizer.zig");

const Cases = struct { encode: []const []const u8, decode: []const []const u32 };

fn hex(out: *std.ArrayList(u8), a: std.mem.Allocator, bytes: []const u8) !void {
    try out.append(a, '"');
    for (bytes) |b| try out.print(a, "{x:0>2}", .{b});
    try out.append(a, '"');
}

fn ids(out: *std.ArrayList(u8), a: std.mem.Allocator, values: []const u32) !void {
    try out.append(a, '[');
    for (values, 0..) |v, i| try out.print(a, "{s}{d}", .{ if (i == 0) "" else ",", v });
    try out.append(a, ']');
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tokenizer-check TOKENIZER CASES OUT\n", .{});
        return error.InvalidArguments;
    }
    const start = std.Io.Clock.awake.now(io);
    var tok = try tokenizer.loadTokenizer(io, a, args[1]);
    defer tok.deinit();
    const loaded = std.Io.Clock.awake.now(io);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 31));
    defer a.free(bytes);
    const cases = try std.json.parseFromSlice(Cases, a, bytes, .{});
    defer cases.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try out.appendSlice(a, "{\"encode\":[");
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var encode_ns: i96 = 0;
    var encoded_bytes: usize = 0;
    for (cases.value.encode, 0..) |case, i| {
        text.clearRetainingCapacity();
        try text.resize(a, case.len / 2);
        _ = try std.fmt.hexToBytes(text.items, case);
        const before = std.Io.Clock.awake.now(io);
        const result = try tok.encode(a, text.items);
        encode_ns += before.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        encoded_bytes += text.items.len;
        defer a.free(result);
        if (i > 0) try out.append(a, ',');
        try ids(&out, a, result);
    }
    try out.appendSlice(a, "],\"decode\":[");
    var decode_ns: i96 = 0;
    for (cases.value.decode, 0..) |case, i| {
        const before = std.Io.Clock.awake.now(io);
        const plain = try tok.decode(a, case, false);
        defer a.free(plain);
        const skipped = try tok.decode(a, case, true);
        defer a.free(skipped);
        decode_ns += before.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        if (i > 0) try out.append(a, ',');
        try out.append(a, '[');
        try hex(&out, a, plain);
        try out.append(a, ',');
        try hex(&out, a, skipped);
        try out.append(a, ']');
    }
    const ms = struct {
        fn of(ns: i96) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    try out.print(a, "],\"load_ms\":{d:.1},\"encode_ms\":{d:.1},\"encode_bytes\":{d},\"decode_ms\":{d:.1}}}", .{ ms(start.durationTo(loaded).nanoseconds), ms(encode_ns), encoded_bytes, ms(decode_ns) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = out.items });
}

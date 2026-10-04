//! Kimi's tiktoken tokenizer over k3_tokenizer_parity.py's hex cases: tf-k3-tokenizer K3_DIR CASES OUT.
const std = @import("std");
const tiktoken = @import("tiktoken");

const Cases = struct { encode: []const []const u8, decode: []const []const u32 };

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-k3-tokenizer K3_DIR CASES OUT\n", .{});
        std.process.exit(2);
    }
    const t0 = std.Io.Clock.awake.now(io);
    var tok = try tiktoken.TikToken.load(io, a, args[1]);
    defer tok.deinit();
    const t1 = std.Io.Clock.awake.now(io);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 31));
    defer a.free(bytes);
    const cases = try std.json.parseFromSlice(Cases, a, bytes, .{});
    defer cases.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try out.appendSlice(a, "{\"encode\":[");
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var ns: i96 = 0;
    for (cases.value.encode, 0..) |case, i| {
        try text.resize(a, case.len / 2);
        _ = try std.fmt.hexToBytes(text.items, case);
        const before = std.Io.Clock.awake.now(io);
        const ids = try tok.encode(text.items);
        ns += before.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        defer a.free(ids);
        try out.appendSlice(a, if (i == 0) "[" else ",[");
        for (ids, 0..) |v, j| try out.print(a, "{s}{d}", .{ if (j == 0) "" else ",", v });
        try out.append(a, ']');
    }
    try out.appendSlice(a, "],\"decode\":[");
    for (cases.value.decode, 0..) |case, i| {
        const plain = try tok.decode(case);
        defer a.free(plain);
        try out.appendSlice(a, if (i == 0) "\"" else ",\"");
        for (plain) |b| try out.print(a, "{x:0>2}", .{b});
        try out.append(a, '"');
    }
    try out.print(a, "],\"load_ms\":{d:.1},\"encode_ms\":{d:.1}}}", .{ @as(f64, @floatFromInt(t0.durationTo(t1).nanoseconds)) / 1e6, @as(f64, @floatFromInt(ns)) / 1e6 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = out.items });
}

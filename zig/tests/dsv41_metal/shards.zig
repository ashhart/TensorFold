//! A safetensors image cut into shards and an index, headers padded so tensor data starts at a chosen offset mod 16.
const std = @import("std");

const Entry = struct { name: []const u8, dtype: []const u8, shape: []const std.json.Value, begin: u64, end: u64 };

fn lessBegin(_: void, x: Entry, y: Entry) bool {
    return x.begin < y.begin;
}

/// `img` as `count` shards under `dir`, each with tensor data at `skew` mod 16 from the file start, and the index.
pub fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, img: []const u8, count: usize, skew: usize) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const hlen: usize = @intCast(std.mem.readInt(u64, img[0..8], .little));
    const data = img[8 + hlen ..];
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, al, img[8..][0..hlen], .{});
    var entries: std.ArrayList(Entry) = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        const o = e.value_ptr.object;
        const off = o.get("data_offsets").?.array.items;
        try entries.append(al, .{ .name = e.key_ptr.*, .dtype = o.get("dtype").?.string, .shape = o.get("shape").?.array.items, .begin = @intCast(off[0].integer), .end = @intCast(off[1].integer) });
    }
    std.mem.sort(Entry, entries.items, {}, lessBegin);
    var index: std.Io.Writer.Allocating = .init(al);
    try index.writer.writeAll("{\"metadata\":{},\"weight_map\":{");
    var first = true;
    for (0..count) |s| {
        const part = entries.items[s * entries.items.len / count ..][0 .. (s + 1) * entries.items.len / count - s * entries.items.len / count];
        var name_buf: [64]u8 = undefined;
        const file = try std.fmt.bufPrint(&name_buf, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ s + 1, count });
        var json: std.Io.Writer.Allocating = .init(al);
        try json.writer.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
        var at: u64 = 0;
        for (part) |e| {
            try json.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ e.name, e.dtype });
            for (e.shape, 0..) |d, i| try json.writer.print("{s}{d}", .{ if (i == 0) "" else ",", d.integer });
            try json.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + e.end - e.begin });
            at += e.end - e.begin;
            try index.writer.print("{s}\"{s}\":\"{s}\"", .{ if (first) "" else ",", e.name, file });
            first = false;
        }
        try json.writer.writeAll("}");
        var head = try al.dupe(u8, json.written());
        // pad with spaces (JSON whitespace) until the data begins at `skew` mod 16
        const pad = (16 + skew - (8 + head.len) % 16) % 16;
        head = try al.realloc(head, head.len + pad);
        @memset(head[head.len - pad ..], ' ');
        const out = try al.alloc(u8, 8 + head.len + @as(usize, @intCast(at)));
        std.mem.writeInt(u64, out[0..8], head.len, .little);
        @memcpy(out[8..][0..head.len], head);
        var pos = 8 + head.len;
        for (part) |e| {
            @memcpy(out[pos..][0 .. e.end - e.begin], data[e.begin..e.end]);
            pos += e.end - e.begin;
        }
        try dir.writeFile(io, .{ .sub_path = file, .data = out });
    }
    try index.writer.writeAll("}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

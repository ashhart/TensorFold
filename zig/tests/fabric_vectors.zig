//! Wire compatibility: MCDMA's own Python producer's replies (tools/zig/fabric_vectors.py), replayed byte for byte in Zig.
const std = @import("std");
const fabric = @import("fabric");
const wire = fabric.wire;
const producer = fabric.producer;
const manifest = fabric.manifest;
const frames = fabric.frames;

const gpa = std.testing.allocator;
const vectors = @embedFile("fabric_vectors.json");

const Capture = struct {
    area: [4096]u8 = undefined,
    last: []const u8 = &.{},

    fn publish(ptr: *anyopaque, _: u32, len: usize) anyerror!void {
        const c: *Capture = @ptrCast(@alignCast(ptr));
        c.last = c.area[0..len];
    }
};

fn hex(arena: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try arena.alloc(u8, s.len / 2);
    _ = try std.fmt.hexToBytes(out, s);
    return out;
}

fn ints(arena: std.mem.Allocator, v: std.json.Value, comptime T: type) ![]T {
    const out = try arena.alloc(T, v.array.items.len);
    for (v.array.items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

test "the Zig responder answers MCDMA's recorded session with identical bytes" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, arena, vectors, .{})).object;
    try std.testing.expectEqual(@as(i64, 4096), root.get("reply_area").?.integer);
    const specs = root.get("tensors").?.array.items;
    var layers: [2]manifest.Layer = undefined;
    var sources: [2]frames.Source = undefined;
    for (specs, &layers, &sources) |spec, *l, *s| {
        const o = spec.object;
        const shape = try ints(arena, o.get("shape").?, u64);
        const rows = try ints(arena, o.get("rows").?, u64);
        const elem: u64 = @intCast(o.get("itemsize").?.integer);
        const seed: usize = @intCast(o.get("seed").?.integer);
        var count: usize = @intCast(elem);
        for (shape) |d| count *= @intCast(d);
        const bytes = try arena.alloc(u8, count);
        for (bytes, 0..) |*b, i| b.* = @intCast((i * 37 + seed) % 256);
        s.* = .{ .bytes = bytes, .shape = shape, .block_dim = 0, .elem = elem, .rows = rows };
        const exported = try arena.dupe(u64, shape);
        exported[0] = rows.len;
        const dims = try arena.alloc(manifest.Dim, o.get("dims").?.array.items.len);
        for (o.get("dims").?.array.items, dims) |d, *x| x.* = std.meta.stringToEnum(manifest.Dim, d.string).?;
        const mla = std.mem.indexOfScalar(manifest.Dim, dims, .latent) != null;
        l.* = .{ .index = @intCast(o.get("index").?.integer), .kind = if (mla) .mla else .attention, .shape = exported, .dims = dims, .dtype = std.meta.stringToEnum(manifest.Dtype, o.get("dtype").?.string).? };
        if (mla) {
            l.latent_size = 64;
            l.rope_size = 8;
        } else {
            l.heads = 2;
            l.total_heads = 4;
            l.head_size = 8;
        }
    }
    const tokens = try ints(arena, root.get("tokens").?, u32);
    var exports = [_]producer.Export{
        .{ .handoff = undefined, .tokens = tokens, .first_token = 0, .tokens_per_row = 16, .layers = &layers, .sources = &sources },
        .{ .handoff = undefined, .tokens = tokens[0..33], .first_token = 16, .tokens_per_row = 16, .layers = &layers, .sources = &sources },
    };
    defer for (exports) |e| gpa.free(e.frames);
    var table: producer.Table = .{ .gpa = gpa, .ttl_ns = std.time.ns_per_s, .clock = struct {
        fn zero() u64 {
            return 0;
        }
    }.zero };
    defer table.deinit();
    var cap: Capture = .{};
    var r: producer.Responder = .{ .gpa = gpa, .table = &table, .out = .{ .area = &cap.area, .ptr = &cap, .publish_fn = Capture.publish }, .model = "org/model-\u{e9}", .tp_rank = 1, .tp_size = 2 };
    defer r.deinit();
    var added: usize = 0;
    var replies: usize = 0;
    for (root.get("events").?.array.items, 1..) |ev, seq| {
        const o = ev.object;
        if (o.get("add")) |id_hex| {
            var id: [16]u8 = undefined;
            _ = try std.fmt.hexToBytes(&id, id_hex.string);
            const failed = o.get("failed").?;
            if (failed == .string) {
                try table.add(id, "req", .{ .failed = failed.string });
            } else {
                exports[added].handoff = id;
                try table.add(id, "req", .{ .ready = &exports[added] });
                added += 1;
            }
            continue;
        }
        const request = try hex(arena, o.get("request").?.string);
        const want = try hex(arena, o.get("reply").?.string);
        r.handle(@intCast(seq), request) catch |err| try r.failed(@intCast(seq), err);
        try std.testing.expectEqualSlices(u8, want, cap.last);
        replies += 1;
    }
    try std.testing.expectEqual(@as(usize, 18), replies);
    for (root.get("digests").?.array.items) |d| {
        const t = try ints(arena, d.object.get("tokens").?, u32);
        try std.testing.expectEqualStrings(d.object.get("sha256").?.string, &wire.tokenSha256(t));
    }
    const h = try hex(arena, root.get("header").?.string);
    var first: [16]u8 = undefined;
    for (&first, 0..) |*b, i| b.* = @intCast(i);
    try std.testing.expectEqualSlices(u8, h, &wire.pack(.{ .kind = .data, .handoff = first, .frame = 3, .frames = 9, .layer = 2, .flags = wire.checked, .row_start = 16, .rows = 4, .nbytes = 8, .crc = 77 }));
}

test "the Zig decoder parses MCDMA's manifests and checks MCDMA's DATA frames" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, arena, vectors, .{})).object;
    var manifests: usize = 0;
    var datas: usize = 0;
    for (root.get("events").?.array.items) |ev| {
        const reply_hex = ev.object.get("reply") orelse continue;
        const reply = try hex(arena, reply_hex.string);
        const head = wire.unpack(reply).ok;
        switch (head.kind) {
            .manifest => {
                const m = try manifest.parse(arena, wire.body(reply, head));
                try std.testing.expectEqualStrings("org/model-\u{e9}", m.model);
                try std.testing.expectEqual(@as(u64, head.frames), m.frames);
                var out: std.Io.Writer.Allocating = .init(gpa);
                defer out.deinit();
                try manifest.write(m, &out.writer);
                try std.testing.expectEqualStrings(wire.body(reply, head), out.written());
                manifests += 1;
            },
            .data => {
                const body = wire.body(reply, head);
                if (head.flags & wire.checked != 0) try std.testing.expectEqual(head.crc, wire.crc32(body));
                datas += 1;
            },
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 2), manifests);
    try std.testing.expectEqual(@as(usize, 5), datas);
}

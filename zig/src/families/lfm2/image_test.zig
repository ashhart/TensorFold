//! The picture goldens (zig/tests/fixtures/lfm2/images.py): RGB, refusals, tiles, patches and ids.

const std = @import("std");
const image = @import("image.zig");
const tokenizer = @import("core").tokenizer;

const Case = struct {
    name: []const u8,
    url: []const u8,
    problem: ?[]const u8 = null,
    width: usize = 0,
    height: usize = 0,
    rgb: []const u8 = "",
    tiles: ?[]const [2]usize = null,
    patches: []const []const u8 = &.{},
    ids: []const [2]u32 = &.{},
};

const Golden = struct {
    marks: struct { image: u32, start: u32, end: u32, thumbnail: u32, tiles: [image.max_grid][image.max_grid]u32 },
    cases: []const Case,
};

fn sha256(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn goldens(arena: std.mem.Allocator) !Golden {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/lfm2/images.json", arena, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(Golden, arena, text, .{});
}

test "pictures match tensorfold's loader and the LFM2-VL processor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = try goldens(arena.allocator());
    const marks = try marksFromTokenizer(arena.allocator(), golden);
    const a = std.testing.allocator;
    for (golden.cases) |case| {
        errdefer std.debug.print("case {s}\n", .{case.name});
        var problem: []const u8 = "";
        var rgb = (try image.load(a, case.url, &problem)) orelse {
            try std.testing.expectEqualStrings(case.problem orelse "", problem);
            continue;
        };
        defer rgb.deinit(a);
        try std.testing.expect(case.problem == null);
        try std.testing.expectEqual(case.width, rgb.width);
        try std.testing.expectEqual(case.height, rgb.height);
        try std.testing.expectEqualStrings(case.rgb, &sha256(rgb.pixels));
        const tiles = case.tiles orelse continue;
        var pic = try image.prepare(a, rgb);
        defer pic.deinit(a);
        try std.testing.expectEqual(tiles.len, pic.tiles.len);
        var at: usize = 0;
        for (tiles, pic.tiles, case.patches) |want, got, digest| {
            try std.testing.expectEqual(want, [2]usize{ got.h, got.w });
            const n = got.h * got.w * image.patch_width;
            try std.testing.expectEqualStrings(digest, &sha256(std.mem.sliceAsBytes(pic.patches[at..][0..n])));
            at += n;
        }
        try std.testing.expectEqual(at, pic.patches.len);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);
        try pic.appendIds(a, &ids, &marks);
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        for (case.ids) |run| try want.appendNTimes(a, run[0], run[1]);
        try std.testing.expectEqualSlices(u32, want.items, ids.items);
        try std.testing.expectEqual(std.mem.count(u32, ids.items, &.{marks.image}), pic.tokens());
    }
}

test "truncated and corrupted pictures are refused or prepared, never trapped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = try goldens(arena.allocator());
    const a = std.testing.allocator;
    const b64 = std.base64.standard;
    var prng = std.Random.DefaultPrng.init(0x4c464d32);
    const r = prng.random();
    var outcomes: [2]usize = .{ 0, 0 }; // refused, prepared
    for (golden.cases) |case| {
        const comma = std.mem.indexOf(u8, case.url, ";base64,") orelse continue;
        if (case.problem != null or case.width * case.height > 40_000) continue;
        const payload = case.url[comma + 8 ..];
        const clean = try arena.allocator().alloc(u8, try b64.Decoder.calcSizeForSlice(payload));
        try b64.Decoder.decode(clean, payload);
        for (0..120) |i| {
            const bytes = try a.dupe(u8, clean);
            defer a.free(bytes);
            var v = bytes;
            if (i % 4 == 0) {
                v = bytes[0..r.uintLessThan(usize, bytes.len)];
            } else for (0..1 + r.uintLessThan(usize, 4)) |_| {
                v[r.uintLessThan(usize, if (r.boolean()) @min(v.len, 600) else v.len)] = r.int(u8);
            }
            if (i % 2 == 1 and std.mem.startsWith(u8, v, "\x89PNG")) refreshCrcs(v);
            const prefix = "data:image/x;base64,";
            const url = try a.alloc(u8, prefix.len + b64.Encoder.calcSize(v.len));
            defer a.free(url);
            @memcpy(url[0..prefix.len], prefix);
            _ = b64.Encoder.encode(url[prefix.len..], v);
            var problem: []const u8 = "";
            var rgb = (try image.load(a, url, &problem)) orelse {
                outcomes[0] += 1;
                continue;
            };
            defer rgb.deinit(a);
            var pic = try image.prepare(a, rgb);
            pic.deinit(a);
            outcomes[1] += 1;
        }
    }
    try std.testing.expect(outcomes[0] + outcomes[1] >= 3000 and outcomes[0] > 0 and outcomes[1] > 0);
}

/// Rewrites each whole chunk's CRC, so a corruption reaches the decoder past the checksum.
fn refreshCrcs(png: []u8) void {
    var at: usize = 8;
    while (png.len >= at + 12) {
        const len = std.mem.readInt(u32, png[at..][0..4], .big);
        if (len > png.len - at - 12) return;
        std.mem.writeInt(u32, png[at + 8 + len ..][0..4], std.hash.Crc32.hash(png[at + 4 .. at + 8 + len]), .big);
        at += 12 + len;
    }
}

/// The marks as `Marks.resolve` finds them in a tokenizer holding the goldens' ids under their names.
fn marksFromTokenizer(arena: std.mem.Allocator, golden: Golden) !image.Marks {
    var json: std.Io.Writer.Allocating = .init(arena);
    const w = &json.writer;
    const m = golden.marks;
    try w.print("{{\"model\": {{\"type\": \"BPE\", \"merges\": [], \"vocab\": {{\"<image>\": {d}, \"<|image_start|>\": {d}, \"<|image_end|>\": {d}, \"<|img_thumbnail|>\": {d}", .{ m.image, m.start, m.end, m.thumbnail });
    for (m.tiles, 1..) |row, r| for (row, 1..) |id, c| try w.print(", \"<|img_row_{d}_col_{d}|>\": {d}", .{ r, c, id });
    try w.writeAll("}}}");
    var t = try tokenizer.parse(arena, json.written());
    defer t.deinit();
    const marks = image.Marks.resolve(&t) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(m.tiles, marks.tiles);
    try std.testing.expectEqual(m.thumbnail, marks.thumbnail);
    return marks;
}

//! The decoder against what 0.6.6 decoded (LFM2's picture goldens and the edge pictures), and against damaged bytes.

const std = @import("std");
const picture = @import("picture.zig");

const Case = struct {
    name: []const u8,
    url: []const u8,
    problem: ?[]const u8 = null,
    width: usize = 0,
    height: usize = 0,
    rgb: []const u8 = "",
};

const Golden = struct { cases: []const Case };

const prefix = ";base64,";

fn goldens(arena: std.mem.Allocator) !Golden {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/lfm2/images.json", arena, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(Golden, arena, text, .{ .ignore_unknown_fields = true });
}

/// A golden's bytes when its URL is a plain base64 data URL; the URL refusals are the caller's to test.
fn payload(arena: std.mem.Allocator, url: []const u8) !?[]u8 {
    if (!std.mem.startsWith(u8, url, "data:")) return null;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return null;
    const media = url[5..comma];
    if (!std.mem.endsWith(u8, media, prefix[0 .. prefix.len - 1]) or std.mem.count(u8, media, ";") != 1) return null;
    const b64 = std.base64.standard.Decoder;
    const out = try arena.alloc(u8, b64.calcSizeForSlice(url[comma + 1 ..]) catch return null);
    b64.decode(out, url[comma + 1 ..]) catch return null;
    return out;
}

/// JPEG cases need the pinned library: required when TENSORFOLD_LIBTURBOJPEG names one, skipped when none loads.
fn jpegReady(a: std.mem.Allocator) !bool {
    if (picture.jpeg.available(std.testing.io, a)) return true;
    if (@import("sys").getenv(picture.jpeg.env) != null) return error.PinnedLibraryDidNotLoad;
    std.debug.print("libjpeg-turbo {s} not found: JPEG pixel cases skipped\n", .{picture.jpeg.pinned});
    return false;
}

fn sha256(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

test "every picture 0.6.6 decoded decodes to the same RGB bytes, and its refusals read the same" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const golden = try goldens(arena);
    const jpeg = try jpegReady(a);
    var checked: [2]usize = .{ 0, 0 }; // decoded, refused
    for (golden.cases) |case| {
        errdefer std.debug.print("case {s}\n", .{case.name});
        const bytes = (try payload(arena, case.url)) orelse continue; // not a base64 picture: the caller's refusal
        if (bytes.len == 0) continue;
        if (picture.sniff(bytes) == .jpeg and !jpeg) continue;
        var rgb = picture.decode(std.testing.io, a, bytes) catch |err| {
            try std.testing.expectEqualStrings(case.problem orelse @errorName(err), picture.problem(err));
            checked[1] += 1;
            continue;
        };
        defer rgb.deinit(a);
        try std.testing.expect(case.problem == null);
        try std.testing.expectEqual(case.width, rgb.width);
        try std.testing.expectEqual(case.height, rgb.height);
        try std.testing.expectEqualStrings(case.rgb, &sha256(rgb.pixels));
        checked[0] += 1;
    }
    try std.testing.expect(checked[0] >= @as(usize, if (jpeg) 29 else 19) and checked[1] >= 4);
}

const Edge = struct { name: []const u8, b64: []const u8, width: usize = 0, height: usize = 0, rgb: ?[]const u8 = null, problem: ?[]const u8 = null };

/// Where this decoder parts from 0.6.6 on purpose: a CMYK JPEG is refused by name; a two-picture MPO gives its first.
const departures = [_]struct { name: []const u8, outcome: []const u8 }{
    .{ .name = "jpeg_cmyk", .outcome = "CMYK JPEG images are unsupported here; send an RGB or grayscale JPEG, or a PNG" },
    .{ .name = "jpeg_mpo_two", .outcome = "40x30 89ec4fbf73b19a585760f7293a178a3f02c38fe177491362e1d84fd3fdd5be70" },
};

test "edge pictures (zig/tests/fixtures/picture/edges.json) get 0.6.6's outcome, but for the named departures" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/edges.json", arena, .limited(1 << 22));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Edge }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    const jpeg = try jpegReady(a);
    var checked: usize = 0;
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const b64 = std.base64.standard.Decoder;
        const bytes = try arena.alloc(u8, try b64.calcSizeForSlice(c.b64));
        try b64.decode(bytes, c.b64);
        const ours = if (picture.decode(std.testing.io, a, bytes)) |got| blk: {
            var rgb = got;
            defer rgb.deinit(a);
            break :blk try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ rgb.width, rgb.height, &sha256(rgb.pixels) });
        } else |err| blk: {
            if (!jpeg and (err == error.NoJpegLibrary or err == error.JpegLibraryVersion)) break :blk null;
            break :blk picture.problem(err);
        };
        const got = ours orelse continue; // a JPEG that needs the library this host lacks
        var want = if (c.rgb) |h| try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ c.width, c.height, h }) else c.problem.?;
        for (departures) |d| if (std.mem.eql(u8, d.name, c.name)) {
            want = d.outcome;
        };
        try std.testing.expectEqualStrings(want, got);
        checked += 1;
    }
    try std.testing.expect(checked >= @as(usize, if (jpeg) 76 else 50));
}

/// WebP cases need the pinned libwebp: required when TENSORFOLD_LIBWEBP names one, skipped when none loads.
fn webpReady(a: std.mem.Allocator) !bool {
    if (picture.webp.available(std.testing.io, a)) return true;
    if (@import("sys").getenv(picture.webp.env) != null) return error.PinnedLibraryDidNotLoad;
    std.debug.print("libwebp {s} not found: WebP pixel cases skipped\n", .{picture.webp.pinned});
    return false;
}

test "WebP pictures (zig/tests/fixtures/picture/webp.json) decode to Pillow's RGB bytes, and its refusals read the same" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/webp.json", arena, .limited(1 << 22));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Edge }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    const ready = try webpReady(a);
    var checked: usize = 0;
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const b64 = std.base64.standard.Decoder;
        const bytes = try arena.alloc(u8, try b64.calcSizeForSlice(c.b64));
        try b64.decode(bytes, c.b64);
        try std.testing.expectEqual(picture.Format.webp, picture.sniff(bytes));
        const got = if (picture.decode(std.testing.io, a, bytes)) |d| blk: {
            var rgb = d;
            defer rgb.deinit(a);
            break :blk try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ rgb.width, rgb.height, &sha256(rgb.pixels) });
        } else |err| blk: {
            if (!ready and (err == error.NoWebpLibrary or err == error.WebpLibraryVersion)) continue;
            break :blk picture.problem(err);
        };
        const want = if (c.rgb) |h| try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ c.width, c.height, h }) else c.problem.?;
        try std.testing.expectEqualStrings(want, got);
        checked += 1;
    }
    try std.testing.expect(checked >= @as(usize, if (ready) 8 else 2)); // without the library: the files refused before it loads
}

test "damaged and cut pictures are refused or decoded, never trapped" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const golden = try goldens(arena);
    const jpeg = try jpegReady(a);
    var prng = std.Random.DefaultPrng.init(0x70696374);
    const r = prng.random();
    var outcomes: [2]usize = .{ 0, 0 };
    for (golden.cases) |case| {
        if (case.problem != null or case.width * case.height > 40_000) continue;
        const clean = (try payload(arena, case.url)) orelse continue;
        if (picture.sniff(clean) == .jpeg and !jpeg) continue;
        for (0..80) |i| {
            const v = try a.dupe(u8, clean);
            defer a.free(v);
            var bytes = v;
            if (i % 4 == 0) {
                bytes = v[0..r.uintLessThan(usize, v.len)];
            } else for (0..1 + r.uintLessThan(usize, 4)) |_| {
                v[r.uintLessThan(usize, if (r.boolean()) @min(v.len, 600) else v.len)] = r.int(u8);
            }
            if (i % 2 == 1 and picture.sniff(bytes) == .png) refreshCrcs(bytes);
            var rgb = picture.decode(std.testing.io, a, bytes) catch {
                outcomes[0] += 1;
                continue;
            };
            rgb.deinit(a);
            outcomes[1] += 1;
        }
    }
    try std.testing.expect(outcomes[0] > 0 and outcomes[1] > 0);
}

/// Each whole chunk's CRC rewritten, so a damaged body reaches the decoder past its checksum.
fn refreshCrcs(png: []u8) void {
    var at: usize = 8;
    while (png.len >= at + 12) {
        const len = std.mem.readInt(u32, png[at..][0..4], .big);
        if (len > png.len - at - 12) return;
        std.mem.writeInt(u32, png[at + 8 + len ..][0..4], std.hash.Crc32.hash(png[at + 4 .. at + 8 + len]), .big);
        at += 12 + len;
    }
}

test "sizes are judged as 0.6.6 judged them" {
    try picture.fits(8192, 2048);
    try std.testing.expectError(error.Oversized, picture.fits(8193, 1));
    try std.testing.expectError(error.Invalid, picture.fits(0, 5)); // Pillow's open found no image
    try std.testing.expectError(error.Oversized, picture.fits(4097, 4096));
    try std.testing.expectError(error.Invalid, picture.fits(9000, 10000)); // Pillow's bomb bound first
    try std.testing.expectError(error.Invalid, picture.fits(0, 100_000_000));
}

test "alpha over white keeps opaque values and whitens clear ones" {
    for (0..256) |c| try std.testing.expectEqual(@as(u8, @intCast(c)), picture.overWhite(@intCast(c), 255));
    try std.testing.expectEqual(@as(u8, 255), picture.overWhite(0, 0));
    try std.testing.expectEqual(@as(u8, 132), picture.overWhite(10, 128));
}

test "orientations transpose as Pillow's do" {
    const a = std.testing.allocator;
    var px: [18]u8 = undefined; // 3 x 2, each pixel its index: 0 1 2 / 3 4 5
    for (0..6) |i| @memset(px[i * 3 ..][0..3], @intCast(i));
    const want = [_][]const u8{
        &.{ 2, 1, 0, 5, 4, 3 }, &.{ 5, 4, 3, 2, 1, 0 }, &.{ 3, 4, 5, 0, 1, 2 }, &.{ 0, 3, 1, 4, 2, 5 },
        &.{ 3, 0, 4, 1, 5, 2 }, &.{ 5, 2, 4, 1, 3, 0 }, &.{ 2, 5, 1, 4, 0, 3 },
    };
    for (want, 2..) |w, o| {
        var out = try picture.finish(a, .{ .width = 3, .height = 2, .channels = 3, .data = &px, .orientation = @intCast(o) });
        defer out.deinit(a);
        for (w, 0..) |v, i| try std.testing.expectEqual(v, out.pixels[i * 3]);
    }
}

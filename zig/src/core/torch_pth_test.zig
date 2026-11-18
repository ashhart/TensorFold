//! torch_pth's nested roots and raw reads on archives built here: names, plain values, views, refusals and damage.

const std = @import("std");
const pth = @import("torch_pth.zig");
const Archive = pth.Archive;

/// A torch.save pickle, written opcode by opcode (protocol 2, as torch.save writes it).
const Pickle = struct {
    b: std.ArrayList(u8) = .empty,

    fn op(p: *Pickle, bytes: []const u8) void {
        p.b.appendSlice(std.testing.allocator, bytes) catch @panic("oom");
    }

    fn str(p: *Pickle, s: []const u8) void {
        p.op("X");
        var n: [4]u8 = undefined;
        std.mem.writeInt(u32, &n, @intCast(s.len), .little);
        p.op(&n);
        p.op(s);
    }

    fn int(p: *Pickle, v: u16) void {
        if (v < 256) p.op(&.{ 'K', @intCast(v) }) else p.op(&.{ 'M', @truncate(v), @intCast(v >> 8) });
    }

    fn ints(p: *Pickle, vs: []const u16) void {
        p.op("(");
        for (vs) |v| p.int(v);
        p.op("t");
    }

    /// An OrderedDict, its items and the `_metadata` attribute torch's state dicts carry (BUILD).
    fn stateDictOpen(p: *Pickle) void {
        p.op("ccollections\nOrderedDict\n)R(");
    }

    fn stateDictClose(p: *Pickle) void {
        p.op("u}");
        p.str("_metadata");
        p.op("ccollections\nOrderedDict\n)R");
        p.str("");
        p.op("}");
        p.str("version");
        p.int(1);
        p.op("sssb");
    }

    /// _rebuild_tensor_v2((storage, Type, key, "cpu", numel), offset, shape, stride, False, OrderedDict()).
    fn tensor(p: *Pickle, storage: []const u8, key: []const u8, numel: u16, offset: u16, shape: []const u16, stride: []const u16) void {
        p.op("ctorch._utils\n_rebuild_tensor_v2\n((");
        p.str("storage");
        p.op("ctorch\n");
        p.op(storage);
        p.op("\n");
        p.str(key);
        p.str("cpu");
        p.int(numel);
        p.op("tQ");
        p.int(offset);
        p.ints(shape);
        p.ints(stride);
        p.op("\x89ccollections\nOrderedDict\n)RtR");
    }

    fn done(p: *Pickle) []const u8 {
        p.op(".");
        return p.b.items;
    }

    fn deinit(p: *Pickle) void {
        p.b.deinit(std.testing.allocator);
    }
};

const Entry = struct { name: []const u8, data: []const u8, method: u16 = 0, crc: ?u32 = null };

fn put16(l: *std.ArrayList(u8), v: u16) void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    l.appendSlice(std.testing.allocator, &b) catch @panic("oom");
}

fn put32(l: *std.ArrayList(u8), v: u32) void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    l.appendSlice(std.testing.allocator, &b) catch @panic("oom");
}

/// A zip of stored members (local headers, central directory, end record), as torch.save lays it out.
fn zipBytes(entries: []const Entry) []u8 {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    var cd: std.ArrayList(u8) = .empty;
    defer cd.deinit(a);
    for (entries) |e| {
        const crc = e.crc orelse std.hash.Crc32.hash(e.data);
        const local: u32 = @intCast(out.items.len);
        for ([_]*std.ArrayList(u8){ &out, &cd }, 0..) |l, central| {
            put32(l, if (central == 0) 0x04034b50 else 0x02014b50);
            if (central == 1) put16(l, 20);
            for ([_]u16{ 20, 0, e.method, 0, 0 }) |v| put16(l, v);
            for ([_]u32{ crc, @intCast(e.data.len), @intCast(e.data.len) }) |v| put32(l, v);
            put16(l, @intCast(e.name.len));
            put16(l, 0);
            if (central == 1) {
                for ([_]u16{ 0, 0, 0 }) |v| put16(l, v);
                put32(l, 0);
                put32(l, local);
            }
            l.appendSlice(a, e.name) catch @panic("oom");
        }
        out.appendSlice(a, e.data) catch @panic("oom");
    }
    const cd_off: u32 = @intCast(out.items.len);
    out.appendSlice(a, cd.items) catch @panic("oom");
    put32(&out, 0x06054b50);
    for ([_]u16{ 0, 0, @intCast(entries.len), @intCast(entries.len) }) |v| put16(&out, v);
    put32(&out, @intCast(cd.items.len));
    put32(&out, cd_off);
    put16(&out, 0);
    return out.toOwnedSlice(a) catch @panic("oom");
}

const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    fn init(bytes: []const u8) !Fixture {
        const io = std.testing.io;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "model.pth", .data = bytes });
        const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
        defer std.testing.allocator.free(dir);
        return .{ .tmp = tmp, .path = try std.fs.path.join(std.testing.allocator, &.{ dir, "model.pth" }) };
    }

    fn deinit(f: *Fixture) void {
        std.testing.allocator.free(f.path);
        f.tmp.cleanup();
    }
};

/// Ten float32 values 0..9 and six bf16 words, as the storages "0" and "1".
fn storages() [2][]const u8 {
    const S = struct {
        const f32s: [40]u8 = blk: {
            var b: [40]u8 = undefined;
            for (0..10) |i| std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(@as(f32, @floatFromInt(i))), .little);
            break :blk b;
        };
        const bf16s = [12]u8{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0 };
    };
    return .{ &S.f32s, &S.bf16s };
}

/// A state dict under "model", a tensor two dicts down under "ema", and plain values beside them.
fn nestedPickle(p: *Pickle) []const u8 {
    p.op("\x80\x02}(");
    p.str("model");
    p.stateDictOpen();
    p.str("enc.w");
    p.tensor("FloatStorage", "0", 10, 4, &.{ 2, 3 }, &.{ 3, 1 });
    p.str("enc.b");
    p.tensor("FloatStorage", "0", 10, 0, &.{2}, &.{1});
    p.stateDictClose();
    p.str("ema");
    p.op("}(");
    p.str("inner");
    p.op("}(");
    p.str("b");
    p.tensor("BFloat16Storage", "1", 6, 2, &.{ 2, 2 }, &.{ 2, 1 });
    p.op("u");
    p.op("u");
    p.str("step");
    p.int(7);
    p.str("config");
    p.op("}(");
    p.str("name");
    p.str("x");
    p.str("layers");
    p.op("](K\x01K\x02e");
    p.op("u");
    p.str("lr");
    p.op("G\x3f\xe0\x00\x00\x00\x00\x00\x00");
    p.op("u");
    return p.done();
}

fn nestedArchive(pickle: []const u8) []u8 {
    const s = storages();
    return zipBytes(&.{
        .{ .name = "ckpt/data.pkl", .data = pickle },
        .{ .name = "ckpt/byteorder", .data = "little" },
        .{ .name = "ckpt/data/0", .data = s[0] },
        .{ .name = "ckpt/data/1", .data = s[1] },
        .{ .name = "ckpt/version", .data = "3\n" },
    });
}

test "openNested names a nested root's tensors by their joined keys and skips plain values and metadata" {
    const gpa = std.testing.allocator;
    var p: Pickle = .{};
    defer p.deinit();
    const bytes = nestedArchive(nestedPickle(&p));
    defer gpa.free(bytes);
    var f = try Fixture.init(bytes);
    defer f.deinit();
    var arc = try Archive.openNested(gpa, std.testing.io, f.path);
    defer arc.close();
    try std.testing.expectEqual(@as(u32, 3), arc.tensors.count());
    try std.testing.expectEqualStrings("ckpt/", arc.prefix);
    const w = try arc.tensor("model/enc.w");
    try std.testing.expectEqual(pth.DType.f32, w.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3 }, w.shape);
    try std.testing.expectEqual(@as(u64, 4), w.offset);
    try std.testing.expectEqual(pth.DType.bf16, (try arc.tensor("ema/inner/b")).dtype);
    for ([_][]const u8{ "step", "config/name", "config/layers", "lr", "model/_metadata", "model", "ema/inner" }) |name|
        try std.testing.expect(arc.tensors.get(name) == null);
}

test "readRaw returns a view's own elements: offset slices of a shared storage, at each width" {
    const gpa = std.testing.allocator;
    var p: Pickle = .{};
    defer p.deinit();
    const bytes = nestedArchive(nestedPickle(&p));
    defer gpa.free(bytes);
    var f = try Fixture.init(bytes);
    defer f.deinit();
    var arc = try Archive.openNested(gpa, std.testing.io, f.path);
    defer arc.close();
    const s = storages();
    const w = try arc.readRaw(gpa, "model/enc.w", .f32, &.{ 2, 3 });
    defer gpa.free(w);
    try std.testing.expectEqualSlices(u8, s[0][16..40], w);
    const b = try arc.readRaw(gpa, "model/enc.b", .f32, &.{2});
    defer gpa.free(b);
    try std.testing.expectEqualSlices(u8, s[0][0..8], b);
    const h = try arc.readRaw(gpa, "ema/inner/b", .bf16, &.{ 2, 2 });
    defer gpa.free(h);
    try std.testing.expectEqualSlices(u8, s[1][4..12], h);
    try std.testing.expectError(error.UnexpectedDType, arc.readRaw(gpa, "ema/inner/b", .f16, &.{ 2, 2 }));
}

test "readRaw refuses a view past its storage, a strided view and a storage of the wrong size" {
    const gpa = std.testing.allocator;
    var p: Pickle = .{};
    defer p.deinit();
    p.op("\x80\x02}(");
    p.str("past");
    p.tensor("FloatStorage", "0", 10, 5, &.{ 2, 3 }, &.{ 3, 1 });
    p.str("strided");
    p.tensor("FloatStorage", "0", 10, 0, &.{ 2, 3 }, &.{ 1, 2 });
    p.str("short");
    p.tensor("FloatStorage", "0", 11, 0, &.{2}, &.{1});
    p.str("missing");
    p.tensor("FloatStorage", "9", 2, 0, &.{2}, &.{1});
    p.op("u");
    const s = storages();
    const bytes = zipBytes(&.{ .{ .name = "ckpt/data.pkl", .data = p.done() }, .{ .name = "ckpt/data/0", .data = s[0] } });
    defer gpa.free(bytes);
    var f = try Fixture.init(bytes);
    defer f.deinit();
    var arc = try Archive.open(gpa, std.testing.io, f.path);
    defer arc.close();
    try std.testing.expectError(error.BadArchive, arc.readRaw(gpa, "past", .f32, &.{ 2, 3 }));
    try std.testing.expectError(error.NonContiguousTensor, arc.readRaw(gpa, "strided", .f32, &.{ 2, 3 }));
    try std.testing.expectError(error.BadArchive, arc.readRaw(gpa, "short", .f32, &.{2}));
    try std.testing.expectError(error.MissingStorage, arc.readRaw(gpa, "missing", .f32, &.{2}));
}

test "open refuses a nested root in flat mode; openNested refuses a name reached twice and roots past eight levels" {
    const gpa = std.testing.allocator;
    var p: Pickle = .{};
    defer p.deinit();
    const nested = nestedArchive(nestedPickle(&p));
    defer gpa.free(nested);
    var f = try Fixture.init(nested);
    defer f.deinit();
    try std.testing.expectError(error.BadPickle, Archive.open(gpa, std.testing.io, f.path));

    var twice: Pickle = .{};
    defer twice.deinit();
    twice.op("\x80\x02}(");
    twice.str("a/b");
    twice.tensor("FloatStorage", "0", 10, 0, &.{2}, &.{1});
    twice.str("a");
    twice.op("}(");
    twice.str("b");
    twice.tensor("FloatStorage", "0", 10, 0, &.{2}, &.{1});
    twice.op("uu");
    const dup = nestedArchive(twice.done());
    defer gpa.free(dup);
    var g = try Fixture.init(dup);
    defer g.deinit();
    try std.testing.expectError(error.BadPickle, Archive.openNested(gpa, std.testing.io, g.path));

    var deep: Pickle = .{};
    defer deep.deinit();
    deep.op("\x80\x02");
    for (0..10) |_| {
        deep.op("}(");
        deep.str("d");
    }
    deep.tensor("FloatStorage", "0", 10, 0, &.{2}, &.{1});
    for (0..10) |_| deep.op("u");
    const deep_bytes = nestedArchive(deep.done());
    defer gpa.free(deep_bytes);
    var h = try Fixture.init(deep_bytes);
    defer h.deinit();
    try std.testing.expectError(error.BadPickle, Archive.openNested(gpa, std.testing.io, h.path));
}

fn expectOpenError(expected: anyerror, bytes: []const u8) !void {
    var f = try Fixture.init(bytes);
    defer f.deinit();
    try std.testing.expectError(expected, Archive.openNested(std.testing.allocator, std.testing.io, f.path));
}

test "damaged archives are refused: CRC mismatches, truncation, missing or doubled pickles, big-endian, bad pickles" {
    const gpa = std.testing.allocator;
    var p: Pickle = .{};
    defer p.deinit();
    const pickle = nestedPickle(&p);
    const s = storages();

    // a storage byte changed after its CRC was written: the open succeeds, the read refuses
    const good = nestedArchive(pickle);
    defer gpa.free(good);
    const flipped = try gpa.dupe(u8, good);
    defer gpa.free(flipped);
    const at = std.mem.indexOf(u8, flipped, s[0]).?;
    flipped[at + 20] ^= 0x01;
    var f = try Fixture.init(flipped);
    defer f.deinit();
    var arc = try Archive.openNested(gpa, std.testing.io, f.path);
    defer arc.close();
    try std.testing.expectError(error.ChecksumMismatch, arc.readRaw(gpa, "model/enc.w", .f32, &.{ 2, 3 }));

    const bad_crc = zipBytes(&.{.{ .name = "ckpt/data.pkl", .data = pickle, .crc = 0 }});
    defer gpa.free(bad_crc);
    try expectOpenError(error.ChecksumMismatch, bad_crc);

    try expectOpenError(error.BadArchive, good[0 .. good.len - 22]); // no end record
    try expectOpenError(error.BadArchive, good[0..10]);

    const none = zipBytes(&.{.{ .name = "ckpt/data/0", .data = s[0] }});
    defer gpa.free(none);
    try expectOpenError(error.BadArchive, none);

    const two = zipBytes(&.{ .{ .name = "a/data.pkl", .data = pickle }, .{ .name = "b/data.pkl", .data = pickle } });
    defer gpa.free(two);
    try expectOpenError(error.BadArchive, two);

    const big = zipBytes(&.{ .{ .name = "ckpt/data.pkl", .data = pickle }, .{ .name = "ckpt/byteorder", .data = "big" } });
    defer gpa.free(big);
    try expectOpenError(error.BigEndianArchive, big);

    const cut = zipBytes(&.{.{ .name = "ckpt/data.pkl", .data = pickle[0 .. pickle.len - 1] }}); // no STOP
    defer gpa.free(cut);
    try expectOpenError(error.BadPickle, cut);

    // a storage member whose local header is said to lie past the end of the file
    const far = try gpa.dupe(u8, good);
    defer gpa.free(far);
    const name = std.mem.lastIndexOf(u8, far, "ckpt/data/0").?; // its central directory entry's name
    std.mem.writeInt(u32, far[name - 4 ..][0..4], 1 << 30, .little);
    try expectOpenError(error.BadArchive, far);
}

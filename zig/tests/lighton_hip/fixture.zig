//! Strict local reference inputs and diagnostic tensor output for the private HIP capture tool.
const std = @import("std");
const core = @import("core");
const Allocator = std.mem.Allocator;

pub const Files = struct {
    a: Allocator,
    io: std.Io,
    reference: []const u8,
    output: std.Io.Dir,
    manifest: std.json.Value,

    pub fn init(a: Allocator, io: std.Io, reference: []const u8, output: []const u8) !Files {
        const from = try std.fs.path.resolve(a, &.{reference});
        const to = try std.fs.path.resolve(a, &.{output});
        if (std.mem.eql(u8, from, to)) return error.OutputIsReference;
        const path = try std.fs.path.join(a, &.{ reference, "manifest.json" });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(8 << 20));
        const manifest = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        _ = try field(manifest, "tensors");
        return .{ .a = a, .io = io, .reference = reference, .output = try std.Io.Dir.cwd().createDirPathOpen(io, output, .{}), .manifest = manifest };
    }

    pub fn read(f: Files, name: []const u8) !core.npy.Array {
        const record = try field(try field(f.manifest, "tensors"), name);
        const file = try string(try field(record, "file"));
        if (!std.mem.eql(u8, std.fs.path.basename(file), file) or !std.mem.endsWith(u8, file, ".npy"))
            return error.BadFixture;
        const path = try std.fs.path.join(f.a, &.{ f.reference, file });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(f.io, path, f.a, .limited(1 << 30));
        const expected = try string(try field(record, "sha256"));
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        const hex = std.fmt.bytesToHex(hash, .lower);
        if (!std.mem.eql(u8, &hex, expected)) return error.FixtureChecksum;
        const array = try core.npy.parse(bytes);
        if (!std.mem.eql(u8, array.descr, try string(try field(record, "dtype")))) return error.BadFixture;
        const shape = try list(try field(record, "shape"));
        if (shape.len != array.rank) return error.BadFixture;
        for (shape, array.shape[0..array.rank]) |dim, actual| if (try integer(dim) != actual) return error.BadFixture;
        return array;
    }

    pub fn verifyCheckpoint(f: Files, model: []const u8) !void {
        const checkpoint = try field(f.manifest, "checkpoint");
        if (!std.mem.eql(u8, try string(try field(checkpoint, "repo")), "lightonai/LightOnOCR-3-1B"))
            return error.WrongReferenceModel;
        for ([_][]const u8{ "assets", "weights_sha256" }) |group| {
            const records = try field(checkpoint, group);
            if (records != .object or records.object.count() == 0) return error.BadFixture;
            var it = records.object.iterator();
            while (it.next()) |entry| {
                const name = entry.key_ptr.*;
                if (!std.mem.eql(u8, std.fs.path.basename(name), name)) return error.BadFixture;
                const path = try std.fs.path.join(f.a, &.{ model, name });
                const file = try std.Io.Dir.cwd().openFile(f.io, path, .{});
                defer file.close(f.io);
                const length = try file.length(f.io);
                var sha = std.crypto.hash.sha2.Sha256.init(.{});
                const buffer = try f.a.alloc(u8, 4 << 20);
                defer f.a.free(buffer);
                var offset: u64 = 0;
                while (offset < length) {
                    const part = buffer[0..@intCast(@min(buffer.len, length - offset))];
                    if (try file.readPositionalAll(f.io, part, offset) != part.len) return error.CheckpointChanged;
                    sha.update(part);
                    offset += part.len;
                }
                const hex = std.fmt.bytesToHex(sha.finalResult(), .lower);
                if (!std.mem.eql(u8, &hex, try string(entry.value_ptr.*))) return error.CheckpointChecksum;
            }
        }
    }

    pub fn floats(f: Files, name: []const u8) !struct { data: []f32, array: core.npy.Array } {
        const array = try f.read(name);
        if (!std.mem.eql(u8, array.descr, "<f4")) return error.BadFixture;
        const data = try f.a.alloc(f32, array.count());
        @memcpy(std.mem.sliceAsBytes(data), array.data);
        for (data) |v| if (!std.math.isFinite(v)) return error.NonfiniteFixture;
        return .{ .data = data, .array = array };
    }

    pub fn write(f: Files, name: []const u8, shape: []const usize, data: []const f32) !void {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (shape.len == 0 or count != data.len) return error.BadCaptureShape;
        for (data) |v| if (!std.math.isFinite(v)) return error.NonfiniteCapture;
        var header: std.Io.Writer.Allocating = .init(f.a);
        try header.writer.writeAll("{'descr': '<f4', 'fortran_order': False, 'shape': (");
        for (shape) |dim| try header.writer.print("{d}, ", .{dim});
        try header.writer.writeAll("), }");
        const dict = header.written();
        const pad = (64 - (10 + dict.len + 1) % 64) % 64;
        const head = 10 + dict.len + pad + 1;
        const bytes = try f.a.alloc(u8, head + data.len * 4);
        @memcpy(bytes[0..8], "\x93NUMPY\x01\x00");
        std.mem.writeInt(u16, bytes[8..10], @intCast(head - 10), .little);
        @memcpy(bytes[10..][0..dict.len], dict);
        @memset(bytes[10 + dict.len .. head - 1], ' ');
        bytes[head - 1] = '\n';
        @memcpy(bytes[head..], std.mem.sliceAsBytes(data));
        try f.output.writeFile(f.io, .{ .sub_path = try std.fmt.allocPrint(f.a, "{s}.npy", .{name}), .data = bytes });
    }

    pub fn compare(f: Files, name: []const u8, got: []const f32) !Stats {
        const want = (try f.floats(name)).data;
        const s = try compareValues(want, got);
        std.debug.print("{s}: max_abs={e:.6} rms={e:.6} cosine={d:.9} different={d}/{d}\n", .{ name, s.max_abs, s.rms, s.cosine, s.different, got.len });
        return s;
    }
};

pub const Stats = struct { max_abs: f64, rms: f64, cosine: f64, different: usize, values: usize };

pub fn compareValues(want: []const f32, got: []const f32) !Stats {
    if (want.len == 0 or want.len != got.len) return error.BadCaptureShape;
    var s: Stats = .{ .max_abs = 0, .rms = 0, .cosine = 0, .different = 0, .values = got.len };
    var diff: f64 = 0;
    var dot: f64 = 0;
    var ww: f64 = 0;
    var gg: f64 = 0;
    for (want, got) |w, g| {
        if (!std.math.isFinite(w) or !std.math.isFinite(g)) return error.NonfiniteCapture;
        const delta = @as(f64, w) - g;
        s.max_abs = @max(s.max_abs, @abs(delta));
        s.different += @intFromBool(w != g);
        diff += delta * delta;
        dot += @as(f64, w) * g;
        ww += @as(f64, w) * w;
        gg += @as(f64, g) * g;
    }
    s.rms = @sqrt(diff / @as(f64, @floatFromInt(got.len)));
    s.cosine = if (ww == 0 or gg == 0) @intFromBool(ww == gg) else dot / @sqrt(ww * gg);
    return s;
}

pub fn field(v: std.json.Value, key: []const u8) !std.json.Value {
    if (v != .object) return error.BadFixture;
    return v.object.get(key) orelse error.BadFixture;
}

pub fn list(v: std.json.Value) ![]std.json.Value {
    if (v != .array) return error.BadFixture;
    return v.array.items;
}

pub fn integer(v: std.json.Value) !usize {
    if (v != .integer or v.integer < 0) return error.BadFixture;
    return std.math.cast(usize, v.integer) orelse error.BadFixture;
}

pub fn string(v: std.json.Value) ![]const u8 {
    if (v != .string) return error.BadFixture;
    return v.string;
}

test "capture statistics distinguish matching zero vectors and finite differences" {
    const same = try compareValues(&.{ 0, 0 }, &.{ 0, 0 });
    try std.testing.expectEqual(@as(f64, 1), same.cosine);
    const different = try compareValues(&.{ 1, 2 }, &.{ 1, 3 });
    try std.testing.expectEqual(@as(f64, 1), different.max_abs);
    try std.testing.expectEqual(@as(usize, 1), different.different);
    try std.testing.expectError(error.NonfiniteCapture, compareValues(&.{1}, &.{std.math.inf(f32)}));
}

pub const Patches = struct { data: []f32, gh: u32, gw: u32 };

pub fn patchify(a: Allocator, shape: []const usize, pixels: []const f32) !Patches {
    const chw = if (shape.len == 4 and shape[0] == 1) shape[1..] else shape;
    if (chw.len != 3 or chw[0] != 3 or chw[1] == 0 or chw[2] == 0 or chw[1] % 14 != 0 or chw[2] % 14 != 0 or
        chw[1] > 1540 or chw[2] > 1540 or pixels.len != 3 * chw[1] * chw[2]) return error.BadFixturePixels;
    const gh = chw[1] / 14;
    const gw = chw[2] / 14;
    const out = try a.alloc(f32, pixels.len);
    for (0..gh) |py| for (0..gw) |px| for (0..3) |ch| for (0..14) |y| for (0..14) |x| {
        out[(py * gw + px) * 588 + (ch * 14 + y) * 14 + x] = pixels[(ch * chw[1] + py * 14 + y) * chw[2] + px * 14 + x];
    };
    return .{ .data = out, .gh = @intCast(gh), .gw = @intCast(gw) };
}

test "patch conversion preserves channel, grid and local raster order" {
    const a = std.testing.allocator;
    const pixels = try a.alloc(f32, 3 * 28 * 42);
    defer a.free(pixels);
    for (pixels, 0..) |*p, i| p.* = @floatFromInt(i);
    const result = try patchify(a, &.{ 1, 3, 28, 42 }, pixels);
    defer a.free(result.data);
    try std.testing.expectEqual(@as(u32, 2), result.gh);
    try std.testing.expectEqual(@as(u32, 3), result.gw);
    try std.testing.expectEqual(pixels[0], result.data[0]);
    try std.testing.expectEqual(pixels[14], result.data[588]);
    try std.testing.expectEqual(pixels[28 * 42], result.data[196]);
    try std.testing.expectEqual(pixels[14 * 42], result.data[3 * 588]);
    try std.testing.expectError(error.BadFixturePixels, patchify(a, &.{ 1, 3, 29, 42 }, pixels));
}

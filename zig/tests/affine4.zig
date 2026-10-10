//! CPU preparation must match actual MLX words and BF16 metadata on generated groups.
const std = @import("std");
const cpu = @import("affine4");
const Pin = struct { name: []const u8, bytes: usize, sha256: []const u8 };
const Manifest = struct { schema: []const u8, n: usize, k: usize, groups: usize, files: struct { weights: Pin, words: Pin, scales: Pin, biases: Pin } };
fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, pin: Pin) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, pin.name });
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(8 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    if (data.len != pin.bytes or !std.mem.eql(u8, pin.sha256, &std.fmt.bytesToHex(digest, .lower))) return error.BadFixture;
    return data;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 or !std.mem.eql(u8, args[1], "--fixtures")) return error.BadOptions;
    const path = try std.fs.path.join(a, &.{ args[2], "manifest.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 20));
    const m = (try std.json.parseFromSlice(Manifest, a, text, .{})).value;
    if (!std.mem.eql(u8, m.schema, "tf-affine4-v1") or m.groups != m.n * m.k / 64) return error.BadFixture;
    const raw = try read(a, init.io, args[2], m.files.weights);
    const weights = try a.alloc(u16, raw.len / 2);
    for (weights, 0..) |*v, i| v.* = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
    var result = try cpu.prepare(a, weights, m.n, m.k);
    defer result.deinit();
    var errors: usize = 0;
    inline for (.{ .{ "words", std.mem.sliceAsBytes(result.words), m.files.words }, .{ "scales", std.mem.sliceAsBytes(result.scales), m.files.scales }, .{ "biases", std.mem.sliceAsBytes(result.biases), m.files.biases } }) |field| {
        const expected = try read(a, init.io, args[2], field[2]);
        if (expected.len != field[1].len) return error.BadFixture;
        var unequal: usize = 0;
        for (expected, field[1]) |old, new| if (old != new) {
            unequal += 1;
        };
        errors += unequal;
        std.debug.print("affine4 {s}: {d} unequal bytes of {d}\n", .{ field[0], unequal, expected.len });
    }
    std.debug.print("affine4: {d} groups, {d} unequal bytes\n", .{ m.groups, errors });
    if (errors != 0) return error.QuantizationBytesDiffer;
}

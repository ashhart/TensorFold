//! Read-only SHA-256 fingerprints of full logits in little-endian F32 representation.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub fn logits(values: []const f32) [32]u8 {
    var hash = Sha256.init(.{});
    if (@import("builtin").target.cpu.arch.endian() == .little) {
        hash.update(std.mem.sliceAsBytes(values));
    } else {
        for (values) |value| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
            hash.update(&bytes);
        }
    }
    return hash.finalResult();
}

pub fn sequence(digests: []const [32]u8) [32]u8 {
    var hash = Sha256.init(.{});
    for (digests) |digest| hash.update(&digest);
    return hash.finalResult();
}

test "logits hash little-endian raw F32 bits, including signed zero, without modifying them" {
    var values = [_]f32{ 1, -0.0, -2.5 };
    const before = values;
    const bytes = [_]u8{ 0, 0, 128, 63, 0, 0, 0, 128, 0, 0, 32, 192 };
    var expected: [32]u8 = undefined;
    Sha256.hash(&bytes, &expected, .{});
    try std.testing.expectEqual(expected, logits(&values));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&values));
    const original = logits(&values);
    values[1] = @bitCast(@as(u32, @bitCast(values[1])) ^ 1);
    try std.testing.expect(!std.mem.eql(u8, &original, &logits(&values)));
    const steps = [_][32]u8{ original, logits(&values) };
    var joined: [64]u8 = undefined;
    @memcpy(joined[0..32], &steps[0]);
    @memcpy(joined[32..], &steps[1]);
    Sha256.hash(&joined, &expected, .{});
    try std.testing.expectEqual(expected, sequence(&steps));
}

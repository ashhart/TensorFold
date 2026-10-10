//! Frozen outputs of the removed Flash Next kernel, captured before migration. Never regenerate in a test.
const std = @import("std");
const core = @import("core_lane");
const Case = struct { sk: usize, pf: usize, output: core.Output, cut: bool, bytes: []const u8, sha256: []const u8 };
pub const cases = [_]Case{
    .{ .sk = 1, .pf = 1, .output = .bf16, .cut = false, .bytes = @embedFile("sk1-pf1-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 1, .pf = 1, .output = .f32, .cut = false, .bytes = @embedFile("sk1-pf1-f32-full.bin"), .sha256 = "3be54c07d002b2b1f1834d5f447f8368f857bcf864c46da3bd2eb64abde5f7db" },
    .{ .sk = 1, .pf = 2, .output = .bf16, .cut = false, .bytes = @embedFile("sk1-pf2-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 1, .pf = 2, .output = .f32, .cut = false, .bytes = @embedFile("sk1-pf2-f32-full.bin"), .sha256 = "3be54c07d002b2b1f1834d5f447f8368f857bcf864c46da3bd2eb64abde5f7db" },
    .{ .sk = 2, .pf = 1, .output = .bf16, .cut = false, .bytes = @embedFile("sk2-pf1-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 2, .pf = 1, .output = .f32, .cut = false, .bytes = @embedFile("sk2-pf1-f32-full.bin"), .sha256 = "4d4477dd12dc353a4a387144747b0a6141aa4b2e07f4cd36e51d9056b92eef51" },
    .{ .sk = 2, .pf = 2, .output = .bf16, .cut = false, .bytes = @embedFile("sk2-pf2-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 2, .pf = 2, .output = .f32, .cut = false, .bytes = @embedFile("sk2-pf2-f32-full.bin"), .sha256 = "4d4477dd12dc353a4a387144747b0a6141aa4b2e07f4cd36e51d9056b92eef51" },
    .{ .sk = 4, .pf = 1, .output = .bf16, .cut = false, .bytes = @embedFile("sk4-pf1-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 4, .pf = 1, .output = .f32, .cut = false, .bytes = @embedFile("sk4-pf1-f32-full.bin"), .sha256 = "638d6867fe44699d8742aa9661a5570f022c90d5b7c5abd5644d2ef42caab82c" },
    .{ .sk = 4, .pf = 2, .output = .bf16, .cut = false, .bytes = @embedFile("sk4-pf2-bf16-full.bin"), .sha256 = "1c374cced29b1e412d463342d56af6b0e74ed3d5fb2f33d6cabef5c03058768d" },
    .{ .sk = 4, .pf = 2, .output = .f32, .cut = true, .bytes = @embedFile("sk4-pf2-f32-cut.bin"), .sha256 = "55be6060067e82f7d89147196b4a3e221ef7ebdd81b645e9fad20176780794f3" },
    .{ .sk = 4, .pf = 2, .output = .f32, .cut = false, .bytes = @embedFile("sk4-pf2-f32-full.bin"), .sha256 = "638d6867fe44699d8742aa9661a5570f022c90d5b7c5abd5644d2ef42caab82c" },
};

pub fn bytes(l: core.Layout) ![]const u8 {
    if (l.format.bits != 6 or l.format.group != 32 or l.k != 224) return error.MissingFrozenCase;
    const cut = l.groups != null;
    if (cut) {
        if (l.n != 96 or !std.mem.eql(usize, &l.groupCut(), &.{ 1, 3 }) or
            !std.mem.eql([2]usize, l.ranges, &.{.{ 1, 1 }})) return error.MissingFrozenCase;
    } else if (l.n != 64 or l.ranges.len != 0) return error.MissingFrozenCase;
    for (cases) |c| if (c.sk == l.sk and c.pf == l.pf and c.output == l.output and c.cut == cut) return c.bytes;
    return error.MissingFrozenCase;
}

pub fn compareWindow(l: core.Layout, rows: usize, got: []const u8) !void {
    if (rows == 0 or rows > 16) return error.MissingFrozenCase;
    const all = try bytes(l);
    const stride = l.n * @as(usize, if (l.output == .f32) 4 else 2);
    const offset = rows * (rows - 1) / 2 * stride;
    const expected = all[offset..][0 .. rows * stride];
    if (!std.mem.eql(u8, got, expected)) return error.FlashNextGoldenMismatch;
}

pub fn verify() !void {
    for (cases) |c| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(c.bytes, &digest, .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), c.sha256)) return error.FrozenFixtureChecksum;
        const n: usize = if (c.cut) 96 else 64;
        const elem: usize = if (c.output == .f32) 4 else 2;
        if (c.bytes.len != 136 * n * elem) return error.FrozenFixtureLength;
    }
}

test "frozen fn_lane outputs have recorded hashes and 208 windows" {
    try verify();
    try std.testing.expectEqual(@as(usize, 13), cases.len);
}

test "frozen window comparison rejects one changed output byte" {
    const l = core.Layout{ .n = 64, .k = 224, .format = .{ .bits = 6, .group = 32 }, .output = .f32 };
    const expected = (try bytes(l))[0 .. 64 * 4];
    try compareWindow(l, 1, expected);
    const changed = try std.testing.allocator.dupe(u8, expected);
    defer std.testing.allocator.free(changed);
    changed[0] ^= 1;
    try std.testing.expectError(error.FlashNextGoldenMismatch, compareWindow(l, 1, changed));
    try std.testing.expectError(error.FlashNextGoldenMismatch, compareWindow(l, 1, expected[1..]));
}

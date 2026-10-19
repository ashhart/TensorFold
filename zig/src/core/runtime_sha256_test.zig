//! Published SHA256 answers and independent standard-library comparisons cover every dispatch and streaming boundary.
const std = @import("std");
const sha = @import("runtime_sha256.zig");
const t = std.testing;

test "runtime SHA256: known answers use scalar and the available hardware backend" {
    const known = [_]struct { input: []const u8, hex: []const u8 }{
        .{ .input = "", .hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        .{ .input = "abc", .hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        .{ .input = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", .hex = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    };
    for ([_]sha.Backend{ .software, sha.available() }) |backend| {
        for (known) |answer| {
            errdefer std.debug.print("SHA backend={s}\n", .{@tagName(backend)});
            var h = sha.Hash.init(backend);
            h.update(answer.input);
            try t.expectEqualStrings(answer.hex, &std.fmt.bytesToHex(h.finalResult(), .lower));
        }
        var h = sha.Hash.init(backend);
        const thousand: [1000]u8 = @splat('a');
        for (0..1000) |_| h.update(&thousand);
        try t.expectEqualStrings("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", &std.fmt.bytesToHex(h.finalResult(), .lower));
    }
    try t.expectEqual(sha.Backend.software, try sha.select(.software));
    try t.expectEqual(sha.available(), try sha.select(.auto));
    if (sha.available() == .software) {
        try t.expectError(error.Sha256HardwareUnavailable, sha.select(.hardware));
    } else try t.expectEqual(sha.available(), try sha.select(.hardware));
}

test "runtime SHA256: the checked baseline executable does not require optional SHA CPU features" {
    const builtin = @import("builtin");
    if (builtin.cpu.arch == .aarch64) try t.expect(!builtin.cpu.has(.aarch64, .sha2));
    if (builtin.cpu.arch == .x86_64) try t.expect(!builtin.cpu.has(.x86, .sha));
}

test "runtime SHA256: unaligned chunk and padding boundaries equal the independent standard hash" {
    var bytes: [1026]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i *% 133 + 17);
    for (0..1025) |n| {
        var want: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes[1..][0..n], &want, .{});
        for ([_]sha.Backend{ .software, sha.available() }) |backend| {
            for ([_]usize{ 1, 7, 55, 63, 64, 65, 127, 257, 1025 }) |chunk| {
                var h = sha.Hash.init(backend);
                h.update("");
                var at: usize = 0;
                while (at < n) {
                    const end = @min(n, at + chunk);
                    h.update(bytes[1 + at .. 1 + end]);
                    at = end;
                }
                try t.expectEqualSlices(u8, &want, &h.finalResult());
            }
        }
    }
}

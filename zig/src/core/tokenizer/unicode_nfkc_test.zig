//! NFKC/NFKD against Rust tokenizers' Unicode 9 goldens, curated by tools/zig/nfkc_fixtures.py; NFC/NFD unchanged.
const std = @import("std");
const unicode = @import("unicode.zig");
const data = @import("unicode_data.zig");

test "Unicode compatibility forms match the HF library on sampled compatibility mappings and composed boundaries" {
    const Case = struct { input: []const u8, nfc: []const u8, nfd: []const u8, nfkc: []const u8, nfkd: []const u8 };
    const Manifest = struct { tokenizers_version: []const u8, normalization_version: []const u8, cases: []Case };
    const a = std.testing.allocator;
    const doc = try std.json.parseFromSlice(Manifest, a, @embedFile("fixtures/nfkc.json"), .{});
    defer doc.deinit();
    try std.testing.expectEqualStrings("9.0", doc.value.normalization_version);
    for (doc.value.cases) |case| {
        inline for (.{ .nfc, .nfd, .nfkc, .nfkd }) |form| {
            const got = try unicode.normalize(a, case.input, form);
            defer a.free(got);
            try std.testing.expectEqualStrings(@field(case, @tagName(form)), got);
        }
    }
}

test "NFKC and NFKD change and are idempotent for every compatibility-mapped code point" {
    const a = std.testing.allocator;
    for (data.compatibility) |entry| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(entry[0], &buf);
        inline for (.{ .nfkc, .nfkd }) |form| {
            const once = try unicode.normalize(a, buf[0..n], form);
            defer a.free(once);
            try std.testing.expect(!std.mem.eql(u8, once, buf[0..n])); // a dropped mapping would pass as idempotent
            const twice = try unicode.normalize(a, once, form);
            defer a.free(twice);
            try std.testing.expectEqualStrings(once, twice);
        }
    }
}

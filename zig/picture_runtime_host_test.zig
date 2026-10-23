//! The actual codec loaders qualify diagnostics without ever entering a decoder.
const std = @import("std");
const runtime = @import("src/core/picture/runtime.zig");
const libs = @import("sys").optional_libraries;
const fixture = @import("codec_fixture");
const env = @import("sys").env;

test "codec diagnostics: loadable, missing, wrong and incomplete libraries reflect actual decoder admission" {
    const a = std.testing.allocator;
    const path = try a.dupeSentinel(u8, fixture.library, 0);
    defer a.free(path);
    for ([_][:0]const u8{ "TENSORFOLD_LIBTURBOJPEG", "TENSORFOLD_LIBWEBP" }) |name| {
        const rc = env.setenv(name, path, 1);
        try std.testing.expectEqual(@as(c_int, 0), rc);
    }
    const rows = try runtime.probe(a, std.testing.io);
    defer a.free(rows);
    const warnings = runtime.warnings(std.testing.io, a);
    for ([_]libs.Id{ .jpeg, .webp }, 0..) |id, i| {
        const row = rows[@backingInt(id)];
        try std.testing.expectEqual(!std.mem.eql(u8, fixture.kind, "missing"), row.found);
        const ready = std.mem.eql(u8, fixture.kind, "ready");
        try std.testing.expectEqual(ready, warnings[i] == null);
        try std.testing.expectEqual(@as(?bool, ready), row.usable);
        if (ready) {
            try std.testing.expect(row.problem == null and warnings[i] == null);
        } else {
            try std.testing.expectEqualStrings(row.problem.?, warnings[i].?);
            try std.testing.expect(std.mem.indexOf(u8, row.problem.?, if (id == .jpeg) "3.1.4.1" else "1.6.0") != null);
            if (std.mem.eql(u8, fixture.kind, "wrong"))
                try std.testing.expect(std.mem.indexOf(u8, row.problem.?, "another release") != null);
        }
    }
    try std.testing.expect(rows[@backingInt(libs.Id.nccl)].usable == null);
}

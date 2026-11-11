//! Runs every actual synchronous readback expression on a recording host driver.
const std = @import("std");
const cases = @import("cuda_readback_sites.zig");

pub fn tests(b: *std.Build, cuda: *std.Build.Module, all: *std.Build.Step) void {
    var code: std.ArrayList(u8) = .empty;
    code.appendSlice(b.allocator,
        \\const std = @import("std");
        \\const cuda = @import("cuda");
        \\const Fake = @import("fake").Fake;
        \\
    ) catch @panic("out of memory");
    for (cases.sites) |site| {
        for ([_]bool{ true, false }) |integrated| {
            var expected: usize = 0;
            for (cases.sites) |other| if (std.mem.eql(u8, other.path, site.path)) {
                expected += 1;
            };
            const expression = readback(site.source, site.ordinal, expected);
            const body = b.fmt(
                \\test "staged copy: {s}:{d} {s}" {{
                \\    const integrated = {};
                \\    {{
                \\        var rig = Fake.init(integrated);
                \\        defer rig.deinit();
                \\        {s}
                \\        const read_at = {s};
                \\        const read_bytes = {s};
                \\        @memset(read_bytes, 0xaa);
                \\        try {s};
                \\        try rig.check(read_at, read_bytes);
                \\    }}
                \\}}
                \\
            , .{ site.path, site.ordinal, if (integrated) "integrated" else "discrete", integrated, site.setup, site.at, site.out, expression });
            code.appendSlice(b.allocator, body) catch @panic("out of memory");
        }
    }
    const fake = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/staged_copy_fake.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const root = b.createModule(.{ .root_source_file = b.addWriteFiles().add("staged_copy_sites.zig", code.items), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "fake", .module = fake } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    const step = b.step("test-staged-copies-host", "Every synchronous readback site on integrated and discrete fake drivers");
    step.dependOn(&run.step);
    const lifetime = b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{ "an unowned device span", "an empty unowned readback", "closing a driver drops" } }));
    step.dependOn(&lifetime.step);
    const tail = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_mtp_verify_tail.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const tail_test = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/mtp_tail_readback_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "tail", .module = tail } } }) }));
    b.step("test-mtp-tail-readback-host", "Actual MTP tail readback waits and preserves integrated/discrete output bytes").dependOn(&tail_test.step);
    step.dependOn(&tail_test.step);
    all.dependOn(&tail_test.step);
    all.dependOn(&run.step);
}

fn readback(source: []const u8, ordinal: usize, expected: usize) []const u8 {
    var chosen: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, source, '\n');
    var count: usize = 0;
    while (lines.next()) |raw| {
        if (std.mem.indexOf(u8, raw, "cuMemcpyDtoH_v2(") == null and
            std.mem.indexOf(u8, raw, "cuda.memory.download(") == null) continue;
        count += 1;
        if (count != ordinal) continue;
        var line = std.mem.trim(u8, raw, " \t;");
        if (std.mem.indexOf(u8, line, "try ")) |at| line = line[at + 4 ..];
        if (std.mem.indexOf(u8, line, " catch ")) |at| line = line[0..at];
        chosen = line;
    }
    if (count != expected) @panic("readback source changed without updating its cases");
    return chosen orelse @panic("readback case no longer has a source expression");
}

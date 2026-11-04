//! The control vector kernel's host checks: the float64 reference, the gates and the launch's limits.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-steer", "Control vector contract: the float64 reference, the gates and the limits");
    all.dependOn(step);
    // A module reaches only under its root: the contract's files sit under zig/src/core, so the copies keep that shape.
    const files = b.addWriteFiles();
    const sources = [_][2][]const u8{
        .{ "zig/src/core/contract/steer.zig", "core/contract/steer.zig" },
        .{ "zig/src/core/contract/steer_records.zig", "core/contract/steer_records.zig" },
        .{ "zig/src/core/contract/contract.zig", "core/contract/contract.zig" },
        .{ "zig/src/core/contract/rows.zig", "core/contract/rows.zig" },
        .{ "zig/src/core/contract/limits.zig", "core/contract/limits.zig" },
        .{ "zig/src/core/bf16_round.zig", "core/bf16_round.zig" },
        .{ "zig/src/core/norms_ref.zig", "core/norms_ref.zig" },
    };
    for (sources) |pair| _ = files.addCopyFile(b.path(pair[0]), pair[1]);
    const root = files.add("core/steer_root.zig",
        \\test {
        \\    _ = @import("contract/steer.zig");
        \\    _ = @import("contract/steer_records.zig");
        \\}
    );
    const module = b.createModule(.{
        .root_source_file = root,
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}

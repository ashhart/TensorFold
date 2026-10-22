//! The compiler's SASS dump must contain every registered x3ld entry on every requested SM.
const std = @import("std");

pub fn check(b: *std.Build, image: std.Build.LazyPath, sms: []const u8, install: *std.Build.Step, cuobjdump: ?[]const u8) void {
    const step = b.step("check-dsv41-symbols", "Check the compiled x3ld image's registered entries for every SM");
    const tool = cuobjdump orelse {
        step.dependOn(&b.addFail("check-dsv41-symbols needs -Dcuobjdump=PATH and -Dnvcc=PATH or -Dfatbins=DIR").step);
        return;
    };
    const dump = b.addSystemCommand(&.{ tool, "--dump-sass" });
    dump.addFileArg(image);
    const run = b.addSystemCommand(&.{"python3"});
    run.addFileArg(b.path("tools/zig/check_cuda_symbols.py"));
    run.addArgs(&.{ "--sms", sms, "--registry" });
    run.addFileArg(b.path("zig/src/core/contract/dsv41_pending_1.zig"));
    run.addFileArg(dump.captureStdOut(.{}));
    step.dependOn(&run.step);
    install.dependOn(step);
}

//! A device draft/plain token-ID check and its GPU-free comparison/setup tests.
const std = @import("std");
pub fn steps(b: *std.Build) void {
    const host = b.addSystemCommand(&.{ "python3", "-B", "-m", "unittest", "discover", "-s", "tools/tests", "-p", "test_draft_parity.py" });
    host.setCwd(b.path("."));
    host.has_side_effects = true;
    for ([_][]const u8{ "tools/draft_parity.py", "tools/draft_parity_native.py", "tools/tests/test_draft_parity.py" }) |path| host.addFileInput(b.path(path));
    b.step("test-draft-parity-host", "Exact draft/plain comparison, accepted-draft refusals and owned-server setup without a GPU").dependOn(&host.step);
    const device = b.addSystemCommand(&.{ "python3", "-B" });
    device.addFileArg(b.path("tools/draft_parity.py"));
    device.addFileInput(b.path("tools/draft_parity_native.py"));
    device.setCwd(b.path("."));
    device.has_side_effects = true;
    device.addPassthruArgs();
    b.step("device-draft-parity", "Run one native binary against a local checkpoint with default drafts and --no-drafts; arguments after --").dependOn(&device.step);
}

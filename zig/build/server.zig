//! The server's shared JSON values, available to server and family modules.
const std = @import("std");

/// The server's JSON values as a module (zig/src/server/json.zig), so a family's serving code shares them.
pub fn json(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/server/json.zig"),
        .target = target,
        .optimize = optimize,
    });
}

/// Attach a server's shared JSON module (tests run apart), and Living Weights' families where the build gave none.
pub fn addTo(
    b: *std.Build,
    m: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const j = json(b, target, optimize);
    m.addImport("json", j);
    if (m.import_table.get("engine_api")) |api| if (api.import_table.get("lanes")) |lanes| m.addImport("lanes", lanes);
    // a build whose engines learn hands the server their own living module; the rest read only its families
    if (m.import_table.get("living") == null) m.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return j;
}

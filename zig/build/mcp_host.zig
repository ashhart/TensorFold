//! test-mcp-host: `tensorfold mcp`'s wire, tools and serve children against a temporary cache, fakes and stand-ins.
const std = @import("std");

/// The tests' module: the server tests' own imports over the host's engines, as tensorfold-native has them.
pub fn add(b: *std.Build, server_tests: *std.Build.Module, engines: *std.Build.Module, checkpoint_cli: *std.Build.Module, step: *std.Build.Step) void {
    if (b.graph.host.result.os.tag == .windows) return; // the serve children and the fakes are POSIX processes and sockets
    const host = b.graph.host;
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/server/mcp_tests.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    for ([_][]const u8{ "engine_api", "tokenizer", "template", "living", "core" }) |name| m.addImport(name, server_tests.import_table.get(name).?);
    m.addImport("native_engines", engines); // so the server's drama prompt is the engines' own, as in the binary
    m.addImport("checkpoint_cli", checkpoint_cli);
    _ = @import("server.zig").addTo(b, m, host, .Debug);
    @import("server.zig").testImports(b, m, host, .Debug);
    // only these tests: the server files they import bring their own, which test-server-cpu already runs
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"mcp "} }));
    run.has_side_effects = true; // child processes and loopback sockets: a cached pass would say nothing about them
    b.step("test-mcp-host", "The MCP server's framing, lifecycle and tools against a temporary cache, a fake hub and a stand-in serve child").dependOn(&run.step);
    step.dependOn(&run.step);
}

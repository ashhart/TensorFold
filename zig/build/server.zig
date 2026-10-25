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
    if (!m.import_table.contains("living")) m.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const j = json(b, target, optimize);
    m.addImport("json", j);
    m.addImport("dsv41_chat", b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve/template.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "json", .module = j }},
    }));
    const family = if (m.import_table.get("native_engines")) |engines| engines.import_table.get("drama3") else null;
    if (family) |drama3| {
        const source = b.addWriteFiles().add("drama_prompt.zig",
            \\pub const Reference = @import("drama3").prompt.Reference;
            \\pub const Built = @import("drama3").prompt.Built;
            \\pub const build = @import("drama3").prompt.build;
        );
        m.addImport("drama_prompt", b.createModule(.{
            .root_source_file = source,
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "drama3", .module = drama3 }},
        }));
    } else {
        m.addImport("drama_prompt", b.createModule(.{
            .root_source_file = b.path("zig/src/families/drama3/prompt.zig"),
            .target = target,
            .optimize = optimize,
        }));
    }
    if (m.import_table.get("engine_api")) |api| if (api.import_table.get("lanes")) |lanes| m.addImport("lanes", lanes);
    // a build whose engines learn hands the server their own living module; the rest read only its families
    if (m.import_table.get("living") == null) m.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return j;
}

/// Tool parsing and actual chat visibility checks share the aggregate's real modules.
pub fn toolHostTests(b: *std.Build, parser: *std.Build.Module, server: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = parser }));
    all.dependOn(&run.step);
    b.step("test-tool-parse-host", "Tool parser visibility, schemas and single-call limits without devices").dependOn(&run.step);
    const http = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"tool visibility HTTP:"} }));
    http.has_side_effects = true;
    b.step("test-tool-visibility-http-host", "Rejected tool envelopes survive actual chat JSON and SSE replies").dependOn(&http.step);
    const deepseek = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"DeepSeek chat:"} }));
    deepseek.has_side_effects = true;
    all.dependOn(&deepseek.step);
    b.step("test-server-deepseek-chat-host", "Native DeepSeek prompt goldens, provenance and missing-template HTTP refusal").dependOn(&deepseek.step);
}

/// The CPU-only fake server and the engine-less CLI into zig-out/server/ (zig build server-cpu), for build.sh.
pub fn cpuArtefacts(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const none = b.createModule(.{ .root_source_file = b.path("zig/src/native/none.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "engine_api", .module = api }} });

    // the fake server: the server module with no native engines, so addTo takes the plain drama prompt
    const server_mod = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "engine_api", .module = api },
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "template", .module = template },
    } });
    _ = addTo(b, server_mod, target, optimize);
    const fake = b.addExecutable(.{ .name = "fake_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/fake_serve.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "server", .module = server_mod }, .{ .name = "engine_api", .module = api } },
    }) });
    const server_install = b.addInstallArtifact(fake, .{ .dest_dir = .{ .override = .{ .custom = "server" } } });

    // the engine-less CLI: main.zig over the none engines, the checkpoint subcommands' own module included
    const cli = @import("cuda.zig").checkpointCli(b, target, optimize, null, none, core);
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "engine_api", .module = api },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "native_engines", .module = none },
            .{ .name = "checkpoint_cli", .module = cli },
        },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = addTo(b, exe.root_module, target, optimize);
    const cli_install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "server" } } });

    const step = b.step("server-cpu", "The CPU-only fake server and engine-less CLI into zig-out/server");
    step.dependOn(&server_install.step);
    step.dependOn(&cli_install.step);
}

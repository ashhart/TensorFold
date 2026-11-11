//! The server's shared JSON values, available to server and family modules.
const std = @import("std");

/// The web launcher uses fake PATH tools and no model, network or device.
pub fn webHostTests(b: *std.Build, all: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/web_cli.zig"),
        .target = b.graph.host,
        .link_libc = true,
    }) });
    const run = b.addRunArtifact(tests);
    run.has_side_effects = true;
    b.step("test-web-cli-host", "Web launcher consent, version pins and fake PATH tools without network").dependOn(&run.step);
    all.dependOn(&run.step);
}

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

/// Isolated server tests share the pure family limits; native servers keep the family's existing file owner.
pub fn testImports(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    m.addImport("nemotron_limits", b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/limits.zig"), .target = target, .optimize = optimize }));
}

/// Tool parsing and actual chat visibility checks share the aggregate's real modules.
pub fn toolHostTests(b: *std.Build, parser: *std.Build.Module, server: *std.Build.Module, all: *std.Build.Step) void {
    testImports(b, parser, b.graph.host, .debug);
    testImports(b, server, b.graph.host, .debug);
    b.step("test-thinking-tail-host", "Finished calls before a think close, whole and streamed without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"thinking tail:"} })).step);
    const environment = b.step("test-environment-report-host", "Named environment discovery, new registry rows and startup reporting");
    environment.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: environment", "stable profile:", "profile line:" } })).step);
    const logprobs = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"logprobs_test"} }));
    b.step("test-logprob-offers-host", "Logprob request capability refusals before engine submission").dependOn(&logprobs.step);
    const limits = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"limit pins:"} }));
    b.step("test-limit-pins-host", "Documented request and flag limits pinned at their bounds").dependOn(&limits.step);
    const deps = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"flag deps:"} }));
    b.step("test-flag-deps-host", "Flag dependencies: needs, refusals and build gates at the CLI").dependOn(&deps.step);
    const models = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"model discovery"} }));
    b.step("test-model-discovery-host", "Loaded context and reasoning offers for each model discovery entry").dependOn(&models.step);
    b.step("test-decision-context-host", "Decision context classification through actual HTTP errors").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: decision context", "prompt past the context", "System One" } })).step);
    b.step("test-capability-env-host", "Environment discovery subset and complete profile registry").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: environment", "stable profile:", "env defaults:" } })).step);
    b.step("test-model-idle-host", "Model discovery and idle reload snapshots without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: model discovery waits", "idle", "model discovery" } })).step);
    b.step("test-effort-capabilities-host", "CLI effort choices and parsed defaults against request fields").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: effort", "parse and capabilities share the table" } })).step);
    b.step("test-effort-discovery-host", "Effective reasoning defaults and template effort discovery").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "capability audit: model discovery", "model discovery", "effort" } })).step);
    const phases = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "queue phase", "render exposes", "render counts", "metrics name" } }));
    b.step("test-phase-metrics-host", "Off-by-default queue phase rendering and existing metric families").dependOn(&phases.step);
    const offers = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "model discovery", "engine capability reader:" } }));
    b.step("test-model-capabilities-host", "Per-model offers agree with the flags used for request refusals").dependOn(&offers.step);
    const drafts = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"draft sources:"} }));
    b.step("test-draft-line-host", "Serving line draft provenance without devices").dependOn(&drafts.step);
    all.dependOn(&drafts.step);
    const scopes = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"auth scopes"} }));
    scopes.has_side_effects = true;
    b.step("test-key-scopes-host", "Key-file scopes: exact route grants, stateless Responses and refusals over sockets").dependOn(&scopes.step);
    const witnessed = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"execution witness"} }));
    witnessed.has_side_effects = true;
    b.step("test-execution-witness-host", "The opt-in execution witness: parts, scopes, reload epochs and refusals over sockets").dependOn(&witnessed.step);
    const profile_line = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"profile line:"} }));
    profile_line.has_side_effects = true;
    b.step("test-profile-line-host", "Profile start-up line scoping against the tree's real backend names").dependOn(&profile_line.step);
    const admitted = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"admission"} }));
    admitted.has_side_effects = true;
    b.step("test-admission-host", "Opt-in in-flight caps and per-key rates: limits, 429s over sockets, metrics").dependOn(&admitted.step);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = parser }));
    all.dependOn(&run.step);
    b.step("test-tool-parse-host", "Tool parser visibility, schemas and single-call limits without devices").dependOn(&run.step);
    const http = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"tool visibility HTTP:"} }));
    http.has_side_effects = true;
    b.step("test-tool-visibility-http-host", "Rejected tool envelopes survive actual chat JSON and SSE replies").dependOn(&http.step);
    const windows = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{ "Nemotron window:", "model discovery", "thinking reserve" } }));
    b.step("test-nemotron-window-host", "Nemotron advertised capacity and unchanged chat admission without devices").dependOn(&windows.step);
    b.step("test-thinking-request-host", "Client refusals and internal note and lesson requests without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"thinking request:"} })).step);
    b.step("test-thinking-benchmark-host", "Benchmark thinking capability and request bodies without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"thinking benchmark:"} })).step);
    const deepseek = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"DeepSeek chat:"} }));
    deepseek.has_side_effects = true;
    all.dependOn(&deepseek.step);
    b.step("test-server-deepseek-chat-host", "Native DeepSeek prompt goldens, provenance and missing-template HTTP refusal").dependOn(&deepseek.step);
    const quoted = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"quoted markers:"} }));
    quoted.has_side_effects = true; // reads TF_TOKENIZER_DIRS on every run
    b.step("test-quoted-markers-host", "Quoted added-token spellings through the chat route with TF_TOKENIZER_DIRS' tokenizers").dependOn(&quoted.step);
}

/// The server tree's module rooted at `root` over no native engines: the fake server's, or a test among its sources.
pub fn plain(b: *std.Build, root: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lanes: *std.Build.Module) *std.Build.Module {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "engine_api", .module = api },
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "template", .module = template },
        .{ .name = "core", .module = core },
    } });
    _ = addTo(b, m, target, optimize);
    testImports(b, m, target, optimize);
    return m;
}

/// The CPU-only fake server and the engine-less CLI into zig-out/server/ (zig build server-cpu), for build.sh.
pub fn cpuArtefacts(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    // the fake server: the server module with no native engines, so addTo takes the plain drama prompt
    const server_mod = plain(b, "zig/src/server/root.zig", target, optimize, lanes);
    const api = server_mod.import_table.get("engine_api").?;
    const tokenizer = server_mod.import_table.get("tokenizer").?;
    const template = server_mod.import_table.get("template").?;
    const core = server_mod.import_table.get("core").?;
    const none = b.createModule(.{ .root_source_file = b.path("zig/src/native/none.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "engine_api", .module = api }} });
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

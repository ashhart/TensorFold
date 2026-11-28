//! tensorfold-native over the Metal engines: the server executable, its step and its host tests.
const std = @import("std");
const server_build = @import("server.zig");
const exlr8_build = @import("exlr8.zig");
const golden_build = @import("golden.zig");
const mlxMetalStd = @import("metal_targets.zig").mlxMetalStd;

pub fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, living: *std.Build.Module, api: *std.Build.Module, engines: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: ?*std.Build.Step, no_compiler: ?*std.Build.Step) *std.Build.Step.Compile {
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize`
    const tokenizer = engines.import_table.get("tokenizer") orelse b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const checkpoint_cli = @import("cuda.zig").checkpointCli(b, target, .ReleaseSafe, null, engines, engine);
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "living", .module = living }, .{ .name = "checkpoint_cli", .module = checkpoint_cli } },
    }) });
    const bom = build_options.createModule();
    exe.root_module.addImport("build_options", bom);
    engines.addImport("build_options", bom); // the served families read the engine version (the pair's identity fingerprint)
    _ = server_build.addTo(b, exe.root_module, target, .ReleaseSafe);
    if (test_step == null) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    const native = b.step("native", "tensorfold-native with the Metal engines into zig-out/native/bin (bundles: -Dcpu=apple_m1)");
    native.dependOn(&install.step);
    if (no_compiler) |fail| native.dependOn(fail);
    // Link libSystem's memcpy to avoid compiler-rt's weak replacement.
    const libc_mem = b.addSystemCommand(&.{ "sh", "-c", "nm -m \"$0\" | grep -q 'external _memcpy (from libSystem)' || { echo 'tensorfold-native links its own memcpy: find the typed std.json int parse or 128-bit float conversion that pulled in compiler-rt'; exit 1; }" });
    libc_mem.addArtifactArg(exe);
    test_step.?.dependOn(&libc_mem.step);
    // core/root.zig, as cuda.zig gives the server tests: the engine's tensorfold.zig shares files with qwen_draw
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = target,
        .optimize = .Debug, // the server's unit tests run with safety checks, as zig_test.sh builds them
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "living", .module = living }, .{ .name = "core", .module = core } },
    });
    // the required-tools tests' 27B mask policy and lane fake, on the engine API's own lanes module
    if (api.import_table.get("lanes")) |lanes_mod| {
        const q27 = @import("qwen27_required.zig").modules(b, lanes_mod);
        server_tests.addImport("qwen27_mask_fake", q27.fake);
        server_tests.addImport("qwen27_masks", q27.masks);
    }
    const server_json = server_build.addTo(b, server_tests, target, .Debug);
    server_build.testImports(b, server_tests, target, .Debug);
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests })).step);
    // the PDF render worker as tensorfold-native spawns it; POSIX only, as Windows renders in process
    if (target.result.os.tag != .windows) {
        const worker_exe = b.addOptions();
        worker_exe.addOptionPath("path", exe.getEmittedBin());
        const pdf_worker = b.createModule(.{ .root_source_file = b.path("zig/src/server/pdf_worker_test.zig"), .target = target, .optimize = .Debug, .link_libc = true });
        for (server_tests.import_table.keys(), server_tests.import_table.values()) |name, module| pdf_worker.addImport(name, module);
        pdf_worker.addOptions("pdf_worker_exe", worker_exe);
        const pdf_worker_run = b.addRunArtifact(b.addTest(.{ .root_module = pdf_worker, .filters = &.{"tensorfold-native PDF render worker:"} }));
        test_step.?.dependOn(&pdf_worker_run.step);
        b.step("test-pdf-worker", "The PDF render worker spawned from tensorfold-native, rendering where libpdfium is found").dependOn(&pdf_worker_run.step);
    }
    const kimi_text = b.addExecutable(.{ .name = "tf-k3-text-golden", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/kimi_golden_main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "json", .module = server_json }, .{ .name = "tokenizer", .module = tokenizer } },
    }) });
    b.step("tf-k3-text-golden", "Kimi release prompt token ids against an explicitly supplied local tokenizer").dependOn(&b.addInstallArtifact(kimi_text, .{}).step);
    const kimi_host = b.step("test-kimi-serving-host", "Kimi native text, HTTP and resident budget refusals on the host");
    for ([_]*std.Build.Module{ server_tests, engines }) |module|
        kimi_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"Kimi"} })).step);

    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_json })).step);
    @import("cortheon_fuzz.zig").add(b, target, api, tokenizer, template, core);
    @import("stream_scan_bench.zig").add(b, target, api, tokenizer, template, core);
    // the parse half's tests, as their own test binary (zig/src/server/tool_parse_test.zig)
    const tool_parse_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } } }) });
    _ = server_build.addTo(b, tool_parse_tests.root_module, target, .Debug);
    server_build.testImports(b, tool_parse_tests.root_module, target, .Debug);
    test_step.?.dependOn(&b.addRunArtifact(tool_parse_tests).step);
    // the chat-template engine's tests (zig/src/core/template/tests.zig), which no other test binary reaches
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/tests.zig"), .target = target, .optimize = .Debug, .link_libc = true }) })).step);
    // the engine seam's own tests (lane_host.zig), as zig/tests/server/zig_test.sh runs them
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    // the frozen server parity goldens, checked by Zig driving fake_serve
    golden_build.step(b, target, server_tests, api, tokenizer, template, engines);
    engines.addImport("lanes", api.import_table.get("lanes").?);
    b.step("test-draft-metal-host", "Metal draft source metadata without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines, .filters = &.{"draft sources:"} })).step);
    b.step("test-qwen35-chunk-startup-metal-host", "Qwen3.5 Metal startup refuses bad TF_QWEN35_CHUNK before checkpoint load").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines, .filters = &.{"qwen35 startup: TF_QWEN35_CHUNK refuses before the checkpoint loader runs"} })).step);
    b.step("test-drafter-skip-metal-host", "Counted shared draft skips without device execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"zero grant:"} })).step);
    b.step("test-gemma4-metal-vtable-host", "Gemma 4 Metal steps its prompt passes (#3021), without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"Gemma Metal steps:"} })).step);
    b.step("test-uocr", "Unlimited-OCR-2's tests (its kernels compile on this GPU)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"uocr:"} })).step);
    const embed_host = b.step("test-qwen3-metal-host", "Qwen3 Metal embedding geometry, batch bounds and native refusals without GPU work");
    const embed_air = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{mlxMetalStd(b)}), "-fno-fast-math", "-c" });
    embed_air.addFileArg(b.path("zig/kernels/metal/qwen3/heads.metal"));
    embed_air.addArg("-o");
    _ = embed_air.addOutputFileArg("qwen3-heads.air");
    embed_host.dependOn(&embed_air.step);
    embed_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = &.{"embedding"} })).step);
    embed_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines, .filters = &.{"Qwen3 Metal"} })).step);
    const flint_draft_host = b.step("test-flint-draft-host", "Flint draft config, taps and first-token metadata without devices");
    for ([_]*std.Build.Module{ engine, api.import_table.get("lanes").? }) |module|
        flint_draft_host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module, .filters = &.{"Flint DFlash2"} })).step);
    const native_tests = b.step("test-native", "The Metal engines module's host-side tests");
    native_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines })).step);
    const capabilities = b.createModule(.{ .root_source_file = b.path("zig/src/native/inkling2_capabilities_test.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    for (engines.import_table.keys(), engines.import_table.values()) |name, module| capabilities.addImport(name, module);
    capabilities.addImport("native_metal", engines);
    const capability_run = b.addRunArtifact(b.addTest(.{ .root_module = capabilities }));
    const capability_step = b.step("test-inkling2-metal-capabilities", "Inkling native Metal checkpoint-format capabilities");
    capability_step.dependOn(&capability_run.step);
    native_tests.dependOn(capability_step);
    test_step.?.dependOn(&capability_run.step);
    const tfq3_caps = b.createModule(.{ .root_source_file = b.path("zig/src/native/tfq3_capabilities_test.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    for (engines.import_table.keys(), engines.import_table.values()) |name, module| tfq3_caps.addImport(name, module);
    tfq3_caps.addImport("native_metal", engines);
    const tfq3_run = b.addRunArtifact(b.addTest(.{ .root_module = tfq3_caps }));
    b.step("test-tfq3-metal-capabilities", "The Metal families that offer TFQ3 checkpoints: glm5_next and qwen4_exp only").dependOn(&tfq3_run.step);
    native_tests.dependOn(&tfq3_run.step);
    test_step.?.dependOn(&tfq3_run.step);
    const reuse = b.addExecutable(.{ .name = "tf-flashnext-reuse", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_reuse.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } },
    }) });
    b.installArtifact(reuse);
    b.step("tf-flashnext-reuse", "Flash Next prompt reuse against fresh prompt passes: replies at every depth and kept states' bytes").dependOn(&b.addInstallArtifact(reuse, .{}).step);
    const batch = b.addExecutable(.{ .name = "tf-flashnext-batch", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_batch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = engine.import_table.get("kernel_sources").? } },
    }) });
    b.step("tf-flashnext-batch", "Flash Next shared rounds against fresh legacy replies with prompt admission and cancellation").dependOn(&b.addInstallArtifact(batch, .{}).step);
    return exe;
}

//! The HTTP server's host tests over the CUDA side's modules, and its goldens and fuzz step where macOS does not.
const std = @import("std");
const golden_build = @import("golden.zig");
const kolibri1Module = @import("kolibri.zig").module;

/// Registers the server host steps; returns the template and server test modules the later CUDA host tests import.
pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, native_modules: anytype, build_options: *std.Build.Step.Options, windows_port: *std.Build.Step, step: *std.Build.Step, server_steps: bool) struct { *std.Build.Module, *std.Build.Module } {
    // the HTTP server's unit tests (zig/src/server/root.zig), which the macOS build runs, on Linux hosts too
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template }, .{ .name = "living", .module = mods.living }, .{ .name = "core", .module = mods.core } },
    });
    const server_json = @import("server.zig").addTo(b, server_tests, host, .debug);
    const startup_receipt = b.step("test-startup-receipt-host", "Monotonic startup receipts and fail-closed readiness without devices");
    const startup_spans = b.createModule(.{ .root_source_file = b.path("zig/src/core/startup_receipt.zig"), .target = host, .link_libc = true });
    startup_receipt.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = startup_spans, .filters = &.{"startup receipt:"} })).step);
    startup_receipt.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"startup receipt:"} })).step);
    @import("q27_profile_host.zig").steps(b, server_tests);
    // the open-connection limit, one address's share under pressure and its flags (#3048), on Linux hosts too
    b.step("test-connection-admission-host", "Connection limit, per-address share, trusted proxies and idle shedding").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "security: ", "connection CLI:" } })).step);
    b.step("test-drama-cfg-host", "Drama CFG CLI, HTTP and voice transitions without GPU execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"CFG"} })).step);
    b.step("test-speech-redraw-deadline-host", "Busy speech redraw expiry without counting an unsubmitted take").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"a busy redraw stops at its latest start"} })).step);
    @import("proposal_vocab_host.zig").steps(b, mods.core);
    const eos_policy = b.step("test-eos-policy-host", "Shared EOS, real HfText and HTTP policy without a GPU");
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"EOS policy:"} })).step);
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.tokenizer, .filters = &.{"EOS policy:"} })).step);
    eos_policy.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = kolibri1Module(b, host, .debug, cuda, mods.core, mods.lanes), .filters = &.{"EOS policy:"} })).step);
    b.step("test-serve-flags-host", "Serve flag parsing, the flat TOML reader and --config expansion, without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "--compact-at:", "flat TOML:", "serve --config:" } })).step);
    b.step("test-dory-cli-host", "Dory CUDA loop option forwarding and invalid-value refusals").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"--device and --segments:"} })).step);
    b.step("test-dory-chat-host", "Dory chat template, EOS, reasoning split and qwen3_coder call on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Dory chat:"} })).step);
    b.step("test-embeddings-protocol-host", "Embedding request and response contracts without an admitted model or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"embeddings protocol:"} })).step);
    b.step("test-embeddings-http-host", "Embedding route capability and borrowed request lifetime without a model or GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"embeddings HTTP:"} })).step);
    @import("rerank_host.zig").steps(b, server_tests);
    b.step("test-ollama-e2e-host", "Ollama routes over a loopback socket with the CPU fake engine").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"ollama HTTP:"} })).step);
    @import("safetensors_header_host.zig").steps(b, mods.core, step);
    b.step("test-runtime-capabilities-host", "Optional library discovery in actual capabilities JSON").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"runtime library:"} })).step);
    b.step("test-vision-cli-host", "Image token budgets through parsing and engine admission options without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"vision image budget:"} })).step);
    windows_port.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"hardware names"} })).step);
    @import("benchmark_group_host.zig").steps(b, &.{ cuda, mods.core, server_tests });
    b.step("test-benchmark-manifest-host", "Manifest identity through the HTTP server without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"manifest"} })).step);
    b.step("test-lesson-questions-host", "Supplied questions through the production lesson without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Lesson questions:"} })).step);
    b.step("test-grammar-metrics-host", "Grammar totals in whole and streamed replies without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"grammar timing:"} })).step);
    b.step("test-slide-input-host", "Text-only lesson admission before any engine work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"slide input:"} })).step);
    b.step("test-slide-subject-host", "Living Weights subject words: own names, questions, numbers and sentence openers").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "a fact's subject", "a told fact's subject", "supplied question", "subject swaps" } })).step);
    b.step("test-pocket-audit-host", "Read-only pocket audit probes, witnesses and CLI on the host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "pocket audit", "audit request:", "audit CLI:", "audit probe bank:", "audit endpoint:" } })).step);
    b.step("test-slide-recovery-host", "Living Weights conversation, stream and refusal-word regressions without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "lesson conversations:", "lesson stream:", "lesson writer:", "lesson refusal:" } })).step);
    b.step("test-issue-934", "Benchmark config admission and receipt regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 934:"} })).step);
    b.step("test-issue-969", "First-EOS classification and incomplete benchmark messages without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"issue 969"} })).step);
    b.step("test-tool-required-host", "Required tool parsing, grammar form and constrained lane replies on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"required tools:"} })).step);
    const q27 = @import("qwen27_required.zig").modules(b, mods.lanes);
    server_tests.addImport("qwen27_mask_fake", q27.fake);
    server_tests.addImport("qwen27_masks", q27.masks);
    b.step("test-qwen27-required-host", "Qwen27 required tool routes through the masked lane fake").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"qwen27 tool calls:"} })).step);
    b.step("test-save-graph-host", "Actual graph receipts paired with generic learner persistence failures").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Save recovery:"} })).step);
    b.step("test-slide-metrics-host", "Fixed untaught drift, supplied recall and opt-in lesson gate").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Slide metrics:"} })).step);
    b.step("test-lesson-gate-host", "Production server rounds and earlier-fact retention").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Learning persistence:"} })).step);
    b.step("test-slide-delete-host", "Single and multiple fact deletion without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "delete", "a graph file", "allocation that fails anywhere in recording" } })).step);
    b.step("test-frame-cache-identity-host", "Image frame identities through the actual shared cache").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api, .filters = &.{"Frame cache:"} })).step);
    b.step("test-originals-host", "Model folder snapshots, holds and slide rollback without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api, .filters = &.{"originals_test."} })).step);
    const media_gate = b.step("test-media-request-ownership-host", "Shared request ownership, image frame caches and media expansion on the CPU");
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native_modules.api })).step);
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "image", "video" } })).step);
    media_gate.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.core, .filters = &.{ "qwen_vl", "video" } })).step);
    b.step("test-http-limits-host", "Total HTTP read deadlines and connection admission without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"HTTP limits:"} })).step);
    b.step("test-request-log-host", "Default text and opt-in JSON access, ended and refused log lines without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "default request log:", "json request log:" } })).step);
    b.step("test-server-recovery-host", "Incomplete replies, token allocation and bounded SSE storage without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"Server recovery:"} })).step);
    b.step("test-prompt-interior-host", "Last-user metadata excludes later messages without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"interior prompt:"} })).step);
    b.step("test-ocr-host", "OCR route readers, shared page steps and dialects without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"ocr "} })).step);
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests }));
    @import("request_starts_host.zig").steps(b, native_modules.engines, server_tests);
    const json_test = b.addRunArtifact(b.addTest(.{ .root_module = server_json }));
    const publication = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"benchmark publication:"} }));
    b.step("test-benchmark-publication-host", "Benchmark command publication boundaries with inert effects").dependOn(&publication.step);
    b.step("test-benchmark-routing-host", "Benchmark command routing from explicit model capabilities").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"benchmark routing:"} })).step);
    b.step("test-benchmark-cancel-host", "A cancelled local benchmark saves its receipt, exits 130 and never publishes").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"a cancelled local run"} })).step);
    const server_step = b.step(
        "test-server-cpu",
        "The HTTP server's unit tests (routes, templates, tool parsing) without a GPU",
    );
    for ([_]*std.Build.Step.Run{ server_test, json_test }) |run| {
        step.dependOn(&run.step);
        server_step.dependOn(&run.step);
    }
    // the server goldens and Cortheon's fuzz step here when the target is not macOS, whose Metal build defines them
    if (server_steps) {
        golden_build.step(b, host, server_tests, native_modules.api, mods.tokenizer, template, native_modules.engines);
        @import("cortheon_fuzz.zig").add(b, host, native_modules.api, mods.tokenizer, template, mods.core);
        @import("stream_scan_bench.zig").add(b, host, native_modules.api, mods.tokenizer, template, mods.core);
        // the PDF render worker from this host's CUDA tensorfold-native, as metal_targets.zig has it; POSIX only
        if (host.result.os.tag != .windows) {
            const server = @import("cuda.zig").nativeServer(b, host, .ReleaseSafe, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, false, null);
            const worker_exe = b.addOptions();
            worker_exe.addOptionPath("path", server.getEmittedBin());
            const pdf_worker = b.createModule(.{ .root_source_file = b.path("zig/src/server/pdf_worker_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
            for (server_tests.import_table.keys(), server_tests.import_table.values()) |name, module| pdf_worker.addImport(name, module);
            pdf_worker.addOptions("pdf_worker_exe", worker_exe);
            const pdf_worker_run = b.addRunArtifact(b.addTest(.{ .root_module = pdf_worker, .filters = &.{"tensorfold-native PDF render worker:"} }));
            pdf_worker_run.step.dependOn(&server.step); // a step edge, so sys.zig's walk gives the server its imports
            b.step("test-pdf-worker", "The PDF render worker spawned from tensorfold-native, rendering where libpdfium is found").dependOn(&pdf_worker_run.step);
        }
    }
    return .{ template, server_tests };
}

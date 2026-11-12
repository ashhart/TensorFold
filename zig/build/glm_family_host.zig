//! Existing GLM family recording controls on the CPU runtime.
const std = @import("std");

pub fn steps(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, mods: anytype, step: *std.Build.Step) void {
    // GLM CUDA's startup watcher (named phases, a warning while NCCL init or a load blocks), from cuda.zig's room
    b.step("test-startup-watch-host", "Blocked startup phase warnings without a driver or communicator").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"startup watcher:"} })).step);
    const repetition = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM repetition:"} }));
    b.step("test-glm-repetition-host", "Full-prefix repetition history, target rows and TP refusal without devices").dependOn(&repetition.step);
    const host = b.graph.host;
    inline for (.{ .{ "test-glm-link-transport-host", "zig/glm_link_transport_host.zig" }, .{ "test-glm-kv-uncached-host", "zig/glm_kv_uncached_host.zig" }, .{ "test-glm-lane-costs-host", "zig/glm_lane_costs_host.zig" } }) |entry| {
        const checks = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(entry[1]),
            .target = host,
            .optimize = .debug,
            .link_libc = true,
            .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda } },
        }) }));
        b.step(entry[0], "GLM startup prices, transport and uncached KV through recording calls").dependOn(&checks.step);
        step.dependOn(&checks.step);
    }
    b.step("test-glm-relay-host", "GLM relay options before device setup").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM relay integration:"} })).step);
    const warm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_warm_peers_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "heat", .module = mods.heat } } });
    const warm_run = b.addRunArtifact(b.addTest(.{ .root_module = warm, .filters = &.{"GLM warm peers:"} }));
    b.step("test-glm-warm-peers-host", "Selected GLM warm-up peers and NCCL posting failures without devices").dependOn(&warm_run.step);
    step.dependOn(&warm_run.step);
    const kda = b.step("test-kda-host", "Shared KDA launch ABI against its frozen predecessor, without CUDA execution");
    kda.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"shared KDA:"} })).step);
    const kda_header = b.createModule(.{ .root_source_file = b.path("zig/kda_header_host.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const kda_header_run = b.addRunArtifact(b.addTest(.{ .root_module = kda_header }));
    kda.dependOn(&kda_header_run.step);
    step.dependOn(&kda_header_run.step);
    const startup = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_start_test.zig"), .target = host, .optimize = .debug });
    const startup_run = b.addRunArtifact(b.addTest(.{ .root_module = startup }));
    b.step("test-glm-startup-host", "Page refusals propagate through rank liveness before engine construction").dependOn(&startup_run.step);
    step.dependOn(&startup_run.step);
    const thin_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM thin weights:"} }));
    b.step("test-glm-thin-weights-host", "TF_GLM_LAYERS's admission bytes by tensor from synthetic shard headers").dependOn(&thin_run.step);
    step.dependOn(&thin_run.step);
    b.step("test-glm-tp3-host", "Uneven KDA head shares through actual projection and load planning, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM TP3:"} })).step);
    // TF_GLM_INDEX_CUDA: the launch rules, and (its own slower step) glm_index_*.cuh as host C++ against plain models
    const index_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM index:"} }));
    b.step("test-glm-index-host", "GLM's prompt indexer launch rules without devices").dependOn(&index_run.step);
    step.dependOn(&index_run.step);
    const index_emulate = b.addExecutable(.{ .name = "glm-index-emulate", .root_module = b.createModule(.{ .target = host, .optimize = .ReleaseSafe, .link_libcpp = true }) });
    index_emulate.root_module.addCSourceFile(.{ .file = b.path("tools/glm_index_emulate/emulate.cpp"), .flags = &.{ "-std=c++20", "-O1" } });
    index_emulate.root_module.addIncludePath(b.path("tools/glm_index_emulate"));
    const index_emulated = b.addRunArtifact(index_emulate);
    index_emulated.expectExitCode(0);
    b.step("test-glm-index-emulate", "GLM's prompt indexer kernels as host C++ against a plain top-K and a model of tf_glm_scores").dependOn(&index_emulated.step);
    const kv8_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM FP8 KV:"} }));
    b.step("test-glm-fp8-kv-host", "GLM latent cache layout, mode agreement and recorded CUDA launches without devices").dependOn(&kv8_run.step);
    step.dependOn(&kv8_run.step);
    const profile_run = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM prompt profile:"} }));
    b.step("test-glm-prompt-profile-host", "Paired prompt-stage event timing, overlap, failure and disabled controls without devices").dependOn(&profile_run.step);
    step.dependOn(&profile_run.step);
    const pages = b.createModule(.{ .root_source_file = b.path("zig/glm_pages_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core } } });
    const pages_run = b.addRunArtifact(b.addTest(.{ .root_module = pages, .filters = &.{"GLM pages:"} }));
    b.step("test-glm-pages-host", "Actual GLM page ownership, refusal and rollback without devices").dependOn(&pages_run.step);
    step.dependOn(&pages_run.step);
    const api = glm.import_table.get("engine_api").?; // the family's own (the shared store's types): one module, one set of types
    const adapter = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_prompt_pages.zig"), .target = host, .link_libc = true, .imports = &.{ .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = api } } });
    const image_cache = b.createModule(.{ .root_source_file = b.path("zig/glm_image_cache_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "core", .module = mods.core }, .{ .name = "prompt_pages", .module = adapter } } });
    const image_cache_run = b.addRunArtifact(b.addTest(.{ .root_module = image_cache }));
    b.step("test-glm-image-cache-host", "Image-content prefix reuse over the shared store and suffix rows across both ranks without devices").dependOn(&image_cache_run.step);
    step.dependOn(&image_cache_run.step);
    const page_contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_glm_controls_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const page_contract_run = b.addRunArtifact(b.addTest(.{ .root_module = page_contract, .filters = &.{"GLM page contract:"} }));
    b.step("test-glm-page-contract-host", "GLM paged contract case and planted-control inventory without devices").dependOn(&page_contract_run.step);
    step.dependOn(&page_contract_run.step);
    const admission = b.createModule(.{ .root_source_file = b.path("zig/glm_admission_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes }, .{ .name = "engine_api", .module = api } } });
    const admission_run = b.addRunArtifact(b.addTest(.{ .root_module = admission, .filters = &.{"GLM CUDA admission:"} }));
    b.step("test-glm-admission-host", "Real GLM admission capabilities through the lane worker without GPU work").dependOn(&admission_run.step);
    step.dependOn(&admission_run.step);
    const copies = b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_snaps.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "core", .module = mods.core } } });
    const snapshots = b.createModule(.{ .root_source_file = b.path("zig/glm_snapshot_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core }, .{ .name = "engine_api", .module = api }, .{ .name = "copies", .module = copies } } });
    const snapshots_run = b.addRunArtifact(b.addTest(.{ .root_module = snapshots }));
    b.step("test-glm-snapshots-host", "Prompt-sized GLM state copies, page budgets and rank ordering without devices").dependOn(&snapshots_run.step);
    step.dependOn(&snapshots_run.step);
    b.step("test-glm-prompt-settings-host", "Actual opt-in settings and rank agreement on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prompt K experiment"} })).step);
    const request_starts = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"glm request starts:"} }));
    b.step("test-glm-request-starts-host", "GLM message-start prompt cuts and the shared-KV startup refusal without devices").dependOn(&request_starts.step);
    const mm_record = b.createModule(.{ .root_source_file = b.path("zig/test_support/glm_mm_cuda.zig"), .target = host, .optimize = .debug });
    const mm_actual = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_mm.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "cuda", .module = mm_record }} });
    const mm_policy = b.createModule(.{ .root_source_file = b.path("zig/glm_mm_policy_host.zig"), .target = host, .optimize = .debug, .imports = &.{ .{ .name = "cuda", .module = mm_record }, .{ .name = "mm", .module = mm_actual } } });
    b.step("test-glm-mm-policy-host", "Actual prompt K policy dispatch without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mm_policy })).step);
    const drafter_host = b.createModule(.{ .root_source_file = b.path("zig/glm_dflash_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-dflash-host", "DFlash rank buffers, partial loader ownership and pending records on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = drafter_host, .filters = &.{"glm dflash:"} })).step);
    const packed_host = b.createModule(.{ .root_source_file = b.path("zig/glm_packed_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "core", .module = mods.core } } });
    b.step("test-glm-packed-host", "Packed GLM launch records, cap parsing and native loader on the CPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = packed_host, .filters = &.{"glm packed:"} })).step);
    const ring_host = b.createModule(.{ .root_source_file = b.path("zig/glm_ring_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm } } });
    b.step("test-glm-ring-host", "Actual GLM MTP rollback pool through recording CUDA copies").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ring_host, .filters = &.{"glm ring:"} })).step);
    const taps_tp = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM prompt taps:"} }));
    b.step("test-glm-prompt-taps-host", "DFlash2 prompt taps hold every row at TP>1 through recording copies and emulated NCCL").dependOn(&taps_tp.step);
    step.dependOn(&taps_tp.step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    const prompt_host = b.createModule(.{ .root_source_file = b.path("zig/exl3_prompt_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = glm }, .{ .name = "lanes", .module = mods.lanes } } });
    const prompt_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_host, .filters = &.{"exl3 prompt:"} }));
    prompt_run.has_side_effects = true;
    b.step("test-exl3-prompt-host", "Actual common prompt dispatch and scratch with recording CUDA calls").dependOn(&prompt_run.step);
}

pub fn compile(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, comptime makeGlm: anytype) void {
    const fp8_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_ref_fp8.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = mods.core } } });
    const fp8_object = b.addObject(.{ .name = "glm-fp8-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_fp8_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = fp8_gate }} }) });
    _ = fp8_object.getEmittedBin();
    b.step("compile-glm-fp8-gate-host", "Compile synthetic FP8 write/read and FP64 device checks with inert images, never execute").dependOn(&fp8_object.step);
    const glm_open = b.addObject(.{ .name = "glm-open-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_open_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }} }) });
    _ = glm_open.getEmittedBin();
    b.step("compile-glm-native-open", "Compile the actual GLM Linux open body with empty GPU images").dependOn(&glm_open.step);
    const prompt_gate = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_mm_fixed_tests.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = makeGlm(b, target, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api) }, .{ .name = "cuda", .module = cuda } } });
    const prompt_gate_compile = b.addObject(.{ .name = "glm-prompt-gate-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/glm_prompt_gate_compile.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "gate", .module = prompt_gate }} }) });
    _ = prompt_gate_compile.getEmittedBin();
    b.step("compile-glm-prompt-gate-host", "Compile the actual device prompt comparison gate with inert images").dependOn(&prompt_gate_compile.step);
}

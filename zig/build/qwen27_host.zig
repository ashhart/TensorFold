//! Host regressions use no shaders, model or GPU execution.
const std = @import("std");
pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const real_metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const recording = b.createModule(.{ .root_source_file = b.path("zig/qwen27_quant_metal.zig"), .target = b.graph.host, .imports = &.{.{ .name = "real_metal", .module = real_metal }} });
    const quant_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const quant = b.createModule(.{ .root_source_file = b.path("zig/qwen27_quant_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = recording }, .{ .name = "kernel_sources", .module = quant_sources } } });
    _ = @import("qwen27.zig").engineModules(b, b.graph.host, recording, quant_sources, quant);
    const quant_test = b.addTest(.{ .root_module = quant, .filters = &.{"qwen27 quant:"} });
    const quant_run = b.addRunArtifact(quant_test);
    test_step.dependOn(&quant_run.step);
    b.step("test-qwen27-quant-host", "Prepared head cache identity and prompt row capacity without a GPU").dependOn(&quant_run.step);
    const metrics = b.createModule(.{ .root_source_file = b.path("zig/src/metal/pass_metrics.zig"), .target = b.graph.host });
    b.step("test-qwen27-pass-metrics-host", "Pass timing and forwarding allocator without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = metrics })).step);
    const window = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_window.zig"), .target = b.graph.host });
    const taps = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/tap_gather.zig"), .target = b.graph.host });
    const pricing_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host });
    b.step("test-qwen27-shared-allocation-host", "All lane scheduling and recorder regressions without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pricing_lanes })).step);
    const allocated = b.createModule(.{ .root_source_file = b.path("zig/qwen27_allocation_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = pricing_lanes }} });
    const window_step = b.step("test-qwen27-window-host", "Device width, copied trees and counted zero-grant drafting without shaders");
    for ([_]*std.Build.Module{ window, taps, allocated }) |m| window_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const rows = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_rows_test.zig"), .target = b.graph.host });
    b.step("test-qwen27-prompt-rows-host", "Device-aware prompt rows beside owned decoding streams without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = rows, .filters = &.{"qwen27 rows:"} })).step);
    const ane_policy = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/ane_policy.zig"), .target = b.graph.host });
    b.step("test-qwen27-ane-policy-host", "Neural Engine opt-in and tensor-unit chip policy without shaders").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ane_policy })).step);
    const draws_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host });
    const draws = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_draws_test.zig"), .target = b.graph.host, .imports = &.{.{ .name = "lanes", .module = draws_lanes }} });
    const draws_run = b.addRunArtifact(b.addTest(.{ .root_module = draws, .filters = &.{"qwen27 draws:"} }));
    test_step.dependOn(&draws_run.step);
    b.step("test-qwen27-draws-host", "A sampled window's rows drawn on host threads equal the walk's tokens and first error").dependOn(&draws_run.step);
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "metal", .module = metal }} });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"qwen27 speed:"} }));
    run.has_side_effects = true;
    b.step("test-qwen27-speed-host", "Prompt tap guards, attention admission and cache failure ownership").dependOn(&run.step);
    const fake = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_speed_metal.zig"), .target = b.graph.host });
    const core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_speed_core.zig"), .target = b.graph.host, .imports = &.{.{ .name = "metal", .module = fake }} });
    const pass = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_pass_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core } } });
    b.step("test-qwen27-pass-host", "Count actual draft scratch allocations without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pass, .filters = &.{"qwen27 pass:"} })).step);
    const memory = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_memory_test.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core } } });
    const memory_run = b.addRunArtifact(b.addTest(.{ .root_module = memory, .filters = &.{"qwen27 memory:"} }));
    test_step.dependOn(&memory_run.step);
    b.step("test-qwen27-memory-host", "Prompt scratch and working-set fit without a GPU").dependOn(&memory_run.step);
    const sources = b.createModule(.{ .root_source_file = b.addWriteFiles().add("qwen27-fake-source.zig", "pub const core_deltanet = \"\";\n"), .target = b.graph.host });
    const delta = b.createModule(.{ .root_source_file = b.path("zig/src/core/deltanet_kernels.zig"), .target = b.graph.host, .imports = &.{.{ .name = "metal", .module = fake }} });
    delta.addImport("kernel_sources", sources);
    const keep_module = b.createModule(.{
        .root_source_file = b.path("zig/src/qwen27_keep_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core }, .{ .name = "core_delta", .module = delta } },
    });
    const bank_module = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_bank_test.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{ .{ .name = "metal", .module = fake }, .{ .name = "core", .module = core }, .{ .name = "core_delta", .module = delta } } });
    b.step("test-qwen27-bank-host", "Actual DeltaNet bank ownership with a recording device, no GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = bank_module, .filters = &.{"qwen27 bank:"} })).step);
    const keep = b.addRunArtifact(b.addTest(.{ .root_module = keep_module, .filters = &.{"qwen27 speed:"} }));
    keep.has_side_effects = true;
    b.step("test-qwen27-keep-host", "Actual shared round keep with a recording Metal facade").dependOn(&keep.step);
    var names: std.ArrayList([]const u8) = .empty;
    tryNames(b, &names, @embedFile("../src/families/qwen27/gdn_test.zig"));
    tryNames(b, &names, @embedFile("../src/families/qwen27/attention_plan.zig"));
    const plans = b.addRunArtifact(b.addTest(.{ .root_module = keep_module, .filters = names.items }));
    plans.has_side_effects = true;
    b.step("test-qwen27-plans-host", "Shared GDN and attention plans with no shaders or GPU").dependOn(&plans.step);

    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host });
    const equal = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_lanes_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    }), .filters = &.{"qwen27 speed:"} }));
    equal.has_side_effects = true;
    b.step("test-qwen27-equality-host", "Shared lanes drafted, resumed and concurrent equality on the fake target").dependOn(&equal.step);
    const resume_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .link_libc = true });
    const resume_api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = resume_lanes }} });
    const prompt_stats = b.createModule(.{ .root_source_file = b.path("zig/src/qwen27_prefill_stats_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = resume_lanes }, .{ .name = "engine_api", .module = resume_api } } });
    const prompt_stats_run = b.addRunArtifact(b.addTest(.{ .root_module = prompt_stats, .filters = &.{"qwen27 prefill stats:"} }));
    prompt_stats_run.has_side_effects = true;
    test_step.dependOn(&prompt_stats_run.step);
    b.step("test-qwen27-prefill-stats-host", "Completed prompt passes through the lane host and server mailbox without a GPU").dependOn(&prompt_stats_run.step);
    const resume_rules = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_resume.zig"), .target = b.graph.host, .optimize = .debug, .imports = &.{.{ .name = "engine_api", .module = resume_api }} });
    b.step("test-qwen27-resume-host", "Slot 0's in-place resume and rewind point: the cache's rule, with no GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = resume_rules, .filters = &.{"qwen27 resume:"} })).step);
}

pub fn compileHost(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module) void {
    const ownership = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_engine.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "tensorfold", .module = engine }} });
    b.step("test-qwen-dense-route-host", "Existing 2B route and untied 9B and 27B route without weights").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ownership, .filters = &.{"a qwen3_5 checkpoint opens"} })).step);
    var dense_names: std.ArrayList([]const u8) = .empty;
    inline for (.{ "config_test.zig", "weights_test.zig", "checkpoint.zig", "projection.zig", "affine.zig", "geometry_test.zig" }) |file| tryNames(b, &dense_names, @embedFile("../src/families/qwen27/" ++ file));
    tryNames(b, &dense_names, @embedFile("../src/families/qwen3_5/config.zig"));
    b.step("test-qwen-dense-geometry-host", "Dense config and full-size packed headers without shaders or a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine, .filters = dense_names.items })).step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_served.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    b.step("test-qwen27-reply-cache-host", "Served reply keeps retain a reusable history within the cache caps").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host, .filters = &.{ "qwen27 reply cache:", "qwen27 hand-off:" } })).step);
    const startup_test = b.addTest(.{ .root_module = host, .filters = &.{"qwen27 startup:"} });
    const startup_step = b.step("test-qwen27-startup-host", "Startup draft stream counts and suffixes without a GPU");
    for ([_]?[]const u8{ null, "1" }) |value| {
        const run = b.addRunArtifact(startup_test);
        if (value) |v| run.setEnvironmentVariable("TF_Q27_DRAFT_SLOTS", v) else run.removeEnvironmentVariable("TF_Q27_DRAFT_SLOTS");
        run.has_side_effects = true;
        startup_step.dependOn(&run.step);
    }
    const floor_logs = b.createModule(.{ .root_source_file = b.path("zig/qwen27_floor_logs.zig"), .target = b.graph.host });
    const floor_root = b.createModule(.{ .root_source_file = b.path("zig/qwen27_floor_entry_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "qwen27_served", .module = host }, .{ .name = "tensorfold", .module = engine }, .{ .name = "qwen_weights_diagnostics", .module = floor_logs } } });
    if (b.graph.host.result.os.tag == .macos) floor_root.linkSystemLibrary("objc", .{});
    const floor_test = b.addTest(.{ .root_module = floor_root, .test_runner = .{ .path = b.path("zig/qwen_vision_weights_runner.zig"), .mode = .simple } });
    const floor_step = b.step("test-qwen27-prompt-floor-host", "Actual startup and lane constructor floor diagnostics without devices");
    for ([_][]const u8{ "16", "128" }) |value| {
        const run = b.addRunArtifact(floor_test);
        run.setEnvironmentVariable("TF_PROMPT_ROWS", value);
        run.has_side_effects = true;
        floor_step.dependOn(&run.step);
    }
    const minimax = b.createModule(.{ .root_source_file = b.path("zig/src/native/minimax_m3.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } } });
    b.step("test-minimax-info-host", "MiniMax logprob offers against its actual wrapped backend without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = minimax, .filters = &.{"MiniMax capabilities:"} })).step);
    const boundary = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen27_bank_boundary_host.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "metal", .module = metal } } }), .filters = &.{"qwen27 bank boundary:"} });
    b.step("test-qwen27-bank-boundary-host", "Runner pending-keep guards with no device or shader execution").dependOn(&b.addRunArtifact(boundary).step);
    const facts = b.createModule(.{ .root_source_file = b.path("zig/src/native/draft_sources_metal_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes } } });
    b.step("test-qwen27-facts-host", "Actual native window and copy facts with initialized recording fields").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = facts, .filters = &.{ "qwen27 window:", "draft sources: Metal Qwen27" } })).step);
    const grants = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_shared_policy_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes } } });
    b.step("test-qwen27-grants-host", "Native shared draft policy counts granted and proposed nodes without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = grants })).step);
    const tree_host = b.addTest(.{ .root_module = engine, .filters = &.{"qwen27 window:"} });
    b.step("test-qwen27-draft-tree-host", "Real selector 31-node tree and verify parents without a GPU").dependOn(&b.addRunArtifact(tree_host).step);
    const object = b.addObject(.{ .name = "qwen27-speed-compile", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/qwen27_speed_compile.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api }, .{ .name = "qwen27_served", .module = host } },
    }) });
    _ = object.getEmittedBin();
    b.step("check-qwen27-speed-host", "Compile the actual Qwen27 startup and encoding without shaders").dependOn(&object.step);
}

fn tryNames(b: *std.Build, names: *std.ArrayList([]const u8), text: []const u8) void {
    var rest = text;
    while (std.mem.indexOf(u8, rest, "test \"")) |at| {
        rest = rest[at + 6 ..];
        const end = std.mem.indexOfScalar(u8, rest, '"').?;
        names.append(b.allocator, rest[0..end]) catch @panic("OOM");
        rest = rest[end + 1 ..];
    }
}

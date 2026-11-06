//! Dense CUDA learner checks, including the real constructor under a fake driver with deliberately invalid images.
const std = @import("std");

pub fn target(b: *std.Build, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module) void {
    const moe = b.addExecutable(.{ .name = "tf-qwen-moe-learn", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/qwen_moe_learn.zig"), .target = t, .optimize = o, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen } } }) });
    b.installArtifact(moe);
    b.step("compile-qwen-moe-learn", "Compile MoE learner qualification without running a device").dependOn(&moe.step);
    const run = b.addRunArtifact(moe);
    const model = b.option([]const u8, "qwen35-learn-model", "MoE checkpoint for learner qualification");
    const ids = b.option([]const u8, "qwen35-learn-ids", "Rendered token IDs as a JSON array");
    if (model != null and ids != null) run.addArgs(&.{ model.?, ids.? });
    b.step("qwen35-learn", "Qualify the Qwen MoE learner on a CUDA checkpoint").dependOn(&run.step);
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/cli/qwen35_learn.zig"), .target = t, .optimize = o, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen35-learn", .root_module = m });
    b.installArtifact(exe);
    b.step("compile-qwen35-learn", "Compile actual dense model qualification without execution").dependOn(&exe.step);
}

pub fn host(b: *std.Build, cuda: *std.Build.Module, qwen: *std.Build.Module, draft_ids: *std.Build.Module, comptime runtime: anytype, comptime family: anytype, comptime n: usize, all: *std.Build.Step) void {
    const t = b.graph.host;
    b.step("test-qwen-moe-learning-host", "MoE descriptor, widened loading and backward host regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{ "MoE learning:", "the 35B-A3B's routed experts", "Dense learning:", "Trainer admission:", "Save recovery: dense" } })).step);
    const num_mod = b.createModule(.{ .root_source_file = b.path("zig/qwen35_gradient_host.zig"), .target = t, .optimize = .debug });
    num_mod.addImport("core", qwen.import_table.get("core").?);
    const numerical = b.addRunArtifact(b.addTest(.{ .root_module = num_mod }));
    const gradient = b.step("test-qwen35-gradient-host", "Hybrid prompt/answer gradient, GDN replay ABI and dense mode on host");
    gradient.dependOn(&numerical.step);
    gradient.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"dense gradient:"} })).step);
    gradient.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{"dense gradient:"} })).step);
    all.dependOn(gradient);

    const pin = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/dense_serving_test.zig"), .target = t, .optimize = .debug }) }));
    b.step("test-qwen-serving-host", "Landed CUDA kernels and dense serving launch pins").dependOn(&pin.step);
    b.step("test-qwen-pocket-host", "Dense device publication and checked reads under CUDA failures").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{ "dense pocket:", "dense media:", "dense save:", "dense read cost:" } })).step);
    const placeholder = b.addWriteFiles().add("fake-driver-only.fatbin", "FAKE DRIVER ONLY, NEVER A DEVICE IMAGE");
    const images: [n]?std.Build.LazyPath = @splat(placeholder);
    const fake_cuda = runtime(b, t, .debug, &images);
    const fake = family(b, t, .debug, fake_cuda, draft_ids);
    const component = b.addObject(.{ .name = "qwen_moe_contract_compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen_moe_contract_compile.zig"), .target = t, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = fake_cuda }, .{ .name = "qwen35", .module = fake.qwen35 } } }) });
    b.step("compile-qwen-moe-contract-host", "Compile group-64 CUDA component bodies with inert images, no execution").dependOn(&component.step);
    const constructors = b.addRunArtifact(b.addTest(.{ .root_module = fake.qwen35, .filters = &.{"dense constructor:"} }));
    b.step("test-dense-constructors-host", "Actual dense constructor ownership and completion failures").dependOn(&constructors.step);
    const reference = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/moe_back_tests.zig"), .target = t, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} }), .filters = &.{"MoE packed reference:"} }));
    all.dependOn(&reference.step);
    b.step("test-qwen-moe-reference-host", "Packed reference adjoints against finite differences without devices").dependOn(&reference.step);
    b.step("test-packed-experts-host", "Packed adjoints layout and launch bounds without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{ "MoE packed:", "Quantized adjoints:" } })).step);
}

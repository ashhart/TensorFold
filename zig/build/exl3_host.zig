//! The EXL3 host tests of the Flash Next module and the CPU-only planner trace writer parity test.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, flashnext: *std.Build.Module) void {
    b.step("test-flashnext-exl3-host", "Flash Next EXL3 admission, store, sizing and loader host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = flashnext, .filters = &.{"exl3: "} })).step);
    const trace_writer_build = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O1" });
    trace_writer_build.addFileArg(b.path("zig/src/families/flashnext/exl3/dplan/exl3_dplan_trace_kernel_test.cpp"));
    trace_writer_build.addArg("-o");
    const trace_writer_binary = trace_writer_build.addOutputFileArg("dplan-trace-writer");
    trace_writer_build.addFileInput(b.path("zig/kernels/cuda/flashnext_exl3/flashnext_exl3_dplan.cu"));
    const trace_writer_run = std.Build.Step.Run.create(b, "dplan-trace-writer");
    trace_writer_run.addFileArg(trace_writer_binary);
    const trace_fixture = trace_writer_run.addOutputFileArg("demand-v4.exdpt");
    const checkpoint_fixture = trace_writer_run.addOutputFileArg("demand-v4-checkpoints.exdpt");
    const trace_writer_tests = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/exl3/dplan/exl3_dplan_trace_writer_test.zig"), .target = host, .optimize = .debug });
    trace_writer_tests.addAnonymousImport("dplan_trace_fixture", .{ .root_source_file = trace_fixture });
    trace_writer_tests.addAnonymousImport("dplan_checkpoint_fixture", .{ .root_source_file = checkpoint_fixture });
    b.step("test-flashnext-exl3-trace-writer-host", "CPU-only CUDA planner writer parity and exact Zig replay; requires c++").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = trace_writer_tests })).step);
}

//! Emit only the SASS targets supported by this unit; no eligible targets produces an inert empty image.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const Kernel = catalog.Kernel;
const torch_flags = catalog.torch_flags;

/// The `-Dsm` entry and SASS profile rule, kept with the catalog so its host tests cover it.
pub const Target = catalog.Target;
pub const Profile = catalog.Profile;
pub const profile = catalog.profile;

pub fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    var choices = std.mem.tokenizeScalar(u8, sms, ',');
    var any = false;
    while (choices.next()) |sm| {
        any = any or Target.parse(sm).number >= k.min_sm;
    }
    if (!any) return b.addWriteFiles().add(b.fmt("unsupported-{s}.fatbin", .{k.name}), "");
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    // CUDA 13's headers with cl.exe need MSVC's conforming preprocessor and M_SQRT2's math define.
    if (b.graph.host.result.os.tag == .windows) run.addArgs(&.{ "-Xcompiler=/Zc:preprocessor", "-D_USE_MATH_DEFINES" });
    run.addArgs(k.flags);
    // Generated CUDA sources resolve the same maintained headers as checked-in units.
    run.addDirectoryArg2(b.path("zig/kernels/cuda"), .{ .prefix = "-I" });
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| {
        const target = Target.parse(sm);
        const p = profile(k, target) orelse continue;
        // arch-specific features (the block-scaled FP4 mma) from sm_120 on; older targets build the trapping fallback
        const a = if (p.arch_specific and target.suffix.len == 0) "a" else "";
        run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    }
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    if (std.mem.eql(u8, k.name, "qwen4_gdn")) {
        const text = @import("../src/families/flashnext/hybrid_sources.zig").Gdn.cuda(b.allocator, @embedFile("../kernels/cuda/flashnext_gdn.cu")) catch @panic("Flash Next GDN source changed; regenerate Qwen4 specialization");
        const controlled = @import("../src/families/qwen4/gdn_control_source.zig").cuda(b.allocator, text) catch @panic("Qwen4 GDN control boundaries changed");
        run.addFileArg(b.addWriteFiles().add("qwen4_gdn.cu", controlled));
    } else if (std.mem.eql(u8, k.name, "qwen4_ple")) {
        const text = @import("../src/families/flashnext/hybrid_ple_source.zig").cuda(b.allocator) catch @panic("Flash Next PLE source changed; update Qwen4 geometry adapter");
        const controlled = @import("../src/families/qwen4/ple_control_source.zig").cuda(b.allocator, text) catch @panic("Qwen4 PLE controls changed");
        run.addFileArg(b.addWriteFiles().add("qwen4_ple.cu", controlled));
    } else run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

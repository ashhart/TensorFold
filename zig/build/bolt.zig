//! libbolt, Nemotron 3.5 Bolt's forward pass: an optional library nvcc builds beside the server from -Dbolt-src.

const std = @import("std");

const sources = [_][]const u8{ "bolt_kernels.cu", "bolt_mma.cu", "bolt_host.cu", "bolt_draft.cu" };

/// `zig build bolt -Dnvcc=NVCC -Dbolt-src=DIR`: native/lib/libbolt.so, which a Bolt checkpoint opens.
pub fn steps(b: *std.Build, nvcc: ?[]const u8) void {
    const step = b.step("bolt", "libbolt (Nemotron 3.5 Bolt's engine, sm_121) from -Dbolt-src with -Dnvcc, beside the server");
    const src = b.option([]const u8, "bolt-src", "the directory of libbolt's .cu sources (bolt_host.cu and its siblings)");
    const tool = nvcc orelse {
        step.dependOn(&b.addFail("zig build bolt needs -Dnvcc=PATH to nvcc").step);
        return;
    };
    const dir = src orelse {
        step.dependOn(&b.addFail("zig build bolt needs -Dbolt-src=DIR, the folder holding bolt_host.cu").step);
        return;
    };
    const root: std.Build.LazyPath = .{ .cwd_relative = dir };
    const run = b.addSystemCommand(&.{ tool, "-shared", "-Xcompiler", "-fPIC", "-O3", "-gencode", "arch=compute_121,code=sm_121", "-I" });
    run.addDirectoryArg(root);
    run.addArg("-o");
    const lib = run.addOutputFileArg("libbolt.so");
    for (sources) |name| run.addFileArg(root.path(b, name));
    run.addArg("-lcuda");
    step.dependOn(&b.addInstallFile(lib, "native/lib/libbolt.so").step);
}

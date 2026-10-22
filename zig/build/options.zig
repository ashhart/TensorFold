//! The backend options a build script may pass on every OS, declared for every target so one script fits all.
const std = @import("std");

/// The CUDA options, declared for every target: only a CUDA target reads them.
pub const Cuda = struct {
    nvcc: ?[]const u8,
    fatbins: ?[]const u8,
    sms: []const u8,
    strip: bool,
    test_dump: bool,
    contract_controls: ?[]const u8,
    cuobjdump: ?[]const u8,
    flashnext_model: ?[]const u8,
    flashnext_kernels: ?[]const u8,
    flashnext_token: []const u8,
    flashnext_ids: ?[]const u8,
    flashnext_count: []const u8,
    flashnext_fixtures: ?[]const u8,
};

/// The CUDA options for every target: a script that passes them where no CUDA step runs is not refused.
pub fn cuda(b: *std.Build) Cuda {
    const sm_help = "SASS targets, comma separated (default 121; 80, 86, 89, 120 and 121 build)";
    const strip_help = "No debug info in the CUDA executables: no build machine paths leave with them";
    return .{
        .nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins"),
        .fatbins = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed"),
        .sms = b.option([]const u8, "sm", sm_help) orelse "121",
        .strip = b.option(bool, "strip", strip_help) orelse false,
        .test_dump = b.option(bool, "cuda-test-dump", "Compile test-only CUDA byte records") orelse false,
        .contract_controls = b.option([]const u8, "contract-controls", "Compile-only controls image"),
        .cuobjdump = b.option([]const u8, "cuobjdump", "CUDA cuobjdump for the compiler's x3ld symbol gate"),
        .flashnext_model = b.option([]const u8, "flashnext_model", "Flash Next checkpoint directory for the embedding check"),
        .flashnext_kernels = b.option([]const u8, "flashnext_kernels", "captured Triton pack for the embedding check"),
        .flashnext_token = b.option([]const u8, "flashnext_token", "token id for the embedding check") orelse "7",
        .flashnext_ids = b.option([]const u8, "flashnext_ids", "file of prompt token ids for the Flash Next prompt check"),
        .flashnext_count = b.option([]const u8, "flashnext_count", "greedy tokens to emit") orelse "8",
        .flashnext_fixtures = b.option([]const u8, "flashnext_fixtures", "capture_sparse.sh's selection fixtures folder"),
    };
}

/// The tensor-parallel options, declared for every target: only the Linux step reads them.
pub const Tp = struct {
    nvcc: ?[]const u8,
    fatbin: ?[]const u8,
};

pub fn tp(b: *std.Build) Tp {
    return .{
        .nvcc = b.option([]const u8, "tp-nvcc", "nvcc that builds the TP mailbox kernel"),
        .fatbin = b.option([]const u8, "tp-fatbin", "absolute path of a prebuilt tp_mailbox.fatbin to embed"),
    };
}

//! tf-xpu-qwen3_asr_contract-test and its controls: each the binding's source with one edit (cl_edit.zig).
const std = @import("std");
const xpu = @import("xpu.zig");

/// A control's image: `image`'s source and options with each edit applied once.
const Control = struct { name: []const u8, image: []const u8, edits: []const [2][]const u8 };

/// The names here are the test's `Mutation` tags: a missing image is a compile error there.
pub const controls = [_]Control{
    .{ .name = "gemm_no_residual", .image = "qwen3_asr", .edits = &.{.{ "if (has_res) t = bfr(t + bf(res[at]));", "if (has_res) t = bfr(t);" }} },
    .{ .name = "gemm_bf16_sums", .image = "qwen3_asr", .edits = &.{.{ "acc[i][j] += part[i][j];", "acc[i][j] = bfr(acc[i][j] + part[i][j]);" }} },
    .{ .name = "gemv_steps_reversed", .image = "qwen3_asr_mv", .edits = &.{.{ "acc += sub_group_broadcast(part, j);", "acc += sub_group_broadcast(part, live - 1 - j);" }} },
    .{ .name = "conv_taps_transposed", .image = "qwen3_asr", .edits = &.{.{ "kh = (kk % 9) / 3, kw = kk % 3;", "kw = (kk % 9) / 3, kh = kk % 3;" }} },
    .{ .name = "flatten_f_major", .image = "qwen3_asr", .edits = &.{.{ "ci = col / f, ff = col % f;", "ff = col / c, ci = col % c;" }} },
    .{ .name = "layernorm_no_bias", .image = "qwen3_asr", .edits = &.{.{ "* rs * bf(w[i]) + bf(b[i]));", "* rs * bf(w[i]));" }} },
    .{ .name = "gelu_is_silu", .image = "qwen3_asr", .edits = &.{.{ "if (i < total) y[i] = to_bf(gelu(bf(x[i])));", "if (i < total) y[i] = to_bf(bf(x[i]) / (1.0f + exp(-bf(x[i]))));" }} },
    .{ .name = "attn_causal", .image = "qwen3_asr", .edits = &.{.{ "s = a * scale;", "s = key <= row ? a * scale : -INFINITY;" }} },
    .{ .name = "rope_turns_back", .image = "qwen3_asr", .edits = &.{.{ "const float rot = dd < 64 ? -other : other;", "const float rot = dd < 64 ? other : -other;" }} },
    .{ .name = "rms_no_weight", .image = "lfm3_gq2", .edits = &.{.{ "to_bf(bf(w[lid + 256 * k]) * bfr(v[k] * rs));", "to_bf(bfr(v[k] * rs));" }} },
    .{ .name = "attn_reads_ahead", .image = "lfm3_gq2", .edits = &.{.{ "const uint hi = min(p + 1, lo + chunk);", "const uint hi = lo + chunk;" }} },
    .{ .name = "swiglu_swapped", .image = "lfm3_gq2", .edits = &.{
        .{ "const float x = bf(a[i]);", "const float x = bf(b[i]);" },
        .{ "out[i] = to_bf(bfr(x / (1.0f + exp(-x))) * bf(b[i]));", "out[i] = to_bf(bfr(x / (1.0f + exp(-x))) * bf(a[i]));" },
    } },
};

/// The test program, in `xpu-tests`: the binding's images through `xpu`, the controls' images in their own module.
pub fn program(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, x: *std.Build.Module, core: *std.Build.Module, step: *std.Build.Step, cl: xpu.Ocloc) void {
    const editor = b.addExecutable(.{ .name = "cl_edit", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/cl_edit.zig"), .target = b.graph.host, .optimize = .ReleaseSafe }) });
    const files = b.addWriteFiles();
    var decls: std.ArrayList(u8) = .empty;
    for (controls) |ctl| {
        const k = xpu.kernelNamed(ctl.image);
        const edit = b.addRunArtifact(editor);
        edit.addFileArg(b.path(b.fmt("zig/kernels/xpu/{s}.cl", .{k.src orelse k.name})));
        const src = edit.addOutputFileArg(b.fmt("{s}.cl", .{ctl.name}));
        for (ctl.edits) |e| edit.addArgs(&e);
        var named = k;
        named.name = ctl.name;
        _ = files.addCopyFile(xpu.spirvOf(b, cl.ocloc, cl.device, cl.dpas, named, src), b.fmt("{s}.spv", .{ctl.name}));
        decls.print(b.allocator, "pub const {s} align(16) = @embedFile(\"{s}.spv\").*;\n", .{ ctl.name, ctl.name }) catch @panic("OOM");
    }
    const images = b.createModule(.{ .root_source_file = files.add("qwen3_asr_contract_images.zig", decls.items), .target = target, .optimize = optimize });
    const mod = b.createModule(.{ .root_source_file = b.path("zig/tests/xpu/qwen3_asr_contract.zig"), .target = target, .optimize = optimize, .link_libc = true });
    mod.addImport("xpu", x);
    mod.addImport("core", core);
    mod.addImport("qwen3_asr_contract_images", images);
    step.dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-xpu-qwen3_asr_contract-test", .root_module = mod }), .{}).step);
}

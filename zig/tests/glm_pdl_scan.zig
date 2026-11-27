//! Every GLM kernel opens with glm_pdl::enter() (weight writers: enter_weights), so any GLM launch may be dependent.
const std = @import("std");
const manifest = @embedFile("files.txt");
const kernel = @import("pdl_kernel_scan.zig");
const files = blk: {
    @setEvalBranchQuota(1000000);
    var out: []const []const u8 = &.{};
    var it = std.mem.tokenizeScalar(u8, manifest, '\n');
    while (it.next()) |f| out = out ++ &[_][]const u8{f};
    break :blk out;
};

/// glm_mm's matmul bodies call mm_kernel, which loads weights, then waits (reviewed weight-first placement).
const weight_first = "mm_kernel<";
/// Kernels that write weights: they wait but let no dependent in early (qmmf and glm_mm read weights first).
const writers = [_][]const u8{ "glm_mm_tile4", "glm_mm_tile6", "glm_mm_tile8", "glm_mm_tile16", "tf_glm_nvfp4_amax", "tf_glm_nvfp4_quantize", "tf_glm_quantize4" };

const open_c = "/" ++ "*"; // a C block comment's ends, spelled apart
const close_c = "*" ++ "/";

const Problem = struct { name: []const u8, why: []const u8 };

fn isWriter(name: []const u8) bool {
    for (writers) |w| if (std.mem.eql(u8, w, name)) return true;
    return false;
}

/// The kernels in `src` whose bodies break the rule; `kernels` counts every definition seen.
pub fn scan(a: std.mem.Allocator, src: []const u8, problems: *std.ArrayList(Problem), kernels: *usize) !void {
    var found: std.ArrayList(kernel.Definition) = .empty;
    try kernel.definitions(a, src, &found);
    kernels.* += found.items.len;
    for (found.items) |d| {
        const writer = isWriter(d.name);
        if (std.mem.startsWith(u8, d.opening, weight_first)) continue;
        if (writer and std.mem.startsWith(u8, d.opening, "glm_pdl::enter_weights();")) continue;
        if (!writer and std.mem.startsWith(u8, d.opening, "glm_pdl::enter();")) continue;
        try problems.append(a, .{ .name = d.name, .why = if (writer) "a weight writer must open with glm_pdl::enter_weights();" else "must open with glm_pdl::enter();" });
    }
}

/// A weight-first prologue between `from` and `to`: its weight loads before grid_wait(), its input loads only after.
fn prologue(src: []const u8, from: []const u8, to: []const u8, weights: []const u8, inputs: []const u8) bool {
    const a = std.mem.indexOf(u8, src, from) orelse return false;
    const b = std.mem.indexOfPos(u8, src, a, to) orelse return false;
    const seg = src[a..b];
    const w = std.mem.indexOf(u8, seg, "grid_wait();") orelse return false;
    const before = seg[0..w];
    return std.mem.indexOf(u8, before, weights) != null and std.mem.indexOf(u8, before, inputs) == null and
        std.mem.indexOf(u8, seg[w..], inputs) != null;
}

/// The weight copies between `from` and the lambda's end name no input row (`x`, `ldx`): they may run before the wait.
fn weightsOnly(src: []const u8, from: []const u8) bool {
    const a = std.mem.indexOf(u8, src, from) orelse return false;
    const b = std.mem.indexOfPos(u8, src, a, "\n    };") orelse return false;
    var it = std.mem.tokenizeAny(u8, src[a..b], " \t\n()[]{}<>,;:+-*/&|!=?.");
    while (it.next()) |t| if (std.mem.eql(u8, t, "x") or std.mem.eql(u8, t, "ldx")) return false;
    return true;
}

fn problemsIn(src: []const u8) !usize {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problems: std.ArrayList(Problem) = .empty;
    var n: usize = 0;
    try scan(arena.allocator(), src, &problems, &n);
    return problems.items.len;
}

test "GLM kernels: every body opens with the dependency wait" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problems: std.ArrayList(Problem) = .empty;
    var kernels: usize = 0;
    inline for (files) |file| {
        const before = problems.items.len;
        try scan(arena.allocator(), @embedFile(file), &problems, &kernels);
        for (problems.items[before..]) |p| std.debug.print("{s}: {s} {s}\n", .{ file, p.name, p.why });
    }
    try std.testing.expect(kernels >= 100);
    try std.testing.expectEqual(@as(usize, 0), problems.items.len);
}

test "weight-first prologues: weights before the wait, the kernel before's outputs only after it" {
    try std.testing.expect(prologue(@embedFile("glm_mm.cu"), "grid_launch();", "for (int it", "loadW(", "loadX("));
    try std.testing.expect(prologue(@embedFile("qmmf_body.cuh"), "#ifdef TF_QMMF_WFIRST", "#else", "load(s, g0 + s, 2)", "load(s, g0 + s, 1)"));
    try std.testing.expect(!prologue("grid_launch(); loadX(0); grid_wait(); loadW(0); for (int it", "grid_launch();", "for (int it", "loadW(", "loadX("));
    try std.testing.expect(!prologue("grid_launch(); loadW(0); loadX(0); grid_wait(); loadX(1); for (int it", "grid_launch();", "for (int it", "loadW(", "loadX("));
    try std.testing.expect(weightsOnly(@embedFile("glm_mm.cu"), "auto loadW = [&]"));
    try std.testing.expect(weightsOnly(@embedFile("qmmf_body.cuh"), "if (!(what & 2)) return;"));
    try std.testing.expect(!weightsOnly("auto loadW = [&](int st) { cp16(pw, x + st * 8);\n    };", "auto loadW = [&]"));
    try std.testing.expect(!weightsOnly("if (!(what & 2)) return; cp16(pw, w + row * ldx);\n    };", "if (!(what & 2)) return;"));
}

test "GLM kernels: late, conditional and missing waits fail; a macro body and a writer pass" {
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("extern \"C\" __global__ void tf_glm_a(int* x) {\n  int v = x[0];\n  glm_pdl::enter();\n  x[1] = v;\n}\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("extern \"C\" __global__ void tf_glm_b(int* x, int c) {\n  if (c) glm_pdl::enter();\n  x[0] = 1;\n}\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("extern \"C\" __global__ void __launch_bounds__(128) tf_glm_c(int* x) { x[0] = 1; }\n"));
    try std.testing.expectEqual(@as(usize, 0), try problemsIn("#define K(NAME)  \\\n  extern \"C\" __global__ void NAME(int* x) { glm_pdl::enter(); \\\n    x[0] = 1;  \\\n  }\n"));
    try std.testing.expectEqual(@as(usize, 0), try problemsIn("extern \"C\" __global__ void glm_mm_tile4(int* x) {\n  // the tiles\n  glm_pdl::enter_weights();\n  x[0] = 1;\n}\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("extern \"C\" __global__ void glm_mm_tile6(int* x) {\n  glm_pdl::enter();\n  x[0] = 1;\n}\n"));
    try std.testing.expectEqual(@as(usize, 0), try problemsIn("template __global__ void k<1>(int*);\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("__global__ void tf_glm_d(int* x) " ++ open_c ++ " shape; " ++ close_c ++ " { x[0] = 1; }\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("__global__ void tf_glm_e(int* x) // shape; rows\n{\n  x[0] = 1;\n}\n"));
    try std.testing.expectEqual(@as(usize, 1), try problemsIn("__global__ void tf_glm_f(int* x " ++ open_c ++ " output); " ++ close_c ++ ") { x[0] = 1; }\n"));
    try std.testing.expectEqual(@as(usize, 0), try problemsIn("__global__ void tf_glm_g(int* x) {\n  glm_pdl::enter();\n  asm volatile(\"{ .reg .u32 r; }\" ::: \"memory\");\n}\n"));
}

test "GLM kernels: leading block comments and later literals keep a valid entry wait visible" {
    try std.testing.expectEqual(@as(usize, 0), try problemsIn("extern \"C\" __global__ void tf_glm_h(int* x) {\n  " ++ open_c ++ " routing belongs to predecessors " ++ close_c ++ "\n  glm_pdl::enter();\n  const char* text = \"braces } and glm_pdl::enter(); in text\";\n  const char quote = '\\'';\n  x[0] = 1;\n}\n"));
}

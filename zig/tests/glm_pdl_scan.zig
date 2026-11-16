//! Every GLM kernel opens with glm_pdl::enter() (weight writers: enter_weights), so any GLM launch may be dependent.
const std = @import("std");
const manifest = @embedFile("files.txt");
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

fn ident(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '#';
}

/// The index past a comment or a string or character literal starting at `k`, or null when none starts there.
fn skip(s: []const u8, k: usize) ?usize {
    const rest = s[k..];
    if (std.mem.startsWith(u8, rest, "//")) return std.mem.indexOfScalarPos(u8, s, k, '\n') orelse s.len;
    if (std.mem.startsWith(u8, rest, open_c)) return if (std.mem.indexOfPos(u8, s, k + 2, close_c)) |e| e + 2 else s.len;
    if (rest.len > 0 and (rest[0] == '"' or rest[0] == '\'')) {
        var e = k + 1;
        while (e < s.len and s[e] != rest[0]) e += if (s[e] == '\\') 2 else 1;
        return @min(e + 1, s.len);
    }
    return null;
}

/// The index of the bracket closing the one at `i` (comments and literals skipped), or null when unbalanced.
fn close(s: []const u8, i: usize, open: u8, shut: u8) ?usize {
    var depth: usize = 0;
    var k = i;
    while (k < s.len) {
        if (skip(s, k)) |next| {
            k = next;
            continue;
        }
        if (s[k] == open) depth += 1 else if (s[k] == shut) {
            depth -= 1;
            if (depth == 0) return k;
        }
        k += 1;
    }
    return null;
}

/// A body's text with macro line breaks, line comments and leading space dropped.
fn opening(a: std.mem.Allocator, body: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var k: usize = 0;
    while (k < body.len) {
        if (body[k] == '\\' and k + 1 < body.len and body[k + 1] == '\n') {
            k += 2;
        } else if (body[k] == '/' and k + 1 < body.len and body[k + 1] == '/') {
            while (k < body.len and body[k] != '\n') k += 1;
        } else {
            try out.append(a, body[k]);
            k += 1;
        }
    }
    return std.mem.trimStart(u8, out.items, " \t\r\n");
}

fn isWriter(name: []const u8) bool {
    for (writers) |w| if (std.mem.eql(u8, w, name)) return true;
    return false;
}

/// The kernels in `src` whose bodies break the rule; `kernels` counts every definition seen.
pub fn scan(a: std.mem.Allocator, src: []const u8, problems: *std.ArrayList(Problem), kernels: *usize) !void {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, src, at, "__global__")) |g| {
        at = g + "__global__".len;
        const line = src[(if (std.mem.lastIndexOfScalar(u8, src[0..g], '\n')) |n| n + 1 else 0)..g];
        if (std.mem.indexOf(u8, line, "//") != null) continue;
        var i = at;
        var name: []const u8 = "";
        while (i < src.len and src[i] != '(') {
            if (ident(src[i])) {
                const start = i;
                while (i < src.len and ident(src[i])) i += 1;
                const tok = src[start..i];
                if (std.mem.eql(u8, tok, "__launch_bounds__")) {
                    const p = std.mem.indexOfScalarPos(u8, src, i, '(') orelse return error.Unparsed;
                    i = (close(src, p, '(', ')') orelse return error.Unparsed) + 1;
                } else name = tok;
            } else if (src[i] == '<') {
                i = (close(src, i, '<', '>') orelse return error.Unparsed) + 1;
            } else i += 1;
        }
        const params_end = close(src, i, '(', ')') orelse return error.Unparsed;
        var k = params_end + 1;
        while (k < src.len and src[k] != '{' and src[k] != ';') k = skip(src, k) orelse k + 1;
        if (k >= src.len or src[k] == ';') continue; // a declaration or an explicit instantiation
        const end = close(src, k, '{', '}') orelse return error.Unparsed;
        kernels.* += 1;
        const head = try opening(a, src[k + 1 .. end]);
        const writer = isWriter(name);
        if (std.mem.startsWith(u8, head, weight_first)) continue;
        if (writer and std.mem.startsWith(u8, head, "glm_pdl::enter_weights();")) continue;
        if (!writer and std.mem.startsWith(u8, head, "glm_pdl::enter();")) continue;
        try problems.append(a, .{ .name = name, .why = if (writer) "a weight writer must open with glm_pdl::enter_weights();" else "must open with glm_pdl::enter();" });
    }
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

//! Address-only variants of the existing expert prompt kernels, with their tensor arithmetic unchanged.
const std = @import("std");

fn replace(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8, count: usize) ![]const u8 {
    if (std.mem.count(u8, text, from) != count) return error.FsrPromptPatch;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// Bind only a named gather entry; the complete library also contains unrelated dense and attention kernels.
fn kernel(a: std.mem.Allocator, text: []const u8, name: []const u8) ![]const u8 {
    const marker = try std.fmt.allocPrint(a, "[[kernel]] void {s}(", .{name});
    if (std.mem.count(u8, text, marker) != 1) return error.FsrPromptPatch;
    const start = std.mem.indexOf(u8, text, marker).?;
    const open = std.mem.indexOfScalarPos(u8, text, start, '{') orelse return error.FsrPromptPatch;
    var level: usize = 1;
    const end = for (text[open + 1 ..], open + 1..) |c, i| {
        if (c == '{') level += 1;
        if (c == '}') level -= 1;
        if (level == 0) break i + 1;
    } else return error.FsrPromptPatch;
    const parameter = "uint sg [[simdgroup_index_in_threadgroup]]";
    const binding = "const device int* FSR_MAP [[buffer(27)]], constant uint* FSR_STRIDES [[buffer(28)]], " ++ parameter;
    var part = try replace(a, text[start..end], parameter, binding, 1);
    part = try replace(a, part, "[[kernel]]", "[[max_total_threads_per_threadgroup(128)]] [[kernel]]", 1);
    return std.mem.concat(a, u8, &.{ text[0..start], part, text[end..] });
}

pub fn patch(a: std.mem.Allocator, original: []const u8, wide: bool) ![]const u8 {
    if (std.mem.indexOf(u8, original, "FSR_MAP") != null) return error.FsrPromptPatch;
    const marker = "// y[r] = x[r] W_e^T for rows sorted";
    const at = std.mem.indexOf(u8, original, marker) orelse return error.FsrPromptPatch;
    var text: []const u8 = original[at..];
    text = try replace(a, text, "uint3 tg, uint sg, uint lane)", "uint3 tg, uint sg, uint lane, const device int* FSR_MAP, constant uint* FSR_STRIDES)", 2);
    text = try replace(a, text, "tile, tg, sg, lane);", "tile, tg, sg, lane, FSR_MAP, FSR_STRIDES);", if (wide) 1 else 3);
    text = try replace(a, text, "tile_g, tile_u, tg, sg, lane);", "tile_g, tile_u, tg, sg, lane, FSR_MAP, FSR_STRIDES);", if (wide) 1 else 2);
    const row = "  const long wrow = long(expert) * N + min(col + t / 2, N - 1);";
    const gather = "  const long wrow = long(min(col + t / 2, N - 1));\n  const size_t slot = size_t(FSR_MAP[expert]);\n  W += slot * FSR_STRIDES[0]; S += slot * FSR_STRIDES[1]; B += slot * FSR_STRIDES[2];";
    const gu = if (wide)
        "  const long wrow = long(min(col + t / 2, N - 1));\n  const size_t slot = size_t(FSR_MAP[expert]);\n  WG_ += slot * FSR_STRIDES[0]; SG += slot * FSR_STRIDES[1]; BG += slot * FSR_STRIDES[2];\n  WU += slot * FSR_STRIDES[3]; SU += slot * FSR_STRIDES[4]; BU += slot * FSR_STRIDES[5];"
    else
        "  const long wrow = long(min(col + t / 2, N - 1));\n  const size_t slot = size_t(FSR_MAP[expert]);\n  WG += slot * FSR_STRIDES[0]; SG += slot * FSR_STRIDES[1]; BG += slot * FSR_STRIDES[2];\n  WU += slot * FSR_STRIDES[3]; SU += slot * FSR_STRIDES[4]; BU += slot * FSR_STRIDES[5];";
    if (std.mem.count(u8, text, row) != 2) return error.FsrPromptPatch;
    const first = std.mem.indexOf(u8, text, row).?;
    text = try std.mem.concat(a, u8, &.{ text[0..first], gather, text[first + row.len ..] });
    text = try replace(a, text, row, gu, 1);
    const names: []const []const u8 = if (wide)
        &.{ "tf_gather_qmmw_nax_##BM", "tf_gather_guw_nax_##BM" }
    else
        &.{ "tf_gather_qmm6_nax_32", "tf_gather_qmm6_nax_64", "tf_gather_qmm6_nax_128", "tf_gather_gu6_nax_32", "tf_gather_gu6_nax_64" };
    for (names) |name| text = try kernel(a, text, name);
    return std.mem.concat(a, u8, &.{ original[0..at], text });
}

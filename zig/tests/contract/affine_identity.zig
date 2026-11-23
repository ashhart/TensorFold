//! Finite bit-identity fixtures and the frozen affine_mm source they replace.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");

const mm = tf.affine_mm;
const Affine = contract.affine.Affine;
const c = contract.spec;

/// 082a90291f's raw core shader and its separately frozen, unchanged GLU epilogue.
const base_main = @embedFile("baselines/affine_mm_main.metal.txt");
const base_glu = @embedFile("baselines/affine_mm_glu_main.metal.txt");

pub const Source = enum { baseline, current, rounding_control };

/// The old generator, including its device-conditional M3 macro, but never the new exact-nonfinite macro.
pub fn source(device: mtl.Device, a: std.mem.Allocator, f: mm.Format, which: Source) ![]u8 {
    if (which != .baseline) {
        var format = f;
        format.exact_nonfinite = true;
        const src = try mm.source(device, a, format);
        if (which == .current) return src;
        defer a.free(src);
        return roundOne(a, src, f.group != 128);
    }
    const raw = try std.mem.concat(a, u8, &.{ base_main, base_glu });
    defer a.free(raw);
    const body = try tf.frags.source(device, a, raw);
    defer a.free(body);
    const glu = switch (f.glu) {
        .plain => "#define TF_GLU_PLAIN 1\n",
        .clipped_silu => "#define TF_GLU_ROUND_SILU 1\n",
        .clipped => "",
    };
    const m3 = if (f.glm_m3_fast and !device.tensorUnits()) "#define TF_GLM_M3_FAST 1\n" else "";
    const map = if (f.mapped_experts) "#define TF_AM_MAPPED_EXPERTS 1\n" else "";
    const out = if (f.out_f32) "float" else "bfloat";
    return std.fmt.allocPrint(a, "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T {s}\n{s}{s}{s}{s}", .{ f.bits, f.group, out, glu, map, m3, body });
}

/// One source edit: the helper is first, so g128 uses it; smaller groups use the second, live main copy.
fn roundOne(a: std.mem.Allocator, src: []const u8, second: bool) ![]u8 {
    const from = "const float p0 = raw[m][0] ? exact[m][j][c] : part[m][j][c];";
    const to = "const float p0 = bfloat(raw[m][0] ? exact[m][j][c] : part[m][j][c]);";
    const first = std.mem.indexOf(u8, src, from) orelse return error.ControlEditMissing;
    const at = if (second) std.mem.indexOfPos(u8, src, first + from.len, from) orelse return error.ControlEditMissing else first;
    return std.mem.concat(a, u8, &.{ src[0..at], to, src[at + from.len ..] });
}

/// Compile all entry points with the same options and ordering as affine_mm's ordinary contract cases.
pub fn compile(device: mtl.Device, src: []const u8) !mm.Pipes {
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: mm.Pipes = undefined;
    for (mm.names, 0..) |name, i| pipes[i] = try mtl.Pipeline.init(device, lib, name, false);
    return pipes;
}

/// Every row/group has a large, exactly cancelling offset pair; no nonfinite group selects the raw path.
pub fn plant(f: Affine) void {
    const k = f.shape.k;
    for (0..f.shape.rows) |r| for (0..k / f.group) |group| {
        const at = r * k + group * f.group;
        f.x[at] = c.toBf16(0x1p120);
        f.x[at + 1] = c.toBf16(-0x1p120);
    };
    for (0..f.shape.n) |n| for (0..k / f.group) |group| {
        const at = n * k + group * f.group;
        f.codes[at] = 15;
        f.codes[at + 1] = 15;
    };
}

//! glm-loader-gate's csf0_v1 experts: a layer's upload made on the host from the checkpoint, as experts32 packs it.
const std = @import("std");
const cuda = @import("cuda");

const Matrix = cuda.experts32.Matrix;
const Y = cuda.experts32.Nvfp4b32Experts;
const at = "model.language_model.layers.{d}.mlp.experts.{d}.";

/// Matrix `module`'s rows [lo, hi) and inputs [klo, khi), its codes and offsets copies of `p.gpa`'s (`free` them).
fn cut(p: anytype, module: []const u8, lo: usize, hi: usize, klo: usize, khi: usize) !Matrix {
    const w = try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight", .{module}));
    const b = try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight_scale.csf0_bases", .{module}));
    const o = try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight_scale.csf0_codes", .{module}));
    const kk = 2 * w.dim(1);
    if (w.dtype != .u8 or lo > hi or hi > w.dim(0) or klo > khi or khi > kk or klo % 64 != 0 or (khi - klo) % 64 != 0) return error.CutOutside;
    const n = hi - lo;
    const k = khi - klo;
    const codes = try p.gpa.alloc(u8, n * k / 2);
    errdefer p.gpa.free(codes);
    const offsets = try p.gpa.alloc(u8, n * (k / 64));
    for (0..n) |r| {
        @memcpy(codes[r * k / 2 ..][0 .. k / 2], w.bytes[(lo + r) * kk / 2 + klo / 2 ..][0 .. k / 2]);
        @memcpy(offsets[r * (k / 64) ..][0 .. k / 64], o.bytes[(lo + r) * (kk / 64) + klo / 64 ..][0 .. k / 64]);
    }
    return .{ .codes = codes, .bases = b.bytes[lo..hi], .offsets = offsets };
}

fn free(p: anytype, m: Matrix) void {
    p.gpa.free(m.codes);
    p.gpa.free(m.offsets);
}

/// Layer `i`'s `experts.up` or `experts.down` upload on the rank of share `s`, as `p.gpa`'s bytes.
pub fn made(p: anytype, c: anytype, s: anytype, i: u32, up: bool) ![]u8 {
    const d: usize = c.hidden;
    const rank = s.moe;
    const ni = rank.len();
    const per = if (up) Y.bytes(1, ni, d, 2) / 4 else Y.bytes(1, d, ni, 1) / 4;
    const out = try p.gpa.alloc(u32, c.experts * per);
    defer p.gpa.free(out);
    for (0..c.experts) |e| {
        const block = out[e * per ..][0..per];
        if (up) {
            const g = try cut(p, try std.fmt.allocPrint(p.a, at ++ "gate_proj", .{ i, e }), rank.lo, rank.hi, 0, d);
            defer free(p, g);
            const u = try cut(p, try std.fmt.allocPrint(p.a, at ++ "up_proj", .{ i, e }), rank.lo, rank.hi, 0, d);
            defer free(p, u);
            Y.packGateUp(block, g, u, ni, d);
        } else {
            const m = try cut(p, try std.fmt.allocPrint(p.a, at ++ "down_proj", .{ i, e }), 0, d, rank.lo, rank.hi);
            defer free(p, m);
            Y.packOne(block, m, d, ni);
        }
    }
    return p.gpa.dupe(u8, std.mem.sliceAsBytes(out));
}

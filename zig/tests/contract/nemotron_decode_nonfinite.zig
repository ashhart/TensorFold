//! Nonfinite attention follows the independent plain program's complete BF16 bytes, including NaN payloads.
const std = @import("std");
const g = @import("contract").gates;
pub const rule = "identical BF16 bytes, including NaN payloads, to independent token-at-a-time attention";
pub fn check(p: anytype, gate: g.Gate) !?[]const u8 {
    if (gate != .nonfinite) return null;
    const got = p.rig.bytes(p.inner.out, p.inner.out_bytes);
    if (!std.mem.eql(u8, got, try p.replaced())) return "nonfinite bytes differ from independent token-at-a-time attention";
    return null;
}
pub fn flipFinite(bits: u16) u16 {
    return if (bits & 0x7f80 != 0x7f80) bits ^ 1 else bits;
}
pub fn controlSource(a: std.mem.Allocator, source: []const u8) ![]u8 {
    const from = "x[(size_t(t) * heads + h) * hd + d] = o[(size_t(h) * rows + t) * hd + d];";
    if (std.mem.count(u8, source, from) != 1) return error.ControlEditMissing;
    const to =
        \\ushort bits = reinterpret_cast<const device ushort*>(o)[(size_t(h) * rows + t) * hd + d];
        \\  if ((bits & 0x7f80u) != 0x7f80u) bits ^= 1u;
        \\  reinterpret_cast<device ushort*>(x)[(size_t(t) * heads + h) * hd + d] = bits;
    ;
    return std.mem.replaceOwned(u8, a, source, from, to);
}

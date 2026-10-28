//! Planted controls for the pair seam's four kernels: one fault in pair_rows.metal's text, and the gate to catch it.
const std = @import("std");
const contract = @import("contract");
const sources = @import("kernel_sources");

pub const Mutation = enum {
    none,
    stage_peer_half,
    gather_own_halves,
    argmax_high_tie,
    argmax_no_flag,
    pick_high_tie,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .stage_peer_half => "control: metal pair stage, this rank stages the peer's half",
            .gather_own_halves => "control: metal pair gather, this rank writes its own half on both sides",
            .argmax_high_tie => "control: metal pair argmax, a tie goes to the highest id",
            .argmax_no_flag => "control: metal pair argmax, a nonfinite logit is not flagged",
            .pick_high_tie => "control: metal pair pick, a tie goes to the higher id",
        };
    }

    /// The gate that must go red.
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return switch (m) {
            .none => null,
            .argmax_no_flag => .nonfinite,
            else => .shapes,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            // The stage's source is a frame row of `stride` = 2*width: the peer's half is in bounds and different.
            .stage_peer_half => .{ .from = "const ulong in = ulong(r) * a.stride;", .to = "const ulong in = ulong(r) * a.stride + a.width;" },
            .gather_own_halves => .{ .from = "out[got + i] = ushort(word >> ((i & 1) * 16));", .to = "out[got + i] = mine[mrow + i];" },
            .argmax_high_tie => .{ .from = "if (ob > best || (ob == best && oa < at)) best = ob, at = oa;", .to = "if (ob > best || (ob == best && oa > at)) best = ob, at = oa;" },
            .argmax_no_flag => .{ .from = "if (!isfinite(v)) { bad = true; continue; }", .to = "if (false) { bad = true; continue; }" },
            .pick_high_tie => .{ .from = "(vp > vm || (vp == vm && ap < am)) ? ap : am;", .to = "(vp > vm || (vp == vm && ap > am)) ? ap : am;" },
        };
    }
};

/// pair_rows.metal's text for the kernel under test; a control's one edit must change it, or it proves nothing.
pub fn source(a: std.mem.Allocator, m: Mutation) ![]const u8 {
    if (m == .none) return sources.core_pair_rows;
    const e = m.edit();
    if (std.mem.indexOf(u8, sources.core_pair_rows, e.from) == null) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, sources.core_pair_rows, e.from, e.to);
}

//! The run-time weight forms built at load from the loaded weights: strips, lanes, bf16 mHC, expert tables.

const std = @import("std");
const cuda = @import("cuda");
const load = @import("cuda_weights.zig");
const plan = @import("plan.zig");
const prepare = @import("prepare.zig");
const named = @import("named.zig");

pub const Source = struct {
    d: *const cuda.Driver,
    weights: *const load.Weights,
    plan: *const plan.Plan,

    fn download(s: *const Source, a: std.mem.Allocator, name: []const u8) ![]u8 {
        const t = s.weights.get(name) orelse return error.MissingWeight;
        const host = try a.alloc(u8, t.len);
        try cuda.DeviceBuffer.download(.{ .d = s.d, .ptr = t.ptr, .len = t.len }, 0, host);
        return host;
    }

    /// (K tiles, N tiles, int16 words a tile) of a group's trellis.
    fn trellis(s: *const Source, w: []const u8) ![3]usize {
        var b: [256]u8 = undefined;
        const t = s.weights.get(try std.fmt.bufPrint(&b, "{s}.trellis", .{w})) orelse return error.MissingWeight;
        if (t.rank != 3) return error.NotTrellis;
        return .{ t.shape[0], t.shape[1], t.shape[2] };
    }
};

/// A persistent role's initial bytes when it is a weight form (null: state, scratch, tables the forward fills).
fn buildArena(a: std.mem.Allocator, s: *const Source, role: []const u8) !?[]const u8 {
    if (!std.mem.startsWith(u8, role, "s.")) return null;
    const lanes = std.mem.endsWith(u8, role, ".lanes");
    if (lanes or std.mem.endsWith(u8, role, ".T")) {
        const w = role[2 .. role.len - (if (lanes) ".lanes".len else ".T".len)];
        const tr = try s.download(a, try std.fmt.allocPrint(a, "{s}.trellis", .{w}));
        const g = try s.trellis(w);
        const st = try prepare.strips(a, tr, g[0], g[1], g[2]);
        return if (lanes) try prepare.lanes(a, st, g[0], g[2] / 8) else st;
    }
    if (std.mem.endsWith(u8, role, ".fn16")) {
        const src = try s.download(a, try std.fmt.allocPrint(a, "{s}.fn", .{role[2 .. role.len - 5]}));
        const out = try a.alloc(u8, src.len / 2);
        for (0..src.len / 4) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], named.bf16Round(@bitCast(std.mem.readInt(u32, src[4 * i ..][0..4], .little))), .little);
        return out;
    }
    if (!std.mem.startsWith(u8, role, "s.L")) return null;
    const ex = std.mem.indexOf(u8, role, ".ex.") orelse return null;
    const layer = role[2..ex];
    const what = role[ex + 4 ..];
    const proj: usize = switch (what[what.len - 1]) {
        'g' => 0,
        'u' => 2,
        'd' => 1,
        else => return null,
    };
    const projs = [_][]const u8{ "w1", "w2", "w3" };
    const w = projs[proj];
    if (std.mem.startsWith(u8, what, "suh") or std.mem.startsWith(u8, what, "svh")) {
        const kind = what[0..3];
        const routed = try s.download(a, try std.fmt.allocPrint(a, "{s}.moe.{s}.{s}", .{ layer, w, kind }));
        const shared = try s.download(a, try std.fmt.allocPrint(a, "{s}.moe.shared.0.{s}.{s}", .{ layer, w, kind }));
        return try std.mem.concat(a, u8, &.{ routed, shared });
    }
    const L = try std.fmt.parseInt(u32, layer[1..], 10);
    const lp = for (s.plan.layers) |*l| {
        if (l.index == L) break l;
    } else return error.LayerNotPlanned;
    const lay = &lp.experts[proj].layout;
    const E = lay.words.len;
    const shared = try std.fmt.allocPrint(a, "{s}.moe.shared.0.{s}", .{ layer, w });
    if (std.mem.startsWith(u8, what, "k2")) {
        const out = try a.alloc(u8, 4 * (E + 1));
        for (lay.words, 0..) |words, i| std.mem.writeInt(i32, out[4 * i ..][0..4], @intCast(words / 8), .little);
        std.mem.writeInt(i32, out[4 * E ..][0..4], @intCast((try s.trellis(shared))[2] / 8), .little);
        return out;
    }
    if (std.mem.startsWith(u8, what, "tp")) {
        // trellis address = ragged base + offset, shared expert last; a uniform stack ("<w>.trellis"): same offsets
        const data = s.weights.get(try std.fmt.allocPrint(a, "{s}.moe.{s}.data", .{ layer, w })) orelse uniform: {
            const t = s.weights.get(try std.fmt.allocPrint(a, "{s}.moe.{s}.trellis", .{ layer, w })) orelse return error.MissingWeight;
            if (t.len != lay.trellisBytes()) return error.ExpertStackSize;
            break :uniform t;
        };
        const sh = s.weights.get(try std.fmt.allocPrint(a, "{s}.trellis", .{shared})) orelse return error.MissingWeight;
        const out = try a.alloc(u8, 8 * (E + 1));
        for (0..E) |e| std.mem.writeInt(u64, out[8 * e ..][0..8], data.ptr + lay.byteOffset(e), .little);
        std.mem.writeInt(u64, out[8 * E ..][0..8], sh.ptr, .little);
        return out;
    }
    return null;
}

// Intermediates belong to this temporary arena. The caller receives exactly one independently owned result.
pub fn build(gpa: std.mem.Allocator, s: *const Source, role: []const u8) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const result = (try buildArena(arena.allocator(), s, role)) orelse return null;
    return try gpa.dupe(u8, result);
}

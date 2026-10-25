//! Planted contract faults transform actual HC output and accumulation statements; production text is unchanged.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub const Control = enum { production, nan, repeat, tail, sums };
pub fn source(a: std.mem.Allocator, text: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, text);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    if (control == .sums) return edit.once(a, text, "total += part[k][t];", "total = float(bfloat(total + part[k][t]));");
    const outputs = [_][2][]const u8{
        .{ "SSP[(r * NT + j) * S + t]", "total" },
        .{ "NORMED[size_t(r) * W + e]", "bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e])" },
        .{ "MIXED[size_t(r) * D + d]", "bfloat(total / float(S))" },
        .{ "A[i]", "bfloat(pf_bsilu(float(G[i])) * float(U[i]))" },
    };
    var out = text;
    for (outputs) |item| {
        const before = try std.fmt.allocPrint(mem, "{s} = {s};", .{ item[0], item[1] });
        const expr = switch (control) {
            .nan => try std.fmt.allocPrint(mem, "(isnan(float({s})) ? 0.0f : float({s}))", .{ item[1], item[1] }),
            .repeat => try std.fmt.allocPrint(mem, "(float({s}) + float(TF_REPEAT))", .{item[1]}),
            .tail => item[1],
            else => unreachable,
        };
        const after = try std.fmt.allocPrint(mem, "{s}{s} = {s};", .{ if (control == .tail) "if (false) " else "", item[0], expr });
        out = try edit.once(mem, out, before, after);
    }
    if (control == .repeat) {
        const args = [_][]const u8{ "uint simdgroup_index_in_threadgroup [[", "uint2 pos [[", "uint i [[" };
        for (args) |arg| {
            const count = std.mem.count(u8, out, arg);
            if (count != @as(usize, if (std.mem.startsWith(u8, arg, "uint2")) 2 else 1)) return error.HybridGeometrySourceChanged;
            out = try std.mem.replaceOwned(u8, mem, out, arg, try std.mem.concat(mem, u8, &.{ "constant uint& TF_REPEAT [[buffer(31)]], ", arg }));
        }
    }
    return a.dupe(u8, out);
}

//! Owned assembled libraries, deduplicated by source bytes and compiler options.
const std = @import("std");
const mtl = @import("metal");
const precision = @import("../src/core/attention_precision.zig");
const ks = @import("kernel_sources");
pub const Entry = struct { family: []const u8, name: []const u8, text: []const u8, options: mtl.CompileOptions, packed_inline: bool };
pub const Set = struct {
    a: std.mem.Allocator,
    units: bool,
    generation: u32,
    entries: std.ArrayList(Entry) = .empty,

    pub fn add(s: *Set, family: []const u8, name: []const u8, text: []const u8, options: mtl.CompileOptions) !void {
        for (s.entries.items) |e| if (std.meta.eql(e.options, options) and std.mem.eql(u8, e.text, text)) return;
        try s.entries.append(s.a, .{ .family = family, .name = try s.a.dupe(u8, name), .text = try s.a.dupe(u8, text), .options = options, .packed_inline = std.mem.indexOf(u8, text, "uint4b_format") != null });
    }

    pub fn raw(s: *Set, family: []const u8, name: []const u8, text: []const u8) !void {
        try s.add(family, name, text, mtl.CompileOptions.mlx());
    }

    pub fn fragments(s: *Set, family: []const u8, name: []const u8, text: []const u8) !void {
        const adapted = try precision.source(s.a, s.units, text, ks.nax, false);
        try s.raw(family, name, adapted);
        if (std.mem.indexOf(u8, text, "TF_ATTN_PV_FP32") != null) {
            const full = try precision.source(s.a, s.units, text, ks.nax, true);
            try s.raw(family, try std.fmt.allocPrint(s.a, "{s}-pv-fp32", .{name}), full);
        }
    }
};

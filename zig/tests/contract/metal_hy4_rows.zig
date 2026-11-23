//! Hy4-flash's row windows, greedy pick, Engram kernels and embedding under the contract (metal_hy4.zig's group).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");
const project = @import("metal_hy4_rows_project.zig");
const pick = @import("metal_hy4_rows_pick.zig");
const engram = @import("metal_hy4_rows_engram.zig");
const apply = @import("metal_hy4_rows_apply.zig");
const embed = @import("metal_hy4_rows_embed.zig");

const g = contract.gates;

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    rows: h.fl.List(project.Rows) = .{},
    picks: h.fl.List(pick.Pick) = .{},
    lookups: h.fl.List(engram.Lookup) = .{},
    gathers: h.fl.List(engram.Gather) = .{},
    applies: h.fl.List(apply.Apply) = .{},
    embeds: h.fl.List(embed.Embed) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const config = try h.text(s.a);
        defer config.deinit();
        const vocab: usize = config.value.text_config.vocab_size;
        try s.rowsLibrary(only, vocab);
        for ([_]bool{ true, false }) |fp8| if (h.fl.wanted(only, engram.Lookup.base(fp8))) {
            try s.lookups.add(s.a, s.kernels, try engram.Lookup.init(s.a, s.device, s.queue, fp8, null));
            if (fp8) try s.lookups.add(s.a, s.kernels, try engram.Lookup.init(s.a, s.device, s.queue, fp8, text.scale));
        };
        for ([_]u8{ 4, 8 }) |bits| if (h.fl.wanted(only, engram.Gather.base(bits))) {
            try s.gathers.add(s.a, s.kernels, try engram.Gather.init(s.a, s.device, s.queue, bits, null));
            if (bits == 4) try s.gathers.add(s.a, s.kernels, try engram.Gather.init(s.a, s.device, s.queue, bits, text.group));
        };
        for ([_]bool{ false, true }) |dots| if (h.fl.wanted(only, apply.Apply.base(dots))) {
            try s.applies.add(s.a, s.kernels, try apply.Apply.init(s.a, s.device, s.queue, dots, null));
            if (!dots) try s.applies.add(s.a, s.kernels, try apply.Apply.init(s.a, s.device, s.queue, dots, text.gain));
        };
        // no other contract entry covers embed_norm's tf_embed_b4_g64 or tf_embed_b8_g64
        for ([_]u8{ 4, 8 }) |bits| if (h.fl.wanted(only, embed.Embed.base(bits))) {
            try s.embeds.add(s.a, s.kernels, try embed.Embed.init(s.a, s.device, s.queue, bits, null, vocab));
            if (bits == 4) try s.embeds.add(s.a, s.kernels, try embed.Embed.init(s.a, s.device, s.queue, bits, text.round, vocab));
        };
    }

    /// Rows.load's pipelines: the 2-row-per-simdgroup windows at 4 and 8 bits (the only ones it builds) and the argmax.
    fn rowsLibrary(s: *Suite, only: ?[]const u8, vocab: usize) !void {
        for ([_]u8{ 4, 8 }) |bits| if (h.fl.wanted(only, project.Rows.base(bits))) {
            const faults = [_]?text.Fault{ null, if (bits == 4) text.tail else null };
            for (faults, 0..) |fault, i| if (i == 0 or fault != null) {
                try s.rows.add(s.a, s.kernels, try project.Rows.init(s.a, s.device, s.queue, bits, fault, vocab));
            };
        };
        if (!h.fl.wanted(only, pick.Pick.base)) return;
        for ([_]?text.Fault{ null, text.tie, text.nan_wins }) |fault| {
            try s.picks.add(s.a, s.kernels, try pick.Pick.init(s.a, s.device, s.queue, fault, vocab));
        }
    }

    pub fn deinit(s: *Suite) void {
        s.rows.deinit(s.a);
        s.picks.deinit(s.a);
        s.lookups.deinit(s.a);
        s.gathers.deinit(s.a);
        s.applies.deinit(s.a);
        s.embeds.deinit(s.a);
    }
};

/// Gate 5's texts for this group's records (`hy4 rows `, `hy4 engram `, `hy4 embed_norm `); null for any other name.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?h.fl.Texts {
    return text.limitsText(a, device, name);
}

//! LFM2 contract registration and lifetime: the prompt selector, shared-round controls and the DSpark drafter's probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const lfm2_argmax = @import("metal_lfm2_argmax.zig");
const lfm2_streams = @import("metal_lfm2_streams.zig");
const lfm2_tap = @import("metal_lfm2_tap.zig");
const lfm2_controls = @import("metal_lfm2_controls.zig");
const lfm2_norm_tap = @import("metal_lfm2_norm_tap.zig");
const lfm2_dspark = @import("metal_lfm2_dspark.zig");
const lfm2_bf16 = @import("metal_lfm2_bf16.zig");
const lfm2_glu = @import("metal_lfm2_glu.zig");
const lfm2_attn = @import("metal_lfm2_attn.zig");
const lfm2_kv = @import("metal_lfm2_kv.zig");
const lfm2_confidence = @import("metal_lfm2_confidence.zig");
const Prompt = @import("metal_lfm2_prompt.zig").Suite;
const support = @import("metal_support.zig");
fn wanted(only: ?[]const u8, name: []const u8) bool {
    return if (only) |o| std.mem.indexOf(u8, name, o) != null else true;
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    prompt: ?Prompt = null,
    argmax: std.ArrayList(*lfm2_argmax.Probe) = .empty,
    streams: std.ArrayList(*lfm2_streams.Probe) = .empty,
    taps: std.ArrayList(*lfm2_tap.Probe) = .empty,
    norm_taps: std.ArrayList(*lfm2_norm_tap.Probe) = .empty,
    drafters: std.ArrayList(*lfm2_dspark.Probe) = .empty,
    blocks: std.ArrayList(*lfm2_attn.Probe) = .empty,
    bf16s: std.ArrayList(*lfm2_bf16.Probe) = .empty,
    glus: std.ArrayList(*lfm2_glu.Probe) = .empty,
    kvs: std.ArrayList(*lfm2_kv.Probe) = .empty,
    confidences: std.ArrayList(*lfm2_confidence.Probe) = .empty,

    fn add(s: *Suite, comptime P: type, list: *std.ArrayList(*P), args: anytype) !void {
        const p = try s.a.create(P);
        errdefer s.a.destroy(p);
        p.* = try @call(.auto, P.init, .{ s.a, s.device, s.queue } ++ args);
        errdefer p.deinit();
        // Ownership transfers after both appends; collect errors never run the partially built kernel list.
        try s.kernels.append(s.a, p.kernel());
        try list.append(s.a, p);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        s.prompt = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
        try s.prompt.?.collect(only);
        try s.collectShared(only);
        try s.collectDrafter(only);
    }

    fn collectShared(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(lfm2_controls.Mutation)) |m| {
            const argmax_side = m == .none or m == .argmax_last_tie or m == .argmax_no_flag;
            if (argmax_side) {
                if (!wanted(only, if (m == .none) lfm2_argmax.name else m.name())) continue;
                try s.add(lfm2_argmax.Probe, &s.argmax, .{m});
                if (m != .none) continue;
            }
            for (std.enums.values(lfm2_streams.Which)) |w| {
                const control_for = (m == .conv_first_state and w == .conv) or (m == .attention_first_cache and w == .attention);
                if (m != .none and !control_for) continue;
                if (!wanted(only, if (m == .none) lfm2_streams.spec(w).name else m.name())) continue;
                try s.add(lfm2_streams.Probe, &s.streams, .{ w, m });
            }
        }
        for ([_]lfm2_controls.Mutation{ .none, .tap_packed_rows, .tap_clobbers_column }) |m| {
            if (!wanted(only, if (m == .none) lfm2_tap.name else m.name())) continue;
            try s.add(lfm2_tap.Probe, &s.taps, .{m});
        }
        for (std.enums.values(lfm2_norm_tap.Which)) |w| for (lfm2_norm_tap.faults(w)) |f| {
            if (!wanted(only, if (f == .none) lfm2_norm_tap.spec(w).name else lfm2_norm_tap.controlName(w, f))) continue;
            try s.add(lfm2_norm_tap.Probe, &s.norm_taps, .{ w, f });
        };
    }

    fn collectDrafter(s: *Suite, only: ?[]const u8) !void {
        const tensor = s.device.tensorUnits();
        for (std.enums.values(lfm2_controls.Mutation)) |m| {
            const drafter_which: ?lfm2_dspark.Which = switch (m) {
                .markov_nonfinite_dropped, .markov_first_prev => .markov_step,
                .markov_bf16_nonfinite_dropped, .markov_bf16_first_prev => .markov_step_bf16,
                .markov_pick_inf_dropped, .markov_pick_empty_flagged => .markov_pick,
                .draft_ids_unzeroed => .draft_ids,
                else => null,
            };
            if (m == .none or drafter_which != null) {
                for (std.enums.values(lfm2_dspark.Which)) |w| {
                    if (m != .none and w != drafter_which.?) continue;
                    if (!wanted(only, if (m == .none) lfm2_dspark.spec(w).name else m.name())) continue;
                    try s.add(lfm2_dspark.Probe, &s.drafters, .{ w, m });
                }
            }
            const bf16_which: ?lfm2_bf16.Which = switch (m) {
                .bf16_rows_short_k, .bf16_rows_last_row => .rows,
                .bf16_tile_untransposed, .bf16_tile_last_column => .tile,
                else => null,
            };
            if (m == .none or bf16_which != null) {
                for (std.enums.values(lfm2_bf16.Which)) |w| {
                    if (m != .none and w != bf16_which.?) continue;
                    if (!wanted(only, if (m == .none) lfm2_bf16.spec(w).name else m.name())) continue;
                    if (w == .tensor and !support.tensor(tensor, lfm2_bf16.spec(w).name)) continue;
                    try s.add(lfm2_bf16.Probe, &s.bf16s, .{ w, m });
                }
            }
            if (m == .none or m == .block_attention_causal) {
                if (!wanted(only, if (m == .none) lfm2_attn.name else m.name())) continue;
                try s.add(lfm2_attn.Probe, &s.blocks, .{m});
            }
        }
        for ([_]lfm2_controls.Mutation{ .none, .context_kv_first_row, .context_kv_theta }) |m| {
            if (!wanted(only, if (m == .none) lfm2_kv.name else m.name())) continue;
            try s.add(lfm2_kv.Probe, &s.kvs, .{m});
        }
        for ([_]lfm2_controls.Mutation{ .none, .bf16_glu_short_k, .bf16_glu_last_row, .bf16_glu_swapped }) |m| {
            if (!wanted(only, if (m == .none) lfm2_glu.name else m.name())) continue;
            try s.add(lfm2_glu.Probe, &s.glus, .{m});
        }
        for ([_]lfm2_controls.Mutation{ .none, .confidence_no_w1 }) |m| {
            if (!wanted(only, if (m == .none) lfm2_confidence.name else m.name())) continue;
            try s.add(lfm2_confidence.Probe, &s.confidences, .{m});
        }
    }

    pub fn deinit(s: *Suite) void {
        if (s.prompt) |*p| p.deinit();
        inline for (.{ "argmax", "streams", "taps", "norm_taps", "drafters", "blocks", "bf16s", "glus", "kvs", "confidences" }) |field| {
            const list = &@field(s, field);
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
            list.deinit(s.a);
        }
    }
};

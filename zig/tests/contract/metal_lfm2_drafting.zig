//! LFM2 DSpark's contract suite keeps every probe alive through all gates, including planted controls.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const controls = @import("metal_lfm2_controls.zig");
const dspark = @import("metal_lfm2_dspark.zig");
const bf16 = @import("metal_lfm2_bf16.zig");
const attn = @import("metal_lfm2_attn.zig");
const kv = @import("metal_lfm2_kv.zig");
const argmax = @import("metal_lfm2_argmax.zig");
const streams = @import("metal_lfm2_streams.zig");
const norm_tap = @import("metal_lfm2_norm_tap.zig");
const tap = @import("metal_lfm2_tap.zig");
const confidence = @import("metal_lfm2_confidence.zig");

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    argmaxes: std.ArrayList(*argmax.Probe) = .empty,
    streams: std.ArrayList(*streams.Probe) = .empty,
    norm_taps: std.ArrayList(*norm_tap.Probe) = .empty,
    taps: std.ArrayList(*tap.Probe) = .empty,
    drafters: std.ArrayList(*dspark.Probe) = .empty,
    blocks: std.ArrayList(*attn.Probe) = .empty,
    bf16s: std.ArrayList(*bf16.Probe) = .empty,
    kvs: std.ArrayList(*kv.Probe) = .empty,
    confidences: std.ArrayList(*confidence.Probe) = .empty,

    pub fn deinit(s: *Suite) void {
        for (s.confidences.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.confidences.deinit(s.a);
        inline for (.{ &s.drafters, &s.blocks, &s.bf16s, &s.kvs }) |list| {
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
        }
        inline for (.{ &s.drafters, &s.blocks, &s.bf16s, &s.kvs }) |list| list.deinit(s.a);
        inline for (.{ &s.taps, &s.norm_taps }) |list| {
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
            list.deinit(s.a);
        }
        inline for (.{ &s.argmaxes, &s.streams }) |list| {
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
        }
        inline for (.{ &s.argmaxes, &s.streams }) |list| list.deinit(s.a);
    }

    fn add(s: *Suite, comptime P: type, list: *std.ArrayList(*P), args: anytype) !void {
        const p = try s.a.create(P);
        errdefer s.a.destroy(p);
        p.* = try @call(.auto, P.init, .{ s.a, s.device, s.queue } ++ args);
        errdefer p.deinit();
        // Ownership transfers after both appends; collect errors never run the partially built kernel list.
        try s.kernels.append(s.a, p.kernel());
        try list.append(s.a, p);
    }

    pub fn collect(s: *Suite, only: ?[]const u8, tensor: bool) !void {
        for (std.enums.values(controls.Mutation)) |m| {
            const argmax_side = m == .none or m == .argmax_last_tie or m == .argmax_no_flag;
            if (argmax_side) {
                if (!wanted(only, if (m == .none) argmax.name else m.name())) continue;
                try s.add(argmax.Probe, &s.argmaxes, .{m});
                if (m != .none) continue;
            }
            for (std.enums.values(streams.Which)) |w| {
                const control_for = (m == .conv_first_state and w == .conv) or (m == .attention_first_cache and w == .attention);
                if (m != .none and !control_for) continue;
                if (!wanted(only, if (m == .none) streams.spec(w).name else m.name())) continue;
                try s.add(streams.Probe, &s.streams, .{ w, m });
            }
        }
        for (std.enums.values(norm_tap.Which)) |w| for ([_]bool{ false, true }) |control| {
            if (!wanted(only, if (control) norm_tap.controlName(w) else norm_tap.spec(w).name)) continue;
            try s.add(norm_tap.Probe, &s.norm_taps, .{ w, control });
        };
        for ([_]controls.Mutation{ .none, .tap_packed_rows, .tap_clobbers_column }) |m| {
            if (!wanted(only, if (m == .none) tap.name else m.name())) continue;
            try s.add(tap.Probe, &s.taps, .{m});
        }
        for (std.enums.values(controls.Mutation)) |m| {
            const drafter_which: ?dspark.Which = switch (m) {
                .markov_nonfinite_dropped, .markov_first_prev => .markov_step,
                .markov_bf16_nonfinite_dropped, .markov_bf16_first_prev => .markov_step_bf16,
                .markov_pick_inf_dropped, .markov_pick_empty_flagged => .markov_pick,
                .draft_ids_unzeroed => .draft_ids,
                else => null,
            };
            if (m == .none or drafter_which != null) {
                for (std.enums.values(dspark.Which)) |w| {
                    if (m != .none and w != drafter_which.?) continue;
                    if (!wanted(only, if (m == .none) dspark.spec(w).name else m.name())) continue;
                    try s.add(dspark.Probe, &s.drafters, .{ w, m });
                }
            }
            const bf16_which: ?bf16.Which = switch (m) {
                .bf16_rows_short_k, .bf16_rows_last_row => .rows,
                .bf16_tile_untransposed, .bf16_tile_last_column => .tile,
                else => null,
            };
            if (m == .none or bf16_which != null) {
                for (std.enums.values(bf16.Which)) |w| {
                    if (w == .tensor and !tensor) continue;
                    if (m != .none and w != bf16_which.?) continue;
                    if (!wanted(only, if (m == .none) bf16.spec(w).name else m.name())) continue;
                    try s.add(bf16.Probe, &s.bf16s, .{ w, m });
                }
            }
            if (m == .none or m == .block_attention_causal) {
                if (!wanted(only, if (m == .none) attn.name else m.name())) continue;
                try s.add(attn.Probe, &s.blocks, .{m});
            }
        }
        for ([_]controls.Mutation{ .none, .context_kv_first_row, .context_kv_theta }) |m| {
            if (!wanted(only, if (m == .none) kv.name else m.name())) continue;
            try s.add(kv.Probe, &s.kvs, .{m});
        }
        for ([_]controls.Mutation{ .none, .confidence_no_w1 }) |m| {
            if (!wanted(only, if (m == .none) confidence.name else m.name())) continue;
            try s.add(confidence.Probe, &s.confidences, .{m});
        }
    }
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, name, only orelse return true) != null;
}

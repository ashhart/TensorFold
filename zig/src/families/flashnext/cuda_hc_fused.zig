//! Prompt-only HC launch pairs; all other paths retain their existing launches and state transitions.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
const zs = @import("cuda_zero_slice.zig");
const raw = @import("cuda_hc_fused_kernels.zig");
pub const Kernels = raw.Kernels;
pub const Writeback = raw.Writeback;
pub const Mode = raw.Mode;
pub const weights = wt;

/// Final-stream capture reads H after the last MoE writeback; intermediate keeps and PLE use separate launches.
pub fn allowed(rows: usize, lanes: usize, hooks: parts.Hooks, ple: bool, kept: parts.HcKeep) bool {
    return rows >= 64 and lanes == 0 and !ple and hooks.logits != .all and hooks.logits_out == 0 and hooks.keep == null and
        hooks.moe_add == null and std.meta.eql(kept, parts.HcKeep{});
}

fn eligible(self: anytype, w: wt.HcW, kept: parts.HcKeep, ple: bool) bool {
    if (comptime !@hasField(@TypeOf(self.tri), "own")) return false;
    if (!allowed(self.rows, self.lanes.len, self.hooks, ple, kept) or self.m.pocket != null) return false;
    if (self.tri.own == null or self.tri.set != null) return false;
    const down = w.down.native orelse return false;
    const up = w.up.native orelse return false;
    return w.has_inj == 1 and w.n_down == wt.low + wt.streams and down.mode == .bf16 and
        down.n == w.n_down and down.k == wt.wide and up.mode == .bf16 and up.n == wt.wide and
        up.k == wt.low and up.act == 0;
}

fn normalized(self: anytype, w: wt.HcW) !void {
    const s = self.s;
    const n = self.rows;
    try self.k.native_lane.matmulF32(self.stream, s.norm, wt.wide, n, w.down.native.?, s.hc_part, s.native_part);
    try self.tri.hcReduceAct(s.hc_part, s.act, s.xs_act, s.inj, 1, n, wt.streams, wt.low, w.n_down, w.has_inj);
    try self.k.native_lane.matmul(self.stream, s.act, wt.low, n, w.up.native.?, s.hc_up, s.native_part);
    try self.tri.hcDenseMix(s.hc_up, s.norm, s.mixed, s.xsm, n, wt.dims, wt.streams);
}

/// Layer entry: pending MoE writeback and this layer's normalization, without deferring work across PLE.
pub fn start(self: anytype, w: wt.HcW, kept: parts.HcKeep, ple: bool) !void {
    if (!eligible(self, w, kept, ple)) {
        try self.writeback(0, self.rows);
        return self.readoutKeep(w, 0, self.rows, kept);
    }
    const s = self.s;
    var in: Writeback = .{ .h = s.h, .pss = s.pss, .inj = s.inj, .stride = self.rows * wt.dims, .mode = .sums };
    if (self.pending != null) {
        if (self.pending_f32) {
            in.mode = .slices;
            in.branch = s.moe_f32;
        } else {
            in.mode = .moe;
            in.y = s.y;
            in.wts = s.wts;
        }
    }
    try self.tri.own.?.hc_fused.writeback(self.stream, in, s.norm, w.scale.ptr, self.rows);
    if (self.pending != null) self.pending_f32 = false;
    self.pending = null;
    try normalized(self, w);
}

/// Mixer exit: its FP32 branch and the following MLP readout, preserving the original slice order.
pub fn afterMixer(self: anytype, w: wt.HcW, keep: parts.LayerKeep, ple: bool) !void {
    const s = self.s;
    const n = self.rows;
    if (eligible(self, w, keep.mlp_hc, ple) and keep.mix_branch == 0) {
        const sl = zs.slices(self.zero_slices, s.cap, n, self.out_slices);
        const in: Writeback = .{
            .h = s.h,
            .pss = s.pss,
            .inj = s.inj,
            .branch = s.part,
            .mode = .slices,
            .stride = sl.stride,
            .slices = sl.count,
        };
        try self.tri.own.?.hc_fused.writeback(self.stream, in, s.norm, w.scale.ptr, n);
        self.zero_slices = false;
        return normalized(self, w);
    }
    if (keep.mix_branch != 0) try self.sumSlices(s.part, keep.mix_branch, true, n * wt.dims, self.out_slices);
    const sl = zs.slices(self.zero_slices, s.cap, n, self.out_slices);
    self.zero_slices = false;
    try self.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", sl.stride, n, wt.dims, wt.streams, 4, 1, 1, sl.count);
    try self.readoutKeep(w, 0, n, keep.mlp_hc);
}

test "HC prompt fusions allow final streams and exclude decode, shared rounds, PLE and intermediate keeps" {
    const t = std.testing;
    try t.expect(allowed(64, 0, .{}, false, .{}));
    try t.expect(allowed(4218, 0, .{}, false, .{}));
    for ([_]usize{ 1, 16, 63 }) |n| try t.expect(!allowed(n, 0, .{}, false, .{}));
    try t.expect(!allowed(64, 1, .{}, false, .{}));
    try t.expect(!allowed(64, 0, .{}, true, .{}));
    try t.expect(!allowed(64, 0, .{ .logits = .all }, false, .{}));
    try t.expect(!allowed(64, 0, .{ .logits_out = 1 }, false, .{}));
    var keep: parts.Keep = undefined;
    try t.expect(!allowed(64, 0, .{ .keep = &keep }, false, .{}));
    inline for (@typeInfo(parts.HcKeep).@"struct".field_names) |field| {
        var kept: parts.HcKeep = .{};
        @field(kept, field) = 1;
        try t.expect(!allowed(64, 0, .{}, false, kept));
    }
}

test "HC prompt fusions admit MTP prompt streams without admitting the head or verify paths" {
    const t = std.testing;
    for ([_]usize{ 64, 65, 1536, 4218 }) |rows| for ([_]parts.Logits{ .none, .last }) |logits| {
        const hooks: parts.Hooks = .{ .streams = 1, .logits = logits };
        try t.expect(allowed(rows, 0, hooks, false, .{}));
        try t.expect(!allowed(rows, 0, hooks, true, .{}));
        try t.expect(!allowed(rows, 1, hooks, false, .{}));
        inline for (@typeInfo(parts.HcKeep).@"struct".field_names) |field| {
            var kept: parts.HcKeep = .{};
            @field(kept, field) = 1;
            try t.expect(!allowed(rows, 0, hooks, false, kept));
        }
    };
    for ([_]usize{ 1, 16, 63 }) |rows| try t.expect(!allowed(rows, 0, .{ .streams = 1 }, false, .{}));
    try t.expect(!allowed(4218, 0, .{ .streams = 1, .logits = .all }, false, .{}));
    try t.expect(!allowed(4218, 0, .{ .streams = 1, .logits_out = 1 }, false, .{}));
    var keep: parts.Keep = undefined;
    try t.expect(!allowed(4218, 0, .{ .streams = 1, .keep = &keep }, false, .{}));
}

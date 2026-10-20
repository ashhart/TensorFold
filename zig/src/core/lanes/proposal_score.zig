//! One normalized-row scoring contract; backend producers supply tensors, queues and existing selection recipes.
const std = @import("std");
const be = @import("backend.zig");
const head = @import("proposal_head.zig");
const Stream = @import("stream.zig").Stream;
pub const Selection = union(enum) { argmax, keyed, topk: u32, coupled_topk: u32 };
pub const Descriptor = struct {
    vocab: u32,
    hidden: u32,
    format: []const u8,
    group: u32,
    scale_recipe: []const u8,
    space: be.Features.Space,
    device: i32,
    queue: u64,
    output: be.Features.Dtype = .bf16,
    tied: bool = false,
    rank: u32 = 0,
    ranks: u32 = 1,
};
/// Producers score the whole batch in order; stream cancellation belongs to the router and host.
pub const Normalized = struct {
    rows: be.Features,
    streams: []const *Stream,
    positions: []const u64,
    queue: u64,
    role: enum { decode, prompt },
};
pub const Buffer = struct { dtype: be.Features.Dtype = .bf16, address: u64, rows: u32, stride: u32, ready: be.Features.Fence = .none };
pub const View = struct {
    buffer: Buffer,
    dtype: be.Features.Dtype,
    vocab: u32,
    logical: u32,
    stride: u32,
    ids: ?[]const u32,
    map: ?head.Map,
    selection: Selection,
    pub fn global(v: View, column: usize) !u32 {
        if (column >= v.logical) return error.ProposalPaddingPick;
        return if (v.map) |m| m.global(column) orelse error.ProposalPaddingPick else @intCast(column);
    }
};
pub const Scorer = struct {
    descriptor: Descriptor,
    generation: ?[32]u8,
    selection: Selection,
    ptr: *anyopaque,
    run: *const fn (*anyopaque, *const head.Plan, Normalized) anyerror!Buffer,

    pub fn calibrated(a: anytype, actual: anytype, descriptor: Descriptor, selection: Selection, ptr: *anyopaque, run: @FieldType(Scorer, "run")) !Scorer {
        if (descriptor.rank != 0 or descriptor.ranks != 1) return error.ProposalPlacementUnsupported;
        if (a.isLegacy() or !a.matches(actual)) return error.ProposalIdentityMismatch;
        if (descriptor.vocab != actual.vocab or descriptor.hidden != actual.hidden or descriptor.group != actual.group or
            !std.mem.eql(u8, descriptor.format, actual.format) or !std.mem.eql(u8, descriptor.scale_recipe, actual.scale_recipe)) return error.ProposalIdentityMismatch;
        switch (selection) {
            .topk, .coupled_topk => |k| if (k == 0 or k > actual.vocab) return error.ProposalScoreShape,
            else => {},
        }
        return .{ .descriptor = descriptor, .generation = a.fingerprint, .selection = selection, .ptr = ptr, .run = run };
    }
    pub fn score(s: *const Scorer, plan: *const head.Plan, input: Normalized) !View {
        if (plan.kind == .paused) return error.ProposalPaused;
        const d = s.descriptor;
        if (d.vocab != plan.vocab or d.hidden == 0 or input.rows.rows == 0 or input.rows.dtype != .bf16 or
            input.rows.row_bytes != try std.math.mul(u32, d.hidden, 2) or input.rows.space != d.space or input.rows.device != d.device or
            input.queue != d.queue or input.streams.len != input.rows.rows or input.positions.len != input.rows.rows or input.rows.buffer == 0) return error.ProposalScoreShape;
        const logical: u32 = if (plan.map) |m| @intCast(m.ids.len) else plan.vocab;
        const result = try s.run(s.ptr, plan, input);
        if (result.dtype != d.output or result.address == 0 or result.rows != input.rows.rows or result.stride < logical) return error.ProposalScoreShape;
        return .{ .buffer = result, .dtype = result.dtype, .vocab = d.vocab, .logical = logical, .stride = result.stride, .ids = if (plan.map) |m| m.ids else null, .map = plan.map, .selection = s.selection };
    }
};

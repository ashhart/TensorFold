//! Owned copies of one plain prompt's DeltaNet marks; only completed copies may become prompt-cache states.
const std = @import("std");
const metal = @import("metal");
const delta = @import("core_delta");
const marks = @import("prompt_marks.zig");
const Budget = @import("gdn_pool.zig").Budget;
const Plan = @import("gdn_plan.zig").Plan;
const gdn = @import("gdn.zig");
comptime {
    std.debug.assert(marks.limit == delta.MARKS);
}

pub const Bank = struct {
    position: marks.Position,
    state: metal.Buffer,
    conv: metal.Buffer,
    list: metal.Buffer,
    layers: usize,
    state_bytes: usize,
    conv_bytes: usize,
    written: usize = 0,
    residency: ?metal.ResidencySet.Follow,

    pub fn init(device: metal.Device, budget: Budget, kernels: delta.Kernels, slot: u32, start: u32, end: u32, rows: [marks.limit]u32, resident: ?metal.ResidencySet.Follow) !Bank {
        if (slot >= budget.slots or budget.shape.c.dk != 128 or budget.shape.c.dv % 4 != 0 or end <= start or end - start > budget.rows) return error.BadPromptMarks;
        if (kernels.chain_wide_marks.simdWidth() != 32 or kernels.chain_wide_marks.maxThreads() < 128 or kernels.mark_conv.maxThreads() < 32) return error.UnsupportedGdnPipeline;
        const position = try marks.Position.init(slot, start, end, rows);
        const size = try marks.layout(budget.layers, budget.shape.state, budget.shape.conv);
        const opts = metal.ResourceOptions.shared | metal.ResourceOptions.untracked;
        const list = try device.buffer(16, opts);
        errdefer list.deinit();
        const state = try device.buffer(size.state, opts);
        errdefer state.deinit();
        const conv = try device.buffer(size.conv, opts);
        @memcpy(list.slice(u32, marks.limit), &rows);
        const bank = Bank{ .position = position, .list = list, .state = state, .conv = conv, .layers = budget.layers, .state_bytes = budget.shape.state * 4, .conv_bytes = budget.shape.conv * 2, .residency = resident };
        if (resident) |f| for ([_]metal.Buffer{ list, state, conv }) |b| f.join(f.ctx, b);
        return bank;
    }

    /// Caller waits for all encoded work on success and failure before dropping these scratch copies.
    pub fn deinit(b: *Bank) void {
        for ([_]metal.Buffer{ b.list, b.state, b.conv }) |buffer| {
            if (b.residency) |f| f.leave(f.ctx, buffer);
            buffer.deinit();
        }
    }

    /// Native prompt marks cover one fully accepted in-place segment, never verification or multiple streams.
    pub fn check(b: *const Bank, plan: Plan) !void {
        if (b.position.ready or b.written != 0 or plan.segments.len != 1 or plan.keeps.len != 1) return error.BadPromptMarks;
        const segment = plan.segments[0];
        if (segment.first != 0 or segment.state_slot != b.position.slot or segment.next_slot != b.position.slot or segment.rows != b.position.end - b.position.start or plan.keeps[0].rows != segment.rows) return error.BadPromptMarks;
    }

    pub fn scan(b: *Bank, layer: usize, r: gdn.Round, e: metal.ComputeEncoder, kernel: metal.Pipeline) !void {
        if (layer != b.written or layer >= b.layers or r.segments != 1) return error.BadPromptMarks;
        e.setPipeline(kernel);
        gdn.bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.segments }, 0);
        e.setValue(r.p, 7);
        gdn.bind(e, .{ r.b.recurrent, r.b.next_state }, 8);
        e.setBuffer(b.list, 0, 10);
        e.setBuffer(b.state, layer * marks.limit * b.state_bytes, 11);
        e.dispatchThreads(.{ .width = 32, .height = r.p.dv / 4, .depth = r.p.nv }, .{ .width = 32, .height = 4 });
        e.barrier();
    }

    /// Read the entry history before ordinary conv publication replaces it.
    pub fn convolution(b: *Bank, layer: usize, r: gdn.Round, e: metal.ComputeEncoder, kernel: metal.Pipeline) !void {
        if (layer != b.written or layer >= b.layers) return error.BadPromptMarks;
        e.setPipeline(kernel);
        gdn.bind(e, .{ r.b.history, r.b.qkv, r.b.segments, r.b.windows }, 0);
        e.setValue(r.p, 4);
        e.setBuffer(b.list, 0, 5);
        e.setBuffer(b.conv, layer * marks.limit * b.conv_bytes, 6);
        e.dispatchThreads(.{ .width = r.conv_width, .height = marks.limit, .depth = 1 }, .{ .width = 32 });
        e.barrier();
        b.written += 1;
    }

    pub fn complete(b: *Bank, slot: u32, stands: u32) !void {
        if (b.written != b.layers or slot != b.position.slot or stands != b.position.end) return error.BadPromptMarks;
        b.position.ready = true;
    }

    pub fn index(b: *const Bank, slot: u32, at: u32, stands: u32) ?usize {
        return b.position.index(slot, at, stands);
    }

    pub fn span(b: *const Bank, layer: usize, mark: usize, recurrent: bool) !gdn.Span {
        if (!b.position.ready or layer >= b.layers or mark >= marks.limit or b.position.rows[mark] == 0) return error.BadPromptMarks;
        const bytes = if (recurrent) b.state_bytes else b.conv_bytes;
        return .{ .buffer = if (recurrent) b.state else b.conv, .offset = (layer * marks.limit + mark) * bytes, .bytes = bytes };
    }
};

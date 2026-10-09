//! Borrowed buffers and caller-owned pipelines encode a round without allocating, committing or reading back.
const std = @import("std");
const metal = @import("metal");
const abi = @import("gdn_contract.zig");
const Plan = @import("gdn_plan.zig").Plan;

pub const Span = struct {
    buffer: metal.Buffer,
    offset: usize = 0,
    bytes: usize,

    fn check(s: Span, required: usize) !void {
        try abi.checkSpan(s.buffer.length(), s.offset, s.bytes, required, 2);
    }

    fn bind(s: Span, encoder: metal.ComputeEncoder, index: usize) void {
        encoder.setBuffer(s.buffer, s.offset, index);
    }

    fn overlaps(a: Span, b: Span) bool {
        return a.buffer.id == b.buffer.id and a.offset < b.offset + b.bytes and b.offset < a.offset + a.bytes;
    }
};

pub const Pipelines = @import("core_delta").Pipelines;

pub const Buffers = struct {
    qkv: Span,
    a: Span,
    b: Span,
    z: Span,
    conv_weight: Span,
    a_log: Span,
    dt: Span,
    norm: Span,
    history: Span,
    state: Span,
    query: Span,
    key: Span,
    value: Span,
    decay: Span,
    mixing: Span,
    tails: Span,
    snapshots: Span,
    recurrent: Span,
    output: Span,
    next_state: Span,
    next_history: Span,
    parents: Span,
    windows: Span,
    row_slots: Span,
    segments: Span,
    keeps: Span,
    kept_rows: Span,

    pub fn check(b: Buffers, sizes: abi.Bytes, kinds: abi.WeightKinds, p: abi.Params, plan: Plan) !void {
        try b.qkv.check(sizes.qkv);
        inline for (.{ "a", "b", "mixing" }) |name| try @field(b, name).check(sizes.heads);
        try b.z.check(sizes.value);
        try b.conv_weight.check(sizes.conv_weight);
        try b.a_log.check(sizes.a_log);
        try b.dt.check(sizes.dt);
        try b.norm.check(sizes.norm);
        inline for (.{ "history", "next_history" }) |name| try @field(b, name).check(sizes.history);
        inline for (.{ "state", "next_state" }) |name| try @field(b, name).check(sizes.state);
        inline for (.{ "query", "key" }) |name| try @field(b, name).check(sizes.query);
        inline for (.{ "value", "recurrent", "output" }) |name| try @field(b, name).check(sizes.value);
        try b.decay.check(sizes.decay);
        try b.tails.check(sizes.tails);
        try b.snapshots.check(sizes.snapshots);
        try b.parents.check(try abi.countBytes(i32, p.rows, 1));
        try b.windows.check(try abi.countBytes(u32, p.rows, p.taps));
        try b.row_slots.check(try abi.countBytes(u32, p.rows, 1));
        try b.segments.check(try abi.countBytes(abi.Segment, plan.segments.len, 1));
        try b.keeps.check(try abi.countBytes(abi.Keep, plan.keeps.len, 1));
        try b.kept_rows.check(try abi.countBytes(u32, plan.kept_rows.len, 1));
        inline for (.{ "state", "next_state", "decay", "snapshots", "parents", "windows", "row_slots", "segments", "keeps", "kept_rows" }) |name| if (@field(b, name).offset % 4 != 0) return error.BadGdnBuffer;
        if ((kinds.conv == .f32 and b.conv_weight.offset % 4 != 0) or (kinds.a_log == .f32 and b.a_log.offset % 4 != 0) or (kinds.dt == .f32 and b.dt.offset % 4 != 0) or (kinds.norm == .f32 and b.norm.offset % 4 != 0)) return error.BadGdnBuffer;
        const reads = [_]Span{ b.qkv, b.a, b.b, b.z, b.conv_weight, b.a_log, b.dt, b.norm, b.history, b.state, b.parents, b.windows, b.row_slots, b.segments, b.keeps, b.kept_rows };
        const writes = [_]Span{ b.query, b.key, b.value, b.decay, b.mixing, b.tails, b.snapshots, b.recurrent, b.output, b.next_state, b.next_history };
        inline for (writes, 0..) |write, index| {
            inline for (reads) |read| if (write.overlaps(read)) return error.AliasedGdnState;
            inline for (writes[0..index]) |other| if (write.overlaps(other)) return error.AliasedGdnState;
        }
    }
};

pub const Round = struct {
    p: abi.Params,
    b: Buffers,
    pipelines: Pipelines,
    segments: usize,
    conv_width: usize,

    pub fn init(shape: abi.Shape, kinds: abi.WeightKinds, p: abi.Params, plan: Plan, b: Buffers, pipelines: Pipelines) !Round {
        if (plan.parents.len != p.rows or plan.segments.len == 0 or plan.keeps.len != plan.segments.len) return error.BadGdnPlan;
        try b.check(try abi.Bytes.init(shape, p, kinds), kinds, p, plan);
        inline for (.{ pipelines.pre, pipelines.post, pipelines.conv_commit }) |pipeline| if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 32) return error.UnsupportedGdnPipeline;
        inline for (.{ pipelines.tree, pipelines.replay }) |pipeline| if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 128) return error.UnsupportedGdnPipeline;
        return .{ .p = p, .b = b, .pipelines = pipelines, .segments = plan.segments.len, .conv_width = shape.conv };
    }

    pub fn prepare(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.pre);
        bind(e, .{ r.b.qkv, r.b.a, r.b.b, r.b.history, r.b.conv_weight, r.b.a_log, r.b.dt, r.b.windows, r.b.row_slots }, 0);
        e.setValue(r.p, 9);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.tails }, 10);
        e.dispatchGroups(.{ .width = 1, .height = 2 * r.p.nk + r.p.nv, .depth = r.p.rows }, .{ .width = 32, .height = 1, .depth = 1 });
        e.barrier();
    }

    pub fn forward(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.tree);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.parents, r.b.segments }, 0);
        e.setValue(r.p, 8);
        bind(e, .{ r.b.recurrent, r.b.snapshots }, 9);
        r.stateGrid(e);
        e.barrier();
        e.setPipeline(r.pipelines.post);
        bind(e, .{ r.b.recurrent, r.b.z, r.b.norm }, 0);
        e.setValue(r.p, 3);
        r.b.output.bind(e, 4);
        e.dispatchGroups(.{ .width = 1, .height = r.p.nv, .depth = r.p.rows }, .{ .width = 32, .height = 1, .depth = 1 });
        e.barrier();
    }

    pub fn forwardChain(r: Round, e: metal.ComputeEncoder, chain: metal.Pipeline) void {
        r.scanChain(e, chain, false);
        r.gate(e);
        r.commitConv(e);
    }
    /// `wide`: tf_delta_gdn_chain_wide's grid, 4 value columns a simdgroup (prompt chunks, dk = 128).
    pub fn scanChain(r: Round, e: metal.ComputeEncoder, chain: metal.Pipeline, wide: bool) void {
        e.setPipeline(chain);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.segments }, 0);
        e.setValue(r.p, 7);
        bind(e, .{ r.b.recurrent, r.b.next_state }, 8);
        if (wide) e.dispatchThreads(.{ .width = 32, .height = r.p.dv / 4, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 4, .depth = 1 }) else r.stateGrid(e);
        e.barrier();
    }
    pub fn gate(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.post);
        bind(e, .{ r.b.recurrent, r.b.z, r.b.norm }, 0);
        e.setValue(r.p, 3);
        r.b.output.bind(e, 4);
        e.dispatchGroups(.{ .width = 1, .height = r.p.nv, .depth = r.p.rows }, .{ .width = 32 });
        e.barrier();
    }
    pub fn commitConv(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.conv_commit);
        bind(e, .{ r.b.history, r.b.qkv, r.b.kept_rows, r.b.keeps, r.b.windows }, 0);
        e.setValue(r.p, 5);
        r.b.next_history.bind(e, 6);
        e.dispatchThreads(.{ .width = r.conv_width, .height = r.segments }, .{ .width = 32 });
        e.barrier();
    }

    pub fn commit(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.replay);
        bind(e, .{ r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.kept_rows, r.b.keeps }, 0);
        e.setValue(r.p, 7);
        r.b.next_state.bind(e, 8);
        r.stateGrid(e);
        e.barrier();
        e.setPipeline(r.pipelines.conv_commit);
        bind(e, .{ r.b.history, r.b.qkv, r.b.kept_rows, r.b.keeps, r.b.windows }, 0);
        e.setValue(r.p, 5);
        r.b.next_history.bind(e, 6);
        e.dispatchThreads(.{ .width = r.conv_width, .height = r.segments, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
        e.barrier();
    }

    fn stateGrid(r: Round, e: metal.ComputeEncoder) void {
        e.dispatchThreads(.{ .width = 32, .height = r.p.dv, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 4, .depth = 1 });
    }
};

pub fn bind(e: metal.ComputeEncoder, buffers: anytype, start: usize) void {
    inline for (buffers, 0..) |buffer, index| buffer.bind(e, start + index);
}

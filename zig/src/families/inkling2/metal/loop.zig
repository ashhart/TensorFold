//! Inkling 2 Mini cross-pass messages and message gates: real Metal dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
const dense = @import("dense.zig");
pub const plan = @import("inkling2_loop_plan");

pub const hidden: u32 = plan.hidden;
pub const rank: u32 = plan.rank;
pub const message_entries: u32 = plan.message_entries;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const RowsArgs = plan.RowsArgs;

/// Bytes of gate scratch a row: normalized [h | m] in BF16, then zeta's 64 BF16.
const gate_row_bytes: usize = (2 * hidden + rank) * 2;

pub const Pipelines = struct {
    messages: mtl.Pipeline,
    gates: mtl.Pipeline,
    zeta: mtl.Pipeline,
    blend: mtl.Pipeline,
    /// The down projection on simdgroup matrices, where the GPU has no tensor units: zeta's sums.
    zeta_mma: ?dense.Mma = null,
    /// The blend on simdgroup matrices, where the GPU has no tensor units.
    blend_mma: ?mtl.Pipeline = null,
    /// The gates' intermediates for up to 1,024 rows of each of `max_gates` tuples.
    scratch: mtl.Buffer,

    pub fn load(device: mtl.Device) !Pipelines {
        var out = try loadSource(device, sources.loop);
        errdefer out.deinit();
        if (device.tensorUnits()) return out;
        out.zeta_mma = try dense.Mma.loadSource(device, sources.dense_mma);
        try out.loadMma(device, sources.loop_mma);
        return out;
    }

    /// The blend on simdgroup matrices from `text`; the contract's controls pass an edited one.
    pub fn loadMma(p: *Pipelines, device: mtl.Device, text: []const u8) !void {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        p.blend_mma = try mtl.Pipeline.init(device, lib, "tf_ink2_loop_blend_mma", false);
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var messages = try mtl.Pipeline.init(device, lib, plan.Kind.messages.symbol(), false);
        errdefer messages.deinit();
        const gate_pipeline = try mtl.Pipeline.init(device, lib, plan.Kind.gates.symbol(), false);
        errdefer gate_pipeline.deinit();
        const zeta = try mtl.Pipeline.init(device, lib, "tf_inkling2_loop_zeta", false);
        errdefer zeta.deinit();
        const blend = try mtl.Pipeline.init(device, lib, "tf_inkling2_loop_blend", false);
        errdefer blend.deinit();
        const scratch = try device.buffer(max_gates * 1024 * gate_row_bytes, mtl.ResourceOptions.shared);
        return .{ .messages = messages, .gates = gate_pipeline, .zeta = zeta, .blend = blend, .scratch = scratch };
    }
    pub fn deinit(p: *Pipelines) void {
        if (p.blend_mma) |pipe| pipe.deinit();
        if (p.zeta_mma) |mma| mma.deinit();
        p.scratch.deinit();
        p.blend.deinit();
        p.zeta.deinit();
        p.gates.deinit();
        p.messages.deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

/// Output layout is [token][12 core layers][q,k,v,mlp][H], all binary32.
pub fn crossPassMessageAttention(p: *const Pipelines, e: mtl.ComputeEncoder, bank0: Ref, bank1: Ref, q_mlp: Ref, kv: Ref, out: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= 1024);
    e.setPipeline(p.messages);
    bind(e, &.{ bank0, bank1, q_mlp, kv, out });
    e.setValue(GateArgs{ .rows = rows, .stride = hidden }, 5);
    e.dispatchGroups(mtl.Size.of(rows * message_entries, 1, 1), mtl.Size.of(256, 1, 1));
}

const GateArgs = extern struct { rows: u32, stride: u32 };

/// One controller tuple; h, m and output are [token][H] binary32 tensors.
pub fn gates(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32) void {
    gatesStrided(p, e, h, m, down, forget, gain, out, rows, hidden);
}

/// As gates, with row r's m at `m_stride` binary32 values from row r - 1's.
pub fn gatesStrided(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32, m_stride: u32) void {
    gatesEach(p, e, &.{.{ .h = h, .m = m, .down = down, .forget = forget, .gain = gain, .out = out }}, rows, m_stride);
}

/// Controller tuples one call gates together: a visit's three attention roles.
pub const max_gates = 3;
pub const Gate = struct { h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref };

/// gatesStrided for each tuple a step at a time, a barrier between steps; the tuples' outputs must be distinct.
pub fn gatesEach(p: *const Pipelines, e: mtl.ComputeEncoder, items: []const Gate, rows: u32, m_stride: u32) void {
    std.debug.assert(rows > 0 and rows <= 1024 and items.len > 0 and items.len <= max_gates);
    const args = GateArgs{ .rows = rows, .stride = m_stride };
    for (items, 0..) |g, i| {
        e.setPipeline(p.gates);
        bind(e, &.{ g.h, g.m, normalizedOf(p, i) });
        e.setValue(args, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
    e.barrier();
    for (items, 0..) |g, i| {
        // The same sums as zeta: a rank is a weight row of 2 H columns.
        const call = dense.Call{ .rows = rows, .n = rank, .k = 2 * hidden, .ldy = rank };
        if (p.zeta_mma) |mma| if (dense.Mma.serves(call)) {
            mma.launch(e, denseRef(normalizedOf(p, i)), denseRef(g.down), denseRef(zetaOf(p, i)), call);
            continue;
        };
        e.setPipeline(p.zeta);
        bind(e, &.{ normalizedOf(p, i), g.down, zetaOf(p, i) });
        e.setValue(args, 3);
        e.dispatchGroups(mtl.Size.of(rank / 8, rows, 1), mtl.Size.of(256, 1, 1));
    }
    e.barrier();
    for (items, 0..) |g, i| {
        bind(e, &.{ g.h, g.m, zetaOf(p, i), g.forget, g.gain, g.out });
        e.setValue(args, 6);
        if (p.blend_mma) |pipe| if (rows >= dense.min_rows) {
            e.setPipeline(pipe);
            e.dispatchGroups(mtl.Size.of(hidden / 16, (rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
            continue;
        };
        e.setPipeline(p.blend);
        e.dispatchThreads(mtl.Size.of(hidden, rows, 1), mtl.Size.of(32, 1, 1));
    }
}

fn denseRef(r: Ref) dense.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

fn normalizedOf(p: *const Pipelines, tuple: usize) Ref {
    return .{ .buf = p.scratch, .off = tuple * 1024 * gate_row_bytes };
}
fn zetaOf(p: *const Pipelines, tuple: usize) Ref {
    return .{ .buf = p.scratch, .off = tuple * 1024 * gate_row_bytes + 1024 * 2 * hidden * 2 };
}

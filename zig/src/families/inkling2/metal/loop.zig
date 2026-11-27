//! Inkling 2 Mini cross-pass messages and message gates: real Metal dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
pub const plan = @import("inkling2_loop_plan");

pub const hidden: u32 = plan.hidden;
pub const rank: u32 = plan.rank;
pub const message_entries: u32 = plan.message_entries;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const RowsArgs = plan.RowsArgs;

/// The rows from which a batch of gates takes the simdgroup-matrix kernel, where it is loaded.
pub const mma_rows: u32 = 8;
const MmaArgs = extern struct { rows: u32, m_stride: u32 };

pub const Pipelines = struct {
    messages: mtl.Pipeline,
    gates: mtl.Pipeline,
    norm: mtl.Pipeline,
    down: mtl.Pipeline,
    apply: mtl.Pipeline,
    roles_norm: mtl.Pipeline,
    roles_zeta: mtl.Pipeline,
    roles_blend: mtl.Pipeline,
    roles_scratch: mtl.Buffer,
    /// The gates of eight rows a threadgroup on simdgroup matrices.
    gates_mma: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        var out = try loadSource(device, sources.loop);
        errdefer out.deinit();
        try out.loadMma(device, sources.loop_mma);
        return out;
    }
    /// The simdgroup-matrix gates from `text`; the contract's controls pass an edited one.
    pub fn loadMma(p: *Pipelines, device: mtl.Device, text: []const u8) !void {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_inkling2_loop_gates_mma", false);
        errdefer pipe.deinit();
        if (pipe.maxThreads() < 256) return error.LoopThreads;
        if (p.gates_mma) |old| old.deinit();
        p.gates_mma = pipe;
    }
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [8]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{ plan.Kind.messages.symbol(), plan.Kind.gates.symbol(), "tf_inkling2_loop_norm", "tf_inkling2_loop_down", "tf_inkling2_loop_apply", "tf_inkling2_loop_roles_norm", "tf_inkling2_loop_roles_zeta", "tf_inkling2_loop_roles_blend" }, [_]usize{ 256, 256, 256, 32, 256, 256, 256, 32 }, &pipes) |name, req_threads, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < req_threads) return error.LoopThreads;
        }
        const scratch = try device.buffer(max_gates * 256 * gate_row_bytes, mtl.ResourceOptions.shared);
        return .{ .roles_norm = pipes[5], .roles_zeta = pipes[6], .roles_blend = pipes[7], .roles_scratch = scratch, .messages = pipes[0], .gates = pipes[1], .norm = pipes[2], .down = pipes[3], .apply = pipes[4] };
    }
    pub fn deinit(p: *Pipelines) void {
        p.roles_scratch.deinit();
        p.roles_blend.deinit();
        p.roles_zeta.deinit();
        p.roles_norm.deinit();
        p.apply.deinit();
        p.down.deinit();
        p.norm.deinit();
        if (p.gates_mma) |pipe| pipe.deinit();
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
    std.debug.assert(rows > 0 and rows <= 256);
    e.setPipeline(p.messages);
    bind(e, &.{ bank0, bank1, q_mlp, kv, out });
    e.setValue(RowsArgs{ .rows = rows }, 5);
    e.dispatchGroups(mtl.Size.of(rows * message_entries, 1, 1), mtl.Size.of(256, 1, 1));
}

/// One controller tuple; h, m and output are [token][H] binary32 tensors.
pub fn gates(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= 256);
    e.setPipeline(p.gates);
    bind(e, &.{ h, m, down, forget, gain, out });
    e.setValue(RowsArgs{ .rows = rows }, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const SplitArgs = extern struct { rows: u32, m_stride: u32 };
/// `gates`' bits for every row in 3 launches; m rows m_stride floats apart; scratch [rows][2H] BF16, [rows][rank] F32.
pub fn gatesSplit(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, m_stride: u32, down: Ref, forget: Ref, gain: Ref, out: Ref, normalized: Ref, zeta: Ref, rows: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and m_stride >= hidden);
    const args = SplitArgs{ .rows = rows, .m_stride = m_stride };
    e.setPipeline(p.norm);
    bind(e, &.{ h, m, normalized });
    e.setValue(args, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    e.barrier();
    e.setPipeline(p.down);
    bind(e, &.{ normalized, down, zeta });
    e.setValue(args, 3);
    e.dispatchGroups(mtl.Size.of(rank / 32, rows, 1), mtl.Size.of(32, 1, 1));
    e.barrier();
    e.setPipeline(p.apply);
    bind(e, &.{ h, m, zeta, forget, gain, out });
    e.setValue(args, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// The gates of `rows` rows (at most 256) at once: h and out hold [row][H], m's rows are `m_stride` values apart.
pub fn gatesRows(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, m: Ref, m_stride: u32, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0 and rows <= 256);
    const pipe = p.gates_mma orelse return error.NoSimdgroupMatrices;
    e.setPipeline(pipe);
    bind(e, &.{ h, m, down, forget, gain, out });
    e.setValue(MmaArgs{ .rows = rows, .m_stride = m_stride }, 6);
    e.dispatchGroups(mtl.Size.of((rows + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
}

const GateArgs = extern struct { rows: u32, stride: u32, roles: u32 };
const gate_row_bytes: usize = (2 * hidden + rank) * 2;
/// Controller tuples one call gates together: a visit's three attention roles.
pub const max_gates = 3;
pub const Gate = struct { h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref };

/// gatesStrided for up to `max_gates` roles fused along grid.z with barriers between steps; outputs must be distinct.
pub fn gatesEach(p: *const Pipelines, e: mtl.ComputeEncoder, items: []const Gate, rows: u32, m_stride: u32) void {
    std.debug.assert(rows > 0 and rows <= 256 and items.len > 0 and items.len <= max_gates);
    const roles: u32 = @intCast(items.len);
    const args = GateArgs{ .rows = rows, .stride = m_stride, .roles = roles };
    const slot0 = items[0];
    const slot1 = if (items.len > 1) items[1] else slot0;
    const slot2 = if (items.len > 2) items[2] else slot0;
    const norm_ref = normalizedOf(p, 0);
    const zeta_ref = zetaOf(p, 0);
    e.setPipeline(p.roles_norm);
    bind(e, &.{ slot0.h, slot1.h, slot2.h, slot0.m, slot1.m, slot2.m, norm_ref });
    e.setValue(args, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, roles), mtl.Size.of(256, 1, 1));
    e.barrier();
    e.setPipeline(p.roles_zeta);
    bind(e, &.{ norm_ref, slot0.down, slot1.down, slot2.down, zeta_ref });
    e.setValue(args, 5);
    e.dispatchGroups(mtl.Size.of((rank + 255) / 256, rows, roles), mtl.Size.of(256, 1, 1));
    e.barrier();
    e.setPipeline(p.roles_blend);
    bind(e, &.{ slot0.h, slot1.h, slot2.h, slot0.m, slot1.m, slot2.m, zeta_ref, slot0.forget, slot1.forget, slot2.forget, slot0.gain, slot1.gain, slot2.gain, slot0.out, slot1.out, slot2.out });
    e.setValue(args, 16);
    e.dispatchThreads(mtl.Size.of(hidden, rows, roles), mtl.Size.of(32, 1, 1));
}

fn normalizedOf(p: *const Pipelines, tuple: usize) Ref {
    return .{ .buf = p.roles_scratch, .off = tuple * 256 * gate_row_bytes };
}
fn zetaOf(p: *const Pipelines, tuple: usize) Ref {
    return .{ .buf = p.roles_scratch, .off = tuple * 256 * gate_row_bytes + 256 * 2 * hidden * 2 };
}

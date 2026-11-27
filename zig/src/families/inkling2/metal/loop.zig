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
        var pipes: [5]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][]const u8{ plan.Kind.messages.symbol(), plan.Kind.gates.symbol(), "tf_inkling2_loop_norm", "tf_inkling2_loop_down", "tf_inkling2_loop_apply" }, [_]usize{ 256, 256, 256, 32, 256 }, &pipes) |name, req_threads, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < req_threads) return error.LoopThreads;
        }
        return .{ .messages = pipes[0], .gates = pipes[1], .norm = pipes[2], .down = pipes[3], .apply = pipes[4] };
    }
    pub fn deinit(p: *Pipelines) void {
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
    e.setPipeline(p.down);
    bind(e, &.{ normalized, down, zeta });
    e.setValue(args, 3);
    e.dispatchGroups(mtl.Size.of(rank / 32, rows, 1), mtl.Size.of(32, 1, 1));
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

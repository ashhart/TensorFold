//! The model's remaining methods.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
const kv = @import("kv.zig");
const ops = @import("ops.zig");
const mhc = @import("mhc.zig");
const csa2 = @import("csa2.zig");
const csa2_index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const model = @import("model.zig");
const lookahead = @import("model_lookahead.zig");
const Model = model.Model;
const CondProf = Model.CondProf;
const Snapshot = Model.Snapshot;
const Pipelines = model.Pipelines;
const Options = model.Options;
const View = model.View;
const Logits = model.Logits;
const Seg = model.Seg;
const HidHash = model.HidHash;
const tensor = model.tensor;
const asF32 = model.asF32;
const convert = model.convert;
const splitStage = model.splitStage;
const completed = model.completed;
const nowNs = model.nowNs;

pub fn cpuHashRows() u32 {
    const S = struct {
        var cached: i64 = -1;
    };
    if (S.cached < 0) S.cached = if (std.c.getenv("TF_DSV41_CPU_HASH")) |e|
        (std.fmt.parseInt(i64, std.mem.span(e), 10) catch 8)
    else
        8;
    return @intCast(@max(0, S.cached));
}
pub fn hashPass(m: *Model, slot: u32, rows: u32) !void {
    const p = &m.pipelines;
    const ws = &m.workspace;
    if (rows <= cpuHashRows()) {
        try engram.encodeHashHost(try ws.historyView(slot), ws.view(.ids), ws.view(.token_map), ws.view(.engram_ids_0), ws.view(.engram_ids_1), rows);
        try m.stageRows(rows);
        return;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const e = m.queue.commandBufferUnretained();
    var enc = e.compute(.serial);
    for (engram.hash.layers, 0..) |meta, index| {
        try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
    }
    try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
    enc.end();
    e.commit();
    e.wait();
    try completed(e);
    try m.stageRows(rows);
}
pub fn hashPassSegs(m: *Model, segs: []const Seg, rows: u32) !void {
    const ws = &m.workspace;
    var r0: usize = 0;
    for (segs) |sg| {
        const cols: usize = engram.hash.columns;
        try engram.encodeHashHost(try ws.historyView(sg.slot), ws.view(.ids).shifted(r0 * 4), ws.view(.token_map), ws.view(.engram_ids_0).shifted(r0 * cols * 4), ws.view(.engram_ids_1).shifted(r0 * cols * 4), @intCast(sg.ids.len));
        r0 += sg.ids.len;
    }
    try m.stageRows(rows);
}
pub fn engramThreads(m: *Model) usize {
    // The mapped default is sixteen for the large gather too.
    var v: usize = if (m.stream) |s| (if (s.mapped()) 16 else 1) else 1;
    if (std.c.getenv("TF_DSV41_ENGRAM_THREADS")) |e| {
        v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
        if (v == 0 or v > 64) v = 1;
    }
    return v;
}
pub fn latchStage() bool {
    const S = struct {
        var cached: i8 = 0;
    };
    if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_LATCH")) |e|
        (if (std.mem.eql(u8, std.mem.span(e), "gather")) -1 else 1)
    else
        1;
    return S.cached > 0;
}
pub fn engramPrefetch() bool {
    const S = struct {
        var cached: i8 = 0;
    };
    if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PF")) |e|
        (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
    else
        1;
    return S.cached > 0;
}
pub fn engramSmallThreads() usize {
    const S = struct {
        var cached: usize = 0;
    };
    if (S.cached != 0) return S.cached;
    var v: usize = 16;
    if (std.c.getenv("TF_DSV41_ENGRAM_SMALL_THREADS")) |e| {
        v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
        if (v == 0 or v > 16) v = 16;
    }
    S.cached = v;
    return v;
}
pub fn engramAdapt() bool {
    const S = struct {
        var cached: i8 = 0;
    };
    if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_ADAPT")) |e|
        (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
    else
        1;
    return S.cached > 0;
}
pub fn engramPendingOn() bool {
    const S = struct {
        var cached: i8 = 0;
    };
    if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PEND")) |e|
        (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
    else
        1;
    return S.cached > 0;
}
pub fn engramPipeOn() bool {
    // Read per forward: the same process can run the control and the arm.
    if (std.c.getenv("TF_DSV41_ENGRAM_PIPE")) |e| return !std.mem.eql(u8, std.mem.span(e), "0");
    return true;
}
pub fn engramPipeRows(m: *Model, rows: u32) bool {
    if (rows <= 8 or !engramPipeOn()) return false;
    if (m.stream == null or m.stream_stage == null) return false;
    if (engram.hash.layers.len != 2) return false;
    if (engram.hash.layers[1].layer_id <= engram.hash.layers[0].layer_id) return false;
    if (m.profile != null or m.observer != null) return false;
    return std.c.getenv("TF_DSV41_LAYER_CB") == null and
        std.c.getenv("TF_DSV41_STAGE_CB") == null and
        std.c.getenv("TF_DSV41_ATTN_PROF") == null;
}
pub fn prefetchPending(m: *Model, slot: u32, pending: u32) void {
    const src = m.stream orelse return;
    if (!src.mapped()) return;
    const ws = &m.workspace;
    const history = ws.historyView(slot) catch return;
    var ids: [2][engram.hash.columns]u32 = undefined;
    engram.peekHashHost(history, ws.view(.token_map), pending, &ids) catch return;
    for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |_, li| src.prefetch(li, &ids[li]);
}
pub fn stageFnv(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}
pub fn stageHash(m: *Model, rows: u32, tag: []const u8) void {
    if (std.c.getenv("TF_DSV41_ENGRAM_STAGE_HASH") == null) return;
    const buf = m.stream_stage orelse return;
    const ws = &m.workspace;
    const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
    const base: [*]const u8 = buf.contents();
    const n_ids: usize = @as(usize, rows) * engram.hash.columns;
    const i0v = ws.view(.engram_ids_0);
    const i1v = ws.view(.engram_ids_1);
    const ids_a: [*]const u32 = @ptrCast(@alignCast(i0v.buffer.contents() + i0v.offset));
    const ids_b: [*]const u32 = @ptrCast(@alignCast(i1v.buffer.contents() + i1v.offset));
    std.log.info("engram stage hash {s} rows {d} ids0 {x} ids1 {x} staged0 {x} staged1 {x}", .{
        tag,                                             rows,
        stageFnv(std.mem.sliceAsBytes(ids_a[0..n_ids])), stageFnv(std.mem.sliceAsBytes(ids_b[0..n_ids])),
        stageFnv(base[0..layer_bytes]),                  stageFnv(base[layer_bytes..][0..layer_bytes]),
    });
}
pub fn stageLayer(m: *Model, rows: u32, li: usize) !void {
    const src = m.stream orelse return;
    const buf = m.stream_stage orelse return error.EngramNotOpen;
    const ws = &m.workspace;
    const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
    const base: [*]u8 = buf.contents();
    const n_ids: usize = @as(usize, rows) * engram.hash.columns;
    const v = ws.view(if (li == 0) work.Field.engram_ids_0 else work.Field.engram_ids_1);
    const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
    const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
    try src.rowsU32Parallel(li, at[0..n_ids], engram.head_dim, out, engramThreads(m));
}
pub fn stageRows(m: *Model, rows: u32) !void {
    const stage_start: u64 = if (m.host_prof) nowNs() else 0;
    defer if (m.host_prof) {
        m.host_stage_ns += nowNs() - stage_start;
    };
    m.stageHash(rows, "pre");
    const src = m.stream orelse return;
    if (lookahead.take(m, m.look_slot, rows)) {
        m.stageHash(rows, "look");
        src.recordWindow(rows);
        return;
    }
    const buf = m.stream_stage orelse return error.EngramNotOpen;
    const ws = &m.workspace;
    const cols: usize = engram.hash.columns;
    const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
    const base: [*]u8 = buf.contents();
    const n_ids: usize = @as(usize, rows) * cols;
    const small = n_ids < 512;
    // The latch's clock is taken HERE, before the page-in. This clock sees prefetch + gather.
    const w0: u64 = if (small and latchStage()) nowNs() else 0;
    const adapt = small and engramAdapt();
    var threads: usize = engramThreads(m);
    var probed = false;
    if (small) {
        if (!adapt) {
            threads = 1;
        } else if (m.engram_par and m.engram_probe_in > 0) {
            m.engram_probe_in -= 1;
            threads = engramSmallThreads();
        } else {
            threads = 1;
            probed = true;
        }
    }
    // The window's page-ins, only where they pay: the prompt path readaheads; the warm band never arrives.
    const pf_start: u64 = if (small and engramPrefetch()) nowNs() else 0;
    // The stage's own major faults, for the HOST_PROF line; the latch below uses the gather's clock.
    const ru0 = std.posix.getrusage(std.posix.rusage.SELF);
    const pf_on = small and m.engram_pf_on and pf_start != 0;
    if (m.host_prof) {
        m.host_stage_pf = if (pf_on) 1 else 0;
        m.host_stage_thr = @intCast(threads);
    }
    if (pf_on) {
        for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
            const v = ws.view(field);
            const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
            src.prefetch(li, at[0..n_ids]);
        }
    }
    const t0: u64 = if (adapt) nowNs() else 0;
    const g0: u64 = if (small) nowNs() else 0;
    const misses0: u64 = if (m.host_prof) src.stats.misses else 0;
    if (!small and engramPipeRows(m, rows)) {
        // The first layer now, the second at its own layer (the GPU runs the earlier layers while that read happens).
        try m.stageLayer(rows, 0);
        m.engram_pending = true;
    } else {
        for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
            const v = ws.view(field);
            const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
            const ids = at[0..n_ids];
            const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
            try src.rowsU32Parallel(li, ids, engram.head_dim, out, threads);
        }
    }
    m.stageHash(rows, "stage");
    if (adapt) {
        const dt = nowNs() - t0;
        if (probed) {
            if (dt > 1_500_000) {
                m.engram_slow_n += 1;
                if (m.engram_slow_n >= 2) {
                    m.engram_par = true;
                    m.engram_probe_in = m.engram_probe_every;
                    m.engram_probe_every = if (m.engram_probe_every >= 128) 128 else m.engram_probe_every * 2;
                    m.engram_slow_n = 0;
                }
            } else {
                m.engram_slow_n = 0;
                m.engram_par = false;
                m.engram_probe_in = 0;
                m.engram_probe_every = 16;
            }
        }
    }
    if (g0 != 0) {
        // The latch's signal is the gather's own clock, taken AFTER the prefetch: the fault counters cannot serve.
        const gather = nowNs() - (if (w0 != 0) w0 else g0);
        if (gather > 400_000) {
            m.engram_pf_on = true;
            m.engram_pf_off = 0;
        } else if (gather < 200_000) {
            if (m.engram_pf_off < 3) m.engram_pf_off += 1;
            if (m.engram_pf_off >= 3) m.engram_pf_on = false;
        } else {
            m.engram_pf_off = 0;
        }
    }
    if (m.host_prof) {
        m.host_stage_misses +%= src.stats.misses -% misses0;
        const faults1: i64 = std.posix.getrusage(std.posix.rusage.SELF).majflt;
        if (faults1 > ru0.majflt) m.host_stage_faults +%= @intCast(faults1 - ru0.majflt);
    }
    src.recordWindow(rows);
}

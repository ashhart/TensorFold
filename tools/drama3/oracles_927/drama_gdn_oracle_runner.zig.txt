//! tf-drama-gdn-canonical-oracle's device side: guarded controller memory and the runner that executes each case.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const gdn = drama.backbone.gdn;
const Allocator = std.mem.Allocator;
const chunk = gdn.chunk_rows;
const gdn_oracle = @import("drama_gdn_canonical_oracle.zig");
const gdn_cases = @import("drama_gdn_oracle_cases.zig");
const dv = gdn_oracle.dv;
const layer_count = gdn_oracle.layer_count;
const guard = gdn_oracle.guard;
const y_guard_rows = gdn_oracle.y_guard_rows;
const fill = gdn_oracle.fill;
const probe_fields = gdn_oracle.probe_fields;
const probe_available = gdn_oracle.probe_available;
const Geometry = gdn_cases.Geometry;
const Op = gdn_cases.Op;
const Case = gdn_cases.Case;
const Cases = gdn_cases.Cases;
const Errors = gdn_oracle.Errors;
const sha256Hex = gdn_oracle.sha256Hex;
const nowNs = gdn_oracle.nowNs;
const notFill = gdn_oracle.notFill;
const nonfiniteBf16 = gdn_oracle.nonfiniteBf16;
const unwrittenBf16 = gdn_oracle.unwrittenBf16;
const nonfiniteF32 = gdn_oracle.nonfiniteF32;
const bitsDifferent = gdn_oracle.bitsDifferent;
const int = gdn_oracle.int;
const sameBank = gdn_oracle.sameBank;

/// One layer's caller-owned controller memory with guards between every buffer, and its frozen-control states.
pub const LayerMem = struct {
    region: cuda.DeviceBuffer,
    old: cuda.DeviceBuffer, // the frozen sequential chain's state (frozen prefill/decode exports)
    state: gdn.State,
    bank_bytes: usize,
    qkv_cache_bytes: usize,
    ba_cache_bytes: usize,

    pub fn offsets(m: LayerMem) [4]usize {
        const b0 = guard;
        const b1 = b0 + m.bank_bytes + guard;
        const q = b1 + m.bank_bytes + guard;
        const a = q + m.qkv_cache_bytes + guard;
        return .{ b0, b1, q, a };
    }

    pub fn total(bank: usize, qkvc: usize, bac: usize) usize {
        return 5 * guard + 2 * bank + qkvc + bac;
    }

    /// [start, len) of the five guards.
    fn guards(m: LayerMem) [5][2]usize {
        const o = m.offsets();
        return .{ .{ 0, guard }, .{ o[0] + m.bank_bytes, guard }, .{ o[1] + m.bank_bytes, guard }, .{ o[2] + m.qkv_cache_bytes, guard }, .{ o[3] + m.ba_cache_bytes, guard } };
    }
};

pub const GeoMem = struct {
    a_log: cuda.DeviceBuffer,
    dt_bias: cuda.DeviceBuffer,
    layers: [layer_count]LayerMem,
    ws: cuda.DeviceBuffer, // guard | workspace (largest layout) | guard
    y: cuda.DeviceBuffer, // guard rows | max_rows | guard rows, bf16 rows of nv * 128
    y_old: cuda.DeviceBuffer,
    y_base: cuda.DeviceBuffer,
    base_state: cuda.DeviceBuffer,
};

fn view(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
}

const Recur = struct { qkv: u64, ld_qkv: usize, ba: u64, ld_ba: usize, a_log: u64, dt_bias: u64, state: u64, y: u64, rows: usize, nk: usize, nv: usize, scale: f32 };

/// The frozen image's exports with backbone-kernels-pre-chunk.zig's launch: grid (128 / 32, nv), 128 threads.
fn launchFrozen(f: cuda.Function, s: cuda.Stream, r: Recur) !void {
    var a: cuda.Args = .{};
    a.add(r.qkv);
    a.add(int(r.ld_qkv));
    a.add(r.ba);
    a.add(int(r.ld_ba));
    a.add(r.a_log);
    a.add(r.dt_bias);
    a.add(r.state);
    a.add(r.y);
    a.add(int(r.rows));
    a.add(int(r.nk));
    a.add(int(r.nv));
    a.add(r.scale);
    try cuda.launch.launch(f, .{ .grid = .{ .x = dv / 32, .y = @intCast(r.nv), .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// What one op observed in one pass.
pub const OpRecord = struct {
    pending_before: usize = 0,
    committed_before: u1 = 0,
    pending_after: usize = 0,
    committed_after: u1 = 0,
    current_bank: ?u1 = null,
    y_sha: [64]u8 = @splat('-'),
    state_sha: [64]u8 = @splat('-'),
    old_y_sha: [64]u8 = @splat('-'),
    old_state_sha: [64]u8 = @splat('-'),
    base_y_sha: [64]u8 = @splat('-'),
    base_state_sha: [64]u8 = @splat('-'),
    base_y_bits_different: usize = 0,
    base_state_bits_different: usize = 0,
    y_unwritten: usize = 0,
    y_nonfinite: usize = 0,
    y_guard_before_changed: usize = 0,
    y_guard_after_changed: usize = 0,
    state_nonfinite: usize = 0,
    ws_front_guard_changed: usize = 0,
    ws_tail_changed: usize = 0,
    layer_guards_changed: usize = 0,
    inputs_unchanged: bool = true,
    elapsed_ns: i128 = 0,
    probe_sha: [probe_fields.len][64]u8 = @splat(@splat('-')),
    probe_nonfinite: [probe_fields.len]usize = @splat(0),
    probe_in_layout: bool = false,

    pub fn sameOutputs(a: OpRecord, b: OpRecord) bool {
        return a.pending_after == b.pending_after and a.committed_after == b.committed_after and sameBank(a.current_bank, b.current_bank) and
            std.mem.eql(u8, &a.y_sha, &b.y_sha) and std.mem.eql(u8, &a.state_sha, &b.state_sha) and
            std.mem.eql(u8, &a.old_y_sha, &b.old_y_sha) and std.mem.eql(u8, &a.old_state_sha, &b.old_state_sha) and
            std.mem.eql(u8, &a.base_y_sha, &b.base_y_sha) and std.mem.eql(u8, &a.base_state_sha, &b.base_state_sha) and
            std.mem.eql(u8, std.mem.sliceAsBytes(&a.probe_sha), std.mem.sliceAsBytes(&b.probe_sha));
    }
};

pub const Runner = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const gdn.Kernels,
    frozen_decode_fn: cuda.Function,
    frozen_prefill_fn: cuda.Function,
    cs: *const Cases,
    stream_bufs: []const [2]cuda.DeviceBuffer, // per stream: qkv, ba
    out: std.Io.Dir,
    errors: *Errors,
    scale: f32,
    // host scratch, sized once for the largest geometry and stream
    scratch: []align(16) u8,
    scratch2: []align(16) u8,

    fn sync(r: *const Runner) !void {
        try r.stream.synchronize();
    }

    fn download(r: *const Runner, ptr: u64, len: usize, dst: []align(16) u8) ![]align(16) u8 {
        const buf = dst[0..len];
        try view(r.d, ptr, len).download(0, buf);
        return buf;
    }

    /// Zero (or NaN-poison) the layer's caches, zero both banks, resetMetadata, then a nonzero state into bank0.
    fn reset(r: *Runner, g: Geometry, m: *GeoMem, layer: usize, init: ?usize, poison_cache: bool, rec: *OpRecord) !void {
        const lm = &m.layers[layer];
        const o = lm.offsets();
        const base = lm.region.ptr;
        try view(r.d, base + o[0], lm.bank_bytes).fill8(0, r.stream.handle);
        try view(r.d, base + o[1], lm.bank_bytes).fill8(0, r.stream.handle);
        const cache_value: u8 = if (poison_cache) fill else 0;
        try view(r.d, base + o[2], lm.qkv_cache_bytes).fill8(cache_value, r.stream.handle);
        try view(r.d, base + o[3], lm.ba_cache_bytes).fill8(cache_value, r.stream.handle);
        try lm.old.fill8(0, r.stream.handle);
        rec.pending_before = lm.state.pending;
        rec.committed_before = lm.state.committed_bank;
        lm.state.resetMetadata();
        try r.sync();
        if (init) |i| {
            const bytes = r.cs.inits[i].file.bytes;
            try view(r.d, lm.state.banks[0], g.stateBytes()).upload(0, bytes);
            try lm.old.upload(0, bytes);
        }
        rec.pending_after = lm.state.pending;
        rec.committed_after = lm.state.committed_bank;
        rec.current_bank = bankOf(lm.state);
        if (lm.state.pending != 0 or lm.state.committed_bank != 0 or !sameBank(rec.current_bank, 0))
            try r.errors.add("layer {d}: after resetMetadata pending {d} committed_bank {d}, current() not bank0", .{ layer, lm.state.pending, lm.state.committed_bank });
        const cur = try r.download(lm.state.current(), g.stateBytes(), r.scratch);
        rec.state_sha = sha256Hex(cur);
        if (init) |i| {
            if (!std.mem.eql(u8, cur, r.cs.inits[i].file.bytes)) try r.errors.add("layer {d}: bank0 does not hold the uploaded initial state", .{layer});
        } else if (notZero(cur)) try r.errors.add("layer {d}: zeroed bank0 is not zero", .{layer});
    }

    fn notZero(bytes: []const u8) bool {
        for (bytes) |b| {
            if (b != 0) return true;
        }
        return false;
    }

    fn bankOf(s: gdn.State) ?u1 {
        const c = s.current();
        if (c == s.banks[0]) return 0;
        if (c == s.banks[1]) return 1;
        return null;
    }

    fn checkInputs(r: *Runner, g: Geometry, m: *GeoMem, op: Op) !bool {
        const s = r.cs.streams[op.stream];
        const bufs = r.stream_bufs[op.stream];
        var same = true;
        for ([_]struct { []align(16) const u8, cuda.DeviceBuffer }{ .{ s.qkv.bytes, bufs[0] }, .{ s.ba.bytes, bufs[1] }, .{ g.a_log.bytes, m.a_log }, .{ g.dt_bias.bytes, m.dt_bias } }) |pair| {
            const host, const dev = pair;
            const got = try r.download(dev.ptr, host.len, r.scratch2);
            if (!std.mem.eql(u8, got, host)) same = false;
        }
        return same;
    }

    fn checkLayerGuards(r: *Runner, m: *GeoMem) !usize {
        var changed: usize = 0;
        for (m.layers) |lm| {
            for (lm.guards()) |gr| changed += notFill(try r.download(lm.region.ptr + gr[0], gr[1], r.scratch2));
        }
        return changed;
    }

    /// Rows [0, n) of the guarded output buffer, after checking coverage and both guards.
    fn checkY(r: *Runner, g: Geometry, buf: cuda.DeviceBuffer, n: usize, rec: *OpRecord) ![]align(16) u8 {
        const rb = g.rowBytes();
        const all = try r.download(buf.ptr, buf.len, r.scratch2);
        rec.y_guard_before_changed = notFill(all[0 .. y_guard_rows * rb]);
        const rows: []align(16) u8 = @alignCast(all[y_guard_rows * rb ..][0 .. n * rb]);
        rec.y_guard_after_changed = notFill(all[(y_guard_rows + n) * rb ..]);
        const values = std.mem.bytesAsSlice(u16, rows);
        rec.y_unwritten = unwrittenBf16(values);
        rec.y_nonfinite = nonfiniteBf16(values);
        return rows;
    }

    pub fn runCase(r: *Runner, c: Case, m: *GeoMem, records: []OpRecord, pass: usize, ys: *[layer_count]std.ArrayList(u8), old_ys: *[layer_count]std.ArrayList(u8), base_ys: *[layer_count]std.ArrayList(u8)) !void {
        const g = r.cs.geometries[c.geometry];
        const ws_proto, const ws_bytes = gdn.Workspace.layout(c.max_rows, g.nk, g.nv);
        const ws_base = m.ws.ptr + guard;
        const ws = ws_proto.place(ws_base);
        const rb = g.rowBytes();
        const y_ptr = m.y.ptr + y_guard_rows * rb;
        var path_buffer: [256]u8 = undefined;

        // a pass starts from both layers' case initial states
        for (0..layer_count) |layer| {
            var rec: OpRecord = .{};
            try r.reset(g, m, layer, c.init[layer], false, &rec);
        }
        for (c.ops, records, 0..) |op, *rec, oi| {
            rec.* = .{};
            const lm = &m.layers[op.layer];
            if (op.kind == .reset) {
                try r.reset(g, m, op.layer, op.init, op.poison_cache, rec);
                rec.layer_guards_changed = try r.checkLayerGuards(m);
                continue;
            }
            const sb = r.stream_bufs[op.stream];
            const s = r.cs.streams[op.stream];
            const run: gdn.Run = .{
                .qkv = sb[0].ptr + op.start * s.ld_qkv * 2,
                .ld_qkv = s.ld_qkv,
                .ba = sb[1].ptr + op.start * s.ld_ba * 2,
                .ld_ba = s.ld_ba,
                .a_log = m.a_log.ptr,
                .dt_bias = m.dt_bias.ptr,
                .y = y_ptr,
                .rows = op.rows,
                .nk = g.nk,
                .nv = g.nv,
                .dv = dv,
                .scale = r.scale,
            };
            const frozen: Recur = .{ .qkv = run.qkv, .ld_qkv = run.ld_qkv, .ba = run.ba, .ld_ba = run.ld_ba, .a_log = run.a_log, .dt_bias = run.dt_bias, .state = lm.old.ptr, .y = m.y_old.ptr, .rows = op.rows, .nk = g.nk, .nv = g.nv, .scale = r.scale };
            rec.pending_before = lm.state.pending;
            rec.committed_before = lm.state.committed_bank;
            try m.y.fill8(fill, r.stream.handle);
            if (op.kind == .prefill) {
                try m.ws.fill8(fill, r.stream.handle);
                try r.sync();
                const t0 = nowNs(r.io);
                try r.kernels.prefill(r.stream, &lm.state, ws, run);
                try r.sync();
                rec.elapsed_ns = nowNs(r.io) - t0;
                rec.ws_front_guard_changed = notFill(try r.download(m.ws.ptr, guard, r.scratch2));
                rec.ws_tail_changed = notFill(try r.download(ws_base + ws_bytes, m.ws.len - guard - ws_bytes, r.scratch2));
                if (op.probe) try r.probe(c, g, ws, ws_base, ws_bytes, op, oi, rec, pass, &path_buffer);
            } else {
                // the unchanged frozen decode on a copy of the state decode must start from, the same rows
                try m.y_base.fill8(fill, r.stream.handle);
                try m.base_state.copyFrom(0, lm.state.current(), g.stateBytes(), r.stream.handle);
                try r.sync();
                const t0 = nowNs(r.io);
                try r.kernels.decode(r.stream, &lm.state, run);
                try r.sync();
                rec.elapsed_ns = nowNs(r.io) - t0;
                var base = frozen;
                base.state = m.base_state.ptr;
                base.y = m.y_base.ptr;
                try launchFrozen(r.frozen_decode_fn, r.stream, base);
                try r.sync();
            }
            rec.pending_after = lm.state.pending;
            rec.committed_after = lm.state.committed_bank;
            rec.current_bank = bankOf(lm.state);
            if (rec.current_bank == null) try r.errors.add("{s} op {d}: State.current() is neither bank", .{ c.name, oi });

            // the controller's outputs, then its state, before any control touches device memory again
            const y = try r.checkY(g, m.y, op.rows, rec);
            rec.y_sha = sha256Hex(y);
            if (pass == 0) try ys[op.layer].appendSlice(r.gpa, y);
            const y_copy = try r.gpa.alignedAlloc(u8, .@"16", y.len);
            defer r.gpa.free(y_copy);
            @memcpy(y_copy, y);
            const state = try r.download(lm.state.current(), g.stateBytes(), r.scratch);
            rec.state_sha = sha256Hex(state);
            rec.state_nonfinite = nonfiniteF32(state);
            if (pass == 0) {
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.state.f32", .{ c.name, oi });
                try r.out.writeFile(r.io, .{ .sub_path = name, .data = state, .flags = .{ .exclusive = true } });
            }
            if (op.kind == .decode) {
                const by = try r.download(m.y_base.ptr, op.rows * rb, r.scratch2);
                rec.base_y_sha = sha256Hex(by);
                rec.base_y_bits_different = bitsDifferent(u16, by, y_copy);
                if (pass == 0) try base_ys[op.layer].appendSlice(r.gpa, by);
                const bs = try r.download(m.base_state.ptr, g.stateBytes(), r.scratch2);
                rec.base_state_sha = sha256Hex(bs);
                rec.base_state_bits_different = bitsDifferent(u32, bs, state);
                if (rec.base_y_bits_different != 0 or rec.base_state_bits_different != 0) {
                    try r.errors.add("{s} op {d}: decode differs from the frozen decode export on identical state and rows ({d} output, {d} state elements)", .{ c.name, oi, rec.base_y_bits_different, rec.base_state_bits_different });
                    if (pass == 0) {
                        const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.decbase.state.f32", .{ c.name, oi });
                        try r.out.writeFile(r.io, .{ .sub_path = name, .data = bs, .flags = .{ .exclusive = true } });
                    }
                }
            }

            // the frozen sequential chain on its own state: frozen prefill export for prefill, frozen decode for decode
            try launchFrozen(if (op.kind == .prefill) r.frozen_prefill_fn else r.frozen_decode_fn, r.stream, frozen);
            try r.sync();
            const oy = try r.download(m.y_old.ptr, op.rows * rb, r.scratch2);
            rec.old_y_sha = sha256Hex(oy);
            if (pass == 0) try old_ys[op.layer].appendSlice(r.gpa, oy);
            const os = try r.download(lm.old.ptr, g.stateBytes(), r.scratch2);
            rec.old_state_sha = sha256Hex(os);
            if (pass == 0 and (c.save_old_states or c.last_op[op.layer] == oi)) {
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.old.state.f32", .{ c.name, oi });
                try r.out.writeFile(r.io, .{ .sub_path = name, .data = os, .flags = .{ .exclusive = true } });
            }

            rec.inputs_unchanged = try r.checkInputs(g, m, op);
            rec.layer_guards_changed = try r.checkLayerGuards(m);
            var prefix_buffer: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buffer, "{s} pass {d} op {d} ({s} layer {d} rows {d})", .{ c.name, pass, oi, @tagName(op.kind), op.layer, op.rows });
            if (!rec.inputs_unchanged) try r.errors.add("{s}: inputs changed", .{prefix});
            if (rec.y_unwritten != 0 or rec.y_nonfinite != 0) try r.errors.add("{s}: output rows unwritten {d}, nonfinite {d}", .{ prefix, rec.y_unwritten, rec.y_nonfinite });
            if (rec.y_guard_before_changed != 0 or rec.y_guard_after_changed != 0) try r.errors.add("{s}: output overrun: {d} bytes before, {d} bytes after the rows changed", .{ prefix, rec.y_guard_before_changed, rec.y_guard_after_changed });
            if (rec.state_nonfinite != 0) try r.errors.add("{s}: {d} nonfinite state values", .{ prefix, rec.state_nonfinite });
            if (rec.ws_front_guard_changed != 0 or rec.ws_tail_changed != 0) try r.errors.add("{s}: workspace wrote outside its layout ({d} front-guard bytes, {d} bytes past layout {d})", .{ prefix, rec.ws_front_guard_changed, rec.ws_tail_changed, ws_bytes });
            if (rec.layer_guards_changed != 0) try r.errors.add("{s}: {d} bank/cache guard bytes changed", .{ prefix, rec.layer_guards_changed });
            if (rec.pending_after >= chunk) try r.errors.add("{s} pass {d} op {d}: pending {d} is not below {d}", .{ c.name, pass, oi, rec.pending_after, chunk });
        }
    }

    fn probe(r: *Runner, c: Case, g: Geometry, ws: gdn.Workspace, ws_base: u64, ws_bytes: usize, op: Op, oi: usize, rec: *OpRecord, pass: usize, path_buffer: *[256]u8) !void {
        if (comptime !probe_available) {
            if (pass == 0) try r.errors.add("{s} op {d}: gate probe requested but gdn.Workspace has no u64 fields g, gc and beta", .{ c.name, oi });
            return;
        } else {
            const bytes = op.rows * g.nv * 4;
            rec.probe_in_layout = true;
            inline for (probe_fields, 0..) |name, fi| {
                const ptr: u64 = @field(ws, name);
                if (ptr < ws_base or ptr + bytes > ws_base + ws_bytes) {
                    rec.probe_in_layout = false;
                    try r.errors.add("{s} op {d}: Workspace.{s} [{d} rows][{d}] f32 is not inside the placed layout", .{ c.name, oi, name, op.rows, g.nv });
                } else {
                    const got = try r.download(ptr, bytes, r.scratch2);
                    rec.probe_sha[fi] = sha256Hex(got);
                    rec.probe_nonfinite[fi] = nonfiniteF32(got);
                    if (pass == 0) {
                        const file = try std.fmt.bufPrint(path_buffer, "{s}/op-{d:0>4}.ws-{s}.f32", .{ c.name, oi, name });
                        try r.out.writeFile(r.io, .{ .sub_path = file, .data = got, .flags = .{ .exclusive = true } });
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------

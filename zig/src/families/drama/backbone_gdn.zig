//! Gated DeltaNet for one layer: canonical 64-row chunked prefill (after SGLang/FLA) and the packed decode recurrence.
const std = @import("std");
const cuda = @import("cuda");

/// Prefill chunk length (fla chunk.py CHUNK_SIZE); the row cache holds one chunk.
pub const chunk_rows: usize = 64;
/// Key and value head width the kernels are written for.
const head_dim: usize = 128;
/// l2norm_fwd's eps.
const l2norm_eps: f32 = 1e-6;
const max_int: usize = std.math.maxInt(i32);
/// Grid y / z limit.
const max_grid_yz: usize = 65535;

/// One layer's recurrent state. All addresses are caller device memory; resetMetadata never touches it.
pub const State = struct {
    banks: [2]u64, // FP32 [nv][128 v][128 k] each, distinct, 16-byte aligned
    qkv_cache: u64, // BF16 [64, (2*nk+nv)*128]
    ba_cache: u64, // BF16 [64, 2*nv]
    pending: usize = 0, // rows of the unfinished chunk in the caches
    committed_bank: u1 = 0,

    /// The latest state: the trial bank while a chunk is unfinished, else the committed bank.
    pub fn current(self: *const State) u64 {
        return self.banks[if (self.pending > 0) ~self.committed_bank else self.committed_bank];
    }

    /// Back to an empty state's metadata. DOES NOT zero device memory: the caller zeroes the banks first.
    pub fn resetMetadata(self: *State) void {
        self.pending = 0;
        self.committed_bank = 0;
    }

    fn committed(self: *const State) u64 {
        return self.banks[self.committed_bank];
    }

    fn other(self: *const State) u64 {
        return self.banks[~self.committed_bank];
    }

    fn commit(self: *State) void {
        self.committed_bank = ~self.committed_bank;
        self.pending = 0;
    }
};

/// Prefill intermediates for segments of up to rows rows, reusable across layers on one stream.
pub const Workspace = struct {
    rows: usize, // capacity, a multiple of chunk_rows
    nk: usize,
    nv: usize,
    qk: u64 = 0, // bf16 [rows][2 nk][128] normalized q heads then k heads
    g: u64 = 0, // f32 [rows][nv]
    gc: u64 = 0, // f32 [rows][nv] chunk-local cumsum of g
    beta: u64 = 0, // f32 [rows][nv] bf16 values
    a: u64 = 0, // bf16 [rows][nv][64] (I + A)^-1
    w: u64 = 0, // bf16 [rows][nv][128]
    u: u64 = 0, // bf16 [rows][nv][128]
    vn: u64 = 0, // bf16 [rows][nv][128] v_new
    h: u64 = 0, // bf16 [rows / 64][nv][128 v][128 k] state at each chunk start

    const device = .{ "qk", "g", "gc", "beta", "a", "w", "u", "vn", "h" };

    /// Region offsets (256-byte aligned) and size for max(64, floor(max_rows / 64) * 64) rows.
    pub fn layout(max_rows: usize, nk: usize, nv: usize) struct { Workspace, usize } {
        const rows = @max(chunk_rows, max_rows / chunk_rows * chunk_rows);
        const vec = rows * nv * head_dim * 2;
        var used: usize = 0;
        var ws: Workspace = .{ .rows = rows, .nk = nk, .nv = nv };
        ws.qk = take(&used, rows * 2 * nk * head_dim * 2);
        ws.g = take(&used, rows * nv * 4);
        ws.gc = take(&used, rows * nv * 4);
        ws.beta = take(&used, rows * nv * 4);
        ws.a = take(&used, rows * nv * chunk_rows * 2);
        ws.w = take(&used, vec);
        ws.u = take(&used, vec);
        ws.vn = take(&used, vec);
        ws.h = take(&used, rows / chunk_rows * nv * head_dim * head_dim * 2);
        return .{ ws, used };
    }

    /// The same layout at device address `base`.
    pub fn place(self: Workspace, base: u64) Workspace {
        var ws = self;
        inline for (device) |name| @field(ws, name) += base;
        return ws;
    }

    fn take(used: *usize, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, used.*, 256);
        used.* = at + bytes;
        return at;
    }
};

/// One call's rows: qkv post-convolution, ba = [b | a], a_log and dt_bias FP32, y BF16 [rows][nv][dv].
pub const Run = struct {
    qkv: u64,
    ld_qkv: usize,
    ba: u64,
    ld_ba: usize,
    a_log: u64,
    dt_bias: u64,
    y: u64,
    rows: usize,
    nk: usize,
    nv: usize,
    dv: usize,
    scale: f32,
};

/// One six-kernel segment from a chunk boundary, state_in -> state_out, emitting rows emit_from.. to y.
const Segment = struct {
    qkv: u64,
    ld_qkv: usize,
    ba: u64,
    ld_ba: usize,
    rows: usize,
    emit_from: usize,
    y: u64,
    state_in: u64,
    state_out: u64,
};

fn int(v: usize) i32 {
    return @intCast(v);
}

fn checkRun(r: Run) !void {
    if (r.dv != head_dim or r.nk == 0 or r.nv == 0 or r.nv % r.nk != 0 or r.nv > max_grid_yz) return error.DramaGdnShape;
    if (r.rows == 0 or r.rows > max_int / (2 * r.nk)) return error.DramaGdnShape;
    if (r.ld_qkv < (2 * r.nk + r.nv) * head_dim or r.ld_qkv > max_int or r.ld_ba < 2 * r.nv or r.ld_ba > max_int) return error.DramaGdnShape;
}

fn checkBanks(st: *const State, nv: usize) !void {
    const bytes = nv * head_dim * head_dim * 4;
    const lo = @min(st.banks[0], st.banks[1]);
    const hi = @max(st.banks[0], st.banks[1]);
    if (st.banks[0] % 16 != 0 or st.banks[1] % 16 != 0 or hi - lo < bytes) return error.DramaGdnState;
    if (st.pending >= chunk_rows) return error.DramaGdnState;
}

pub const Kernels = struct {
    l2norm_fn: cuda.Function,
    gate_fn: cuda.Function,
    kkt_fn: cuda.Function,
    wu_fn: cuda.Function,
    h_fn: cuda.Function,
    o_fn: cuda.Function,
    append_fn: cuda.Function,
    decode_fn: cuda.Function,
    /// Programmatic dependent launch on every launch: only for an image whose kernels wait for their predecessor.
    pdl: bool,

    /// Resolves the exports of the drama_backbone glue image.
    pub fn init(module: *cuda.Module, pdl: bool) !Kernels {
        return .{
            .l2norm_fn = try module.function("drama_gdn_chunk_l2norm"),
            .gate_fn = try module.function("drama_gdn_chunk_gate"),
            .kkt_fn = try module.function("drama_gdn_chunk_kkt"),
            .wu_fn = try module.function("drama_gdn_chunk_wu"),
            .h_fn = try module.function("drama_gdn_chunk_h"),
            .o_fn = try module.function("drama_gdn_chunk_o"),
            .append_fn = try module.function("drama_gdn_chunk_append"),
            .decode_fn = try module.function("drama_gdn_recur"),
            .pdl = pdl,
        };
    }

    fn go(self: *const Kernels, s: cuda.Stream, f: cuda.Function, grid: [3]usize, block: u32, a: *cuda.Args) !void {
        const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = block }, .pdl = self.pdl }, s, a);
    }

    /// Prefill rows on canonical chunk boundaries (see the file comment). Validates everything before the first launch.
    pub fn prefill(self: *const Kernels, stream: cuda.Stream, state: *State, workspace: Workspace, run: Run) !void {
        try checkRun(run);
        try checkBanks(state, run.nv);
        if (state.qkv_cache == 0 or state.ba_cache == 0) return error.DramaGdnState;
        if (workspace.nk != run.nk or workspace.nv != run.nv or workspace.rows == 0 or workspace.rows % chunk_rows != 0) return error.DramaGdnWorkspace;
        const head = if (state.pending > 0) @min(chunk_rows - state.pending, run.rows) else 0;
        const bulk = (run.rows - head) / chunk_rows * chunk_rows;
        if (bulk > workspace.rows or bulk / chunk_rows > max_grid_yz) return error.DramaGdnWorkspace;
        const y_row = run.nv * run.dv * 2;
        var done: usize = 0;
        if (head > 0) {
            try self.unfinished(stream, state, workspace, run, 0, head);
            done = head;
        }
        if (bulk > 0) {
            try self.chain(stream, workspace, run, .{
                .qkv = run.qkv + done * run.ld_qkv * 2,
                .ld_qkv = run.ld_qkv,
                .ba = run.ba + done * run.ld_ba * 2,
                .ld_ba = run.ld_ba,
                .rows = bulk,
                .emit_from = 0,
                .y = run.y + done * y_row,
                .state_in = state.committed(),
                .state_out = state.other(),
            });
            state.commit();
            done += bulk;
        }
        if (done < run.rows) try self.unfinished(stream, state, workspace, run, done, run.rows - done);
    }

    /// Packed decode recurrence in place on the latest state, adopting an unfinished chunk's trial bank first.
    pub fn decode(self: *const Kernels, stream: cuda.Stream, state: *State, run: Run) !void {
        try checkRun(run);
        try checkBanks(state, run.nv);
        if (state.pending > 0) state.commit();
        var a: cuda.Args = .{};
        a.add(run.qkv);
        a.add(int(run.ld_qkv));
        a.add(run.ba);
        a.add(int(run.ld_ba));
        a.add(run.a_log);
        a.add(run.dt_bias);
        a.add(state.committed());
        a.add(run.y);
        a.add(int(run.rows));
        a.add(int(run.nk));
        a.add(int(run.nv));
        a.add(run.scale);
        try self.go(stream, self.decode_fn, .{ run.dv / 32, run.nv, 1 }, 128, &a);
    }

    /// Appends run rows to the unfinished chunk and recomputes it into the other bank; a full chunk commits.
    fn unfinished(self: *const Kernels, stream: cuda.Stream, state: *State, ws: Workspace, run: Run, from: usize, count: usize) !void {
        const at = state.pending;
        const qkv_cols = (2 * run.nk + run.nv) * head_dim;
        const ba_cols = 2 * run.nv;
        var a: cuda.Args = .{};
        a.add(run.qkv + from * run.ld_qkv * 2);
        a.add(int(run.ld_qkv));
        a.add(run.ba + from * run.ld_ba * 2);
        a.add(int(run.ld_ba));
        a.add(state.qkv_cache);
        a.add(state.ba_cache);
        a.add(int(at));
        a.add(int(qkv_cols));
        a.add(int(ba_cols));
        try self.go(stream, self.append_fn, .{ count, 1, 1 }, 256, &a);
        const rows = at + count;
        try self.chain(stream, ws, run, .{
            .qkv = state.qkv_cache,
            .ld_qkv = qkv_cols,
            .ba = state.ba_cache,
            .ld_ba = ba_cols,
            .rows = rows,
            .emit_from = at,
            .y = run.y + from * run.nv * run.dv * 2,
            .state_in = state.committed(),
            .state_out = state.other(),
        });
        if (rows == chunk_rows) state.commit() else state.pending = rows;
    }

    /// l2norm -> gate -> kkt -> wu -> h -> o over one segment.
    fn chain(self: *const Kernels, stream: cuda.Stream, ws: Workspace, run: Run, seg: Segment) !void {
        const nt = (seg.rows + chunk_rows - 1) / chunk_rows;
        const rows = int(seg.rows);
        const nk = int(run.nk);
        const nv = int(run.nv);
        {
            var a: cuda.Args = .{};
            a.add(seg.qkv);
            a.add(int(seg.ld_qkv));
            a.add(ws.qk);
            a.add(rows);
            a.add(nk);
            a.add(l2norm_eps);
            try self.go(stream, self.l2norm_fn, .{ (seg.rows * 2 * run.nk + 15) / 16, 1, 1 }, 256, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(seg.ba);
            a.add(int(seg.ld_ba));
            a.add(run.a_log);
            a.add(run.dt_bias);
            a.add(ws.g);
            a.add(ws.gc);
            a.add(ws.beta);
            a.add(rows);
            a.add(nv);
            try self.go(stream, self.gate_fn, .{ nt, run.nv, 1 }, 64, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(ws.qk);
            a.add(ws.gc);
            a.add(ws.beta);
            a.add(ws.a);
            a.add(rows);
            a.add(nk);
            a.add(nv);
            try self.go(stream, self.kkt_fn, .{ nt, run.nv, 1 }, 32, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(seg.qkv);
            a.add(int(seg.ld_qkv));
            a.add(ws.qk);
            a.add(ws.gc);
            a.add(ws.beta);
            a.add(ws.a);
            a.add(ws.w);
            a.add(ws.u);
            a.add(rows);
            a.add(nk);
            a.add(nv);
            try self.go(stream, self.wu_fn, .{ nt, run.nv, 1 }, 128, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(ws.qk);
            a.add(ws.w);
            a.add(ws.u);
            a.add(ws.gc);
            a.add(seg.state_in);
            a.add(seg.state_out);
            a.add(ws.h);
            a.add(ws.vn);
            a.add(rows);
            a.add(nk);
            a.add(nv);
            try self.go(stream, self.h_fn, .{ head_dim / 32, run.nv, 1 }, 128, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(ws.qk);
            a.add(ws.vn);
            a.add(ws.h);
            a.add(ws.gc);
            a.add(seg.y);
            a.add(rows);
            a.add(int(seg.emit_from));
            a.add(nk);
            a.add(nv);
            a.add(run.scale);
            try self.go(stream, self.o_fn, .{ head_dim / 64, nt, run.nv }, 128, &a);
        }
    }
};

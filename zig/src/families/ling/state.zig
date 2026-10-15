//! Ling-3.1-flash's CUDA memory at TP=1: a forward's Buffers (any sequences) and one sequence's State, an arena each.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");

const kda = graph.kda;
const latent = graph.latent;

/// A verify window's most rows on one sequence: the KDA saves and projection rows a commit replays.
pub const window_rows = 16;
/// The most rows of one roped attention call (a prompt chunk's rows go in blocks of these).
pub const att_rows = 128;
/// MLA's cache row: the 512-wide latent, then the 64 roped key columns.
pub const cache_width = 512 + latent.rope_width;

/// Sub-allocations of one zeroed device allocation, 256-byte aligned.
const Arena = struct {
    buf: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
        var total: usize = 0;
        for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
        const buf = try cuda.DeviceBuffer.alloc(d, @max(total, 256));
        errdefer {
            var b = buf;
            b.free();
        }
        try buf.fill8(0, null);
        var at: usize = 0;
        for (sizes, out) |n, *p| {
            p.* = buf.ptr + at;
            at += std.mem.alignForward(usize, n, 256);
        }
        return .{ .buf = buf };
    }
};

/// The KDA projection row: q | k | v (3 heads 128), then beta's logits, padded to 8 columns.
pub fn kdaStride(w: *const weights.Weights) usize {
    const h: usize = w.config.kda_heads;
    return 3 * h * kda.dk + std.mem.alignForward(usize, h, 8);
}

/// The largest split-K partials any loaded projection takes at 1 .. rows rows.
fn partBytes(m: *const graph.Modules, w: *const weights.Weights, rows: usize) usize {
    var most: usize = 16;
    const one = struct {
        fn of(mo: *const graph.Modules, p: weights.Proj, r: usize, best: *usize) void {
            const q = switch (p) {
                .fp8g, .nvfp4, .fp8, .bf16 => |x| x,
                else => return,
            };
            for (1..r + 1) |k| best.* = @max(best.*, mo.lane.partBytes(k, q));
        }
    };
    for (w.layers) |l| {
        switch (l.mixer) {
            .kda => |a| for ([_]weights.Proj{ a.q, a.k, a.v, a.b, a.f, a.g, a.o }) |p| one.of(m, p, rows, &most),
            .mla => |a| for ([_]?weights.Proj{ a.q, a.q_a, a.q_b, a.kv_a, a.gate, a.o }) |p| if (p) |x| one.of(m, x, rows, &most),
        }
        const mlps: [2]?weights.Mlp = switch (l.ffn) {
            .dense => |x| .{ x, null },
            .moe => |e| .{ e.shared, if (e.routed == .lanes and e.routed.lanes.len > 0) e.routed.lanes[0] else null },
        };
        for (mlps) |x| if (x) |p| for ([_]weights.Proj{ p.gate, p.up, p.down }) |q| one.of(m, q, rows, &most);
    }
    if (w.head) |hd| one.of(m, hd, rows, &most);
    return most;
}

/// A forward's scratch for up to `rows` rows on any sequences: prompt rows (`prompt`) or decode rows.
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    prompt: bool,
    nch: usize, // the most 512-key chunks one attention call reads
    ids: u64, // i32 [rows]
    x: u64, // bf16 [rows][hidden]: the residual stream
    normed: u64,
    y: u64, // a block's output rows
    sh: u64, // the shared expert's rows
    tmp: u64, // one projection's rows before they go into their columns
    part: u64, // the lane's split-K partials
    kp: u64, // bf16 [rows][kdaStride]: a prompt's KDA projection rows (a window's stay in the State)
    ka: u64, // bf16 [rows][kda n]: the decay's projection
    kg: u64, // bf16 [rows][kda n]: the output gate's projection
    saves: kda.Saves, // out bf16; a prompt's k, v, g, b (a window's stay in the State); state_out set per layer
    wide: kda.Wide,
    qa_in: u64, // bf16 [rows][q_lora]
    qr: u64, // bf16 [rows][q_lora]: q_a_layernorm's rows
    q: u64, // bf16 [rows][heads 192]
    kv: u64, // bf16 [rows][576]: kv_a_proj_with_mqa's rows
    lat: u64, // bf16 [rows][576]: normed latent, roped key columns
    qa: u64, // bf16 [rows][heads][576]: absorbed then roped queries
    ol: u64, // bf16 [rows][heads][512]
    vn: u64, // bf16 [rows][heads 128]
    g: u64, // bf16 [rows][heads]: g_proj's rows
    partials: latent.Partials,
    dummy: u64,
    at: u64, // i32: an attention block's first position
    gu: u64, // bf16 [rows][2 width]: [gate | up]
    act: u64, // bf16 [rows][width]
    xs: u64, // fp32 [rows][width / 64]
    mlog: u64, // fp32 [rows][experts]
    rpart: u64, // fp32 [router_ks][rows][experts]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [pairs][moe_inter]
    ey: u64, // bf16 [pairs][hidden], or every expert's rows [experts][rows][hidden] for per-expert lanes
    logits: u64, // bf16 [rows][vocab]

    pub fn init(d: *const cuda.Driver, m: *const graph.Modules, w: *const weights.Weights, rows: usize, prompt: bool, capacity: usize) !Buffers {
        const c = &w.config;
        const D: usize = c.hidden;
        const H: usize = c.kda_heads;
        const n = H * kda.dk;
        const HM: usize = c.mla_heads;
        const ql: usize = @max(c.q_lora, 1);
        const E: usize = c.experts;
        const slots: usize = c.topk + 1;
        const pairs = rows * slots;
        const width: usize = @max(@max(c.dense_inter, c.shared_inter), c.moe_inter);
        var lanes = false;
        for (w.layers) |l| if (l.ffn == .moe and l.ffn.moe.routed == .lanes) {
            lanes = true;
        };
        const ar = @min(rows, att_rows);
        const nch = latent.chunksFor(capacity);
        const hist = if (pairs > cuda.grouped.small) (pairs + cuda.grouped.small - 1) / cuda.grouped.small * (E + 1) else 1;
        const sizes = [_]usize{
            rows * 4,                                        rows * D * 2,                   rows * D * 2,                rows * D * 2,
            rows * D * 2,                                    rows * @max(n, width) * 2,      partBytes(m, w, rows),       if (prompt) rows * kdaStride(w) * 2 else 16,
            rows * n * 2,                                    rows * n * 2,                   rows * n * 2,                rows * n * 4,
            rows * n * 2,                                    rows * n * 4,                   rows * H * 4,                rows * n * 4,
            rows * n * 2,                                    rows * ql * 2,                  rows * ql * 2,               rows * HM * c.qkDim() * 2,
            rows * cache_width * 2,                          rows * cache_width * 2,         rows * HM * cache_width * 2, rows * HM * 512 * 2,
            rows * HM * c.v_dim * 2,                         rows * HM * 2,                  nch * ar * HM * 512 * 4,     nch * ar * HM * 4,
            nch * ar * HM * 4,                               4,                              4,                           rows * 2 * width * 2,
            rows * width * 2,                                rows * (width / 64 + 1) * 4,    rows * E * 4,                glue_router_ks * rows * E * 4,
            pairs * 4,                                       pairs * 4,                      pairs * 4,                   cuda.grouped.maxItems(pairs, E + 1, 16) * 12,
            8,                                               pairs * 4,                      hist * 4,                    pairs * c.moe_inter * 2,
            @max(pairs, if (lanes) E * rows else 0) * D * 2, rows * @as(usize, c.vocab) * 2,
        };
        var p: [sizes.len]u64 = undefined;
        const a = try Arena.init(d, &sizes, &p);
        return .{
            .arena = a,
            .rows = rows,
            .prompt = prompt,
            .nch = nch,
            .ids = p[0],
            .x = p[1],
            .normed = p[2],
            .y = p[3],
            .sh = p[4],
            .tmp = p[5],
            .part = p[6],
            .kp = p[7],
            .ka = p[8],
            .kg = p[9],
            .saves = .{ .out = p[10], .state_out = 0, .k = p[11], .v = p[12], .g = p[13], .b = p[14] },
            .wide = .{ .q = p[15], .y = p[16] },
            .qa_in = p[17],
            .qr = p[18],
            .q = p[19],
            .kv = p[20],
            .lat = p[21],
            .qa = p[22],
            .ol = p[23],
            .vn = p[24],
            .g = p[25],
            .partials = .{ .po = p[26], .pm = p[27], .pl = p[28] },
            .dummy = p[29],
            .at = p[30],
            .gu = p[31],
            .act = p[32],
            .xs = p[33],
            .mlog = p[34],
            .rpart = p[35],
            .pick = p[36],
            .wts = p[37],
            .plan = .{ .members = p[38], .items = p[39], .counts = p[40], .rank = p[41], .hist = p[42] },
            .eact = p[43],
            .ey = p[44],
            .logits = p[45],
        };
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }
};

const glue_router_ks = graph.glue.router_ks;

/// One sequence's committed caches: KDA conv windows and states (two copies), a window's saves, MLA latent rows.
pub const State = struct {
    arena: Arena,
    gpa: std.mem.Allocator,
    capacity: usize,
    pos: usize = 0,
    pending: usize = 0, // rows of the last forward until its commit
    prompt: bool = false, // whether those rows were prompt rows (their KDA states already committed)
    pos_dev: u64, // i32
    conv: u64, // bf16 [kda layers][3][3 heads 128]
    rec: u64, // fp32 [2][kda layers][heads][128][128]
    proj: u64, // bf16 [kda layers][window_rows][kdaStride]
    save_k: u64, // fp32 [kda layers][window_rows][heads 128]
    save_v: u64, // bf16, as save_k
    save_g: u64, // fp32, as save_k
    save_b: u64, // fp32 [kda layers][window_rows][heads]
    cache: u64, // bf16 [mla layers][capacity][576]
    cur: []u1, // each KDA layer's committed copy of rec
    kda_layers: usize,
    mla_layers: usize,
    heads: usize,
    stride: usize, // kdaStride
    rec_layer: usize, // bytes of one layer's state

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *const graph.Graph, capacity: usize) !State {
        const c = &g.w.config;
        const nk = g.kda_layers;
        const nm = g.mla_layers;
        const H: usize = c.kda_heads;
        const n = H * kda.dk;
        const ps = kdaStride(g.w);
        const W = window_rows;
        const sizes = [_]usize{ 4, nk * 3 * 3 * n * 2, 2 * nk * n * kda.dv * 4, nk * W * ps * 2, nk * W * n * 4, nk * W * n * 2, nk * W * n * 4, nk * W * H * 4, nm * capacity * cache_width * 2 };
        var p: [sizes.len]u64 = undefined;
        var a = try Arena.init(d, &sizes, &p);
        errdefer a.buf.free();
        const cur = try gpa.alloc(u1, nk);
        @memset(cur, 0);
        return .{ .arena = a, .gpa = gpa, .capacity = capacity, .pos_dev = p[0], .conv = p[1], .rec = p[2], .proj = p[3], .save_k = p[4], .save_v = p[5], .save_g = p[6], .save_b = p[7], .cache = p[8], .cur = cur, .kda_layers = nk, .mla_layers = nm, .heads = H, .stride = ps, .rec_layer = n * kda.dv * 4 };
    }

    pub fn deinit(st: *State) void {
        st.arena.buf.free();
        st.gpa.free(st.cur);
        st.* = undefined;
    }

    /// A fresh sequence: zero conv windows and states, position 0 (cache rows past it are never read).
    pub fn reset(st: *State, s: cuda.Stream) !void {
        const d = st.arena.buf.d;
        try d.check(d.api.cuMemsetD8Async(st.conv, 0, st.kda_layers * 3 * st.convWidth() * 2, s.handle), "cuMemsetD8Async");
        try d.check(d.api.cuMemsetD8Async(st.rec, 0, 2 * st.kda_layers * st.rec_layer, s.handle), "cuMemsetD8Async");
        @memset(st.cur, 0);
        st.pending = 0;
        st.prompt = false;
        try st.setPos(s, 0);
    }

    pub fn setPos(st: *State, s: cuda.Stream, pos: usize) !void {
        st.pos = pos;
        const v: i32 = @intCast(pos);
        try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.pos_dev, @bitCast(v), 1, s.handle), "cuMemsetD32Async");
    }

    /// The conv window's channels: q | k | v.
    pub fn convWidth(st: *const State) usize {
        return 3 * st.heads * kda.dk;
    }

    pub fn recAt(st: *const State, copy: u1, li: usize) u64 {
        return st.rec + (@as(usize, copy) * st.kda_layers + li) * st.rec_layer;
    }

    pub fn convAt(st: *const State, li: usize) u64 {
        return st.conv + li * 3 * st.convWidth() * 2;
    }

    pub fn cacheAt(st: *const State, mi: usize) u64 {
        return st.cache + mi * st.capacity * cache_width * 2;
    }

    /// Layer li's window saves: projection rows, k, v, decay, beta.
    pub fn saves(st: *const State, li: usize) struct { p: u64, k: u64, v: u64, g: u64, b: u64 } {
        const n = st.heads * kda.dk;
        const kv = li * window_rows * n;
        return .{ .p = st.proj + li * window_rows * st.stride * 2, .k = st.save_k + kv * 4, .v = st.save_v + kv * 2, .g = st.save_g + kv * 4, .b = st.save_b + li * window_rows * st.heads * 4 };
    }
};

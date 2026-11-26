//! Native HuggingFace FP8 E4M3 128x128 block-scaled safetensors loader for GLM-5.3 (zai-org/GLM-5.3).
const std = @import("std");
const Io = std.Io;
const mtl = @import("metal");
const ex = @import("../../core/exlr8.zig");
const job = @import("../../core/exlr8_job.zig");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const pack = ex.pack;

/// Format numbers matching `zig/kernels/metal/core/exlr8_dense.metal`.
const FMT_FP8_BLOCK: u32 = 4;
const FMT_BF16: u32 = 6;
const FMT_FP32: u32 = 7;

/// Per-MoE-layer FP8 E4M3 128x128 block-scaled expert weights held on GPU for this rank's channel slice.
pub const Fp8Moe = struct {
    gate_w: mtl.Buffer, // [257, loc, hidden] u8
    gate_s: mtl.Buffer, // [257, loc/128, hidden/128] f32
    up_w: mtl.Buffer, // [257, loc, hidden] u8
    up_s: mtl.Buffer, // [257, loc/128, hidden/128] f32
    down_w: mtl.Buffer, // [257, hidden, loc] u8
    down_s: mtl.Buffer, // [257, hidden/128, loc/128] f32
    router_bf16_off: u64 = 0,

    pub fn bytes(m: *const Fp8Moe) u64 {
        return m.gate_w.length() + m.gate_s.length() + m.up_w.length() + m.up_s.length() + m.down_w.length() + m.down_s.length();
    }

    pub fn deinit(m: *Fp8Moe) void {
        inline for (.{ m.gate_w, m.gate_s, m.up_w, m.up_s, m.down_w, m.down_s }) |b| b.deinit();
    }
};

/// Whether `dir` contains a standard HF safetensors index (`model.safetensors.index.json`).
pub fn isHfCheckpoint(io: Io, gpa: std.mem.Allocator, dir: []const u8) bool {
    const path = std.fmt.allocPrint(gpa, "{s}/model.safetensors.index.json", .{dir}) catch return false;
    defer gpa.free(path);
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

const OpenShard = struct {
    file: Io.File,
    meta: st.Metadata,

    fn close(s: *OpenShard, io: Io) void {
        s.file.close(io);
        s.meta.deinit();
    }
};

/// Indexed view over `model.safetensors.index.json`; opens only the shards needed by the active layer.
pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    root: []const u8,
    arena: std.heap.ArenaAllocator,
    /// tensor_name -> shard filename (owned by arena).
    map: std.StringHashMapUnmanaged([]const u8) = .empty,
    /// Open shard cache for the current layer (kept small to stay far below macOS's fd limit).
    open: std.StringHashMapUnmanaged(OpenShard) = .empty,

    pub fn init(gpa: std.mem.Allocator, io: Io, root: []const u8) !Store {
        const path = try std.fmt.allocPrint(gpa, "{s}/model.safetensors.index.json", .{root});
        defer gpa.free(path);
        const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(text);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{ .allocate = .alloc_always });
        if (parsed != .object) return error.BadSafetensorsIndex;
        const wm = parsed.object.get("weight_map") orelse return error.BadSafetensorsIndex;
        if (wm != .object) return error.BadSafetensorsIndex;
        var map: std.StringHashMapUnmanaged([]const u8) = .empty;
        try map.ensureTotalCapacity(a, @intCast(wm.object.count()));
        var it = wm.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadSafetensorsIndex;
            map.putAssumeCapacity(e.key_ptr.*, e.value_ptr.string);
        }
        return .{ .gpa = gpa, .io = io, .root = root, .arena = arena, .map = map };
    }

    pub fn deinit(s: *Store) void {
        s.clearOpen();
        s.open.deinit(s.gpa);
        s.arena.deinit();
    }

    /// Close currently open layer shards to keep open file descriptors tiny (< 10).
    pub fn clearOpen(s: *Store) void {
        var it = s.open.valueIterator();
        while (it.next()) |sh| sh.close(s.io);
        s.open.clearRetainingCapacity();
    }

    /// Ensure the shard containing `name` is opened and return both shard handle and tensor header entry.
    pub fn locate(s: *Store, name: []const u8) !struct { shard: *const OpenShard, entry: st.Entry } {
        const fname = s.map.get(name) orelse return error.MissingGlmTensor;
        const gop = try s.open.getOrPut(s.gpa, fname);
        if (!gop.found_existing) {
            const full = try std.fmt.allocPrint(s.gpa, "{s}/{s}", .{ s.root, fname });
            defer s.gpa.free(full);
            var meta = try st.Metadata.read(s.gpa, s.io, full);
            errdefer meta.deinit();
            const file = try Io.Dir.cwd().openFile(s.io, full, .{});
            gop.value_ptr.* = .{ .file = file, .meta = meta };
        }
        const entry = gop.value_ptr.meta.names.get(name) orelse return error.MissingGlmTensor;
        return .{ .shard = gop.value_ptr, .entry = entry };
    }

    /// Pre-open all shards referenced by a list of tensor names so parallel threads can read lock-free.
    fn preopen(s: *Store, name: []const u8) !void {
        _ = try s.locate(name);
    }

    /// Read-only lookup after `preopen` (safe to call from worker threads with no map mutation).
    fn locateOpen(s: *const Store, name: []const u8) !struct { shard: *const OpenShard, entry: st.Entry } {
        const fname = s.map.get(name) orelse return error.MissingGlmTensor;
        const sh = s.open.getPtr(fname) orelse return error.MissingGlmTensor;
        const entry = sh.meta.names.get(name) orelse return error.MissingGlmTensor;
        return .{ .shard = sh, .entry = entry };
    }

    fn readExact(io: Io, file: Io.File, dst: []u8, offset: u64) !void {
        if (try file.readPositionalAll(io, dst, offset) != dst.len) return error.UnexpectedEof;
    }

    /// Read rows `[r0, r1)` of a 2D tensor `[R, C]` (all columns) into `dst`.
    fn readRows(io: Io, sh: *const OpenShard, e: st.Entry, r0: usize, r1: usize, cols: usize, elem: usize, dst: []u8) !void {
        const rows = r1 - r0;
        const off = sh.meta.data + e.begin + r0 * cols * elem;
        std.debug.assert(dst.len == rows * cols * elem);
        try readExact(io, sh.file, dst, off);
    }

    /// Read columns `[c0, c1)` of every row `[r0, r1)` of a 2D tensor `[R, C]` into packed `dst` (batched pread).
    fn readSubmatrix(io: Io, sh: *const OpenShard, e: st.Entry, r0: usize, r1: usize, total_cols: usize, c0: usize, c1: usize, elem: usize, dst: []u8, scratch: []u8) !void {
        const rows = r1 - r0;
        const cols = c1 - c0;
        if (c0 == 0 and c1 == total_cols) {
            return readRows(io, sh, e, r0, r1, total_cols, elem, dst);
        }
        const row_bytes = total_cols * elem;
        const dst_row = cols * elem;
        const col_off = c0 * elem;
        const batch_rows = @max(@as(usize, 1), scratch.len / row_bytes);
        var rr: usize = 0;
        while (rr < rows) {
            const n = @min(batch_rows, rows - rr);
            const buf = scratch[0 .. n * row_bytes];
            try readExact(io, sh.file, buf, sh.meta.data + e.begin + (r0 + rr) * row_bytes);
            for (0..n) |i| {
                @memcpy(dst[(rr + i) * dst_row ..][0..dst_row], buf[i * row_bytes + col_off ..][0..dst_row]);
            }
            rr += n;
        }
    }
};

fn packReserve(p: *pack.Pack, len: usize) !u64 {
    const at = std.mem.alignForward(usize, p.bytes.items.len, 64);
    try p.bytes.resize(p.gpa, at + len);
    return at;
}

fn loadVector(s: *Store, p: *pack.Pack, name: []const u8, expect_dtype: st.DType, expect_len: usize) !u64 {
    const loc = try s.locate(name);
    if (loc.entry.dtype != expect_dtype or loc.entry.dim(0) != expect_len) return error.BadGlmTensorShape;
    const nbytes = expect_len * expect_dtype.size();
    const at = try packReserve(p, nbytes);
    try Store.readExact(s.io, loc.shard.file, p.bytes.items[at..][0..nbytes], loc.shard.meta.data + loc.entry.begin);
    return at;
}

/// Pack a dense matrix slice `out=[o0, o1), in=[i0, i1)` (FP8 block-128 or BF16/F32) into `p`.
fn loadDenseLinear(s: *Store, p: *pack.Pack, prefix: []const u8, o: [2]usize, i: [2]usize, scratch: []u8) !pack.Rows {
    var wname_buf: [256]u8 = undefined;
    const wname = try std.fmt.bufPrint(&wname_buf, "{s}.weight", .{prefix});
    const wloc = try s.locate(wname);
    const total_out = wloc.entry.dim(0);
    const total_in = wloc.entry.dim(1);
    const outs = o[1] - o[0];
    const ins = i[1] - i[0];
    if (o[1] > total_out or i[1] > total_in or o[0] >= o[1] or i[0] >= i[1]) return error.BadGlmTensorShape;
    var dj: job.DenseJob = .{ .format = FMT_BF16, .out0 = 0, .outs = @intCast(outs), .in0 = 0, .ins = @intCast(ins), .out_total = @intCast(outs), .in_total = @intCast(ins), .w = 0, .s = 0 };
    switch (wloc.entry.dtype) {
        .f8_e4m3 => {
            if (o[0] % 128 != 0 or i[0] % 128 != 0 or ins % 128 != 0) return error.BadGlmSliceAlign;
            var sname_buf: [256]u8 = undefined;
            const sname = try std.fmt.bufPrint(&sname_buf, "{s}.weight_scale_inv", .{prefix});
            const sloc = try s.locate(sname);
            if (sloc.entry.dtype != .f32) return error.BadGlmTensorShape;
            const total_sb_cols = (total_in + 127) / 128;
            const sb_r0 = o[0] / 128;
            const sb_r1 = (o[1] + 127) / 128;
            const sb_c0 = i[0] / 128;
            const sb_c1 = (i[1] + 127) / 128;
            const w_at = try packReserve(p, outs * ins);
            try Store.readSubmatrix(s.io, wloc.shard, wloc.entry, o[0], o[1], total_in, i[0], i[1], 1, p.bytes.items[w_at..][0 .. outs * ins], scratch);
            const s_bytes = (sb_r1 - sb_r0) * (sb_c1 - sb_c0) * 4;
            const s_at = try packReserve(p, s_bytes);
            try Store.readSubmatrix(s.io, sloc.shard, sloc.entry, sb_r0, sb_r1, total_sb_cols, sb_c0, sb_c1, 4, p.bytes.items[s_at..][0..s_bytes], scratch);
            dj.format = FMT_FP8_BLOCK;
            dj.w = w_at;
            dj.s = s_at;
        },
        .bf16, .f32 => {
            const elem = wloc.entry.dtype.size();
            const w_at = try packReserve(p, outs * ins * elem);
            try Store.readSubmatrix(s.io, wloc.shard, wloc.entry, o[0], o[1], total_in, i[0], i[1], elem, p.bytes.items[w_at..][0 .. outs * ins * elem], scratch);
            dj.format = if (wloc.entry.dtype == .bf16) FMT_BF16 else FMT_FP32;
            dj.w = w_at;
        },
        else => return error.UnsupportedGlmDType,
    }
    const at: u32 = @intCast(p.dense_jobs.items.len);
    try p.dense_jobs.append(p.gpa, dj);
    return .{ .job = at, .in = @intCast(ins), .out = @intCast(outs) };
}

/// Load one GLM-5.3 HF FP8 layer's dense weights into `p` and (if MoE) its 257 experts' FP8 slices onto GPU.
pub fn loadLayer(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    s: *Store,
    c: *const cfg.Config,
    n: u32,
    p: *pack.Pack,
) !struct { layer: wt.Layer, fp8_moe: ?Fp8Moe } {
    defer s.clearOpen();
    const scratch = try gpa.alloc(u8, 4 << 20);
    defer gpa.free(scratch);
    const full = c.full.isSet(n);
    const h0 = c.head0();
    const nh = c.localHeads();
    const qh = c.qHead();
    var buf: [256]u8 = undefined;
    const in_norm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{n}), .bf16, c.hidden);
    const post_norm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{n}), .bf16, c.hidden);
    var l: wt.Layer = .{ .number = n, .full = full, .in_norm = in_norm, .post_norm = post_norm, .attn = undefined, .mlp = undefined };

    // Attention fused FP8 x projections: q_a, kv_a, and (on full layers) indexer wk; wp BF16 is split out.
    const x_lo: u32 = @intCast(p.dense_jobs.items.len);
    const x_specs = [_]struct { sub: []const u8, out: usize, col: u32 }{
        .{ .sub = "self_attn.q_a_proj", .out = c.q_lora, .col = 0 },
        .{ .sub = "self_attn.kv_a_proj_with_mqa", .out = c.kv_lora + c.rope, .col = wt.kv_col },
        .{ .sub = "self_attn.indexer.wk", .out = c.i_dim, .col = wt.wk_col },
    };
    for (x_specs[0..if (full) 3 else 2]) |sp| {
        const r = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ n, sp.sub }), .{ 0, sp.out }, .{ 0, c.hidden }, scratch);
        p.dense_jobs.items[r.job].y_col = sp.col;
    }
    var wp_rows: ?pack.Rows = null;
    var wp_bf16_off: u64 = 0;
    if (full) {
        const r = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.indexer.weights_proj", .{n}), .{ 0, c.i_heads }, .{ 0, c.hidden }, scratch);
        p.dense_jobs.items[r.job].y_col = wt.wp_col;
        wp_rows = r;
        if (p.dense_jobs.items[r.job].format == FMT_BF16) wp_bf16_off = p.dense_jobs.items[r.job].w;
    }
    const kv_rows = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.kv_b_proj", .{n}), .{ 448 * h0, 448 * (h0 + nh) }, .{ 0, c.kv_lora }, scratch);
    const kvj = p.dense_jobs.items[kv_rows.job];
    _ = p.dense_jobs.pop();
    const uv_lo: u32 = @intCast(p.dense_jobs.items.len);
    for (0..nh) |h| {
        var j = kvj;
        j.out0 = @intCast(448 * h + c.nope);
        j.outs = c.v_dim;
        j.x_col = @intCast(h * c.kv_lora);
        j.y_col = @intCast(h * c.v_dim);
        try p.dense_jobs.append(p.gpa, j);
    }
    const uv_hi: u32 = @intCast(p.dense_jobs.items.len);
    const q_b = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_b_proj", .{n}), .{ qh * h0, qh * (h0 + nh) }, .{ 0, c.q_lora }, scratch);
    const wq_b = if (full) try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.indexer.wq_b", .{n}), .{ 0, c.i_heads * c.i_dim }, .{ 0, c.q_lora }, scratch) else null;
    const o_proj = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.o_proj", .{n}), .{ 0, c.hidden }, .{ c.v_dim * h0, c.v_dim * (h0 + nh) }, scratch);
    l.attn = .{
        .x_jobs = .{ .lo = x_lo, .hi = x_lo + @as(u32, if (full) 3 else 2) },
        .q_b = .{ .rows = q_b },
        .wq_b = wq_b,
        .wp = wp_rows,
        .wp_bf16_off = wp_bf16_off,
        .kvb_w = kvj.w,
        .kvb_s = kvj.s,
        .uv_jobs = .{ .lo = uv_lo, .hi = uv_hi },
        .o = .{ .rows = o_proj },
        .q_norm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_a_layernorm.weight", .{n}), .bf16, c.q_lora),
        .kv_norm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.kv_a_layernorm.weight", .{n}), .bf16, c.kv_lora),
        .k_norm = if (full) try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.indexer.k_norm.weight", .{n}), .bf16, c.i_dim) else 0,
        .k_bias = if (full) try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.indexer.k_norm.bias", .{n}), .bf16, c.i_dim) else 0,
    };

    var fp8_moe: ?Fp8Moe = null;
    if (!c.isMoe(n)) {
        const d_rows: [2]usize = .{ c.dense0(), c.dense0() + c.localDense() };
        l.mlp = .{ .dense = .{
            .gate = .{ .rows = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate_proj", .{n}), d_rows, .{ 0, c.hidden }, scratch) },
            .up = .{ .rows = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.up_proj", .{n}), d_rows, .{ 0, c.hidden }, scratch) },
            .down = .{ .rows = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.down_proj", .{n}), .{ 0, c.hidden }, d_rows, scratch) },
        } };
    } else {
        const router = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate", .{n}), .{ 0, c.experts }, .{ 0, c.hidden }, scratch);
        const bias = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{n}), .f32, c.experts);
        l.mlp = .{ .moe = .{ .router = router, .bias = bias, .signs = 0, .experts = &.{} } };
        fp8_moe = try loadFp8Experts(gpa, device, s, c, n);
        const rj = p.dense_jobs.items[router.job];
        if (rj.format == FMT_BF16) fp8_moe.?.router_bf16_off = rj.w;
    }

    if (n == c.layers) {
        const total_in: usize = 2 * c.hidden;
        const can_shard = c.tp > 1 and (total_in % (c.tp * 256) == 0);
        const local_in: usize = if (can_shard) total_in / c.tp else total_in;
        const in0: usize = if (can_shard) @as(usize, c.tp_rank) * local_in else 0;
        const eh_r = try loadDenseLinear(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.eh_proj", .{n}), .{ 0, c.hidden }, .{ in0, in0 + local_in }, scratch);
        p.dense_jobs.items[eh_r.job].x_col = @intCast(in0);
        const eh_dj = p.dense_jobs.items[eh_r.job];
        var eh_q8_w: u64 = 0;
        var eh_q8_s: u64 = 0;
        if (eh_dj.format == FMT_BF16 and (local_in % 512 == 0)) {
            eh_q8_w = try packReserve(p, c.hidden * local_in);
            eh_q8_s = try packReserve(p, c.hidden * (local_in / 64) * 2);
            try wt.quantizeBf16DraftG64(p.bytes.items[eh_dj.w..][0 .. c.hidden * local_in * 2], c.hidden, local_in, p.bytes.items[eh_q8_w..][0 .. c.hidden * local_in], p.bytes.items[eh_q8_s..][0 .. c.hidden * (local_in / 64) * 2], true);
        }
        l.mtp = .{
            .enorm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.enorm.weight", .{n}), .bf16, c.hidden),
            .hnorm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.hnorm.weight", .{n}), .bf16, c.hidden),
            .eh = .{ .rows = eh_r },
            .head_norm = try loadVector(s, p, try std.fmt.bufPrint(&buf, "model.layers.{d}.shared_head.norm.weight", .{n}), .bf16, c.hidden),
            .eh_bf16_off = if (eh_dj.format == FMT_BF16) eh_dj.w else 0,
            .eh_q8_w = eh_q8_w,
            .eh_q8_s = eh_q8_s,
            .eh_sharded = can_shard,
            .eh_x_col = @intCast(in0),
            .eh_ins = @intCast(local_in),
        };
    }
    return .{ .layer = l, .fp8_moe = fp8_moe };
}

const ExpertTask = struct {
    s: *const Store,
    c: *const cfg.Config,
    layer: u32,
    e0: u32,
    e1: u32,
    gate_w: []u8,
    gate_s: []u8,
    up_w: []u8,
    up_s: []u8,
    down_w: []u8,
    down_s: []u8,
    gpa: std.mem.Allocator,
    err: ?anyerror = null,

    fn run(t: *ExpertTask) void {
        t.exec() catch |e| {
            t.err = e;
        };
    }

    fn exec(t: *ExpertTask) !void {
        const scratch = try t.gpa.alloc(u8, 2 << 20);
        defer t.gpa.free(scratch);
        const c = t.c;
        const hid: usize = c.hidden;
        const hid_b: usize = hid / 128;
        const loc: usize = @as(usize, c.localHalves()) * 128;
        const loc_b: usize = c.localHalves();
        const col0: usize = @as(usize, c.half0()) * 128;
        const col1: usize = col0 + loc;
        const sb0: usize = c.half0();
        const sb1: usize = sb0 + loc_b;
        const total_moe: usize = c.moe_inter;
        const total_sb: usize = total_moe / 128;
        const gu_w_sz = loc * hid;
        const gu_s_sz = loc_b * hid_b * 4;
        const dn_w_sz = hid * loc;
        const dn_s_sz = hid_b * loc_b * 4;
        var pbuf: [256]u8 = undefined;
        var nbuf: [280]u8 = undefined;
        for (t.e0..t.e1) |ei| {
            const pfx = if (ei == c.experts)
                try std.fmt.bufPrint(&pbuf, "model.layers.{d}.mlp.shared_experts", .{t.layer})
            else
                try std.fmt.bufPrint(&pbuf, "model.layers.{d}.mlp.experts.{d}", .{ t.layer, ei });
            // Gate [2048, 6144] -> rows [col0, col1)
            {
                const w = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.gate_proj.weight", .{pfx}));
                const sc = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.gate_proj.weight_scale_inv", .{pfx}));
                try Store.readRows(t.s.io, w.shard, w.entry, col0, col1, hid, 1, t.gate_w[ei * gu_w_sz ..][0..gu_w_sz]);
                try Store.readRows(t.s.io, sc.shard, sc.entry, sb0, sb1, hid_b, 4, t.gate_s[ei * gu_s_sz ..][0..gu_s_sz]);
            }
            // Up [2048, 6144] -> rows [col0, col1)
            {
                const w = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.up_proj.weight", .{pfx}));
                const sc = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.up_proj.weight_scale_inv", .{pfx}));
                try Store.readRows(t.s.io, w.shard, w.entry, col0, col1, hid, 1, t.up_w[ei * gu_w_sz ..][0..gu_w_sz]);
                try Store.readRows(t.s.io, sc.shard, sc.entry, sb0, sb1, hid_b, 4, t.up_s[ei * gu_s_sz ..][0..gu_s_sz]);
            }
            // Down [6144, 2048] -> cols [col0, col1)
            {
                const w = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.down_proj.weight", .{pfx}));
                const sc = try t.s.locateOpen(try std.fmt.bufPrint(&nbuf, "{s}.down_proj.weight_scale_inv", .{pfx}));
                try Store.readSubmatrix(t.s.io, w.shard, w.entry, 0, hid, total_moe, col0, col1, 1, t.down_w[ei * dn_w_sz ..][0..dn_w_sz], scratch);
                try Store.readSubmatrix(t.s.io, sc.shard, sc.entry, 0, hid_b, total_sb, sb0, sb1, 4, t.down_s[ei * dn_s_sz ..][0..dn_s_sz], scratch);
            }
        }
    }
};

/// Parallel-read all 257 experts' FP8 slices directly into 6 shared Metal GPU buffers (zero extra host copy).
fn loadFp8Experts(gpa: std.mem.Allocator, device: mtl.Device, s: *Store, c: *const cfg.Config, n: u32) !Fp8Moe {
    const n_exp: usize = c.experts + 1;
    const hid: usize = c.hidden;
    const hid_b: usize = hid / 128;
    const loc: usize = @as(usize, c.localHalves()) * 128;
    const loc_b: usize = c.localHalves();
    // Pre-open all shards touched by this layer's 257 experts on the main thread (typically 2-3 shards).
    var pbuf: [256]u8 = undefined;
    var nbuf: [280]u8 = undefined;
    for (0..n_exp) |ei| {
        const pfx = if (ei == c.experts)
            try std.fmt.bufPrint(&pbuf, "model.layers.{d}.mlp.shared_experts", .{n})
        else
            try std.fmt.bufPrint(&pbuf, "model.layers.{d}.mlp.experts.{d}", .{ n, ei });
        inline for (.{ "gate_proj.weight", "gate_proj.weight_scale_inv", "up_proj.weight", "up_proj.weight_scale_inv", "down_proj.weight", "down_proj.weight_scale_inv" }) |suf| {
            try s.preopen(try std.fmt.bufPrint(&nbuf, "{s}.{s}", .{ pfx, suf }));
        }
    }
    const opts = mtl.ResourceOptions.shared;
    var m: Fp8Moe = .{
        .gate_w = try device.buffer(n_exp * loc * hid, opts),
        .gate_s = try device.buffer(n_exp * loc_b * hid_b * 4, opts),
        .up_w = try device.buffer(n_exp * loc * hid, opts),
        .up_s = try device.buffer(n_exp * loc_b * hid_b * 4, opts),
        .down_w = try device.buffer(n_exp * hid * loc, opts),
        .down_s = try device.buffer(n_exp * hid_b * loc_b * 4, opts),
    };
    errdefer m.deinit();
    const num_threads: usize = 8;
    var tasks: [num_threads]ExpertTask = undefined;
    var threads: [num_threads]std.Thread = undefined;
    const per = (n_exp + num_threads - 1) / num_threads;
    var spawned: usize = 0;
    for (0..num_threads) |ti| {
        const e0 = ti * per;
        if (e0 >= n_exp) break;
        const e1 = @min(n_exp, e0 + per);
        tasks[ti] = .{
            .s = s,
            .c = c,
            .layer = n,
            .e0 = @intCast(e0),
            .e1 = @intCast(e1),
            .gate_w = m.gate_w.contents()[0 .. n_exp * loc * hid],
            .gate_s = m.gate_s.contents()[0 .. n_exp * loc_b * hid_b * 4],
            .up_w = m.up_w.contents()[0 .. n_exp * loc * hid],
            .up_s = m.up_s.contents()[0 .. n_exp * loc_b * hid_b * 4],
            .down_w = m.down_w.contents()[0 .. n_exp * hid * loc],
            .down_s = m.down_s.contents()[0 .. n_exp * hid_b * loc_b * 4],
            .gpa = gpa,
        };
        threads[ti] = try std.Thread.spawn(.{}, ExpertTask.run, .{&tasks[ti]});
        spawned += 1;
    }
    for (threads[0..spawned]) |th| th.join();
    for (tasks[0..spawned]) |t| if (t.err) |e| return e;
    return m;
}

/// Load `embed_tokens` (BF16), `norm` (BF16), and sliced `lm_head` (exact BF16 + optional draft-only Q4/Q8 g64).
pub fn loadVocab(gpa: std.mem.Allocator, device: mtl.Device, bq: mtl.Queue, s: *Store, c: *const cfg.Config, load_embed: bool, load_head: bool, load_draft_q4: bool) !struct {
    k: @import("gpu.zig").Packed,
    embed_bf16: u64,
    lm_bf16_off: u64,
    lm_q4_w: ?mtl.Buffer,
    lm_q4_s: ?mtl.Buffer,
    lm_is_q8: bool,
    groups: [1]@import("gpu.zig").Groups,
    norm: u64,
    part: [2]u32,
} {
    defer s.clearOpen();
    const scratch = try gpa.alloc(u8, 4 << 20);
    defer gpa.free(scratch);
    var p = pack.Pack.init(gpa);
    defer p.deinit();
    const part = c.vocabPart();
    var embed_bf16: u64 = 0;
    if (load_embed) {
        const eloc = try s.locate("model.embed_tokens.weight");
        if (eloc.entry.dtype != .bf16 or eloc.entry.dim(0) != c.vocab or eloc.entry.dim(1) != c.hidden) return error.UnsupportedGlmEmbedding;
        const ebytes = @as(usize, c.vocab) * c.hidden * 2;
        embed_bf16 = try packReserve(&p, ebytes);
        try Store.readExact(s.io, eloc.shard.file, p.bytes.items[embed_bf16..][0..ebytes], eloc.shard.meta.data + eloc.entry.begin);
    }
    const norm = if (load_head) try loadVector(s, &p, "model.norm.weight", .bf16, c.hidden) else 0;
    var groups: [1]@import("gpu.zig").Groups = .{.{ .off = 0, .count = 0 }};
    var dispatches: [1]pack.Range = .{.{ .lo = 0, .hi = 0 }};
    var lm_bf16_off: u64 = 0;
    var lm_q4_w: ?mtl.Buffer = null;
    var lm_q4_s: ?mtl.Buffer = null;
    const lm_is_q8 = @import("sys").environment.read(bool, "TF_GLM_LM_Q8", false) catch false;
    errdefer {
        if (lm_q4_w) |b| b.deinit();
        if (lm_q4_s) |b| b.deinit();
    }
    const gpu_mod = @import("gpu.zig");
    if (load_head) {
        const head = try loadDenseLinear(s, &p, "lm_head", .{ part[0], part[1] }, .{ 0, c.hidden }, scratch);
        dispatches[0] = .{ .lo = head.job, .hi = head.job + 1 };
        const dj = p.dense_jobs.items[head.job];
        if (dj.format == FMT_BF16) lm_bf16_off = dj.w;
        if (load_draft_q4 and lm_bf16_off != 0 and (c.hidden % 1024 == 0)) {
            const outs: usize = part[1] - part[0];
            const w_bytes = if (lm_is_q8) outs * c.hidden else outs * (c.hidden / 2);
            const s_bytes = outs * (c.hidden / 64) * 2;
            const sw = try gpa.alloc(u8, w_bytes);
            defer gpa.free(sw);
            const ss = try gpa.alloc(u8, s_bytes);
            defer gpa.free(ss);
            try wt.quantizeBf16DraftG64(p.bytes.items[lm_bf16_off..][0 .. outs * c.hidden * 2], outs, c.hidden, sw, ss, lm_is_q8);
            lm_q4_w = try gpu_mod.uploadPrivate(device, bq, sw);
            lm_q4_s = try gpu_mod.uploadPrivate(device, bq, ss);
        }
    }
    const k = try gpu_mod.Packed.initWithQueue(gpa, device, bq, &p, &dispatches, &groups);
    return .{ .k = k, .embed_bf16 = embed_bf16, .lm_bf16_off = lm_bf16_off, .lm_q4_w = lm_q4_w, .lm_q4_s = lm_q4_s, .lm_is_q8 = lm_is_q8, .groups = groups, .norm = norm, .part = part };
}

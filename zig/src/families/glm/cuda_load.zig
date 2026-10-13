//! GLM-5.3-Flash's weights for one of two ranks from the unsplit EXL3 checkpoint (weights.py load, split.py cuts).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const lay = cuda.glm_layer;
const dsa = cuda.glm_dsa;
const head_ = cuda.glm_head;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const B16 = cuda.glm_bmm.B16;
const Tensor = core.safetensors.Tensor;

pub const world = 2;
const chunk_bytes = 1 << 30;

/// Device memory: small tensors packed in 1 GiB chunks (256-byte aligned), large ones allocated alone.
pub const Pool = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    chunks: std.ArrayList(cuda.DeviceBuffer) = .empty,
    at: u64 = 0,
    end: u64 = 0,
    bytes: usize = 0,

    pub fn deinit(p: *Pool) void {
        for (p.chunks.items) |*b| b.free();
        p.chunks.deinit(p.gpa);
    }

    /// `n` bytes of device memory, unwritten; a span past 1/16 of a chunk gets its own allocation (no stranded tail).
    pub fn reserve(p: *Pool, n: usize) !u64 {
        const len = std.mem.alignForward(usize, @max(n, 1), 256);
        if (len > chunk_bytes / 16) {
            try p.chunks.ensureUnusedCapacity(p.gpa, 1);
            const b = try cuda.DeviceBuffer.alloc(p.d, len);
            p.chunks.appendAssumeCapacity(b);
            p.bytes += n;
            return b.ptr;
        }
        if (p.at + len > p.end) {
            try p.chunks.ensureUnusedCapacity(p.gpa, 1);
            const b = try cuda.DeviceBuffer.alloc(p.d, chunk_bytes);
            p.chunks.appendAssumeCapacity(b);
            p.at = b.ptr;
            p.end = b.ptr + b.len;
        }
        const ptr = p.at;
        p.at += len;
        p.bytes += n;
        return ptr;
    }

    pub fn write(p: *Pool, ptr: u64, bytes: []const u8) !void {
        const view: cuda.DeviceBuffer = .{ .d = p.d, .ptr = ptr, .len = bytes.len, .borrowed = true };
        try view.upload(0, bytes);
    }

    pub fn put(p: *Pool, bytes: []const u8) !u64 {
        const ptr = try p.reserve(bytes.len);
        try p.write(ptr, bytes);
        return ptr;
    }
};

/// The bytes of rows [rank n / 2, (rank + 1) n / 2) of a tensor's first axis (split.py's "row" cut).
pub fn rowHalf(t: Tensor, rank: u32) ![]const u8 {
    if (t.rank == 0 or t.shape[0] % world != 0) return error.GlmCut;
    const half = t.bytes.len / world;
    return t.bytes[rank * half ..][0..half];
}

/// Axis 1 of [outer, n, inner bytes] in `parts` blocks, each block's rank half in order: dim1 (1) or fused runs (3).
pub fn cutAxis1(a: std.mem.Allocator, src: []const u8, outer: usize, n: usize, inner: usize, parts: usize, rank: u32) ![]u8 {
    const out = try a.alloc(u8, src.len / world);
    try cutInto(out, src, outer, n, inner, parts, rank);
    return out;
}

/// cutAxis1 into `out` (src.len / 2 bytes).
pub fn cutInto(out: []u8, src: []const u8, outer: usize, n: usize, inner: usize, parts: usize, rank: u32) !void {
    if (n % (parts * world) != 0 or src.len != outer * n * inner or out.len * world != src.len) return error.GlmCut;
    const per = n / parts;
    const half = per / world;
    for (0..outer) |o| for (0..parts) |j| {
        @memcpy(out[(o * parts + j) * half * inner ..][0 .. half * inner], src[(o * n + j * per + rank * half) * inner ..][0 .. half * inner]);
    };
}

/// Trellis words [K/16, N/16, W] -> [N/128, K/16, 8, W] (linear.py strips): each 128-column block's tiles in k order.
pub fn strips(a: std.mem.Allocator, src: []const u8, kt: usize, nt: usize, wb: usize) ![]u8 {
    if (nt % 8 != 0 or src.len != kt * nt * wb) return error.GlmCut;
    const out = try a.alloc(u8, src.len);
    for (0..nt / 8) |jb| for (0..kt) |i| {
        @memcpy(out[(jb * kt + i) * 8 * wb ..][0 .. 8 * wb], src[(i * nt + jb * 8) * wb ..][0 .. 8 * wb]);
    };
    return out;
}

fn bf16Bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// A tensor's values as bf16 bytes (torch's .to(bfloat16): round to nearest even); bf16 bytes pass through.
pub fn toBf16(a: std.mem.Allocator, dtype: core.safetensors.DType, bytes: []const u8) ![]const u8 {
    switch (dtype) {
        .bf16 => return bytes,
        .f16 => {
            const out = try a.alloc(u16, bytes.len / 2);
            for (out, 0..) |*o, i| o.* = bf16Bits(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))));
            return std.mem.sliceAsBytes(out);
        },
        .f32 => {
            const out = try a.alloc(u16, bytes.len / 4);
            for (out, 0..) |*o, i| o.* = bf16Bits(@bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)));
            return std.mem.sliceAsBytes(out);
        },
        else => return error.GlmDtype,
    }
}

/// How an EXL3 group's tensors split (split.py EXL3_RULES): whole, output columns, input rows, or fused q|k|v runs.
pub const Cut = enum { rep, out, in, qkv };

/// One rank's reads: headers mapped, bytes read past the page cache (GB10's allocations do not reclaim it).
pub const Reader = struct {
    ck: core.Checkpoint,
    rank: u32,
    tmp: std.heap.ArenaAllocator,
    pool: *Pool,
    files: []?core.direct_io.File, // ck.files' direct handles, opened on first read

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, pool: *Pool) !Reader {
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        errdefer ck.close();
        const files = try gpa.alloc(?core.direct_io.File, ck.files.items.len);
        @memset(files, null);
        return .{ .ck = ck, .rank = rank, .tmp = .init(gpa), .pool = pool, .files = files };
    }

    pub fn close(r: *Reader) void {
        for (r.files) |*f| if (f.*) |*x| x.close();
        r.ck.gpa.free(r.files);
        r.tmp.deinit();
        r.ck.close();
    }

    fn a(r: *Reader) std.mem.Allocator {
        return r.tmp.allocator();
    }

    /// `p` ++ `name`'s header (p: the layer's prefix), its dtype checked; the bytes are not read.
    fn head(r: *Reader, p: []const u8, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const full = try std.fmt.allocPrint(r.a(), "{s}{s}", .{ p, name });
        const x = try r.ck.get(full);
        if (x.dtype != dtype) {
            std.log.err("{s}: {t}, expected {t}", .{ full, x.dtype, dtype });
            return error.GlmDtype;
        }
        return x;
    }

    /// `head` with the bytes read into the arena.
    fn t(r: *Reader, p: []const u8, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        var x = try r.head(p, name, dtype);
        var buf: core.direct_io.Buffer = &.{};
        x.bytes = try r.readInto(&buf, x);
        return x;
    }

    /// A tensor's bytes through `buf` (grown in the arena when short); valid until `buf`'s next read.
    fn readInto(r: *Reader, buf: *core.direct_io.Buffer, x: Tensor) ![]const u8 {
        const p = @intFromPtr(x.bytes.ptr);
        for (r.ck.files.items, r.files) |*f, *h| {
            const base = @intFromPtr(f.map.memory.ptr);
            if (p < base or p + x.bytes.len > base + f.map.memory.len) continue;
            if (h.* == null) h.* = try core.direct_io.File.open(f.path);
            const off = p - base;
            const need = core.direct_io.span(off, x.bytes.len);
            if (buf.len < need) buf.* = try r.a().alignedAlloc(u8, .fromByteUnits(core.direct_io.alignment), need);
            return h.*.?.read(buf.*, off, x.bytes.len);
        }
        return error.GlmTensorFile;
    }

    fn put(r: *Reader, bytes: []const u8) !u64 {
        return r.pool.put(bytes);
    }

    /// A plain tensor whole, converted to bf16.
    fn bf(r: *Reader, p: []const u8, name: []const u8, dtype: core.safetensors.DType) !u64 {
        const x = try r.t(p, name, dtype);
        return r.put(try toBf16(r.a(), x.dtype, x.bytes));
    }

    /// A plain matrix [n, k] as the bf16 matmul's weight; `rows`: the rank's half of its rows.
    fn b16(r: *Reader, p: []const u8, name: []const u8, rows: bool) !B16 {
        const x = try r.t(p, name, .f16);
        const bytes = if (rows) try rowHalf(x, r.rank) else x.bytes;
        const n = if (rows) x.shape[0] / world else x.shape[0];
        return r.packed_(try toBf16(r.a(), x.dtype, bytes), n, x.shape[1]);
    }

    /// bf16 [n, k] row-major uploaded in the matmul's fragment order.
    fn packed_(r: *Reader, bytes: []const u8, n: usize, k: usize) !B16 {
        const out = try r.a().alloc(u8, bytes.len);
        try cuda.glm_bmm.pack(out, bytes, n, k);
        return .{ .w = try r.put(out), .n = @intCast(n), .k = @intCast(k), .frag = true };
    }

    /// One EXL3 group (Exl3Linear.from_tensors, strips layout) cut for this rank.
    fn linear(r: *Reader, p: []const u8, group: []const u8, cut: Cut) !exl3.Weight {
        const g = try std.fmt.allocPrint(r.a(), "{s}{s}", .{ p, group });
        _ = try r.head(g, ".mul1", .i32);
        const tr = try r.t(g, ".trellis", .i16);
        var suh = (try r.t(g, ".suh", .f16)).bytes;
        const svh_t = try r.t(g, ".svh", .f16);
        var svh = svh_t.bytes;
        if (tr.rank != 3 or tr.shape[2] % 8 != 0) return error.GlmTrellis;
        var kt = tr.shape[0];
        var nt = tr.shape[1];
        const wb = tr.shape[2] * 2;
        var words = tr.bytes;
        switch (cut) {
            .rep => {},
            .out, .qkv => {
                const parts: usize = if (cut == .qkv) 3 else 1;
                words = try cutAxis1(r.a(), words, kt, nt, wb, parts, r.rank);
                svh = if (cut == .qkv) try cutAxis1(r.a(), svh, 1, svh.len / 2, 2, 3, r.rank) else try rowHalf(svh_t, r.rank);
                nt /= world;
            },
            .in => {
                words = try rowHalf(tr, r.rank);
                suh = suh[r.rank * suh.len / world ..][0 .. suh.len / world];
                kt /= world;
            },
        }
        if (kt % 8 != 0 or nt % 8 != 0 or suh.len != kt * 32 or svh.len != nt * 32) return error.GlmTrellis;
        const k = 16 * kt;
        const n = 16 * nt;
        const halves: u16 = @intCast(tr.shape[2] / 8);
        const st = exl3.strides(k, n, halves, true);
        return .{
            .trellis = try r.put(try strips(r.a(), words, kt, nt, wb)),
            .suh = try r.put(suh),
            .svh = try r.put(svh),
            .k = @intCast(k),
            .n = @intCast(n),
            .halves = halves,
            .codebook = .mul1,
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }

    fn hc(r: *Reader, p: []const u8, site: []const u8) !lay.Hc {
        const n = try std.fmt.allocPrint(r.a(), "hc_{s}_", .{site});
        const f = struct {
            fn get(rd: *Reader, pre: []const u8, base: []const u8, field: []const u8) !u64 {
                const full = try std.fmt.allocPrint(rd.a(), "{s}{s}", .{ base, field });
                return rd.put((try rd.t(pre, full, .f32)).bytes);
            }
        }.get;
        return .{ .fn_w = try f(r, p, n, "fn"), .base = try f(r, p, n, "base"), .scale = try f(r, p, n, "scale") };
    }

    fn kda(r: *Reader, p: []const u8, heads: u32) !lay.KdaW {
        const qkv = try r.linear(p, "self_attn.qkv_proj", .qkv);
        // f_a, g_a and the rank's b rows stacked: one matmul (each column's sum is its own)
        const fa = try r.t(p, "self_attn.f_a_proj.weight", .f16);
        const ga = try r.t(p, "self_attn.g_a_proj.weight", .f16);
        const bt = try r.t(p, "self_attn.b_proj.weight", .f16);
        if (fa.shape[1] != ga.shape[1] or fa.shape[1] != bt.shape[1]) return error.GlmShape;
        const b_rows = try rowHalf(bt, r.rank);
        const stacked = try std.mem.concat(r.a(), u8, &.{ fa.bytes, ga.bytes, b_rows });
        const fa_n: u32 = @intCast(fa.shape[0]);
        const ga_n: u32 = @intCast(ga.shape[0]);
        const fab = try r.packed_(try toBf16(r.a(), .f16, stacked), fa.shape[0] + ga.shape[0] + bt.shape[0] / world, fa.shape[1]);
        const conv = try r.t(p, "self_attn.conv1d.weight", .bf16);
        const taps = conv.shape[2];
        const fa_off = 3 * heads * 128;
        return .{
            .qkv = qkv,
            .fab = fab,
            .fb = try r.b16(p, "self_attn.f_b_proj.weight", true),
            .gb = try r.b16(p, "self_attn.g_b_proj.weight", true),
            .o = try r.linear(p, "self_attn.o_proj", .in),
            .conv = try r.put(try cutAxis1(r.a(), conv.bytes, 1, conv.shape[0], taps * 2, 3, r.rank)),
            .a_log = try r.put(try rowHalf(try r.t(p, "self_attn.A_log", .f32), r.rank)),
            .dt_bias = try r.put(try rowHalf(try r.t(p, "self_attn.dt_bias", .f32), r.rank)),
            .norm = try r.put((try r.t(p, "self_attn.o_norm.weight", .bf16)).bytes),
            .heads = heads,
            .fa_off = fa_off,
            .ga_off = fa_off + fa_n,
            .b_off = fa_off + fa_n + ga_n,
            .p_stride = qkv.n + fab.n,
        };
    }

    fn dsaW(r: *Reader, p: []const u8, c: *const config.Config) !dsa.DsaW {
        // kv_b's rank rows, head by head: qk_dim key rows then v_dim value rows (latent.AbsorbW [H, dim, 512] each)
        const kvb = try r.t(p, "self_attn.kv_b_proj.weight", .f16);
        const mine = try toBf16(r.a(), .f16, try rowHalf(kvb, r.rank));
        const lw: usize = c.kv_lora;
        const heads: usize = c.mla_heads / world;
        const qk: usize = c.nope;
        const vd: usize = c.v_dim;
        if (kvb.shape[1] != lw or mine.len != heads * (qk + vd) * lw * 2) return error.GlmShape;
        const wk = try r.a().alloc(u8, heads * qk * lw * 2);
        const wv = try r.a().alloc(u8, heads * vd * lw * 2);
        for (0..heads) |h| {
            const at = mine[h * (qk + vd) * lw * 2 ..];
            @memcpy(wk[h * qk * lw * 2 ..][0 .. qk * lw * 2], at[0 .. qk * lw * 2]);
            @memcpy(wv[h * vd * lw * 2 ..][0 .. vd * lw * 2], at[qk * lw * 2 ..][0 .. vd * lw * 2]);
        }
        const wkx = try r.t(p, "self_attn.indexer.wk.weight", .f16);
        const wpx = try r.t(p, "self_attn.indexer.weights_proj.weight", .f16);
        if (wkx.shape[1] != wpx.shape[1]) return error.GlmShape;
        const kw = try std.mem.concat(r.a(), u8, &.{ try toBf16(r.a(), .f16, wkx.bytes), try toBf16(r.a(), .f16, wpx.bytes) });
        return .{
            .q_a = try r.linear(p, "self_attn.q_a_proj", .rep),
            .kv_a = try r.linear(p, "self_attn.kv_a_proj_with_mqa", .rep),
            .q_b = try r.linear(p, "self_attn.q_b_proj", .out),
            .o = try r.linear(p, "self_attn.o_proj", .in),
            .q_norm = try r.put((try r.t(p, "self_attn.q_a_layernorm.weight", .bf16)).bytes),
            .kv_norm = try r.put((try r.t(p, "self_attn.kv_a_layernorm.weight", .bf16)).bytes),
            .wk = try r.put(wk),
            .wv = try r.put(wv),
            .index = .{
                .kw = try r.packed_(kw, wkx.shape[0] + wpx.shape[0], wkx.shape[1]),
                .qb = try r.b16(p, "self_attn.indexer.wq_b.weight", false),
                .gate = try r.bf(p, "self_attn.indexer.index_kpool_compress_gate", .f16),
                .ln_w = try r.put((try r.t(p, "self_attn.indexer.k_norm.weight", .f16)).bytes),
                .ln_b = try r.put((try r.t(p, "self_attn.indexer.k_norm.bias", .f16)).bytes),
                .ape = try r.bf(p, "self_attn.indexer.index_kpool_compress_ape", .f32),
            },
        };
    }

    /// The routed experts (experts.prepare): each projection's rank-cut trellises packed in one span, pointers stacked.
    fn experts(r: *Reader, p: []const u8, c: *const config.Config) !exl3e.Layer {
        const e_n: usize = c.experts;
        const dims: usize = c.hidden;
        const width: usize = c.moe_inter / world;
        var tables: [3]u64 = undefined;
        var k2s: [3]u64 = undefined;
        var scales: [3][2][]u8 = undefined; // suh, svh stacked [E, n] fp16
        inline for (.{ "gate", "up", "down" }, 0..) |proj, j| {
            const down = j == 2;
            const n_suh = if (down) width else dims;
            const n_svh = if (down) dims else width;
            scales[j] = .{ try r.a().alloc(u8, e_n * n_suh * 2), try r.a().alloc(u8, e_n * n_svh * 2) };
            const trs = try r.a().alloc(Tensor, e_n);
            const k2 = try r.a().alloc(i32, e_n);
            var total: usize = 0;
            for (0..e_n) |e| {
                const g = try std.fmt.allocPrint(r.a(), "{s}mlp.experts.{d}.{s}_proj", .{ p, e, proj });
                _ = try r.head(g, ".mul1", .i32);
                const tr = try r.head(g, ".trellis", .i16);
                const suh = try r.t(g, ".suh", .f16);
                const svh = try r.t(g, ".svh", .f16);
                if (tr.rank != 3 or tr.shape[2] % 8 != 0) return error.GlmTrellis;
                k2[e] = @intCast(tr.shape[2] / 8);
                trs[e] = tr;
                const su = if (down) try rowHalf(suh, r.rank) else suh.bytes;
                const sv = if (down) svh.bytes else try rowHalf(svh, r.rank);
                if (su.len != n_suh * 2 or sv.len != n_svh * 2) return error.GlmShape;
                @memcpy(scales[j][0][e * n_suh * 2 ..][0..su.len], su);
                @memcpy(scales[j][1][e * n_svh * 2 ..][0..sv.len], sv);
                total += tr.bytes.len / world;
            }
            const base = try r.pool.reserve(total);
            const ptrs = try r.a().alloc(u64, e_n);
            var raw: core.direct_io.Buffer = &.{};
            var buf: []u8 = &.{};
            var off: usize = 0;
            for (trs, ptrs) |hd, *q| { // each trellis read, then cut, through two reused host buffers
                var tr = hd;
                tr.bytes = try r.readInto(&raw, hd);
                const half = tr.bytes.len / world;
                if (buf.len < half) buf = try r.a().alloc(u8, half);
                const bytes = if (down) try rowHalf(tr, r.rank) else blk: {
                    try cutInto(buf[0..half], tr.bytes, tr.shape[0], tr.shape[1], tr.shape[2] * 2, 1, r.rank);
                    break :blk buf[0..half];
                };
                q.* = base + off;
                try r.pool.write(q.*, bytes);
                off += half;
            }
            tables[j] = try r.put(std.mem.sliceAsBytes(ptrs));
            k2s[j] = try r.put(std.mem.sliceAsBytes(k2));
        }
        return .{
            .gate = tables[0],
            .up = tables[1],
            .down = tables[2],
            .gate_k2 = k2s[0],
            .up_k2 = k2s[1],
            .down_k2 = k2s[2],
            .suh_g = try r.put(scales[0][0]),
            .suh_u = try r.put(scales[1][0]),
            .svh_g = try r.put(scales[0][1]),
            .svh_u = try r.put(scales[1][1]),
            .suh_d = try r.put(scales[2][0]),
            .svh_d = try r.put(scales[2][1]),
            .experts = e_n,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .limit = c.swiglu_limit,
            .act_mode = 0,
        };
    }

    /// Layer `i` as the engine holds it (weights.load's layer()).
    pub fn layer(r: *Reader, i: u32, c: *const config.Config) !lay.LayerW {
        defer _ = r.tmp.reset(.retain_capacity);
        const p = try std.fmt.allocPrint(r.a(), "model.language_model.layers.{d}.", .{i});
        return .{
            .attn_hc = try r.hc(p, "attn"),
            .ffn_hc = try r.hc(p, "ffn"),
            .in_norm = try r.put((try r.t(p, "input_layernorm.weight", .bf16)).bytes),
            .post_norm = try r.put((try r.t(p, "post_attention_layernorm.weight", .bf16)).bytes),
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try r.kda(p, c.kda_heads / world) },
                .mla => .{ .dsa = try r.dsaW(p, c) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = .{
                .router = try r.bf(p, "mlp.gate.weight", .f16),
                .bias = try r.put((try r.t(p, "mlp.gate.e_score_correction_bias", .f32)).bytes),
                .experts = try r.experts(p, c),
                .gate = try r.linear(p, "mlp.shared_experts.gate_proj", .out),
                .up = try r.linear(p, "mlp.shared_experts.up_proj", .out),
                .down = try r.linear(p, "mlp.shared_experts.down_proj", .in),
            } } else .{ .mlp = .{
                .gate = try r.linear(p, "mlp.gate_proj", .out),
                .up = try r.linear(p, "mlp.up_proj", .out),
                .down = try r.linear(p, "mlp.down_proj", .in),
            } },
        };
    }

    /// The replicated embedding table, the final norm and this rank's half of lm_head's rows.
    pub fn globals(r: *Reader, c: *const config.Config) !struct { embed: u64, head: head_.HeadW } {
        defer _ = r.tmp.reset(.retain_capacity);
        const emb = try r.t("", "model.language_model.embed_tokens.weight", .bf16);
        const lm = try r.t("", "lm_head.weight", .bf16);
        if (lm.shape[0] != c.vocab or lm.shape[1] != c.hidden) return error.GlmShape;
        const n: u32 = c.vocab / world;
        return .{ .embed = try r.put(emb.bytes), .head = .{
            .norm = try r.put((try r.t("", "model.language_model.norm.weight", .bf16)).bytes),
            .head = try r.packed_(try rowHalf(lm, r.rank), n, c.hidden),
            .offset = r.rank * n,
        } };
    }
};

test "a fused q|k|v group's rank runs and a dim1 cut" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // axis 1 holds 12 one-byte columns: q 0-3, k 4-7, v 8-11; rank 1 takes the second half of each part
    const src = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const q = try cutAxis1(a, &src, 1, 12, 1, 3, 1);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 6, 7, 10, 11 }, q);
    // two rows of four: rank 0 keeps each row's first half
    const d = try cutAxis1(a, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, 2, 4, 1, 1, 0);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 4, 5 }, d);
    try std.testing.expectError(error.GlmCut, cutAxis1(a, &src, 1, 12, 1, 4, 0));
}

test "strips put each 128-column block's tiles in k order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // kt 2, nt 16 one-byte tiles: tile (i, j) holds 16 i + j
    var src: [32]u8 = undefined;
    for (&src, 0..) |*v, x| v.* = @intCast(x);
    const out = try strips(a, &src, 2, 16, 1);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 }, out[0..16]);
    try std.testing.expectEqualSlices(u8, &.{ 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 }, out[16..32]);
}

test "f16 and f32 round to bf16 to nearest even" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // f32 1 + 2^-8 (a tie) rounds down to 1.0; 1 + 3 * 2^-9 rounds up
    const f = [_]u32{ 0x3F808000, 0x3F80C000 };
    const out = try toBf16(a, .f32, std.mem.sliceAsBytes(&f));
    try std.testing.expectEqualSlices(u16, &.{ 0x3F80, 0x3F81 }, @alignCast(std.mem.bytesAsSlice(u16, out)));
    const h = [_]u16{0x3C01}; // f16 1 + 2^-10: below bf16's half step, rounds to 1.0
    const outh = try toBf16(a, .f16, std.mem.sliceAsBytes(&h));
    try std.testing.expectEqualSlices(u16, &.{0x3F80}, @alignCast(std.mem.bytesAsSlice(u16, outh)));
}

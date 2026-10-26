//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const prompt_w = fz.prompt_w;
const variants = fz.variants;
const frags = fz.frags;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const PLE_TAIL = fz.PLE_TAIL;
const CS_ROW = fz.CS_ROW;
const Buf = fz.Buf;
const Variant = fz.Variant;
const Run = fz.Run;
const Select = fz.Select;
const Hc = fz.Hc;
const Layer = fz.Layer;
const Model = fz.Model;
const PMAX = fz.PMAX;

pub const Prompt = struct {
    r: *Run,
    qmm6: mtl.Pipeline,
    gather64: mtl.Pipeline,
    gather32: mtl.Pipeline,
    qmm6_128: mtl.Pipeline,
    gather128: mtl.Pipeline,
    gu64: mtl.Pipeline,
    seg_queue: ?mtl.Queue = null, // a later staggered segment's queue (made at first use)
    last: Buf = undefined, // the last chunk's streams before the final mixer (each segment's own)
    gu32: mtl.Pipeline,
    wide: ?*prompt_w.Wide = null, // weights at other widths than 6/32 (prompt_w.zig)
    fsr_width: variants.Experts = .{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 6, 32, 6, 32 } },
    fused_gu: bool = true, // the experts' gate and up products and activation in one pass (same bits)
    expert_bm: usize = 0, // expert tile rows (0: by the average rows an expert)
    mtp_w: ?[3][3]Buf = null, // the MTP head's attention projection, fce, fch in MLX layout (pack_mtp_mlx.safetensors)
    tall_tiles: bool = true, // 128-row tiles (a dequantized weight block serves twice the rows; the same sums)
    router_mm: mtl.Pipeline,
    attn256: mtl.Pipeline,
    splitk: mtl.Pipeline,
    parts_sum: mtl.Pipeline,
    pl: [16]mtl.Pipeline, // normed, act, mix, router, route, offsets, sort, gather rows, act2, scatter, copy, DeltaNet pre/scan/post, scan4, scan8
    scan8: bool = true, // scan4 with a state row over 16 lanes, 8 rows a simdgroup
    scan4: bool = true, // the DeltaNet recurrence with four state rows a simdgroup and both reductions at once
    gdn: Variant,
    gdn_grid: mtl.Size,
    gdn_tg: mtl.Size,
    sel: ?Select = null, // long prompts: selection for the chunk's rows
    step: usize = PMAX, // rows a chunk
    fast_attn: bool = true, // prompt rows' scores and attention on the tensor units (false: the decode's kernels)
    sattn: mtl.Pipeline,
    scores_nax: mtl.Pipeline,
    skip: u32 = 0, // timing knock-outs: 1 experts, 2 DeltaNet, 4 attention, 8 projections, 16 hyper-connections, 32 routing, 64 router, 128 sort, 256 row moves, 512 shared expert, 1024 sparse attention
    proj: [LAYERS][3]Buf,
    out: [LAYERS][3]Buf,
    ple_kv: [3]Buf,
    b: struct { ids: Buf, pids: Buf, h: [2]Buf, ssp: Buf, normed: Buf, dn: Buf, hact: Buf, inj_a: Buf, inj_m: Buf, up: Buf, mixed: Buf, p: Buf, gout: Buf, branch: Buf, cso: Buf, q: Buf, kout: Buf, iq: Buf, po: Buf, pm: Buf, aout: Buf, pos: Buf, pos3: Buf, nk: Buf, zeros: Buf, kvmeta: Buf, lg: Buf, pick: Buf, wts: Buf, cnt: Buf, off: Buf, cur: Buf, row_of: Buf, xs: Buf, g: Buf, u: Buf, a: Buf, ds: Buf, sg: Buf, su: Buf, sa: Buf, ydown: Buf, emb: Buf, kvp: Buf, gated: Buf, hout: Buf, cin: Buf, rows: Buf, part: Buf, qn: Buf, kn: Buf, v: Buf, gg: Buf, beta: Buf, ys: Buf, mids: Buf, n_add: Buf },

    pub fn init(r: *Run, dir: []const u8, header: []const u8) !Prompt {
        var p: Prompt = undefined; // every field is set below: `undefined` skips the declared defaults
        p.r = r;
        p.fsr_width = .{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 6, 32, 6, 32 } };
        p.wide = null;
        p.skip = 0;
        p.sel = null;
        p.step = PMAX;
        p.fast_attn = true;
        p.scan4 = true;
        p.scan8 = true;
        p.tall_tiles = false; // measured: no faster than 64-row tiles (same bits)
        p.fused_gu = true;
        const alib = try r.ownResource(try mtl.Library.fromSource(r.device, try frags.source(r.device, r.arena, ks.flashnext_attn), mtl.CompileOptions.mlx()));
        p.sattn = try r.ownResource(try mtl.Pipeline.init(r.device, alib, "tf_sattn_nax", false));
        p.scores_nax = try r.ownResource(try mtl.Pipeline.init(r.device, alib, "tf_idx_scores_nax", false));
        const qtext = if (r.window.ssd_experts != null) try @import("fsr_prompt_source.zig").patch(r.arena, ks.flashnext_qmm6, false) else ks.flashnext_qmm6;
        const qlib = try r.ownResource(try mtl.Library.fromSource(r.device, try frags.source(r.device, r.arena, qtext), mtl.CompileOptions.mlx()));
        p.qmm6 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_t_nax", false));
        p.gather64 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_64", false));
        p.gather32 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_32", false));
        p.qmm6_128 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_t_nax_128", false));
        p.gather128 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_gather_qmm6_nax_128", false));
        p.gu64 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_gather_gu6_nax_64", false));
        p.gu32 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_gather_gu6_nax_32", false));
        p.router_mm = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_mm_bf16_f32_t_nax", false));
        p.attn256 = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_attn256_nax", false));
        p.splitk = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_qmm6_splitk_nax", false));
        p.parts_sum = try r.ownResource(try mtl.Pipeline.init(r.device, qlib, "tf_parts_sum", false));
        try frags.check(r.device, r.queue, r.arena);
        const glib = try r.ownResource(try mtl.Library.fromSource(r.device, try std.mem.concat(r.arena, u8, &.{ header, ks.flashnext_prompt }), mtl.CompileOptions.mlx()));
        const names = [_][:0]const u8{ "pf_hc_normed", "pf_hc_act", "pf_hc_mix", "pf_router", "pf_route", "pf_offsets", "pf_sort", "pf_gather_rows", "pf_act", "pf_scatter_y", "pf_copy", "pf_gdn_pre", "pf_gdn_scan", "pf_gdn_post", "pf_gdn_scan4", "pf_gdn_scan8" };
        for (names, 0..) |n, i| p.pl[i] = try r.ownResource(try mtl.Pipeline.init(r.device, glib, n, false));
        // DeltaNet at the chunk's rows, storing only the last row's recurrent state (in row 0)
        const gs = r.roles.get("q4_gdn@gdn|8") orelse return error.NoSite;
        const f = try Run.variantText(r.arena, gs.v); // the embedded text when the run is checked-in, else the recorded file
        const from = "SO[((size_t(r) * NV + hv)";
        if (std.mem.count(u8, f, from) != 1) return error.GdnPatch;
        const patched = try std.mem.replaceOwned(u8, r.arena, f, from, "if (r == R - 1) SO[((size_t(0) * NV + hv)");
        const dlib = try r.ownResource(try mtl.Library.fromSource(r.device, patched, mtl.CompileOptions.mlx()));
        p.gdn = gs.v.*;
        p.gdn.pipe = try r.ownResource(try mtl.Pipeline.init(r.device, dlib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{gs.v.name}, 0), false));
        p.gdn_grid = gs.grid;
        p.gdn_tg = gs.tg;
        try r.indexFile(try std.fmt.allocPrintSentinel(r.arena, "{s}/pack_mlx.safetensors", .{dir}, 0));
        for (0..LAYERS) |i| {
            const kind = if (i % 4 != 3) "gdn" else "att";
            const o_name = if (i % 4 != 3) "out" else "o";
            const in_name = if (i % 4 != 3) "in" else "proj";
            for (0..3) |k| {
                const suf = [_][]const u8{ "mw", "ms", "mb" };
                p.proj[i][k] = try r.loadf("L{d}.{s}.{s}.{s}", .{ i, kind, in_name, suf[k] });
                p.out[i][k] = try r.loadf("L{d}.{s}.{s}.{s}", .{ i, kind, o_name, suf[k] });
            }
        }
        p.ple_kv = .{ try r.load("ple.kv.mw"), try r.load("ple.kv.ms"), try r.load("ple.kv.mb") };
        p.mtp_w = null;
        p.expert_bm = 0;
        if (r.indexFile(try std.fmt.allocPrintSentinel(r.arena, "{s}/pack_mtp_mlx.safetensors", .{dir}, 0))) |_| {
            var w: [3][3]Buf = undefined;
            for ([_][]const u8{ "mtp.att.proj", "mtp.fce", "mtp.fch" }, 0..) |name, i| {
                for ([_][]const u8{ "mw", "ms", "mb" }, 0..) |suf, k| w[i][k] = try r.loadf("{s}.{s}", .{ name, suf });
            }
            p.mtp_w = w;
        } else |_| {}
        p.b = try buffers(r);
        p.sel = try Select.init(r, PMAX);
        p.seg_queue = null;
        p.last = p.b.h[0];
        return p;
    }

    /// Another staggered segment's prompt: the same kernels and weights, its own buffers, selection and queue.
    pub fn sibling(p: *const Prompt) !Prompt {
        var s = p.*;
        s.b = try buffers(p.r);
        s.sel = try Select.init(p.r, PMAX);
        s.seg_queue = null;
        s.last = s.b.h[0];
        return s;
    }

    /// A prompt's own buffers for chunks of up to PMAX rows (each staggered segment has its own set).
    pub fn buffers(r: anytype) !@FieldType(Prompt, "b") {
        const sizes = @import("flashnext_memory").prompt.sizes;
        var out: @FieldType(Prompt, "b") = undefined;
        inline for (@typeInfo(@TypeOf(out)).@"struct".field_names) |name| {
            const n = @field(sizes, name);
            if (comptime std.mem.eql(u8, name, "h")) {
                inline for (0..2) |i| @field(out, name)[i] = .{ .b = try r.buffer(n[i]) };
            } else @field(out, name) = .{ .b = try r.buffer(n) };
        }
        return out;
    }

    pub fn barrier(p: *Prompt) void {
        if (!p.r.serial) p.r.enc.barrier();
    }

    /// The decode kernels' attention partials (3.2 GB at PMAX rows), made at first use: the tensor-op path never...
    pub fn partials(p: *Prompt) !void {
        if (p.b.po.b.length() >= PMAX * 24 * 16 * 256 * 4) return;
        p.b.po = .{ .b = try p.r.buffer(PMAX * 24 * 16 * 256 * 4) };
        p.b.pm = .{ .b = try p.r.buffer(PMAX * 24 * 16 * 2 * 4) };
    }

    pub fn bind(p: *Prompt, pipe: mtl.Pipeline, bufs: []const Buf) void {
        p.r.enc.setPipeline(pipe);
        for (bufs, 0..) |b, j| p.r.enc.setBuffer(b.b, b.off, j);
    }

    /// y[rows, n] (row stride ldy, 0: n) = x[rows, k] W^T, W 6-bit g32 in MLX's layout.
    pub fn qmm(p: *Prompt, x: Buf, w: [3]Buf, k: usize, n: usize, rows: usize, y: Buf, ldy: usize) void {
        if (p.skip & 8 != 0) return;
        if (p.wide) |wd| if (wd.wide(w[0])) |ww| return prompt_w.qmm(p, wd, ww, x, w, k, n, rows, y, ldy);
        const tall = p.tall_tiles and rows >= 512;
        p.bind(if (tall) p.qmm6_128 else p.qmm6, &.{ w[0], w[1], w[2], x });
        const prm = [4]i32{ @intCast(k), @intCast(n), @intCast(rows), @intCast(ldy) };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
        p.r.enc.setBuffer(y.b, y.off, 5);
        const bm: usize = if (tall) 128 else 64;
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, (rows + bm - 1) / bm, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// y[pairs, n] = x[slot] W_e^T over the sorted slots, the experts' first slots in b.off.
    pub fn gather(p: *Prompt, x: Buf, w: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
        if (p.skip & 1 != 0) return;
        // tiles of 32, 64 or 128 rows as experts average 32 and 128 rows (128: FZ prompt chunks of thousands of rows)
        const bm: usize = if (p.expert_bm != 0) p.expert_bm else if (p.tall_tiles and pairs >= 512 * 128) 128 else if (pairs >= 512 * 32) 64 else 32;
        if (p.r.fsr) |stream| if (stream.pool != null) return @import("fsr_gather.zig").gather(p, stream, bm, x, w, k, n, pairs, y);
        if (p.wide) |wd| if (wd.wide(w[0])) |ww| return prompt_w.gather(p, wd.of(ww.bits, ww.group), bm, x, w, k, n, pairs, y);
        p.bind(if (bm == 128) p.gather128 else if (bm == 64) p.gather64 else p.gather32, &.{ x, w[0], w[1], w[2], p.b.off });
        const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 5);
        p.r.enc.setBuffer(y.b, y.off, 6);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// a[pairs, n] = act(x[slot] Wg_e^T, x[slot] Wu_e^T) over the sorted slots in one pass (gather's tiles and sums).
    pub fn gatherGU(p: *Prompt, x: Buf, wg: []const Buf, wu: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
        if (p.skip & 1 != 0) return;
        const bm: usize = if (p.expert_bm == 32 or p.expert_bm == 64) p.expert_bm else if (pairs >= 512 * 32) 64 else 32;
        if (p.r.fsr) |stream| if (stream.pool != null) return @import("fsr_gather.zig").gate(p, stream, bm, x, wg, wu, k, n, pairs, y);
        if (p.wide) |wd| if (wd.wide(wg[0])) |ww| return prompt_w.gatherGU(p, wd.of(ww.bits, ww.group), bm, x, wg, wu, k, n, pairs, y);
        p.bind(if (bm == 64) p.gu64 else p.gu32, &.{ x, wg[0], wg[1], wg[2], wu[0], wu[1], wu[2], p.b.off });
        const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
        p.r.enc.setBytes(std.mem.asBytes(&prm), 8);
        p.r.enc.setBuffer(y.b, y.off, 9);
        p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
        p.barrier();
    }

    /// The block input from streams hn (after hc_norm wrote b.ssp): normed, down + inject, act, up, the stream mix.
    pub fn hc(p: *Prompt, m: *Model, hn: Buf, w: Hc, rows: usize, inj: Buf) void {
        if (p.skip & 16 != 0) return;
        const b = &p.b;
        const nd = w.dw.b.length() / (WIDE * (if (w.w) |hw| hw[0][0] else 6) / 8); // the down rows (+ inject rows)
        p.bind(p.pl[0], &.{ hn, b.ssp, w.scale, m.t.eps, b.normed });
        p.r.enc.dispatchThreads(mtl.Size.of(WIDE, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        if (p.skip & 8 == 0) { // the down projection's 10,240-deep sum in 8 parts, added in order
            const parts: usize = 8;
            p.bind(if (w.w) |hw| p.wide.?.of(@intCast(hw[0][0]), @intCast(hw[0][1])).splitk else p.splitk, &.{ w.dw, w.ds, w.db, b.normed });
            const prm = [4]i32{ WIDE, @intCast(nd), @intCast(rows), @intCast(parts) };
            p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
            p.r.enc.setBuffer(b.part.b, b.part.off, 5);
            p.r.enc.dispatchThreads(mtl.Size.of(((nd + 63) / 64) * 128, (rows + 63) / 64, parts), mtl.Size.of(128, 1, 1));
            p.barrier();
            p.bind(p.parts_sum, &.{b.part});
            const ps = [2]i32{ @intCast(parts), @intCast(rows * nd) };
            p.r.enc.setBytes(std.mem.asBytes(&ps), 1);
            p.r.enc.setBuffer(b.dn.b, b.dn.off, 2);
            p.r.enc.dispatchThreads(mtl.Size.of(rows * nd, 1, 1), mtl.Size.of(256, 1, 1));
            p.barrier();
        }
        p.bind(p.pl[1], &.{b.dn});
        const ndi: i32 = @intCast(nd);
        p.r.enc.setBytes(std.mem.asBytes(&ndi), 1);
        p.r.enc.setBuffer(b.hact.b, b.hact.off, 2);
        p.r.enc.setBuffer(inj.b, inj.off, 3);
        p.r.enc.dispatchThreads(mtl.Size.of(nd, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        p.qmm(b.hact, .{ w.uw, w.us, w.ub }, 320, WIDE, rows, b.up, 0);
        p.bind(p.pl[2], &.{ b.up, b.normed, b.mixed });
        p.r.enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
    }

    /// Routed experts by expert over the chunk's pairs, the shared expert dense; outputs in the combine's layout.
    pub fn moe(p: *Prompt, L: *Layer, rows: usize, layer: usize) !void {
        p.fsr_width = L.xw orelse .{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 6, 32, 6, 32 } };
        const b = &p.b;
        const r = p.r;
        const pairs = rows * 10;
        if (p.skip & 32 != 0) {
            p.gather(b.xs, L.ex[0..3], D, 640, pairs, b.g);
            p.gather(b.xs, L.ex[3..6], D, 640, pairs, b.u);
            p.gather(b.a, L.ex[12..15], 640, D, pairs, b.ds);
            return;
        }
        if (p.skip & 64 == 0) p.bind(p.router_mm, &.{ L.router, b.mixed }) else p.bind(p.pl[10], &.{ b.lg, b.lg });
        const rp = [3]i32{ D, 513, @intCast(rows) };
        r.enc.setBytes(std.mem.asBytes(&rp), 2);
        r.enc.setBuffer(b.lg.b, b.lg.off, 3);
        r.enc.dispatchThreads(if (p.skip & 64 == 0) mtl.Size.of(((513 + 63) / 64) * 128, (rows + 63) / 64, 1) else mtl.Size.of(1, 1, 1), mtl.Size.of(if (p.skip & 64 == 0) 128 else 1, 1, 1));
        p.barrier();
        if (p.skip & 128 == 0) {
            p.bind(p.pl[4], &.{ b.lg, b.pick, b.wts, b.cnt });
            r.enc.dispatchThreads(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
            p.barrier();
            p.bind(p.pl[5], &.{ b.cnt, b.off, b.cur });
            r.enc.dispatchThreads(mtl.Size.of(512, 1, 1), mtl.Size.of(512, 1, 1));
            p.barrier();
            p.bind(p.pl[6], &.{ b.pick, b.cur, b.row_of });
            const np: i32 = @intCast(pairs);
            r.enc.setBytes(std.mem.asBytes(&np), 3);
            r.enc.dispatchThreads(mtl.Size.of(pairs, 1, 1), mtl.Size.of(256, 1, 1));
            p.barrier();
        }
        if (r.fsr) |stream| try stream.prompt(r, layer, b.pick, rows);
        if (p.skip & 256 == 0) {
            p.bind(p.pl[7], &.{ b.mixed, b.row_of, b.xs });
            r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
            p.barrier();
        }
        if (p.fused_gu) p.gatherGU(b.xs, L.ex[0..3], L.ex[3..6], D, 640, pairs, b.a) else {
            p.gather(b.xs, L.ex[0..3], D, 640, pairs, b.g);
            p.gather(b.xs, L.ex[3..6], D, 640, pairs, b.u);
            p.bind(p.pl[8], &.{ b.g, b.u, b.a });
            r.enc.dispatchThreads(mtl.Size.of(pairs * 640, 1, 1), mtl.Size.of(256, 1, 1));
            p.barrier();
        }
        p.gather(b.a, L.ex[12..15], 640, D, pairs, b.ds);
        if (p.skip & 256 == 0) {
            p.bind(p.pl[9], &.{ b.ds, b.row_of, b.ydown });
            r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
            p.barrier();
        }
        if (p.skip & 512 != 0) return;
        p.qmm(b.mixed, .{ L.ex[6], L.ex[7], L.ex[8] }, D, 640, rows, b.sg, 0);
        p.qmm(b.mixed, .{ L.ex[9], L.ex[10], L.ex[11] }, D, 640, rows, b.su, 0);
        p.bind(p.pl[8], &.{ b.sg, b.su, b.sa });
        r.enc.dispatchThreads(mtl.Size.of(rows * 640, 1, 1), mtl.Size.of(256, 1, 1));
        p.barrier();
        p.qmm(b.sa, .{ L.ex[15], L.ex[16], L.ex[17] }, 640, D, rows, .{ .b = b.ydown.b, .off = 10 * D * 2 }, 11 * D);
    }

    /// Each kernel class alone, 48 times in one command buffer, on the buffers the last chunk left (layer 0's and...
    pub const classes = @import("replay_prompt_chunks.zig").classes;

    pub const copyWords = @import("replay_prompt_chunks.zig").copyWords;

    /// One prompt chunk from the model's position: caches, DeltaNet states, the n-gram tail and history move past it.
    pub const chunk = @import("replay_prompt_chunks.zig").chunk;

    /// One staggered segment of a chunk: its prompt, rows and first position; the DeltaNet state slot and row it...
    pub const Seg = @import("replay_prompt_segments.zig").Seg;

    /// The DeltaNet pre and scan kernels' mark arguments: the segment's mark rows and layer `li`'s slots (any buffer...
    pub const bindMarks = @import("replay_prompt_segments.zig").bindMarks;

    /// A chunk's host inputs at position `pos`: token ids, positions, the cache meta, the n-gram ids after `hist`.
    pub const prep = @import("replay_prompt_segments.zig").prep;

    /// A segment's layer i up to its mixer: the pending expert write (or the n-gram block at layer 1) and the...
    pub const segPre = @import("replay_prompt_segments.zig").segPre;

    /// A segment's layer-i mixer, DeltaNet or attention, at the segment's position and state slots.
    pub const segMixer = @import("replay_prompt_segments.zig").segMixer;

    /// A segment's layer i after its mixer: the branch write, the MLP hyper-connection and the experts.
    pub const segPost = @import("replay_prompt_segments.zig").segPost;

    /// Speed-up mode's prefill (chunkPair): this Mac's segment of a chunk whose rows are split across two Macs.
    pub const Pair = @import("replay_prompt_segments.zig").Pair;

    pub const bytesOf = @import("replay_prompt_segments.zig").bytesOf;

    /// Segment 0, after layer i's mixer: once the GPU gets here the host sends the state (DeltaNet) or new key,...
    pub const pairSend = @import("replay_prompt_segments.zig").pairSend;

    /// Segment 1, before layer i's mixer (its n-gram block at layer 1): the GPU waits for the layer's flag, then...
    pub const pairTake = @import("replay_prompt_segments.zig").pairTake;

    /// Flash Next's hooks for core/segments.zig: a segment's embedding, layer parts and final streams, encoded into...
    pub const Hooks = @import("replay_prompt_segments.zig").Hooks;

    /// A chunk as one to segments.MAX staggered segments, one queue each (core/segments.zig): segment k runs each...
    pub const chunkN = @import("replay_prompt_chunks.zig").chunkN;

    /// chunkN, with each DeltaNet layer also writing its state after each of `marks` rows (ascending, 1..n) to m.marks.
    pub const chunkMarked = @import("replay_prompt_chunks.zig").chunkMarked;

    /// Speed-up mode's prefill: a chunk's rows split across two Macs (tp.zig).
    pub const chunkPair = @import("replay_prompt_chunks.zig").chunkPair;

    /// Speed-up mode: once each Mac has written the MTP head's prompt keys for its segment's rows (`mine`: first...
    pub const pairMtp = @import("replay_prompt_chunks.zig").pairMtp;

    /// Write the MTP keys and values for these prompt rows; the final row's full head runs later.
    pub const mtpKeys = @import("replay_prompt_chunks.zig").mtpKeys;
};

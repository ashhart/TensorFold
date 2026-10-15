//! Gemma 4 dense glue kernels against gemma4_dump.py's fixtures: embed, norm, q/k/v, attention, tails, softcap.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const cfg = g4.config;
const npy = tf.npy;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Ctx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    queue: mtl.Queue,
    ckpt: *tf.checkpoint_metal.Checkpoint,
    k: g4.kernels.Kernels,
    dump: []const u8,
    failed: usize = 0,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    fn buf(c: *Ctx, bytes: usize) !mtl.Buffer {
        const b = try c.device.buffer(@max(bytes, 16), shared);
        try c.owned.append(c.gpa, b);
        return b;
    }

    /// A dumped array as a GPU buffer: bf16 bits (the dump writes bf16 as exact float32), or its raw f32/i32 data.
    fn load(c: *Ctx, name: []const u8, as_bf16: bool) !struct { b: mtl.Buffer, n: usize } {
        const path = try std.fmt.allocPrint(c.gpa, "{s}/{s}.npy", .{ c.dump, name });
        defer c.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .unlimited);
        defer c.gpa.free(bytes);
        const a = try npy.parse(bytes);
        const n = a.count();
        if (as_bf16) {
            if (!std.mem.eql(u8, a.descr, "<f4")) return error.UnexpectedDtype;
            const b = try c.buf(n * 2);
            const out = b.slice(u16, n);
            for (out, 0..) |*o, i| o.* = @truncate(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little) >> 16);
            return .{ .b = b, .n = n };
        }
        const b = try c.buf(n * 4);
        @memcpy(b.slice(u8, n * 4), a.data[0 .. n * 4]);
        return .{ .b = b, .n = n };
    }

    fn tensor(c: *Ctx, name: []const u8) !mtl.Buffer {
        const t = try c.ckpt.get(name);
        const b = try c.buf(t.bytes);
        @memcpy(b.slice(u8, t.bytes), t.host(u8)[0..t.bytes]);
        return b;
    }

    fn run(c: *Ctx, body: anytype) !void {
        const cb = c.queue.commandBuffer();
        const e = cb.compute(.serial);
        try body.encode(e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failure: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }

    /// Counts bf16 outputs in `got` that differ from the dumped `want`.
    fn compare(c: *Ctx, label: []const u8, got: mtl.Buffer, want_name: []const u8) !void {
        const want = try c.load(want_name, true);
        const a = got.slice(u16, want.n);
        const b = want.b.slice(u16, want.n);
        var diff: usize = 0;
        for (a, b) |x, y| diff += @intFromBool(x != y);
        c.failed += diff;
        std.debug.print("{s} ({s}): {d} of {d} differ\n", .{ label, want_name, diff, want.n });
    }
};

var name_buf: [8][128]u8 = undefined;
fn nm(slot: usize, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(&name_buf[slot], f, args) catch unreachable;
}

fn bf16Of(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

fn rowsCheck(c: *Ctx, rows: usize, p0: usize) !void {
    const kd = &c.k.sliding;
    const tag = nm(0, "r{d}", .{rows});
    const ids = try c.load(nm(1, "{s}.tokens", .{tag}), false);
    const emb = try c.buf(rows * cfg.hidden * 2);
    const ew = try c.tensor("language_model.model.embed_tokens.weight");
    const es = try c.tensor("language_model.model.embed_tokens.scales");
    const eb = try c.tensor("language_model.model.embed_tokens.biases");
    const scale = bf16Of(@sqrt(@as(f32, cfg.hidden)));
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        ids: mtl.Buffer,
        w: mtl.Buffer,
        s: mtl.Buffer,
        b: mtl.Buffer,
        out: mtl.Buffer,
        scale: f32,
        rows: usize,
        fn encode(x: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(x.kd.embed);
            for ([_]mtl.Buffer{ x.ids, x.w, x.s, x.b, x.out }, 0..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue(x.scale, 5);
            e.dispatchThreads(.{ .width = cfg.hidden / 2, .height = x.rows }, .{ .width = 256 });
        }
    }{ .kd = kd, .ids = ids.b, .w = ew, .s = es, .b = eb, .out = emb, .scale = scale, .rows = rows });
    try c.compare("embed", emb, nm(2, "{s}.embed", .{tag}));
    const normed = try c.buf(rows * cfg.hidden * 2);
    const w0 = try c.tensor("language_model.model.layers.0.input_layernorm.weight");
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        x: mtl.Buffer,
        w: mtl.Buffer,
        out: mtl.Buffer,
        rows: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.kd.rms);
            for ([_]mtl.Buffer{ s.x, s.w, s.out }, 0..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue(cfg.eps, 3);
            e.setBuffer(s.out, 0, 4);
            e.setValue(@as(i32, 0), 5);
            e.dispatchGroups(.{ .width = s.rows }, .{ .width = 1024 });
        }
    }{ .kd = kd, .x = emb, .w = w0, .out = normed, .rows = rows });
    try c.compare("rms looped", normed, nm(2, "{s}.normed0", .{tag}));
    for ([_]usize{ 0, 5 }) |layer| try layerCheck(c, rows, p0, layer);
    const raw = try c.load(nm(3, "{s}.head_raw", .{tag}), true);
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        x: mtl.Buffer,
        n: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.kd.softcap);
            e.setBuffer(s.x, 0, 0);
            e.setValue(cfg.softcap, 1);
            e.dispatchThreads(.{ .width = s.n }, .{ .width = 256 });
        }
    }{ .kd = kd, .x = raw.b, .n = raw.n });
    try c.compare("softcap", raw.b, nm(4, "{s}.logits", .{tag}));
}

fn layerCheck(c: *Ctx, rows: usize, p0: usize, layer: usize) !void {
    const global = cfg.global(layer);
    const kd = c.k.kind(global);
    const dh = cfg.headDim(layer);
    const nk = cfg.kvHeads(layer);
    const pre = nm(0, "r{d}.L{d}", .{ rows, layer });
    var pbuf: [64]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&pbuf, "{s}", .{pre});
    const lw = nm(5, "language_model.model.layers.{d}.", .{layer});
    var lbuf: [64]u8 = undefined;
    const lp = try std.fmt.bufPrint(&lbuf, "{s}", .{lw});
    // q/k/v: head norms and RoPE at each row's position
    const qkv = try c.load(nm(1, "{s}.qkv", .{prefix}), true);
    const qw = try c.tensor(nm(2, "{s}self_attn.q_norm.weight", .{lp}));
    const kw = try c.tensor(nm(2, "{s}self_attn.k_norm.weight", .{lp}));
    const invf = try c.load(nm(2, "L{d}.inv_freq", .{layer}), false);
    const pos = try c.buf(rows * 4);
    for (pos.slice(i32, rows), 0..) |*p, i| p.* = @intCast(p0 + i);
    const q = try c.buf(rows * cfg.query_heads * dh * 2);
    const k = try c.buf(nk * rows * dh * 2);
    const v = try c.buf(nk * rows * dh * 2);
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        bs: [8]mtl.Buffer,
        slots: usize,
        rows: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.kd.qkv);
            for (s.bs[0..5], 0..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue(cfg.eps, 5);
            for (s.bs[5..8], 6..) |bb, i| e.setBuffer(bb, 0, i);
            for (s.bs[6..8], 9..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue([3]i32{ 0, 0, 0 }, 11);
            e.dispatchGroups(.{ .width = s.slots, .height = s.rows }, .{ .width = 32 });
        }
    }{ .kd = kd, .bs = .{ qkv.b, qw, kw, invf.b, pos, q, k, v }, .slots = cfg.query_heads + 2 * nk, .rows = rows });
    try c.compare("q", q, nm(3, "{s}.q", .{prefix}));
    try c.compare("k", k, nm(3, "{s}.k", .{prefix}));
    try c.compare("v", v, nm(3, "{s}.v", .{prefix}));
    try attentionCheck(c, kd, prefix, rows, p0, global, dh, nk);
    try tailsCheck(c, kd, prefix, lp, layer, rows);
}

fn attentionCheck(c: *Ctx, kd: *const g4.kernels.Kind, prefix: []const u8, rows: usize, p0: usize, global: bool, dh: usize, nk: usize) !void {
    const ring: usize = if (global) 0 else cfg.window + 128;
    const cap: usize = if (global) p0 + 64 else ring;
    const kb = try c.buf(nk * cap * dh * 2);
    const vb = try c.buf(nk * cap * dh * 2);
    const before_k = try c.load(nm(1, "{s}.keys_before", .{prefix}), true);
    const before_v = try c.load(nm(2, "{s}.values_before", .{prefix}), true);
    const first: usize = if (global) 0 else (if (p0 + 1 > cfg.window) p0 + 1 - cfg.window else 0);
    const n_before = p0 - first;
    for (0..nk) |h| for (0..n_before) |j| {
        const p = first + j;
        const slot = if (global) p else p % ring;
        const src = (h * n_before + j) * dh;
        const dst = (h * cap + slot) * dh;
        @memcpy(kb.slice(u16, nk * cap * dh)[dst..][0..dh], before_k.b.slice(u16, before_k.n)[src..][0..dh]);
        @memcpy(vb.slice(u16, nk * cap * dh)[dst..][0..dh], before_v.b.slice(u16, before_v.n)[src..][0..dh]);
    };
    const q = try c.load(nm(3, "{s}.q", .{prefix}), true);
    const kn = try c.load(nm(4, "{s}.k", .{prefix}), true);
    const vn = try c.load(nm(6, "{s}.v", .{prefix}), true);
    const pos = try c.buf(rows * 4);
    const lo = try c.buf(rows * 4);
    var low: usize = std.math.maxInt(usize);
    for (0..rows) |i| {
        const p = p0 + i;
        pos.slice(i32, rows)[i] = @intCast(p);
        const l: usize = if (global) 0 else (if (p + 1 > cfg.window) p + 1 - cfg.window else 0);
        lo.slice(i32, rows)[i] = @intCast(l);
        low = @min(low, l);
    }
    const first_chunk = low / kd.chunk;
    const chunks = (p0 + rows - 1) / kd.chunk - first_chunk + 1;
    const meta = try c.buf(32);
    meta.slice(i32, 8)[0..5].* = .{ @intCast(first_chunk), @intCast(chunks), @intCast(ring), @intCast(rows), @intCast(p0) };
    const slots = cfg.query_heads * rows * chunks;
    const pm = try c.buf(slots * 4);
    const pl = try c.buf(slots * 4);
    const po = try c.buf(slots * dh * 4);
    const out = try c.buf(rows * cfg.query_heads * dh * 2);
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        bs: [13]mtl.Buffer,
        cap: i32,
        chunks: usize,
        nk: usize,
        rows: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.kd.partial);
            for (s.bs[0..8], 0..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue(s.cap, 8);
            for (s.bs[8..11], 9..) |bb, i| e.setBuffer(bb, 0, i);
            e.dispatchGroups(.{ .width = s.chunks, .height = s.nk, .depth = s.rows }, .{ .width = 32 * s.kd.heads_per_kv * s.kd.split });
            e.barrier();
            e.setPipeline(s.kd.merge);
            for ([_]mtl.Buffer{ s.bs[8], s.bs[9], s.bs[10], s.bs[7], s.bs[11], s.bs[8] }, 0..) |bb, i| e.setBuffer(bb, 0, i);
            e.setValue(@as(i32, 0), 6);
            e.dispatchGroups(.{ .width = 1, .height = cfg.query_heads, .depth = s.rows }, .{ .width = 32 });
        }
    }{ .kd = kd, .bs = .{ q.b, kb, vb, kn.b, vn.b, pos, lo, meta, pm, pl, po, out, out }, .cap = @intCast(cap), .chunks = chunks, .nk = nk, .rows = rows });
    try c.compare("attention", out, nm(5, "{s}.o_in", .{prefix}));
}

fn tailsCheck(c: *Ctx, kd: *const g4.kernels.Kind, prefix: []const u8, lp: []const u8, layer: usize, rows: usize) !void {
    const h = try c.load(nm(1, "{s}.h", .{prefix}), true);
    const o = try c.load(nm(2, "{s}.o", .{prefix}), true);
    const wa = try c.tensor(nm(3, "{s}post_attention_layernorm.weight", .{lp}));
    const w1 = try c.tensor(nm(3, "{s}pre_feedforward_layernorm.weight", .{lp}));
    const hn = try c.buf(rows * cfg.hidden * 2);
    const n1 = try c.buf(rows * cfg.hidden * 2);
    const Tail = struct {
        pipe: mtl.Pipeline,
        bs: []const mtl.Buffer,
        eps_at: usize,
        rows: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.pipe);
            for (s.bs, 0..) |bb, i| e.setBuffer(bb, 0, if (i < s.eps_at) i else i + 1);
            e.setValue(cfg.eps, s.eps_at);
            e.setBuffer(s.bs[0], 0, s.bs.len + 1);
            e.setValue(@as(i32, 0), s.bs.len + 2);
            e.setBuffer(s.bs[0], 0, s.bs.len + 3); // dense_tail: no tap
            e.setValue(@as(i32, 0), s.bs.len + 4);
            e.dispatchGroups(.{ .width = s.rows }, .{ .width = 256 });
        }
    };
    try c.run(Tail{ .pipe = kd.attn_tail, .bs = &.{ h.b, o.b, wa, w1, hn, n1 }, .eps_at = 4, .rows = rows });
    try c.compare("attn_tail hn", hn, nm(4, "{s}.hn", .{prefix}));
    try c.compare("attn_tail norm", n1, nm(4, "{s}.n_mlp", .{prefix}));
    const gu = try c.load(nm(1, "{s}.gate_up", .{prefix}), true);
    const act = try c.buf(rows * cfg.intermediate * 2);
    try c.run(struct {
        kd: *const g4.kernels.Kind,
        gu: mtl.Buffer,
        act: mtl.Buffer,
        rows: usize,
        fn encode(s: @This(), e: mtl.ComputeEncoder) !void {
            e.setPipeline(s.kd.geglu);
            e.setBuffer(s.gu, 0, 0);
            e.setBuffer(s.act, 0, 1);
            e.setValue(@as(u32, cfg.intermediate), 2);
            e.setBuffer(s.act, 0, 3);
            e.setValue(@as(i32, 0), 4);
            e.dispatchGroups(.{ .width = cfg.intermediate / 64, .height = s.rows }, .{ .width = 64 });
        }
    }{ .kd = kd, .gu = gu.b, .act = act, .rows = rows });
    try c.compare("geglu", act, nm(4, "{s}.act", .{prefix}));
    const hn_in = try c.load(nm(1, "{s}.hn", .{prefix}), true);
    const y = try c.load(nm(2, "{s}.y", .{prefix}), true);
    const wp = try c.tensor(nm(3, "{s}post_feedforward_layernorm.weight", .{lp}));
    const sc = try c.tensor(nm(3, "{s}layer_scalar", .{lp}));
    const wn = try c.tensor(nm(3, "language_model.model.layers.{d}.input_layernorm.weight", .{layer + 1}));
    const h2 = try c.buf(rows * cfg.hidden * 2);
    const nx = try c.buf(rows * cfg.hidden * 2);
    try c.run(Tail{ .pipe = kd.dense_tail, .bs = &.{ hn_in.b, y.b, wp, sc, wn, h2, nx }, .eps_at = 5, .rows = rows });
    try c.compare("dense_tail h", h2, nm(4, "{s}.h2", .{prefix}));
    try c.compare("dense_tail next", nx, nm(4, "{s}.normed2", .{prefix}));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model = args.next() orelse return error.Usage;
    const dump = args.next() orelse return error.Usage;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ckpt = tf.checkpoint_metal.Checkpoint.init(gpa);
    defer ckpt.deinit();
    const files = try tf.checkpoint_host.shardFiles(gpa, init.io, model);
    defer tf.checkpoint_host.freeShardFiles(gpa, files);
    for (files) |path| try ckpt.addFileSelected(device, path, "", "language_model.");
    var c = Ctx{ .gpa = gpa, .io = init.io, .device = device, .queue = queue, .ckpt = &ckpt, .k = try g4.kernels.Kernels.init(gpa, device), .dump = dump };
    defer c.k.deinit();
    defer {
        for (c.owned.items) |b| b.deinit();
        c.owned.deinit(gpa);
    }
    for ([_]bool{ false, true }) |global| {
        var mine: [cfg.global_dim / 2]f32 = undefined;
        const n: usize = @as(usize, if (global) cfg.global_dim else cfg.sliding_dim) / 2;
        try g4.kernels.inverseFrequencies(&c.k, queue, device, global, mine[0..n]);
        const ref = try c.load(if (global) "L5.inv_freq" else "L0.inv_freq", false);
        var diff: usize = 0;
        for (mine[0..n], ref.b.slice(u32, n)) |m, r| diff += @intFromBool(@as(u32, @bitCast(m)) != r);
        c.failed += diff;
        std.debug.print("rope table {s}: {d} of {d} differ\n", .{ if (global) "global" else "sliding", diff, n });
    }
    for ([_]usize{ 1, 2, 16 }) |rows| {
        std.debug.print("== rows {d}\n", .{rows});
        try rowsCheck(&c, rows, 1100);
    }
    std.debug.print("{s}: {d} differing outputs\n", .{ if (c.failed == 0) "all bits equal" else "bits differ", c.failed });
    if (c.failed > 0) std.process.exit(1);
}

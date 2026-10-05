//! Flash Next one-row greedy steps on the Python engine's own kernels (tools/zig/flashnext_dump.py): every launch is
//! a recorded variant, every weight is the pack's or the checkpoint's; tokens are checked against the Python engine's.
const std = @import("std");
const mtl = @import("metal");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const D = 2560;
const WIDE = 4 * D;
const LAYERS = 48;
const VOCAB = 248320;
const CAP = 8192; // keys an attention layer holds in this runner
const PLE_TAIL = 9;
const GROUPS = 8;

const Buf = struct { b: mtl.Buffer, off: usize = 0 };
const Entry = struct { fd: std.c.fd_t, at: usize, len: usize };
const Variant = struct { inputs: [][]const u8, outputs: [][]const u8, meta: [][]const u8, pipe: mtl.Pipeline };
const Site = struct { v: *Variant, grid: mtl.Size, tg: mtl.Size };

const glue_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void fz_kv_write(device const bfloat* kout [[buffer(0)]], device const bfloat* p [[buffer(1)]],
    \\    device bfloat* keys [[buffer(2)]], device bfloat* vals [[buffer(3)]], device bfloat* raw [[buffer(4)]],
    \\    constant uint* meta [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    \\  const uint pos = meta[0], cap = meta[1];
    \\  if (i < 512) { const uint at = ((i >> 8) * cap + pos) * 256 + (i & 255); keys[at] = kout[i]; vals[at] = p[12800 + i]; }
    \\  if (i < 128) raw[pos * 128 + i] = p[13824 + i];
    \\}
    \\kernel void fz_argmax(device const bfloat* x [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = t; i < n; i += 1024) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) out[0] = at;
    \\}
;

fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const Run = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    index: std.StringHashMapUnmanaged(Entry) = .empty,
    variants: std.StringHashMapUnmanaged(*Variant) = .empty,
    sites: std.StringHashMapUnmanaged(Site) = .empty,
    shapes: std.StringHashMapUnmanaged(mtl.Buffer) = .empty,
    loaded: usize = 0,
    enc: mtl.ComputeEncoder = undefined,
    kv_pipe: mtl.Pipeline = undefined,
    argmax_pipe: mtl.Pipeline = undefined,

    fn buffer(r: *Run, len: usize) !mtl.Buffer {
        const b = try r.device.buffer(@max(len, 64), opts);
        @memset(b.contents()[0..@max(len, 64)], 0);
        return b;
    }

    /// Every tensor of a safetensors file in the index, at its absolute offset.
    fn indexFile(r: *Run, path: [:0]const u8) !void {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try r.arena.alloc(u8, n);
        try readAll(fd, text, 8);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, text, .{});
        var it = parsed.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const offs = kv.value_ptr.object.get("data_offsets").?.array.items;
            const lo: usize = @intCast(offs[0].integer);
            const hi: usize = @intCast(offs[1].integer);
            try r.index.put(r.arena, kv.key_ptr.*, .{ .fd = fd, .at = 8 + n + lo, .len = hi - lo });
        }
    }

    fn entry(r: *Run, name: []const u8) !Entry {
        return r.index.get(name) orelse {
            std.log.err("no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    fn load(r: *Run, name: []const u8) !Buf {
        const e = try r.entry(name);
        const b = try r.device.buffer(@max(e.len, 64), opts);
        try readAll(e.fd, b.contents()[0..e.len], e.at);
        r.loaded += e.len;
        return .{ .b = b };
    }

    fn loadf(r: *Run, comptime fmt: []const u8, args: anytype) !Buf {
        var name: [160]u8 = undefined;
        return r.load(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// Shards `first..first+count` of `suffix` concatenated into one buffer (PleTables' group).
    fn group(r: *Run, first: usize, count: usize, suffix: []const u8) !Buf {
        var total: usize = 0;
        var name: [160]u8 = undefined;
        for (first..first + count) |s| total += (try r.entry(try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ s, suffix }))).len;
        const b = try r.device.buffer(total, opts);
        var at: usize = 0;
        for (first..first + count) |s| {
            const e = try r.entry(try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ s, suffix }));
            try readAll(e.fd, b.contents()[at .. at + e.len], e.at);
            at += e.len;
        }
        r.loaded += total;
        return .{ .b = b };
    }

    fn compile(r: *Run, dir: []const u8) !void {
        const path = try std.fmt.allocPrintSentinel(r.arena, "{s}/plan.json", .{dir}, 0);
        const f = try mtl.MappedFile.open(path);
        const plan = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, f.bytes[0..f.size], .{});
        var vit = plan.object.get("variants").?.object.iterator();
        while (vit.next()) |kv| {
            const o = kv.value_ptr.object;
            const src_path = try std.fmt.allocPrintSentinel(r.arena, "{s}/{s}", .{ dir, o.get("file").?.string }, 0);
            const src = try mtl.MappedFile.open(src_path);
            const lib = try mtl.Library.fromSource(r.device, src.bytes[0..src.size], mtl.CompileOptions.mlx());
            const v = try r.arena.create(Variant);
            v.* = .{ .inputs = try strings(r.arena, o.get("inputs").?), .outputs = try strings(r.arena, o.get("outputs").?), .meta = try strings(r.arena, o.get("meta").?), .pipe = try mtl.Pipeline.init(r.device, lib, kv.key_ptr.*, false) };
            try r.variants.put(r.arena, kv.key_ptr.*, v);
        }
        var sit = plan.object.get("sites").?.object.iterator();
        while (sit.next()) |kv| {
            const o = kv.value_ptr.object;
            try r.sites.put(r.arena, kv.key_ptr.*, .{ .v = r.variants.get(o.get("function").?.string).?, .grid = size3(o.get("grid").?), .tg = size3(o.get("threadgroup").?) });
        }
        const glue = try mtl.Library.fromSource(r.device, glue_source, mtl.CompileOptions.mlx());
        r.kv_pipe = try mtl.Pipeline.init(r.device, glue, "fz_kv_write", false);
        r.argmax_pipe = try mtl.Pipeline.init(r.device, glue, "fz_argmax", false);
    }

    fn strings(a: std.mem.Allocator, v: std.json.Value) ![][]const u8 {
        const out = try a.alloc([]const u8, v.array.items.len);
        for (v.array.items, 0..) |s, i| out[i] = s.string;
        return out;
    }

    fn size3(v: std.json.Value) mtl.Size {
        const a = v.array.items;
        return mtl.Size.of(@intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer));
    }

    /// One launch of `site` with its inputs and outputs in the variant's order (shape buffers after their array).
    fn call(r: *Run, site: []const u8, ins: []const Buf, outs: []const Buf) !void {
        const s = r.sites.get(site) orelse {
            std.log.err("no launch site {s}", .{site});
            return error.NoSite;
        };
        const v = s.v;
        if (ins.len != v.inputs.len or outs.len != v.outputs.len) {
            std.log.err("{s}: {d} inputs and {d} outputs given, the kernel takes {d} and {d}", .{ site, ins.len, outs.len, v.inputs.len, v.outputs.len });
            return error.Arity;
        }
        r.enc.setPipeline(v.pipe);
        var at: usize = 0;
        for (v.inputs, ins) |name, b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suffix| {
                for (v.meta) |m| {
                    if (m.len == name.len + suffix.len and std.mem.startsWith(u8, m, name) and std.mem.endsWith(u8, m, suffix)) {
                        r.enc.setBuffer(r.shapes.get(m) orelse return error.NoShape, 0, at);
                        at += 1;
                    }
                }
            }
        }
        for (outs) |b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
        }
        r.enc.dispatchThreads(s.grid, s.tg);
        r.enc.barrier();
    }
};

const Hc = struct { scale: Buf, dw: Buf, ds: Buf, db: Buf, uw: Buf, us: Buf, ub: Buf };
const Lane = struct { wq: Buf, sbt: Buf };
const Layer = struct {
    ahc: Hc,
    mhc: Hc,
    linear: bool,
    proj: Lane,
    out: Lane,
    conv: Buf = undefined,
    alog: Buf = undefined,
    dt: Buf = undefined,
    norm: Buf = undefined,
    qn: Buf = undefined,
    kn: Buf = undefined,
    iqn: Buf = undefined,
    router: Buf,
    ex: [18]Buf, // gate, up, sgate, sup (w,s,b) then down, sdown (w,s,b)
    // state
    cs: [2]Buf = undefined,
    so: [2]Buf = undefined,
    keys: Buf = undefined,
    vals: Buf = undefined,
    raw: Buf = undefined,
};

fn hcOf(r: *Run, comptime fmt: []const u8, args: anytype) !Hc {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    const parts = [_][]const u8{ "scale", "down.w", "down.s", "down.b", "up.w", "up.s", "up.b" };
    var out: [7]Buf = undefined;
    for (parts, 0..) |p, i| out[i] = try r.load(try std.fmt.bufPrint(&full, "{s}.{s}", .{ stem, p }));
    return .{ .scale = out[0], .dw = out[1], .ds = out[2], .db = out[3], .uw = out[4], .us = out[5], .ub = out[6] };
}

fn laneOf(r: *Run, comptime fmt: []const u8, args: anytype) !Lane {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    return .{ .wq = try r.load(try std.fmt.bufPrint(&full, "{s}.wq", .{stem})), .sbt = try r.load(try std.fmt.bufPrint(&full, "{s}.sbt", .{stem})) };
}

const Ple = struct {
    kv: Lane,
    ks: Buf,
    qs: Buf,
    cs: Buf,
    conv: Buf,
    starts: Buf,
    tables: [3 * GROUPS]Buf,
    cin: Buf,
    hist: [2]i64,
    eos: i64,
    mult: [3]i64,
    sizes: [16]i64,
    offsets: [16]i64,
};

/// The step's scratch: every intermediate a one-row forward writes.
const Tmp = struct {
    h: [2]Buf,
    ssp: Buf,
    part: Buf,
    mixed: Buf,
    inj_a: Buf,
    inj_m: Buf,
    xs: Buf,
    p: Buf,
    gout: Buf,
    branch: Buf,
    lg: Buf,
    act: Buf,
    pick: Buf,
    wts: Buf,
    ydown: Buf,
    q: Buf,
    kout: Buf,
    iq: Buf,
    po: Buf,
    pm: Buf,
    aout: Buf,
    emb: Buf,
    kvp: Buf,
    gated: Buf,
    hout: Buf,
    logits: Buf,
    pick_out: Buf,
    // constants and per-step values
    rows1: Buf,
    mdims: Buf,
    eps: Buf,
    ids8: Buf,
    pos8: Buf,
    nk8: Buf,
    zero8: Buf,
    ids81: Buf,
    scale: Buf,
    log2base: Buf,
    ple_ids: Buf,
    kvmeta: Buf,
    vocab: Buf,
};

fn f32Buf(r: *Run, v: f32) !Buf {
    const b = try r.buffer(4);
    b.slice(f32, 1)[0] = v;
    return .{ .b = b };
}

fn i32Buf(r: *Run, vals: []const i32) !Buf {
    const b = try r.buffer(vals.len * 4);
    @memcpy(b.slice(i32, vals.len), vals);
    return .{ .b = b };
}

const Model = struct {
    r: *Run,
    layers: [LAYERS]Layer,
    mix: Hc,
    head: Lane,
    embed: [3]Buf,
    ple: Ple,
    t: Tmp,
    pos: usize = 0,
    parity: usize = 0,

    fn lane(m: *Model, x: Buf, k: usize, l: Lane, site: []const u8, y: Buf) !void {
        const xs_site = if (k == D) "lane_qmm_xsum#[1, 2560]" else "lane_qmm_xsum#[1, 6144]";
        try m.r.call(xs_site, &.{ x, m.t.mdims }, &.{m.t.xs});
        try m.r.call(site, &.{ x, m.t.xs, l.wq, l.sbt, m.t.mdims }, &.{y});
    }

    fn hcProject(m: *Model, hn: Buf, hc: Hc, down_site: []const u8, up_site: []const u8, inj: ?Buf) !void {
        const t = &m.t;
        try m.r.call(down_site, &.{ hn, t.ssp, hc.scale, hc.dw, hc.ds, hc.db, t.eps, t.rows1 }, &.{t.part});
        if (inj) |out_inj| {
            try m.r.call(up_site, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, t.rows1 }, &.{ t.mixed, out_inj });
        } else {
            try m.r.call(up_site, &.{ hn, t.ssp, hc.scale, t.part, hc.uw, hc.us, hc.ub, t.eps, t.rows1 }, &.{t.mixed});
        }
    }

    fn grouped(m: *Model, h: Buf, out: Buf) !void {
        const t = &m.t;
        try m.r.call("q4_hc_norm_grouped#[1, 10240]", &.{ h, t.inj_m, t.ydown, t.wts, t.lg }, &.{ out, t.ssp });
    }

    fn pleIds(m: *Model, tok: i64) void {
        const p = &m.ple;
        const seq = [3]i64{ p.hist[0], p.hist[1], tok };
        var before: i64 = -1;
        if (p.hist[0] == p.eos) before = 0;
        if (p.hist[1] == p.eos) before = 1;
        const in_seg = 2 - (before + 1);
        var sh: [3]i64 = undefined;
        for (0..3) |s| sh[s] = if (in_seg >= @as(i64, @intCast(s))) seq[2 - s] else p.eos;
        const out = m.t.ple_ids.b.slice(u32, 16);
        for (2..4) |ng| {
            var mixed: i64 = sh[0] *% p.mult[0];
            for (1..ng) |q| mixed ^= sh[q] *% p.mult[q];
            for (0..8) |k| {
                const hh = (ng - 2) * 8 + k;
                out[hh] = @intCast(@mod(mixed, p.sizes[hh]) + p.offsets[hh]);
            }
        }
        p.hist = .{ p.hist[1], tok };
    }

    /// One token through every layer; returns the argmax of its logits.
    fn step(m: *Model, tok: u32) !u32 {
        const r = m.r;
        const t = &m.t;
        t.ids8.b.slice(u32, 8)[0] = tok;
        t.pos8.b.slice(i32, 8)[0] = @intCast(m.pos);
        t.nk8.b.slice(i32, 8)[0] = @intCast(m.pos + 1);
        t.kvmeta.b.slice(u32, 2)[0] = @intCast(m.pos);
        m.pleIds(tok);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(.concurrent);
        var cur: usize = 0; // t.h[cur] holds the streams
        try r.call("qa_embed_rows@embed", &.{ t.ids8, m.embed[0], m.embed[1], m.embed[2] }, &.{t.h[0]});
        var pending: enum { none, grouped } = .none;
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            if (i == 1) { // the PLE layer: write the pending MoE back, then the n-gram gate and conv
                try m.grouped(t.h[cur], t.h[1 - cur]);
                cur = 1 - cur;
                pending = .none;
                const p = &m.ple;
                var tabs: [2 + 3 * GROUPS]Buf = undefined;
                tabs[0] = t.ple_ids;
                tabs[1] = p.starts;
                for (0..3 * GROUPS) |j| tabs[2 + j] = p.tables[j];
                try r.call("qa_ple_lookup@ple", &tabs, &.{t.emb});
                try m.lane(t.emb, D, p.kv, "lane_qmm_bytes_grouped@ple.kv", t.kvp);
                try r.call("q4_ple_gate@ple", &.{ t.kvp, t.h[cur], p.ks, p.qs, p.cs, t.eps }, &.{ t.gated, .{ .b = p.cin.b, .off = PLE_TAIL * WIDE * 2 } });
                try r.call("q4_ple_conv@ple", &.{ p.cin, p.conv, t.gated, t.h[cur] }, &.{t.hout});
                try r.call("q4_hc_norm_none#[1, 10240]", &.{t.hout}, &.{ t.h[1 - cur], t.ssp });
            } else if (pending == .none) {
                try r.call("q4_hc_norm_none#[1, 10240]", &.{t.h[cur]}, &.{ t.h[1 - cur], t.ssp });
            } else {
                try m.grouped(t.h[cur], t.h[1 - cur]);
            }
            cur = 1 - cur;
            try m.hcProject(t.h[cur], L.ahc, "qa_hc_down_row@ahc", "qa_hc_up_row@ahc", t.inj_a);
            if (L.linear) {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@gdn.in", t.p);
                const a = m.parity;
                try r.call("q4_gdn_pipe@gdn", &.{ t.p, L.cs[a], L.so[a], L.conv, L.alog, L.dt, L.norm, t.eps, t.rows1 }, &.{ t.gout, L.cs[1 - a], L.so[1 - a] });
                try m.lane(t.gout, 6144, L.out, "lane_qmm_bytes_grouped@gdn.out", t.branch);
            } else {
                try m.lane(t.mixed, D, L.proj, "lane_qmm_bytes_grouped@att.proj", t.p);
                try r.call("q4_attn_prep@att", &.{ t.p, t.pos8, L.qn, L.kn, L.iqn, t.eps, t.log2base }, &.{ t.q, t.kout, t.iq });
                r.enc.setPipeline(r.kv_pipe);
                for ([_]Buf{ t.kout, t.p, L.keys, L.vals, L.raw, t.kvmeta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
                r.enc.dispatchThreads(mtl.Size.of(512, 1, 1), mtl.Size.of(256, 1, 1));
                r.enc.barrier();
                try r.call("q4_attn_parts#[1, 24, 256]", &.{ t.q, L.keys, L.vals, t.ids81, t.nk8, t.zero8, t.scale }, &.{ t.po, t.pm });
                try r.call("q4_attn_merge_gate#[1, 24, 16, 256]", &.{ t.po, t.pm, t.p }, &.{t.aout});
                try m.lane(t.aout, 6144, L.out, "lane_qmm_bytes_grouped@att.o", t.branch);
            }
            try r.call("q4_hc_norm_plain#[1, 10240]", &.{ t.h[cur], t.inj_a, t.branch }, &.{ t.h[1 - cur], t.ssp });
            cur = 1 - cur;
            try m.hcProject(t.h[cur], L.mhc, "qa_hc_down_row@mhc", "qa_hc_up_row@mhc", t.inj_m);
            try r.call("q4_router_float@moe", &.{ t.mixed, L.router, t.rows1 }, &.{t.lg});
            const e = L.ex;
            try r.call("qa_expert_gateup@moe.gate", &.{ t.mixed, t.lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11] }, &.{ t.act, t.pick, t.wts });
            try r.call("qa_expert_down_y@moe.down", &.{ t.act, t.pick, e[12], e[13], e[14], e[15], e[16], e[17], t.rows1 }, &.{t.ydown});
            pending = .grouped;
        }
        try m.grouped(t.h[cur], t.h[1 - cur]);
        cur = 1 - cur;
        try m.hcProject(t.h[cur], m.mix, "qa_hc_down_row@mix", "qa_hc_up_row@mix", t.inj_a);
        try m.lane(t.mixed, D, m.head, "lane_qmm_bytes_grouped@head", t.logits);
        r.enc.setPipeline(r.argmax_pipe);
        r.enc.setBuffer(t.logits.b, 0, 0);
        r.enc.setBuffer(t.pick_out.b, 0, 1);
        r.enc.setBuffer(t.vocab.b, 0, 2);
        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
        r.enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        const cin = m.ple.cin.b.contents();
        std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[WIDE * 2 .. (PLE_TAIL + 1) * WIDE * 2]);
        m.parity = 1 - m.parity;
        m.pos += 1;
        return t.pick_out.b.slice(u32, 1)[0];
    }
};

fn jsonInt(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |x| x,
        .float => |x| @intFromFloat(x),
        else => 0,
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-flashnext-run MODEL_DIR DUMP_DIR\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const device = try mtl.Device.init();
    var r = Run{ .gpa = gpa, .arena = arena, .device = device, .queue = try device.queue() };
    const t0 = mtl.clock.seconds();
    try r.compile(args[2]);
    const t1 = mtl.clock.seconds();

    // the checkpoint's shards and the pack
    const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{args[1]}, 0));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
    var files: std.StringHashMapUnmanaged(void) = .empty;
    var wit = index.object.get("weight_map").?.object.iterator();
    while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    var fit = files.keyIterator();
    while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ args[1], name.* }, 0));
    try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{args[2]}, 0));

    const ref_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{args[2]}, 0));
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, arena, ref_file.bytes[0..ref_file.size], .{});
    const ple_ref = ref.object.get("ple").?.object;

    const m = try arena.create(Model);
    m.r = &r;
    for (0..LAYERS) |i| {
        const linear = i % 4 != 3;
        var L: Layer = .{ .ahc = try hcOf(&r, "L{d}.ahc", .{i}), .mhc = try hcOf(&r, "L{d}.mhc", .{i}), .linear = linear, .proj = undefined, .out = undefined, .router = try r.loadf("L{d}.moe.router", .{i}), .ex = undefined };
        if (linear) {
            L.proj = try laneOf(&r, "L{d}.gdn.in", .{i});
            L.out = try laneOf(&r, "L{d}.gdn.out", .{i});
            L.conv = try r.loadf("L{d}.gdn.conv", .{i});
            L.alog = try r.loadf("L{d}.gdn.alog", .{i});
            L.dt = try r.loadf("L{d}.gdn.dt", .{i});
            L.norm = try r.loadf("L{d}.gdn.norm", .{i});
            for (0..2) |j| {
                L.cs[j] = .{ .b = try r.buffer(8 * 3 * WIDE * 2) };
                L.so[j] = .{ .b = try r.buffer(8 * 48 * 128 * 128 * 4) };
            }
        } else {
            L.proj = try laneOf(&r, "L{d}.att.proj", .{i});
            L.out = try laneOf(&r, "L{d}.att.o", .{i});
            L.qn = try r.loadf("L{d}.att.qn", .{i});
            L.kn = try r.loadf("L{d}.att.kn", .{i});
            L.iqn = try r.loadf("L{d}.att.iqn", .{i});
            L.keys = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
            L.vals = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
            L.raw = .{ .b = try r.buffer(CAP * 128 * 2) };
        }
        const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
        // gateup takes gate, up, shared gate, shared up; down takes down, shared down
        const order = [_]usize{ 0, 1, 2, 3, 4, 5 };
        for (order, 0..) |pi, j| {
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                L.ex[j * 3 + k] = try r.loadf("language_model.model.layers.{d}.mlp.{s}.{s}", .{ i, projs[pi], suffix });
            }
        }
        m.layers[i] = L;
        if (i % 8 == 7) std.debug.print("layer {d} loaded ({d:.1} GB so far)\n", .{ i, @as(f64, @floatFromInt(r.loaded)) / 1e9 });
    }
    m.mix = try hcOf(&r, "mix", .{});
    m.head = try laneOf(&r, "head", .{});
    m.embed = .{ try r.load("language_model.model.embed_tokens.weight"), try r.load("language_model.model.embed_tokens.scales"), try r.load("language_model.model.embed_tokens.biases") };
    m.ple = .{
        .kv = try laneOf(&r, "ple.kv", .{}),
        .ks = try r.load("ple.ks"),
        .qs = try r.load("ple.qs"),
        .cs = try r.load("ple.cs"),
        .conv = try r.load("ple.conv"),
        .starts = try r.load("ple.starts"),
        .tables = undefined,
        .cin = .{ .b = try r.buffer((PLE_TAIL + 1) * WIDE * 2) },
        .hist = undefined,
        .eos = jsonInt(ple_ref.get("eos").?),
        .mult = undefined,
        .sizes = undefined,
        .offsets = undefined,
    };
    m.ple.hist = .{ m.ple.eos, m.ple.eos };
    for (0..3) |k| m.ple.mult[k] = jsonInt(ple_ref.get("multipliers").?.array.items[k]);
    for (0..16) |k| {
        m.ple.sizes[k] = jsonInt(ple_ref.get("sizes").?.array.items[k]);
        m.ple.offsets[k] = jsonInt(ple_ref.get("offsets").?.array.items[k]);
    }
    for (0..GROUPS) |g| {
        m.ple.tables[3 * g + 0] = try r.group(16 * g, 16, "weight");
        m.ple.tables[3 * g + 1] = try r.group(16 * g, 16, "scales");
        m.ple.tables[3 * g + 2] = try r.group(16 * g, 16, "biases");
    }
    const big = struct {
        fn of(rr: *Run, n: usize) !Buf {
            return .{ .b = try rr.buffer(n) };
        }
    };
    m.t = .{
        .h = .{ try big.of(&r, WIDE * 2), try big.of(&r, WIDE * 2) },
        .ssp = try big.of(&r, 10 * 4 * 4),
        .part = try big.of(&r, 10 * 324 * 4),
        .mixed = try big.of(&r, D * 2),
        .inj_a = try big.of(&r, 8 * 2),
        .inj_m = try big.of(&r, 8 * 2),
        .xs = try big.of(&r, 192 * 16 * 4),
        .p = try big.of(&r, 16480 * 2),
        .gout = try big.of(&r, 6144 * 2),
        .branch = try big.of(&r, D * 2),
        .lg = try big.of(&r, 513 * 4),
        .act = try big.of(&r, 11 * 640 * 2),
        .pick = try big.of(&r, 10 * 4),
        .wts = try big.of(&r, 10 * 4),
        .ydown = try big.of(&r, 11 * D * 2),
        .q = try big.of(&r, 24 * 256 * 2),
        .kout = try big.of(&r, 2 * 256 * 2),
        .iq = try big.of(&r, 4 * 128 * 2),
        .po = try big.of(&r, 24 * 16 * 256 * 4),
        .pm = try big.of(&r, 24 * 16 * 2 * 4),
        .aout = try big.of(&r, 6144 * 2),
        .emb = try big.of(&r, D * 2),
        .kvp = try big.of(&r, (WIDE + D) * 2),
        .gated = try big.of(&r, WIDE * 2),
        .hout = try big.of(&r, WIDE * 2),
        .logits = try big.of(&r, VOCAB * 2),
        .pick_out = try big.of(&r, 16),
        .rows1 = try i32Buf(&r, &.{1}),
        .mdims = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
        .eps = try r.load("eps"),
        .ids8 = try i32Buf(&r, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }),
        .pos8 = try i32Buf(&r, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }),
        .nk8 = try i32Buf(&r, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }),
        .zero8 = try i32Buf(&r, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }),
        .ids81 = try i32Buf(&r, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }),
        .scale = try f32Buf(&r, @floatCast(ref.object.get("attention_scale").?.float)),
        .log2base = try f32Buf(&r, 23.253496170043945),
        .ple_ids = try i32Buf(&r, &(@as([16]i32, @splat(0)))),
        .kvmeta = try i32Buf(&r, &.{ 0, CAP }),
        .vocab = try i32Buf(&r, &.{VOCAB}),
    };
    try r.shapes.put(arena, "Kc_shape", (try i32Buf(&r, &.{ 1, 2, CAP, 256 })).b);
    try r.shapes.put(arena, "IDS_shape", (try i32Buf(&r, &.{ 8, 1 })).b);
    const t2 = mtl.clock.seconds();
    std.debug.print("compiled in {d:.2} s, loaded {d:.1} GB in {d:.1} s\n", .{ t1 - t0, @as(f64, @floatFromInt(r.loaded)) / 1e9, t2 - t1 });

    const prompt = ref.object.get("prompt").?.array.items;
    const want = ref.object.get("tokens").?.array.items;
    var next: u32 = 0;
    for (prompt) |tok| next = try m.step(@intCast(tok.integer));
    var got: std.ArrayList(u32) = .empty;
    try got.append(gpa, next);
    const t3 = mtl.clock.seconds();
    while (got.items.len < want.len) try got.append(gpa, try m.step(got.items[got.items.len - 1]));
    const t4 = mtl.clock.seconds();
    var same: usize = 0;
    while (same < want.len and got.items[same] == @as(u32, @intCast(want[same].integer))) same += 1;
    const steps: f64 = @floatFromInt(want.len - 1);
    std.debug.print("tokens equal to Python's: {d}/{d}; decode {d:.1} tok/s ({d:.2} ms a step)\n", .{ same, want.len, steps / (t4 - t3), (t4 - t3) * 1e3 / steps });
    if (same != want.len) {
        std.debug.print("first difference at {d}: got", .{same});
        for (got.items[same..@min(same + 8, got.items.len)]) |x| std.debug.print(" {d}", .{x});
        std.debug.print(", want", .{});
        for (want[same..@min(same + 8, want.len)]) |x| std.debug.print(" {d}", .{x.integer});
        std.debug.print("\n", .{});
        std.process.exit(1);
    }
}

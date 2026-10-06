//! FZ_DBENCH: Flash Next's dense projection classes and DeltaNet window step at 1-16 rows, one Mac's launches and
//! speed-up mode's (rank 0): the recorded kernels against fz_lane and fz_gdn, bit for bit on every layer and width,
//! then timed (each class's layers in one command buffer); GB/s from the weights' bytes, beside a streaming read.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;
const split = fz.split;
const dense = fz.dense;
const gdn = fz.gdn_step;
const Buf = fz.Buf;
const Run = fz.Run;
const Model = fz.Model;
const Lane = fz.Lane;

const stream_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void fz_stream(const device uint4* a [[buffer(0)]], const device uint4* b [[buffer(1)]],
    \\    device uint* sink [[buffer(2)]], constant uint2& n [[buffer(3)]],
    \\    uint i [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    \\  uint4 acc = uint4(0);
    \\  for (uint k = i; k < n.x; k += gs) acc ^= a[k];
    \\  for (uint k = i; k < n.y; k += gs) acc ^= b[k];
    \\  if (acc.x == 0x9e3779b9u && acc.y == acc.z) sink[0] = acc.w;
    \\}
;

/// A dense pass: the recorded kernel or fz_lane, one Mac's layout or speed-up mode's (rank 0), or a streaming read.
const Mode = enum { one, tp, stream, new, tpnew };

/// One projection class: its role, input width, input and output buffers, and every layer's weights.
const Pass = struct { name: []const u8, role: []const u8, k: usize, x: Buf, y: Buf, lanes: []const Lane };

const tp_in = [_][2]usize{ .{ 0, 32 }, .{ 64, 32 }, .{ 128, 96 }, .{ 320, 96 }, .{ 512, 3 } };

const Bench = struct {
    r: *Run,
    m: *Model,
    stream: mtl.Pipeline,
    sink: Buf,
    part: Buf,
    gdn_new: mtl.Pipeline,
    gdn_kept: mtl.Pipeline,
    rec: [2]Buf,
    ar: Buf,

    fn setRows(b: *Bench, rows: usize) void {
        b.r.rows = rows;
        b.m.t.rows.b.slice(i32, 1)[0] = @intCast(rows);
        b.m.t.mdims.b.slice(i32, 2)[0] = @intCast(rows);
    }

    /// Bytes a pass reads (speed-up mode: rank 0's tiles or K half).
    fn bytes(p: Pass, mode: Mode) f64 {
        var n: usize = 0;
        for (p.lanes) |l| n += (l.wq.b.length() - l.wq.off) + (l.sbt.b.length() - l.sbt.off);
        const all: f64 = @floatFromInt(n);
        if (mode != .tp and mode != .tpnew) return all;
        if (std.mem.endsWith(u8, p.role, "gdn.in")) return all * 259.0 / 515.0;
        if (std.mem.endsWith(u8, p.role, "gdn.out") or std.mem.endsWith(u8, p.role, "head")) return all / 2;
        return all;
    }

    /// Where a pass writes (the speed-up out-projection's fp32 partial, or the class's output), and its bytes at `rows`.
    fn out(b: *Bench, p: Pass, mode: Mode, rows: usize) struct { buf: Buf, len: usize } {
        const n = (p.lanes[0].wq.b.length() - p.lanes[0].wq.off) * 4 / (3 * p.k);
        if ((mode == .tp or mode == .tpnew) and std.mem.endsWith(u8, p.role, "gdn.out")) return .{ .buf = b.part, .len = rows * fz.D * 4 };
        return .{ .buf = p.y, .len = rows * n * 2 };
    }

    fn layout(b: *Bench, p: Pass, l: Lane, tp: bool) !dense.Layout {
        const n = (l.wq.b.length() - l.wq.off) * 4 / (3 * p.k);
        const sk = (try dense.recorded(b.r, p.role)).sk;
        const pf = b.r.lane_pf;
        if (!tp) return .{ .n = n, .k = p.k, .sk = sk, .pf = pf };
        if (std.mem.endsWith(u8, p.role, "gdn.in")) return .{ .n = n, .k = p.k, .sk = 8, .ranges = &tp_in, .pf = pf };
        if (std.mem.endsWith(u8, p.role, "gdn.out")) return .{ .n = n, .k = p.k, .sk = 8, .ranges = &.{.{ 0, 80 }}, .groups = .{ 0, 96 }, .pf = pf };
        if (std.mem.endsWith(u8, p.role, "head")) return .{ .n = n, .k = p.k, .sk = 1, .ranges = &.{.{ 0, fz.VOCAB / 64 }}, .pf = pf };
        return .{ .n = n, .k = p.k, .sk = sk, .pf = pf };
    }

    fn encodeLane(b: *Bench, p: Pass, l: Lane, mode: Mode) !void {
        const r = b.r;
        const t = &b.m.t;
        switch (mode) {
            .one => try b.m.lane(p.x, p.k, l, p.role, p.y),
            .new, .tpnew => try dense.project(r, try b.layout(p, l, mode == .tpnew), p.x, l, t.mdims, b.out(p, mode, 1).buf),
            .stream => {
                r.enc.setPipeline(b.stream);
                r.enc.setBuffer(l.wq.b, l.wq.off, 0);
                r.enc.setBuffer(l.sbt.b, l.sbt.off, 1);
                r.enc.setBuffer(b.sink.b, 0, 2);
                const n = [2]u32{ @intCast((l.wq.b.length() - l.wq.off) / 16), @intCast((l.sbt.b.length() - l.sbt.off) / 16) };
                r.enc.setBytes(std.mem.asBytes(&n), 3);
                r.enc.dispatchThreads(mtl.Size.of(160 * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
            },
            .tp => {
                const ins = [_]Buf{ p.x, t.xs, l.wq, l.sbt, t.mdims };
                if (std.mem.endsWith(u8, p.role, "gdn.in")) {
                    try split.laneTiles(r, p.role, &ins, p.y, &tp_in, 8, null);
                } else if (std.mem.endsWith(u8, p.role, "gdn.out")) {
                    try split.laneTiles(r, p.role, &ins, b.part, &.{.{ 0, 80 }}, 8, .{ 0, 96 });
                } else if (std.mem.endsWith(u8, p.role, "head")) {
                    try split.laneTiles(r, p.role, &ins, p.y, &.{.{ 0, fz.VOCAB / 64 }}, 1, null);
                } else try b.m.lane(p.x, p.k, l, p.role, p.y);
            },
        }
    }

    fn encode(b: *Bench, p: Pass, mode: Mode) !void {
        for (p.lanes) |l| try b.encodeLane(p, l, mode);
    }

    /// The DeltaNet step over every linear layer: the recorded kernel (0), fz_gdn (1), fz_gdn storing only the last
    /// row's state (2); speed-up mode's rank 0 heads when `tp`.
    fn encodeGdn(b: *Bench, rows: usize, tp: bool, which: usize) !void {
        for (&b.m.layers) |*L| if (L.linear) try b.gdnLayer(L, rows, tp, which);
    }

    fn gdnLayer(b: *Bench, L: *fz.Layer, rows: usize, tp: bool, which: usize) !void {
        const t = &b.m.t;
        const ins = [_]Buf{ t.p, L.cs[0], L.so[0], L.conv, L.alog, L.dt, L.norm, t.eps, t.rows };
        const outs = [_]Buf{ t.gout, L.cs[1], L.so[1] };
        const heads: usize = if (tp) 24 else 48;
        switch (which) {
            0 => if (tp) try split.gdnHeads(b.r, "q4_gdn@gdn", rows, &ins, &outs, 0) else try b.r.callAs("q4_gdn@gdn", rows, &ins, &outs),
            1 => gdn.step(b.r, b.gdn_new, &ins, &outs, 0, heads),
            else => { // the kept state: replays ar[0] rows (left 0 here) and stores one state, not one a row
                const kins = [_]Buf{ t.p, L.cs[0], L.so[1], L.conv, L.alog, L.dt, L.norm, t.eps, t.rows };
                gdn.stepKept(b.r, b.gdn_kept, &kins, &.{ t.gout, L.cs[1] }, b.rec[0], b.rec[1], b.ar, 0, heads);
            },
        }
    }

    fn encodeVariant(b: *Bench, p: Pass, tp: bool, pipe: mtl.Pipeline) !void {
        const r = b.r;
        for (p.lanes) |l| {
            const lay = try b.layout(p, l, tp);
            r.enc.setPipeline(pipe);
            for ([_]Buf{ p.x, l.wq, l.sbt, b.m.t.mdims, b.out(p, if (tp) .tpnew else .new, 1).buf }, 0..) |bf, j| r.enc.setBuffer(bf.b, bf.off, j);
            r.enc.dispatchThreads(mtl.Size.of(dense.tiles(lay) * 32 * lay.sk, 1, 1), mtl.Size.of(32 * lay.sk, 1, 1));
        }
    }

    fn encodeExperts(b: *Bench, lgp: Buf) !void {
        const t = &b.m.t;
        for (&b.m.layers, 0..) |*L, i| try b.r.experts("qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", t.mixed, lgp.at(i * fz.MAXR * 513 * 4), L.ex, t.act, t.pick, t.wts, t.rows, t.ydown);
    }

    fn once(b: *Bench, comptime f: anytype, args: anytype) !f64 {
        const cb = b.r.queue.commandBuffer();
        b.r.enc = cb.compute(.serial);
        try @call(.auto, f, args);
        b.r.enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("bench: {s}", .{msg});
            return error.GpuFailed;
        }
        return cb.gpuSeconds();
    }

    /// GPU seconds of one `f(args)`: the median of five command buffers that each run it twelve times on a serial
    /// encoder (a lone pass of a few short kernels runs before the GPU's clocks rise).
    fn timed(b: *Bench, comptime f: anytype, args: anytype) !f64 {
        var s: [5]f64 = undefined;
        for (&s) |*v| {
            const cb = b.r.queue.commandBuffer();
            b.r.enc = cb.compute(.serial);
            for (0..12) |_| try @call(.auto, f, args);
            b.r.enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("bench: {s}", .{msg});
                return error.GpuFailed;
            }
            v.* = cb.gpuSeconds() / 12;
        }
        std.mem.sort(f64, &s, {}, std.sort.asc(f64));
        return s[2];
    }
};

fn poison(buf: Buf, len: usize) void {
    @memset(buf.b.contents()[buf.off .. buf.off + len], 0xA5);
}

fn differ(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(x != y);
    return n;
}

pub fn run(r: *Run, m: *Model, prompt: []const u32, gpa: std.mem.Allocator, arena: std.mem.Allocator) !void {
    const lib = try mtl.Library.fromSource(r.device, stream_source, mtl.CompileOptions.mlx());
    var b: Bench = .{
        .r = r, .m = m, .stream = try mtl.Pipeline.init(r.device, lib, "fz_stream", false), .sink = .{ .b = try r.buffer(64) },
        .part = .{ .b = try r.buffer(fz.MAXR * fz.D * 4) }, .gdn_new = try gdn.compile(r, false), .gdn_kept = try gdn.compile(r, true),
        .rec = .{ .{ .b = try r.buffer(gdn.RECORD) }, .{ .b = try r.buffer(gdn.RECORD) } }, .ar = .{ .b = try r.buffer(64) },
    };
    var pk: [fz.MAXR]u32 = undefined;
    m.reset();
    for (0..3) |_| try m.window(prompt[0..fz.MAXR], &pk); // real values in every window buffer
    for (&m.layers) |*L| if (L.linear) { // a real state and conv window to start from: the window's last row's
        const so = L.so[1].b.contents()[L.so[1].off + (fz.MAXR - 1) * fz.SO_ROW ..][0..fz.SO_ROW];
        @memcpy(L.so[0].b.contents()[L.so[0].off..][0..fz.SO_ROW], so);
        const cs = L.cs[1].b.contents()[L.cs[1].off + (fz.MAXR - 1) * fz.CS_ROW ..][0..fz.CS_ROW];
        @memcpy(L.cs[0].b.contents()[L.cs[0].off..][0..fz.CS_ROW], cs);
    };
    var gin: std.ArrayList(Lane) = .empty;
    var gout: std.ArrayList(Lane) = .empty;
    var ain: std.ArrayList(Lane) = .empty;
    var aout: std.ArrayList(Lane) = .empty;
    for (&m.layers) |*L| {
        try (if (L.linear) &gin else &ain).append(arena, L.proj);
        try (if (L.linear) &gout else &aout).append(arena, L.out);
    }
    const t = &m.t;
    const passes = [_]Pass{
        .{ .name = "gdn.in", .role = "lane_qmm_bytes_grouped@gdn.in", .k = fz.D, .x = t.mixed, .y = t.p, .lanes = gin.items },
        .{ .name = "gdn.out", .role = "lane_qmm_bytes_grouped@gdn.out", .k = 6144, .x = t.gout, .y = t.branch, .lanes = gout.items },
        .{ .name = "att.proj", .role = "lane_qmm_bytes_grouped@att.proj", .k = fz.D, .x = t.mixed, .y = t.p, .lanes = ain.items },
        .{ .name = "att.o", .role = "lane_qmm_bytes_grouped@att.o", .k = 6144, .x = t.aout, .y = t.branch, .lanes = aout.items },
        .{ .name = "ple.kv", .role = "lane_qmm_bytes_grouped@ple.kv", .k = fz.D, .x = t.emb, .y = t.kvp, .lanes = &.{m.ple.kv} },
        .{ .name = "head", .role = "lane_qmm_bytes_grouped@head", .k = fz.D, .x = t.mixed, .y = t.logits, .lanes = &.{m.head} },
    };
    const pfs = [_]usize{ 1, 2 };

    const bits = std.mem.eql(u8, std.mem.span(std.c.getenv("FZ_DBENCH").?), "bits");
    // bits: fz_lane against the recorded kernel, every layer, one Mac and speed-up layouts, every read-ahead depth
    const keep = try gpa.alloc(u8, fz.MAXR * fz.VOCAB * 2);
    defer gpa.free(keep);
    if (bits) for ([_]usize{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
        b.setRows(rows);
        for (passes) |p| for ([_]bool{ false, true }) |tp| {
            var bad: [pfs.len]usize = @splat(0);
            for (p.lanes) |l| {
                const o = b.out(p, if (tp) .tp else .one, rows);
                poison(o.buf, o.len);
                _ = try b.once(Bench.encodeLane, .{ &b, p, l, if (tp) Mode.tp else Mode.one });
                @memcpy(keep[0..o.len], o.buf.b.contents()[o.buf.off .. o.buf.off + o.len]);
                for (pfs, 0..) |pf, i| {
                    r.lane_pf = pf;
                    poison(o.buf, o.len);
                    _ = try b.once(Bench.encodeLane, .{ &b, p, l, if (tp) Mode.tpnew else Mode.new });
                    bad[i] += differ(keep[0..o.len], o.buf.b.contents()[o.buf.off .. o.buf.off + o.len]);
                }
            }
            std.debug.print("bits rows {d:2} {s:8} {s}: bytes differing at read-ahead 1/2: {any}\n", .{ rows, p.name, if (tp) "speed-up" else "one Mac ", bad });
        };
    };

    // bits: fz_gdn against the recorded step: outputs, conv windows and states, every layer and width
    const so_len = fz.MAXR * fz.SO_ROW;
    const cs_len = fz.MAXR * fz.CS_ROW;
    const ref = try gpa.alloc(u8, so_len + cs_len + fz.MAXR * 6144 * 2);
    defer gpa.free(ref);
    if (bits) for (1..fz.MAXR + 1) |rows| {
        b.setRows(rows);
        for ([_]bool{ false, true }) |tp| {
            var bad: [3]usize = @splat(0);
            for (&m.layers) |*L| if (L.linear) {
                const bufs = [_]Buf{ L.so[1], L.cs[1], t.gout };
                const lens = [_]usize{ rows * fz.SO_ROW, rows * fz.CS_ROW, rows * 6144 * 2 };
                const offs = [_]usize{ 0, so_len, so_len + cs_len };
                _ = try b.once(Bench.encodeLane, .{ &b, passes[0], L.proj, Mode.one }); // this layer's own inputs
                for (bufs, lens) |bf, n| poison(bf, n);
                _ = try b.once(Bench.gdnLayer, .{ &b, L, rows, tp, 0 });
                for (bufs, lens, offs) |bf, n, o| @memcpy(ref[o .. o + n], bf.b.contents()[bf.off .. bf.off + n]);
                for (bufs, lens) |bf, n| poison(bf, n);
                _ = try b.once(Bench.gdnLayer, .{ &b, L, rows, tp, 1 });
                for (bufs, lens, offs, 0..) |bf, n, o, i| bad[i] += differ(ref[o .. o + n], bf.b.contents()[bf.off .. bf.off + n]);
            };
            std.debug.print("bits rows {d:2} DeltaNet {s}: bytes differing in states / conv windows / outputs: {any}\n", .{ rows, if (tp) "speed-up" else "one Mac ", bad });
        }
    };
    if (bits) return;

    // time
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
        b.setRows(rows);
        var sum: [5]f64 = @splat(0);
        var sum_b: [5]f64 = @splat(0);
        for (passes) |p| {
            var us: [5]f64 = undefined;
            for ([_]Mode{ .one, .tp, .stream, .new, .tpnew }, 0..) |mode, i| {
                r.lane_pf = 1;
                const s = try b.timed(Bench.encode, .{ &b, p, mode });
                us[i] = s * 1e6 / @as(f64, @floatFromInt(p.lanes.len));
                if (!std.mem.eql(u8, p.name, "head")) {
                    sum[i] += s;
                    sum_b[i] += Bench.bytes(p, mode);
                }
            }
            const n: f64 = @floatFromInt(p.lanes.len);
            const gbs = struct {
                fn of(pp: Pass, mode: Mode, nn: f64, u: f64) f64 {
                    return Bench.bytes(pp, mode) / nn / u / 1e3;
                }
            };
            std.debug.print("rows {d:2} {s:8}: one Mac {d:6.1} us {d:4.0} GB/s, new {d:6.1} us {d:4.0} | speed-up {d:6.1} us {d:4.0}, new {d:6.1} us {d:4.0} | stream {d:6.1} us {d:4.0}\n", .{ rows, p.name, us[0], gbs.of(p, .one, n, us[0]), us[3], gbs.of(p, .new, n, us[3]), us[1], gbs.of(p, .tp, n, us[1]), us[4], gbs.of(p, .tpnew, n, us[4]), us[2], gbs.of(p, .stream, n, us[2]) });
        }
        std.debug.print("rows {d:2} dense (no head): one Mac {d:.3} ms, new {d:.3} | speed-up {d:.3} ms, new {d:.3} | stream {d:.3} ms ({d:.0} GB/s)\n", .{ rows, sum[0] * 1e3, sum[3] * 1e3, sum[1] * 1e3, sum[4] * 1e3, sum[2] * 1e3, sum_b[2] / sum[2] / 1e9 });
    }
    for ([_]usize{ 1, 2, 4 }) |rows| { // fz_lane's read-ahead depth
        b.setRows(rows);
        for (passes) |p| {
            var us: [pfs.len * 2]f64 = undefined;
            for (pfs, 0..) |pf, i| {
                r.lane_pf = pf;
                us[i] = try b.timed(Bench.encode, .{ &b, p, Mode.new }) * 1e6 / @as(f64, @floatFromInt(p.lanes.len));
                us[pfs.len + i] = try b.timed(Bench.encode, .{ &b, p, Mode.tpnew }) * 1e6 / @as(f64, @floatFromInt(p.lanes.len));
            }
            std.debug.print("rows {d:2} {s:8} read-ahead 1/2: one Mac {d:6.1} {d:6.1} us | speed-up {d:6.1} {d:6.1} us\n", .{ rows, p.name, us[0], us[1], us[2], us[3] });
        }
    }
    r.lane_pf = 1;
    const variants = [_]struct { name: []const u8, from: []const []const u8, to: []const []const u8 }{
        .{ .name = "read-ahead 1", .from = &.{}, .to = &.{} },
        .{ .name = "no x sums", .from = &.{ "fz_lane_xsum(X, fm, (g), M)", "fz_lane_xsum(X, fm + 8, (g), M)" }, .to = &.{ "0.0f", "0.0f" } },
        .{ .name = "no weights", .from = &.{"      stage[lane * (GS / 4) + c] = word;"}, .to = &.{""} },
    };
    for ([_]usize{ 1, 8 }) |rows| {
        b.setRows(rows);
        for (passes[0..2]) |p| for ([_]bool{ false, true }) |tp| for (variants) |v| {
            const l0 = try b.layout(p, p.lanes[0], tp);
            var text = try dense.source(arena, l0);
            for (v.from, v.to) |from, to| text = try std.mem.replaceOwned(u8, arena, text, from, to);
            const vlib = try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx());
            const pipe = try mtl.Pipeline.init(r.device, vlib, "fz_lane", false);
            const s = try b.timed(Bench.encodeVariant, .{ &b, p, tp, pipe });
            std.debug.print("rows {d:2} {s:8} {s}: {s:20} {d:6.1} us\n", .{ rows, p.name, if (tp) "speed-up" else "one Mac ", v.name, s * 1e6 / @as(f64, @floatFromInt(p.lanes.len)) });
        };
    }
    { // routed and shared experts by rows: chain rows' real routing, every layer's own logits
        r.ar = .{ .b = try r.buffer(64) };
        const lgp: Buf = .{ .b = try r.buffer(fz.LAYERS * fz.MAXR * 513 * 4) };
        const pkp: Buf = .{ .b = try r.buffer(fz.LAYERS * fz.MAXR * 10 * 4) };
        r.lg_probe = lgp;
        r.probe = pkp;
        try m.window(prompt[0..8], &pk);
        r.lg_probe = null;
        r.probe = null;
        const picks = pkp.b.slice(u32, fz.LAYERS * fz.MAXR * 10);
        const per_expert: f64 = 2 * 640 * 2560 * 0.75 + 2 * 640 * 80 * 4 + 2560 * 640 * 0.75 + 2560 * 20 * 4;
        for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            b.setRows(rows);
            var distinct: usize = 0;
            for (0..fz.LAYERS) |l| {
                var seen: [512]bool = @splat(false);
                for (picks[l * fz.MAXR * 10 ..][0 .. rows * 10]) |e| {
                    distinct += @intFromBool(!seen[e]);
                    seen[e] = true;
                }
            }
            const s = try b.timed(Bench.encodeExperts, .{ &b, lgp });
            const gb = (@as(f64, @floatFromInt(distinct)) + fz.LAYERS) * per_expert;
            std.debug.print("rows {d:2} experts x48: {d:.3} ms ({d:.0} GB/s); {d:.1} distinct routed a layer\n", .{ rows, s * 1e3, gb / s / 1e9, @as(f64, @floatFromInt(distinct)) / fz.LAYERS });
        }
    }
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 8, 12, 16 }) |rows| {
        b.setRows(rows);
        var ms: [6]f64 = undefined;
        for (0..3) |which| for ([_]bool{ false, true }, 0..) |tp, j| {
            ms[which * 2 + j] = try b.timed(Bench.encodeGdn, .{ &b, rows, tp, which }) * 1e3;
        };
        std.debug.print("rows {d:2} DeltaNet x36 ms: one Mac recorded {d:.3}, fz_gdn {d:.3}, kept state {d:.3} | speed-up recorded {d:.3}, fz_gdn {d:.3}, kept state {d:.3}\n", .{ rows, ms[0], ms[2], ms[4], ms[1], ms[3], ms[5] });
    }
}

const op_source =
    \\#include <metal_stdlib>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\// one trial a simdgroup: lane_qmm's 16 x 32 x 32 op on A (bf16 [16][32]) and codes B ([32 columns][32 bytes])
    \\[[kernel]] void fz_op_probe(const device bfloat* A [[buffer(0)]], const device uint* B [[buffer(1)]],
    \\    device float* P [[buffer(2)]], uint lanei [[thread_index_in_simdgroup]], uint tgx [[threadgroup_position_in_grid]]) {
    \\  const int lane = int(lanei);
    \\  const short qid = short(lane) >> 2;
    \\  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
    \\  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
    \\  constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroup> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)A + tgx * 512, dextents<int32_t, 2>(32, 16));
    \\  threadgroup uint stage[256];
    \\  for (int c = 0; c < 8; c++) stage[lane * 8 + c] = B[tgx * 256 + lane * 8 + c];
    \\  simdgroup_barrier(mem_flags::mem_threadgroup);
    \\  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(32, 32));
    \\  auto a = tA.slice(0, 0);
    \\  auto D = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    \\  op.run(a, b, D);
    \\  for (int f = 0; f < 2; f++) for (int r = 0; r < 2; r++) for (int j = 0; j < 4; j++)
    \\    P[tgx * 512 + (fm + 8 * r) * 32 + f * 16 + fn + j] = D[f * 8 + r * 4 + j];
    \\}
;

fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// FZ_OPORDER: the tensor op's sum over its 32 inputs against fp32 summation orders, on random bf16 x 6-bit codes.
pub fn opOrder(device: mtl.Device) !void {
    const trials: usize = 8192;
    const lib = try mtl.Library.fromSource(device, op_source, mtl.CompileOptions.mlx());
    const pipe = try mtl.Pipeline.init(device, lib, "fz_op_probe", false);
    const queue = try device.queue();
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const A = try device.buffer(trials * 512 * 2, opts);
    const B = try device.buffer(trials * 1024, opts);
    const P = try device.buffer(trials * 512 * 4, opts);
    var seed: u64 = 0x9e3779b97f4a7c15;
    const a16 = A.slice(u16, trials * 512);
    const b8 = B.contents()[0 .. trials * 1024];
    for (a16, 0..) |*v, i| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const x: u32 = @intCast(seed >> 32);
        const spread: u32 = if (i % 3 == 0) 24 else 6; // exponents near 2^0, or spread over 2^-12..2^12
        const e: u32 = 127 - spread / 2 + (x % (spread + 1));
        v.* = @intCast(((x >> 31) << 15) | (e << 7) | ((x >> 8) & 0x7f));
    }
    for (b8) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @intCast((seed >> 40) & 63);
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    enc.setBuffer(A, 0, 0);
    enc.setBuffer(B, 0, 1);
    enc.setBuffer(P, 0, 2);
    enc.dispatchThreads(mtl.Size.of(trials * 32, 1, 1), mtl.Size.of(32, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    const p = P.slice(f32, trials * 512);
    const names = [_][]const u8{ "fp32 k = 0..31", "fp32 k = 31..0", "pairwise tree", "exact 4s, fp32 in order", "exact 8s, fp32 in order", "exact 16s, fp32 in order", "exact, rounded once", "fma chain from k = 0" };
    var same: [names.len]usize = @splat(0);
    var worst: f64 = 0;
    for (0..trials) |t| for (0..16) |m| for (0..32) |n| {
        var prod: [32]f32 = undefined;
        for (0..32) |k| prod[k] = bf16(a16[t * 512 + m * 32 + k]) * @as(f32, @floatFromInt(b8[t * 1024 + n * 32 + k]));
        var c: [names.len]f32 = undefined;
        c[0] = 0;
        for (prod) |x| c[0] += x;
        c[1] = 0;
        var k: usize = 32;
        while (k > 0) : (k -= 1) c[1] += prod[k - 1];
        var lvl = prod;
        var len: usize = 32;
        while (len > 1) : (len /= 2) for (0..len / 2) |i| {
            lvl[i] = lvl[2 * i] + lvl[2 * i + 1];
        };
        c[2] = lvl[0];
        for ([_]usize{ 4, 8, 16 }, 3..) |blk, ci| {
            var s: f32 = 0;
            var at: usize = 0;
            while (at < 32) : (at += blk) {
                var e: f64 = 0;
                for (prod[at .. at + blk]) |x| e += x;
                s += @as(f32, @floatCast(e));
            }
            c[ci] = s;
        }
        var e: f64 = 0;
        for (prod) |x| e += x;
        c[6] = @floatCast(e);
        c[7] = 0;
        for (0..32) |kk| c[7] = @mulAdd(f32, bf16(a16[t * 512 + m * 32 + kk]), @as(f32, @floatFromInt(b8[t * 1024 + n * 32 + kk])), c[7]);
        const got = p[t * 512 + m * 32 + n];
        for (c, 0..) |v, ci| same[ci] += @intFromBool(@as(u32, @bitCast(v)) == @as(u32, @bitCast(got)));
        worst = @max(worst, @abs(@as(f64, got) - e) / @max(1e-30, @abs(e)));
    };
    const total = trials * 512;
    for (names, same) |nm, s| std.debug.print("op sum vs {s:26}: {d} of {d} equal\n", .{ nm, s, total });
    std.debug.print("largest relative difference from the exact sum: {e:.3}\n", .{worst});
}

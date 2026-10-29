//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const segments = @import("../../core/segments.zig");
const tpm = @import("tp.zig");
const variants = fz.variants;
const Tp2 = fz.Tp2;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const PLE_TAIL = fz.PLE_TAIL;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const MARKS = fz.MARKS;
const Marks = fz.Marks;
const Model = fz.Model;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const barrier = Prompt.barrier;
const bind = Prompt.bind;
const qmm = Prompt.qmm;
const gather = Prompt.gather;
const hc = Prompt.hc;
const Seg = Prompt.Seg;
const bindMarks = Prompt.bindMarks;
const prep = Prompt.prep;
const Pair = Prompt.Pair;
const bytesOf = Prompt.bytesOf;
const Hooks = Prompt.Hooks;

pub fn classes(p: *Prompt, m: *Model, rows: usize) !void {
    if (p.r.fsr != null) return error.FsrProfileNeedsActualRoutes;
    const r = p.r;
    const b = &p.b;
    const L0 = &m.layers[0];
    const L3 = &m.layers[3];
    const pairs = rows * 10;
    const names = [_][]const u8{ "router", "top-k+offsets+sort", "row gather+scatter", "expert gate+up gathers", "expert act", "expert down gather", "shared expert", "hyper-connection", "DeltaNet in+out projections", "DeltaNet pre", "DeltaNet scan", "DeltaNet post", "attention proj+o", "attention (tensor units)", "hc norms (3)" };
    for (names, 0..) |name, which| {
        const cb = r.queue.commandBuffer();
        r.begin(cb, if (r.serial) .serial else .concurrent);
        errdefer if (r.fsr != null) r.abort();
        for (0..48) |_| {
            switch (which) {
                0 => {
                    p.bind(p.router_mm, &.{ L0.router, b.mixed });
                    const rp = [3]i32{ D, 513, @intCast(rows) };
                    r.enc.setBytes(std.mem.asBytes(&rp), 2);
                    r.enc.setBuffer(b.lg.b, b.lg.off, 3);
                    r.enc.dispatchThreads(mtl.Size.of(((513 + 63) / 64) * 128, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                    p.barrier();
                },
                1 => {
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
                },
                2 => {
                    p.bind(p.pl[7], &.{ b.mixed, b.row_of, b.xs });
                    r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
                    p.barrier();
                    p.bind(p.pl[9], &.{ b.ds, b.row_of, b.ydown });
                    r.enc.dispatchThreads(mtl.Size.of(D / 8, pairs, 1), mtl.Size.of(64, 1, 1));
                    p.barrier();
                },
                3 => {
                    p.gather(b.xs, L0.ex[0..3], D, 640, pairs, b.g);
                    p.gather(b.xs, L0.ex[3..6], D, 640, pairs, b.u);
                },
                4 => {
                    p.bind(p.pl[8], &.{ b.g, b.u, b.a });
                    r.enc.dispatchThreads(mtl.Size.of(pairs * 640, 1, 1), mtl.Size.of(256, 1, 1));
                    p.barrier();
                },
                5 => p.gather(b.a, L0.ex[12..15], 640, D, pairs, b.ds),
                6 => {
                    p.qmm(b.mixed, .{ L0.ex[6], L0.ex[7], L0.ex[8] }, D, 640, rows, b.sg, 0);
                    p.qmm(b.mixed, .{ L0.ex[9], L0.ex[10], L0.ex[11] }, D, 640, rows, b.su, 0);
                    p.bind(p.pl[8], &.{ b.sg, b.su, b.sa });
                    r.enc.dispatchThreads(mtl.Size.of(rows * 640, 1, 1), mtl.Size.of(256, 1, 1));
                    p.barrier();
                    p.qmm(b.sa, .{ L0.ex[15], L0.ex[16], L0.ex[17] }, 640, D, rows, .{ .b = b.ydown.b, .off = 10 * D * 2 }, 11 * D);
                },
                7 => p.hc(m, b.h[0], L0.ahc, rows, b.inj_a),
                8 => {
                    p.qmm(b.mixed, p.proj[0], D, 16480, rows, b.p, 0);
                    p.qmm(b.gout, p.out[0], 6144, D, rows, b.branch, 0);
                },
                9, 10, 11 => {
                    const ri: i32 = @intCast(rows);
                    if (which == 9) {
                        p.bind(p.pl[11], &.{ b.p, L0.cs[0], L0.conv, L0.alog, L0.dt });
                        r.enc.setBytes(std.mem.asBytes(&ri), 5);
                        for ([_]Buf{ b.qn, b.kn, b.v, b.gg, b.beta, L0.cs[1] }, 6..) |bb, j| r.enc.setBuffer(bb.b, bb.off, j);
                        p.bindMarks(m, @splat(0), 0, 12, 13, false);
                        r.enc.dispatchThreads(mtl.Size.of(80 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                    } else if (which == 10) {
                        p.bind(p.pl[if (p.scan4) 14 else 12], &.{ b.qn, b.kn, b.v, b.gg, b.beta, L0.so[0] });
                        r.enc.setBytes(std.mem.asBytes(&ri), 6);
                        r.enc.setBuffer(b.ys.b, b.ys.off, 7);
                        r.enc.setBuffer(L0.so[1].b, L0.so[1].off, 8);
                        p.bindMarks(m, @splat(0), 0, 9, 10, true);
                        if (p.scan4) r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 256, 1, 1), mtl.Size.of(256, 1, 1)) else r.enc.dispatchThreads(mtl.Size.of(48 * 4 * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
                    } else {
                        p.bind(p.pl[13], &.{ b.ys, b.p, L0.norm, m.t.eps, b.gout });
                        r.enc.dispatchThreads(mtl.Size.of(48 * 128, rows, 1), mtl.Size.of(128, 1, 1));
                    }
                    p.barrier();
                },
                12 => {
                    p.qmm(b.mixed, p.proj[3], D, 13952, rows, b.p, 0);
                    p.qmm(b.aout, p.out[3], 6144, D, rows, b.branch, 0);
                },
                13 => {
                    p.bind(p.attn256, &.{ b.q, L3.keys, L3.vals, b.p });
                    const ap = [4]i32{ @intCast(rows), @intCast(rows), 0, @intCast(m.cap) };
                    r.enc.setBytes(std.mem.asBytes(&ap), 4);
                    r.enc.setBuffer(m.t.scale.b, m.t.scale.off, 5);
                    r.enc.setBuffer(b.aout.b, b.aout.off, 6);
                    r.enc.dispatchThreads(mtl.Size.of(((rows + 63) / 64) * 128, 24, 1), mtl.Size.of(128, 1, 1));
                    p.barrier();
                },
                14 => {
                    try r.callRows("q4_hc_norm_none#[10240]", rows, &.{b.h[0]}, &.{ b.h[1], b.ssp }, null);
                    try r.callRows("q4_hc_norm_plain#[10240]", rows, &.{ b.h[1], b.inj_a, b.branch }, &.{ b.h[0], b.ssp }, null);
                    try r.callRows("q4_hc_norm_grouped#[10240]", rows, &.{ b.h[0], b.inj_m, b.ydown, b.wts, b.lg }, &.{ b.h[1], b.ssp }, null);
                },
                else => {},
            }
        }
        r.enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("  {s:28} {d:7.2} ms a chunk (48 layers' worth)\n", .{ name, cb.gpuSeconds() * 1e3 });
    }
}

pub fn copyWords(p: *Prompt, src: Buf, dst: Buf, words: usize) void {
    p.bind(p.pl[10], &.{ src, dst });
    p.r.enc.dispatchThreads(mtl.Size.of(words, 1, 1), mtl.Size.of(256, 1, 1));
    p.barrier();
}

pub fn chunk(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32) !u32 {
    return chunkN(&.{p}, m, gpa, tokens);
}

pub fn chunkN(ps: []const *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32) !u32 {
    return chunkMarked(ps, m, gpa, tokens, &.{});
}

pub fn chunkMarked(ps: []const *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, marks: []const u32) !u32 {
    const N = ps.len;
    if (marks.len > MARKS or (marks.len > 0 and m.marks == null)) return error.Marks;
    if (N == 0 or N > segments.MAX) return error.Segments;
    const r = ps[0].r;
    const n = tokens.len;
    if (n < N or n > N * PMAX) return error.ChunkSize;
    const a = m.state;
    var queues: [segments.MAX]mtl.Queue = undefined;
    var segs: [segments.MAX]Seg = undefined;
    for (0..N) |k| {
        const at = segments.start(n, N, k);
        const rows = segments.rows(n, N, k);
        var hist = m.ple.hist; // the two tokens before the segment
        for (tokens[at - @min(at, 2) .. at]) |tok| hist = .{ hist[1], tok };
        try ps[k].prep(m, gpa, tokens[at .. at + rows], m.pos + at, hist);
        queues[k] = if (k == 0) r.queue else ps[k].seg_queue orelse blk: {
            const q = try r.ownResource(try r.device.queue());
            ps[k].seg_queue = q;
            break :blk q;
        };
        const ra = if (k % 2 == 0) a else 1 - a; // DeltaNet slots alternate as serial chunks would
        segs[k] = .{ .p = ps[k], .rows = rows, .pos = m.pos + at, .ra = ra, .rr = if (k == 0) m.state_row else 0, .wa = 1 - ra, .cur = 0, .pending = false };
        for (marks, 0..) |mk, j| if (mk > at and mk <= at + rows) {
            segs[k].mk[j] = @intCast(mk - at);
        };
    }
    const cin_old = m.ple.cin.b.contents()[m.ple.cin.off..];
    @memcpy(ps[0].b.cin.b.contents()[0 .. PLE_TAIL * WIDE * 2], cin_old[0 .. PLE_TAIL * WIDE * 2]);
    var hooks: Hooks = .{ .m = m, .segs = segs[0..N] };
    if (r.fsr != null) try @import("fsr_prompt.zig").run(m, segs[0..N]) else m.gpu_seconds += try segments.run(r.device, queues[0..N], LAYERS, if (r.serial) .serial else .concurrent, &hooks);
    if (ps[0].wide) |wd| try wd.check(); // a weight the load left out of the width table fails the pass
    const sl = &segs[N - 1];
    m.state = if (N % 2 == 0) a else 1 - a; // each segment flipped it once
    m.state_row = 0;
    m.pos += n;
    @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], sl.p.b.cin.b.contents()[sl.rows * WIDE * 2 .. (sl.rows + PLE_TAIL) * WIDE * 2]);
    for (tokens) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
    return m.t.picks.b.slice(u32, 1)[0];
}

pub fn chunkPair(p: *Prompt, m: *Model, gpa: std.mem.Allocator, tokens: []const u32, tp: *Tp2, marks: []const u32) !u32 {
    const r = p.r;
    const n = tokens.len;
    if (n < 2 or n > 2 * PMAX) return error.ChunkSize;
    if (marks.len > MARKS or (marks.len > 0 and m.marks == null)) return error.Marks;
    const k: usize = tp.rank;
    const a = m.state;
    const at = segments.start(n, 2, k);
    const rows = segments.rows(n, 2, k);
    const rows0 = segments.rows(n, 2, 0);
    tp.call +%= 1;
    const pr: Pair = .{ .tp = tp, .call = tp.call, .k = k, .pos0 = m.pos, .rows0 = rows0, .a = a };
    var hist = m.ple.hist;
    for (tokens[at - @min(at, 2) .. at]) |tok| hist = .{ hist[1], tok };
    try p.prep(m, gpa, tokens[at .. at + rows], m.pos + at, hist);
    const ra = if (k == 0) a else 1 - a;
    var segs = [1]Seg{.{ .p = p, .rows = rows, .pos = m.pos + at, .ra = ra, .rr = if (k == 0) m.state_row else 0, .wa = 1 - ra, .cur = 0, .pending = false }};
    for (marks, 0..) |mk, j| if (mk > at and mk <= at + rows) {
        segs[0].mk[j] = @intCast(mk - at);
    };
    const cin_old = m.ple.cin.b.contents()[m.ple.cin.off..];
    if (k == 0) @memcpy(p.b.cin.b.contents()[0 .. PLE_TAIL * WIDE * 2], cin_old[0 .. PLE_TAIL * WIDE * 2]);
    var hooks: Hooks = .{ .m = m, .segs = &segs, .pair = &pr };
    if (tp.trace) std.debug.print("TP rank{d} chunk {d}: rows {d} at {d} (pos {d}), state slot {d}\n", .{ k, pr.call, rows, at, m.pos + at, a });
    m.gpu_seconds += try segments.run(r.device, &.{r.queue}, LAYERS, if (r.serial) .serial else .concurrent, &hooks);
    if (tp.trace) std.debug.print("TP rank{d} chunk {d}: GPU done, gave_up {d}\n", .{ k, pr.call, tp.gaveUp() });
    const t = &m.t;
    var pick: u32 = undefined;
    if (k == 1) { // the chunk's end: hand it back
        var ws: std.ArrayList(tpm.Write) = .empty;
        defer ws.deinit(gpa);
        const pos1 = m.pos + at;
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            const slot = tpm.layerSlot(i);
            if (L.linear) {
                try ws.append(gpa, .{ .src = bytesOf(L.cs[a]), .len = CS_ROW, .dst = slot });
                try ws.append(gpa, .{ .src = bytesOf(L.so[a]), .len = SO_ROW, .dst = slot + CS_ROW });
            } else {
                for (0..2) |hd| {
                    try ws.append(gpa, .{ .src = bytesOf(L.keys) + (hd * m.cap + pos1) * 512, .len = rows * 512, .dst = slot + hd * tpm.KV_HEAD });
                    try ws.append(gpa, .{ .src = bytesOf(L.vals) + (hd * m.cap + pos1) * 512, .len = rows * 512, .dst = slot + (2 + hd) * tpm.KV_HEAD });
                }
                try ws.append(gpa, .{ .src = bytesOf(L.raw) + pos1 * 256, .len = rows * 256, .dst = slot + 4 * tpm.KV_HEAD });
            }
        }
        try ws.append(gpa, .{ .src = bytesOf(p.b.cin) + rows * WIDE * 2, .len = PLE_TAIL * WIDE * 2, .dst = tpm.TAIL });
        try ws.append(gpa, .{ .src = bytesOf(p.last) + (rows - 1) * WIDE * 2, .len = WIDE * 2, .dst = tpm.LAST });
        try ws.append(gpa, .{ .src = bytesOf(t.picks), .len = 16, .dst = tpm.LAST + WIDE * 2 });
        try tp.sendNow(ws.items, tpm.BACK_FLAG, pr.call);
        pick = t.picks.b.slice(u32, 1)[0];
        @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], p.b.cin.b.contents()[p.b.cin.off + rows * WIDE * 2 ..][0 .. PLE_TAIL * WIDE * 2]);
        m.last = .{ .b = p.last.b, .off = p.last.off + (rows - 1) * WIDE * 2 };
    } else { // the peer's half of the chunk into place
        try tp.hostWait(tpm.BACK_FLAG, pr.call);
        const w = tp.window();
        const rows1 = n - rows0;
        const pos1 = m.pos + rows0;
        const cb = r.queue.commandBuffer();
        r.begin(cb, if (r.serial) .serial else .concurrent);
        errdefer if (r.fsr != null) r.abort();
        for (0..LAYERS) |i| {
            const L = &m.layers[i];
            const slot = tpm.layerSlot(i);
            if (L.linear) {
                p.copyWords(.{ .b = w, .off = slot }, L.cs[a], CS_ROW / 4);
                p.copyWords(.{ .b = w, .off = slot + CS_ROW }, L.so[a], SO_ROW / 4);
            } else {
                for (0..2) |hd| {
                    p.copyWords(.{ .b = w, .off = slot + hd * tpm.KV_HEAD }, .{ .b = L.keys.b, .off = L.keys.off + (hd * m.cap + pos1) * 512 }, rows1 * 128);
                    p.copyWords(.{ .b = w, .off = slot + (2 + hd) * tpm.KV_HEAD }, .{ .b = L.vals.b, .off = L.vals.off + (hd * m.cap + pos1) * 512 }, rows1 * 128);
                }
                p.copyWords(.{ .b = w, .off = slot + 4 * tpm.KV_HEAD }, .{ .b = L.raw.b, .off = L.raw.off + pos1 * 256 }, rows1 * 64);
            }
        }
        try m.finish(cb);
        @memcpy(cin_old[0 .. PLE_TAIL * WIDE * 2], tp.bytes(tpm.TAIL)[0 .. PLE_TAIL * WIDE * 2]);
        pick = std.mem.readInt(u32, tp.bytes(tpm.LAST + WIDE * 2)[0..4], .little);
        m.last = .{ .b = w, .off = tpm.LAST };
    }
    m.state = a; // two segments: each flipped it once
    m.state_row = 0;
    m.pos += n;
    for (tokens) |tok| m.ple.hist = .{ m.ple.hist[1], tok };
    return pick;
}

pub fn pairMtp(p: *Prompt, m: *Model, tp: *Tp2, mine: [2]usize, theirs: [2]usize) !void {
    const r = p.r;
    const h = &m.mtp;
    var ws: [5]tpm.Write = undefined;
    var n: usize = 0;
    if (mine[1] > 0) {
        for (0..2) |hd| {
            ws[n] = .{ .src = bytesOf(h.keys) + (hd * m.cap + mine[0]) * 512, .len = mine[1] * 512, .dst = tpm.MTP + hd * tpm.KV_HEAD };
            ws[n + 1] = .{ .src = bytesOf(h.vals) + (hd * m.cap + mine[0]) * 512, .len = mine[1] * 512, .dst = tpm.MTP + (2 + hd) * tpm.KV_HEAD };
            n += 2;
        }
        ws[n] = .{ .src = bytesOf(h.raw) + mine[0] * 256, .len = mine[1] * 256, .dst = tpm.MTP + 4 * tpm.KV_HEAD };
        n += 1;
    }
    try tp.sendNow(ws[0..n], tpm.MTP_FLAG, tp.call);
    try tp.hostWait(tpm.MTP_FLAG, tp.call);
    if (theirs[1] == 0) return;
    const w = tp.window();
    const cb = r.queue.commandBuffer();
    r.begin(cb, if (r.serial) .serial else .concurrent);
    errdefer if (r.fsr != null) r.abort();
    for (0..2) |hd| {
        p.copyWords(.{ .b = w, .off = tpm.MTP + hd * tpm.KV_HEAD }, .{ .b = h.keys.b, .off = h.keys.off + (hd * m.cap + theirs[0]) * 512 }, theirs[1] * 128);
        p.copyWords(.{ .b = w, .off = tpm.MTP + (2 + hd) * tpm.KV_HEAD }, .{ .b = h.vals.b, .off = h.vals.off + (hd * m.cap + theirs[0]) * 512 }, theirs[1] * 128);
    }
    p.copyWords(.{ .b = w, .off = tpm.MTP + 4 * tpm.KV_HEAD }, .{ .b = h.raw.b, .off = h.raw.off + theirs[0] * 256 }, theirs[1] * 64);
    try m.finish(cb);
}

pub fn mtpKeys(p: *Prompt, m: *Model, start: usize, nexts: []const u32, streams: Buf) !void {
    const r = p.r;
    const b = &p.b;
    const t = &m.t;
    const h = &m.mtp;
    const n = nexts.len;
    if (n == 0) return;
    if (n > PMAX) return error.ChunkSize;
    @memcpy(b.mids.b.slice(u32, n), nexts);
    for (0..n) |i| b.pos.b.slice(i32, PMAX)[i] = @intCast(start + i);
    const kvm = b.kvmeta.b.slice(u32, 3);
    kvm[0], kvm[1], kvm[2] = .{ @intCast(start), @intCast(m.cap), @intCast(n) };
    b.n_add.b.slice(u32, 1)[0] = @intCast(n * WIDE);
    const cb = r.queue.commandBuffer();
    r.begin(cb, if (r.serial) .serial else .concurrent);
    errdefer if (r.fsr != null) r.abort();
    var ce: [2]variants.Const = undefined;
    try r.callRows("mtp:qa_embed_rows@embed", n, &.{ b.mids, m.embed[0], m.embed[1], m.embed[2] }, &.{b.emb}, try variants.oneRow(r, "mtp:qa_embed_rows@embed", variants.width(&ce, "BITS", "GS", m.embed_w)));
    try r.callRows("mtp:q4_rms_rows@mtp.enorm", n, &.{ b.emb, h.enorm, t.eps }, &.{b.branch}, null);
    if (p.mtp_w) |w| p.qmm(b.branch, w[1], D, D, n, b.aout, 0) else r.denseRows(b.branch, D, try h.fce.recorded(), n, b.aout);
    try r.callRows("mtp:q4_rms_rows@mtp.hnorm", n, &.{ streams, h.hnorm, t.eps }, &.{b.normed}, null);
    if (p.mtp_w) |w| p.qmm(b.normed, w[2], D, D, 4 * n, b.gated, 0) else r.denseRows(b.normed, D, try h.fch.recorded(), 4 * n, b.gated);
    p.bind(r.add_pipe, &.{ b.aout, b.gated, b.xs, b.n_add });
    r.enc.dispatchThreads(mtl.Size.of(n * WIDE, 1, 1), mtl.Size.of(256, 1, 1));
    p.barrier();
    try r.callRows("mtp:q4_hc_norm_none#[10240]", n, &.{b.xs}, &.{ b.hout, b.ssp }, null);
    p.hc(m, b.hout, h.ahc, n, b.inj_a);
    if (p.mtp_w) |w| p.qmm(b.mixed, w[0], D, 13952, n, b.p, 0) else r.denseRows(b.mixed, D, try h.proj.recorded(), n, b.p);
    try r.callRows("mtp:q4_attn_prep@mtp.att", n, &.{ b.p, b.pos, h.qn, h.kn, h.iqn, t.eps, t.log2base }, &.{ b.q, b.kout, b.iq }, null);
    p.bind(r.kv_pipe, &.{ b.kout, b.p, h.keys, h.vals, h.raw, b.kvmeta });
    r.enc.dispatchThreads(mtl.Size.of(512 * n, 1, 1), mtl.Size.of(256, 1, 1));
    try m.finish(cb);
    if (p.wide) |wd| try wd.check();
}

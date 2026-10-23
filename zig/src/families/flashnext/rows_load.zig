//! Rows mode without tensor units (0.6.6's pre-M5 decode): every dense lane also as MLX-layout members for rows_w.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const replay = @import("replay.zig");
pub const rows_w = @import("rows_w.zig");
pub const rows_hc = @import("rows_hc.zig");
const variants = @import("variants.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Model = replay.Model;
const Prompt = replay.Prompt;
const D = replay.D;
const KO = 6144; // the attention and DeltaNet output projections' inputs

/// Whether decode reads its dense projections on the row kernels: no tensor units and FZ_M3_LANES unset.
pub fn on(device: mtl.Device) bool {
    return !device.tensorUnits() and platform.c.getenv("FZ_M3_LANES") == null;
}

/// `d` on the run's encoder at `rows` rows, unless the role is knocked out.
pub fn launch(r: *Run, d: *const rows_w.Dense, role: []const u8, x: Buf, rows: usize, y: Buf) !void {
    if (r.skip & Run.class(role) != 0) return;
    d.shared.limit = r.limit orelse d.shared.no_limit;
    try rows_w.encodeDense(r.enc, d, x, rows, y, .auto, !r.serial);
    if (!r.serial) r.enc.barrier();
}

/// A hyper-connection's down and up projections on the per-row kernels at `rows` rows (`rows_buf` holds the count).
pub fn hc(
    r: *Run,
    sh: *rows_w.Shared,
    h: replay.Hc,
    down: []const u8,
    hn: Buf,
    ssp: Buf,
    eps: Buf,
    rows_buf: Buf,
    rows: usize,
    part: Buf,
    mixed: Buf,
    inj: Buf,
) !void {
    if (r.skip & Run.class(down) != 0 or r.hcskip & 6 != 0) return;
    try rows_hc.encode(
        r.enc,
        sh.a,
        sh.device,
        &sh.hc,
        weightsOf(h),
        hn,
        ssp,
        eps,
        rows_buf,
        rows,
        part,
        mixed,
        inj,
        !r.serial,
        sh.hc_tiles and r.limit == null, // a gated window's rows: the per-row kernels leave the unverified ones alone
    );
    if (!r.serial) r.enc.barrier();
}

fn weightsOf(h: replay.Hc) rows_hc.Weights {
    return .{ .scale = h.scale, .dw = h.dw, .ds = h.ds, .db = h.db, .uw = h.uw, .us = h.us, .ub = h.ub, .w = h.w };
}

/// A hyper-connection on the per-row kernels; a 4/32 pair's tiles checked on 12 rows (_hc_mma_exact), else off.
fn hcOn(r: *Run, sh: *rows_w.Shared, h: *replay.Hc) !void {
    for ([_]usize{ 1, 2 }) |rows| _ = try rows_hc.pipes(
        sh.a,
        sh.device,
        &sh.hc,
        try rows_hc.specOf(h.dw, h.uw, h.w, sh.device.generation(), rows),
    );
    h.rows = sh;
    const s = try rows_hc.specOf(h.dw, h.uw, h.w, sh.device.generation(), 12);
    if (!sh.hc_tiles or !rows_hc.q4(s)) return;
    const rows = 12;
    const wide = 4 * D;
    var prng = std.Random.DefaultPrng.init(0x7e58);
    const hn = try r.buffer(rows * wide * 2);
    for (hn.slice(u16, rows * wide)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) - 0.5)) >> 16);
    const ssp = try r.buffer(rows * 40 * 4);
    @memset(ssp.slice(f32, rows * 40), 64);
    const small = try r.buffer(16);
    small.slice(f32, 1)[0] = 1e-6;
    small.slice(i32, 4)[1] = rows;
    const part = try r.buffer(10 * rows * s.nd * 4);
    var outs: [2][2]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = .{ try r.buffer(rows * D * 2), try r.buffer(rows * 4 * 2) };
    defer for ([_]mtl.Buffer{ hn, ssp, small, part, outs[0][0], outs[0][1], outs[1][0], outs[1][1] }) |b| r.release(b);
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (outs, [_]bool{ false, true }) |o, tiles| {
        try rows_hc.encode(
            e,
            sh.a,
            sh.device,
            &sh.hc,
            weightsOf(h.*),
            .{ .b = hn },
            .{ .b = ssp },
            .{ .b = small },
            .{ .b = small, .off = 4 },
            rows,
            .{ .b = part },
            .{ .b = o[0] },
            .{ .b = o[1] },
            false,
            tiles,
        );
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    for (0..2) |j| if (!std.mem.eql(
        u8,
        outs[0][j].contents()[0..outs[0][j].length()],
        outs[1][j].contents()[0..outs[1][j].length()],
    )) {
        std.log.warn(
            "rows: the 4/32 hyper-connection tiles differ from the per-row kernels here; per-row kernels kept",
            .{},
        );
        sh.hc_tiles = false;
    };
}

fn dense(r: *Run, sh: *rows_w.Shared, members: []rows_w.Member, k: usize) !*const rows_w.Dense {
    var n: usize = 0;
    for (members) |mb| {
        try rows_w.check(mb.p);
        if (mb.p.k != k or mb.col != n) return error.RowsMembers;
        n += mb.p.n;
    }
    const d = try r.arena.create(rows_w.Dense);
    d.* = .{ .members = members, .n = n, .k = k, .shared = sh };
    if (rows_w.merge(members)) |p| { // one launch for the parts, which a small M3 launch costs most of
        d.members = try r.arena.alloc(rows_w.Member, 1);
        d.members[0] = .{ .p = p, .col = 0 };
    }
    return d;
}

/// A pack_mlx projection over k inputs: its `.parts` members at their widths, else one 6/32 matrix.
fn fromPack(r: *Run, sh: *rows_w.Shared, w: [3]Buf, stem: []const u8, k: usize) !*const rows_w.Dense {
    const parts = try std.fmt.allocPrint(r.arena, "{s}.parts", .{stem});
    if (!r.index.contains(parts)) {
        const n = (try r.entry(try std.fmt.allocPrint(r.arena, "{s}.mw", .{stem}))).len * 8 / (k * 6);
        const one = try r.arena.alloc(rows_w.Member, 1);
        one[0] = .{ .p = .{ .w = w[0], .s = w[1], .b = w[2], .n = n, .k = k, .bits = 6, .group = 32 }, .col = 0 };
        return dense(r, sh, one, k);
    }
    const t = try variants.table(r, parts);
    const members = try r.arena.alloc(rows_w.Member, t.len / 3);
    var at = [3]usize{ w[0].off, w[1].off, w[2].off };
    var col: usize = 0;
    for (members, 0..) |*mb, i| {
        const n: usize = t[i * 3];
        const bits: u8 = @intCast(t[i * 3 + 1]);
        const group: u16 = @intCast(t[i * 3 + 2]);
        if (bits == 0 or group == 0) return error.RowsMembers;
        mb.* = .{ .p = .{
            .w = .{ .b = w[0].b, .off = at[0] },
            .s = .{ .b = w[1].b, .off = at[1] },
            .b = .{ .b = w[2].b, .off = at[2] },
            .n = n,
            .k = k,
            .bits = bits,
            .group = group,
        }, .col = col };
        at[0] += n * k * bits / 8;
        at[1] += n * (k / group) * 2;
        at[2] += n * (k / group) * 2;
        col += n;
    }
    return dense(r, sh, members, k);
}

/// An n by k matrix read straight from the checkpoint, its width read off its tensors' sizes.
fn raw(r: *Run, stem: []const u8, n: usize, k: usize) !rows_w.Product {
    var ws: [3]Buf = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, &ws) |suf, *b| b.* = try r.loadf("{s}.{s}", .{ stem, suf });
    const wl = (try r.entry(try std.fmt.allocPrint(r.arena, "{s}.weight", .{stem}))).len;
    const sl = (try r.entry(try std.fmt.allocPrint(r.arena, "{s}.scales", .{stem}))).len;
    if (wl * 8 % (n * k) != 0 or sl % (2 * n) != 0 or k % (sl / (2 * n)) != 0) return error.RowsRawShape;
    return .{
        .w = ws[0],
        .s = ws[1],
        .b = ws[2],
        .n = n,
        .k = k,
        .bits = @intCast(wl * 8 / (n * k)),
        .group = @intCast(k / (sl / (2 * n))),
    };
}

fn single(r: *Run, sh: *rows_w.Shared, p: rows_w.Product) !*const rows_w.Dense {
    const one = try r.arena.alloc(rows_w.Member, 1);
    one[0] = .{ .p = p, .col = 0 };
    return dense(r, sh, one, p.k);
}

/// The head's rows at the draft ids, gathered once (the pack's mtp.draft at the head's own width, MLX layout).
fn draft(r: *Run, head: rows_w.Product, ids: Buf, n: usize) !rows_w.Product {
    const id = ids.b.slice(u32, n);
    const rb = [3]usize{ head.k * head.bits / 8, head.k / head.group * 2, head.k / head.group * 2 };
    var out: [3]Buf = undefined;
    for ([_]Buf{ head.w, head.s, head.b }, rb, &out) |src, len, *dst| {
        dst.* = .{ .b = try r.buffer(n * len) };
        const from = src.b.contents()[src.off..];
        const to = dst.b.contents();
        for (id, 0..) |v, i| {
            if (v >= head.n) return error.DraftIdPastVocab;
            @memcpy(to[i * len ..][0..len], from[v * len ..][0..len]);
        }
    }
    return .{ .w = out[0], .s = out[1], .b = out[2], .n = n, .k = head.k, .bits = head.bits, .group = head.group };
}

/// Every member compiled; its MMA twin and four-row kernel checked on 8 rows against the loop, else not taken.
fn warm(r: *Run, d: *const rows_w.Dense) !void {
    const sh = d.shared;
    const rows = 8;
    var prng = std.Random.DefaultPrng.init(0x7e57);
    const x = try r.buffer(rows * d.k * 2);
    for (x.slice(u16, rows * d.k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) - 0.5)) >> 16);
    const y = [3]mtl.Buffer{ try r.buffer(rows * d.n * 2), try r.buffer(rows * d.n * 2), try r.buffer(rows * d.n * 2) };
    defer for ([_]mtl.Buffer{ x, y[0], y[1], y[2] }) |b| r.release(b);
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (d.members) |mb| _ = try rows_w.pipes(
        sh.a,
        sh.device,
        &sh.cache,
        rows_w.specOf(mb.p, sh.device.generation(), 1, d.n),
    );
    for ([_]rows_w.Mode{ .loop, .twin, .four }, y) |mode, out| try rows_w.encodeDense(
        e,
        d,
        .{ .b = x },
        rows,
        .{ .b = out },
        mode,
        false,
    );
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("rows kernels failed at load: {s}", .{why});
        return error.GpuFailed;
    }
    for (d.members) |*mb| for (0..rows) |i| {
        const at = (i * d.n + mb.col) * 2;
        const loop = y[0].contents()[at..][0 .. mb.p.n * 2];
        if (mb.twin and rows_w.twinOk(mb.p) and !std.mem.eql(u8, loop, y[1].contents()[at..][0 .. mb.p.n * 2])) {
            std.log.warn(
                "rows: [{d}, {d}] q{d}/g{d}: the MMA twin differs from the per-row loop here; not taken",
                .{ mb.p.n, mb.p.k, mb.p.bits, mb.p.group },
            );
            mb.twin = false;
        }
        if (mb.four and !std.mem.eql(u8, loop, y[2].contents()[at..][0 .. mb.p.n * 2])) {
            std.log.warn(
                "rows: [{d}, {d}] q{d}/g{d}: four rows a simdgroup differ from the per-row loop here; not taken",
                .{ mb.p.n, mb.p.k, mb.p.bits, mb.p.group },
            );
            mb.four = false;
        }
    };
}

/// Rows mode for `m`'s dense projections (the lanes stay loaded for TP and the 6-bit fast paths); off: nothing.
pub fn attach(p: *const Prompt, m: *Model, mtp_root: []const u8) !void {
    const r = m.r;
    if (!on(r.device)) return;
    const sh = try r.arena.create(rows_w.Shared);
    const sums = rows_w.Shared.max_rows * (rows_w.Shared.max_k / 8) * 4;
    sh.* = .{
        .a = r.arena,
        .device = r.device,
        .sums = .{ .{ .b = try r.buffer(sums) }, .{ .b = try r.buffer(sums) } },
        .no_limit = undefined,
        .limit = undefined,
    };
    sh.no_limit = .{ .b = try r.buffer(16) };
    sh.no_limit.b.slice(i32, 1)[0] = 1 << 20;
    sh.limit = sh.no_limit;
    r.rows_cleanup = .{ .ctx = sh, .call = rows_w.Shared.deinit };
    for (&m.layers, 0..) |*L, i| {
        const kind = if (L.linear) "gdn" else "att";
        L.proj.rows = try fromPack(
            r,
            sh,
            p.proj[i],
            try std.fmt.allocPrint(r.arena, "L{d}.{s}.{s}", .{ i, kind, if (L.linear) "in" else "proj" }),
            D,
        );
        L.out.rows = try fromPack(
            r,
            sh,
            p.out[i],
            try std.fmt.allocPrint(r.arena, "L{d}.{s}.{s}", .{ i, kind, if (L.linear) "out" else "o" }),
            KO,
        );
    }
    m.ple.kv.rows = try fromPack(r, sh, p.ple_kv, "ple.kv", D);
    const head = try raw(r, "language_model.lm_head", replay.VOCAB, D);
    m.head.rows = try single(r, sh, head);
    if (p.mtp_w) |mw| {
        m.mtp.proj.rows = try fromPack(r, sh, mw[0], "mtp.att.proj", D);
        m.mtp.fce.rows = try fromPack(r, sh, mw[1], "mtp.fce", D);
        m.mtp.fch.rows = try fromPack(r, sh, mw[2], "mtp.fch", D);
        const o = try std.fmt.allocPrint(r.arena, "{s}.layers.0.self_attn.o_proj", .{mtp_root});
        m.mtp.out.rows = try single(r, sh, try raw(r, o, D, KO));
        m.mtp.draft.rows = try single(r, sh, try draft(r, head, m.mtp.ids, m.mtp.ids_n));
    }
    for (&m.layers) |*L| {
        for ([_]?*const rows_w.Dense{ L.proj.rows, L.out.rows }) |d| try warm(r, d.?);
        try hcOn(r, sh, &L.ahc);
        try hcOn(r, sh, &L.mhc);
    }
    for ([_]*replay.Hc{ &m.mix, &m.mtp.ahc, &m.mtp.mhc, &m.mtp.mix }) |h| try hcOn(r, sh, h);
    for ([_]replay.Lane{
        m.ple.kv,
        m.head,
        m.mtp.proj,
        m.mtp.fce,
        m.mtp.fch,
        m.mtp.out,
        m.mtp.draft,
    }) |l| if (l.rows) |d| try warm(r, d);
}

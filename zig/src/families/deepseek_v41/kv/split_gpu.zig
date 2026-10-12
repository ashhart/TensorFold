//! tf-kv-split-test KVSPLIT_FATBIN: split KV exchange on real GPUs, a process a rank; exit 0: rows and tokens equal.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const sessions = @import("sessions");
const split = @import("split.zig");
const DevicePool = @import("device.zig").DevicePool;

const DeviceKernels = @import("split_gpu_kernels.zig").DeviceKernels;

/// The replicated pool's row t of slot s in family f, byte i (8-byte words, so a whole row is filled fast).
fn word(f: u64, s: u64, t: u64, i: u64) u64 {
    return (f << 56) ^ (s << 48) ^ (t << 20) ^ (i *% 0x9E3779B97F4A7C15);
}

const page = 256;
const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

const Rank = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    comm: tp.Collective,
    stream: cuda.Stream,
    pool: *sessions.Pool,
    dp: *DevicePool,
    x: split.Exchange,
    /// TF_DSV41_KV_SPLIT_COMPACT=force over the same kernels
    xp: split.Exchange,
    lens: [3]u64,
    bad: u64 = 0,
    checks: u64 = 0,
    shown: u32 = 0,
    blocks: u32 = 0,
    /// each section's mismatches and checks a family (`run`)
    tally: [std.enums.values(Section).len][2][2]u64 = std.mem.zeroes([std.enums.values(Section).len][2][2]u64),

    const Section = enum { dense_eager, dense_graph, packed_vs_dense, dedup, unions, history };

    /// One section's call; its mismatches and checks land in the tally under (section, family).
    fn run(r: *Rank, sec: Section, fi: u32, comptime f: anytype, args: anytype) !void {
        const b0, const c0 = .{ r.bad, r.checks };
        _ = try @call(.auto, f, .{r} ++ args);
        const t = &r.tally[@backingInt(sec)][fi];
        t[0] += r.bad - b0;
        t[1] += r.checks - c0;
    }

    fn rowsOf(r: *const Rank, fi: usize, s: usize) u32 {
        return @intCast(r.lens[s] / fams[fi].ratio);
    }

    /// Selections for R rows (row r on slot r % 3), K entries, some -1; the same on every rank (same seed).
    fn select(r: *const Rank, rnd: std.Random, fi: usize, R: u32, K: u32, sel: []i32, rslot: []i32) void {
        for (rslot[0..R], 0..) |*s, i| s.* = @intCast(i % 3);
        for (sel[0 .. R * K], 0..) |*t, i| {
            const n = r.rowsOf(fi, @intCast(rslot[i / K]));
            t.* = if (rnd.uintLessThan(u32, 16) == 0) -1 else @intCast(rnd.uintLessThan(u32, n));
        }
    }

    /// Received rows and tokens against the reference: tok = owner*n + i, row = the replicated pool's.
    fn check(r: *Rank, fi: usize, sel: []const i32, rslot: []const i32, K: u32, tok: []const i32, recv: []const u8, psh: u32) void {
        const n: i64 = @intCast(sel.len);
        for (sel, tok, 0..) |t, o, i| {
            r.checks += 1;
            if (t < 0) {
                r.bad += @intFromBool(o != -1);
                continue;
            }
            const owner: i64 = split.ownerOf(@intCast(t), psh, 2);
            if (o != owner * n + @as(i64, @intCast(i))) {
                r.bad += 1;
                continue;
            }
            const row = std.mem.bytesAsSlice(u64, recv[@as(usize, @intCast(o)) * 584 ..][0..584]);
            const before = r.bad;
            for (row, 0..) |w, j| r.bad += @intFromBool(w != word(fi, @intCast(rslot[i / K]), @intCast(t), j));
            if (r.bad != before and r.shown < 6) {
                r.shown += 1;
                std.debug.print("MISMATCH fam {d} slot {d} entry {d} t {d} owner {d}: got {x} {x}, want {x} {x}\n", .{ fi, rslot[i / K], i, t, owner, row[0], row[1], word(fi, @intCast(rslot[i / K]), @intCast(t), 0), word(fi, @intCast(rslot[i / K]), @intCast(t), 1) });
            }
        }
    }

    fn dense(r: *Rank, rnd: std.Random, fi: u32, R: u32, graphs: bool) !void {
        const K = 512;
        const n = R * K;
        const g = r.gpa;
        const sel = try g.alloc(i32, n);
        defer g.free(sel);
        const rslot = try g.alloc(i32, R);
        defer g.free(rslot);
        const tok = try g.alloc(i32, n);
        defer g.free(tok);
        const recv_h = try g.alloc(u8, 2 * n * 584);
        defer g.free(recv_h);
        var dsel = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dsel.free();
        var dslot = try cuda.DeviceBuffer.alloc(r.d, R * 4);
        defer dslot.free();
        var dtok = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dtok.free();
        var send = try cuda.DeviceBuffer.alloc(r.d, n * 584);
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * n * 584);
        defer recv.free();
        const v = r.dp.view(fi, 0);
        const a: split.DenseArgs = .{ .sel = dsel.ptr, .rows = R, .k = K, .table = v.pt, .pts = @intCast(v.pts), .rslot = dslot.ptr, .psh = @intCast(v.psh), .base = v.cv, .row_bytes = 584, .world = 2, .send = send.ptr, .tok = dtok.ptr };
        var exec: ?cuda.graph.Exec = null;
        defer if (exec) |*e| e.deinit();
        if (graphs) {
            try cuda.graph.beginCapture(r.stream, .global);
            try r.x.dense(a, recv.ptr, r.stream.handle);
            var gr = try cuda.graph.endCapture(r.stream);
            defer gr.deinit();
            exec = try gr.instantiate();
        }
        for (0..if (graphs) 3 else 1) |_| {
            r.select(rnd, fi, R, K, sel, rslot);
            try dsel.upload(0, std.mem.sliceAsBytes(sel));
            try dslot.upload(0, std.mem.sliceAsBytes(rslot));
            if (exec) |e| try e.launchOn(r.stream) else try r.x.dense(a, recv.ptr, r.stream.handle);
            try r.stream.synchronize();
            try dtok.download(0, std.mem.sliceAsBytes(tok));
            try recv.download(0, recv_h);
            r.check(fi, sel, rslot, K, tok, recv_h, @intCast(v.psh));
        }
    }

    /// Packed vs dense in one captured graph, eager then replayed; every 3rd round all entries on one residue.
    fn packedVsDense(r: *Rank, rnd: std.Random, fi: u32, R: u32) !void {
        const K = 512;
        const n = R * K;
        const g = r.gpa;
        const sel = try g.alloc(i32, n);
        defer g.free(sel);
        const rslot = try g.alloc(i32, R);
        defer g.free(rslot);
        const tok_d = try g.alloc(i32, n);
        defer g.free(tok_d);
        const tok_p = try g.alloc(i32, n);
        defer g.free(tok_p);
        const rd = try g.alloc(u8, 2 * n * 584);
        defer g.free(rd);
        const rp = try g.alloc(u8, 2 * n * 584);
        defer g.free(rp);
        var dsel = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dsel.free();
        var dslot = try cuda.DeviceBuffer.alloc(r.d, R * 4);
        defer dslot.free();
        var dtok_d = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dtok_d.free();
        var dtok_p = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dtok_p.free();
        var dlens = try cuda.DeviceBuffer.alloc(r.d, 2 * 4);
        defer dlens.free();
        var send_d = try cuda.DeviceBuffer.alloc(r.d, n * 584);
        defer send_d.free();
        var send_p = try cuda.DeviceBuffer.alloc(r.d, n * 584);
        defer send_p.free();
        var recv_d = try cuda.DeviceBuffer.alloc(r.d, 2 * n * 584);
        defer recv_d.free();
        var recv_p = try cuda.DeviceBuffer.alloc(r.d, 2 * n * 584);
        defer recv_p.free();
        const v = r.dp.view(fi, 0);
        const psh: u32 = @intCast(v.psh);
        const a: split.DenseArgs = .{ .sel = dsel.ptr, .rows = R, .k = K, .table = v.pt, .pts = @intCast(v.pts), .rslot = dslot.ptr, .psh = psh, .base = v.cv, .row_bytes = 584, .world = 2, .send = send_d.ptr, .tok = dtok_d.ptr };
        var pa = split.PackArgs.of(a, r.comm.rank(), dlens.ptr);
        pa.send = send_p.ptr;
        pa.tok = dtok_p.ptr;
        if (!r.xp.packs(@as(u64, n) * 584)) return error.ForceDoesNotPack;
        try cuda.graph.beginCapture(r.stream, .global);
        try r.x.dense(a, recv_d.ptr, r.stream.handle);
        try r.xp.window(pa, recv_p.ptr, r.stream.handle);
        var gr = try cuda.graph.endCapture(r.stream);
        defer gr.deinit();
        var exec = try gr.instantiate();
        defer exec.deinit();
        for (0..4) |round| {
            r.select(rnd, fi, R, K, sel, rslot);
            if (round % 3 == 2) for (sel) |*t| if (t.* >= 0 and split.ownerOf(@intCast(t.*), psh, 2) != 0) {
                t.* = if (t.* >= @as(i32, 1) << @intCast(psh)) t.* - (@as(i32, 1) << @intCast(psh)) else -1; // the page before: residue 0
            };
            try dsel.upload(0, std.mem.sliceAsBytes(sel));
            try dslot.upload(0, std.mem.sliceAsBytes(rslot));
            try recv_p.fill8(0xEE, r.stream.handle); // past each owner's rows: never read
            if (round == 0) {
                try r.x.dense(a, recv_d.ptr, r.stream.handle);
                try r.xp.densePacked(pa, recv_p.ptr, r.stream.handle);
            } else try exec.launchOn(r.stream);
            try r.stream.synchronize();
            try dtok_d.download(0, std.mem.sliceAsBytes(tok_d));
            try dtok_p.download(0, std.mem.sliceAsBytes(tok_p));
            try recv_d.download(0, rd);
            try recv_p.download(0, rp);
            var lens: [2]i32 = undefined;
            try dlens.download(0, std.mem.asBytes(&lens));
            r.check(fi, sel, rslot, K, tok_d, rd, psh);
            var count: [2]u32 = .{ 0, 0 };
            for (sel, tok_d, tok_p, 0..) |t, od, op, i| {
                r.checks += 1;
                if (t < 0) {
                    r.bad += @intFromBool(op != -1);
                    continue;
                }
                const o = split.ownerOf(@intCast(t), psh, 2);
                const want: i32 = @intCast(o * n + count[o]);
                count[o] += 1;
                if (op != want or od < 0) {
                    r.bad += 1;
                    continue;
                }
                const before = r.bad;
                const x = std.mem.bytesAsSlice(u64, rd[@as(usize, @intCast(od)) * 584 ..][0..584]);
                const y = std.mem.bytesAsSlice(u64, rp[@as(usize, @intCast(op)) * 584 ..][0..584]);
                for (x, y, 0..) |p, q, j| r.bad += @intFromBool(p != q or q != word(fi, @intCast(rslot[i / K]), @intCast(t), j));
                if (r.bad != before and r.shown < 6) {
                    r.shown += 1;
                    std.debug.print("PACKED MISMATCH fam {d} R {d} round {d} entry {d} t {d} owner {d} place {d}\n", .{ fi, R, round, i, t, o, op });
                }
            }
            for (0..2) |o| r.bad += @intFromBool(lens[o] != @as(i32, @intCast(count[o] * 584)));
            if (round % 3 == 2) r.bad += @intFromBool(lens[1] != 0);
        }
    }

    /// Verify-like selections: a slot's rows share one base list, `fresh` in 16 entries replaced.
    fn verifyLike(r: *const Rank, rnd: std.Random, fi: usize, R: u32, K: u32, fresh: u32, sel: []i32, rslot: []i32) void {
        for (rslot[0..R], 0..) |*s, i| s.* = @intCast(i * 3 / R);
        var base: [3][512]i32 = undefined;
        for (&base, 0..) |*b, s| for (b[0..K]) |*t| {
            t.* = @intCast(rnd.uintLessThan(u32, r.rowsOf(fi, s)));
        };
        for (sel[0 .. R * K], 0..) |*t, i| {
            const s: usize = @intCast(rslot[i / K]);
            t.* = if (rnd.uintLessThan(u32, 16) < fresh) @intCast(rnd.uintLessThan(u32, r.rowsOf(fi, s))) else base[s][i % K];
        }
    }

    /// Dedup (graph replays, verify-like selections) against the pool rows, then dedup / dense / packed times a call.
    fn dedupVsDense(r: *Rank, rnd: std.Random, io: std.Io, fi: u32, R: u32) !void {
        const K = 512;
        const n = R * K;
        const g = r.gpa;
        const v = r.dp.view(fi, 0);
        const psh: u32 = @intCast(v.psh);
        const map_rows = 3 * (@as(u64, @intCast(v.pts)) << @intCast(psh));
        var scratch = try cuda.DeviceBuffer.alloc(r.d, 4 * (map_rows + 2 * n + (n + 255) / 256 * split.max_world));
        defer scratch.free();
        try scratch.fill32(std.math.maxInt(i32), null);
        // by lengths where every rank takes them (RoCE), else forced (NCCL): never silently dense
        const on = try split.Exchange.init(r.comm, r.x.kernels, .on);
        var xd = if (on.packs(@as(u64, n) * 584)) on else try split.Exchange.init(r.comm, r.x.kernels, .force);
        xd.dedup = .{ .map = scratch.ptr, .map_rows = map_rows, .first = scratch.ptr + 4 * map_rows, .place = scratch.ptr + 4 * (map_rows + n), .bcnt = scratch.ptr + 4 * (map_rows + 2 * n), .entries = n, .slots = 3 };
        const xo = try split.Exchange.init(r.comm, r.x.kernels, .on);
        const sel = try g.alloc(i32, n);
        defer g.free(sel);
        const rslot = try g.alloc(i32, R);
        defer g.free(rslot);
        const tok = try g.alloc(i32, n);
        defer g.free(tok);
        const rh = try g.alloc(u8, 2 * n * 584);
        defer g.free(rh);
        var dsel = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dsel.free();
        var dslot = try cuda.DeviceBuffer.alloc(r.d, R * 4);
        defer dslot.free();
        var dtok = try cuda.DeviceBuffer.alloc(r.d, n * 4);
        defer dtok.free();
        var dlens = try cuda.DeviceBuffer.alloc(r.d, 2 * 4);
        defer dlens.free();
        var send = try cuda.DeviceBuffer.alloc(r.d, n * 584);
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * n * 584);
        defer recv.free();
        const a: split.DenseArgs = .{ .sel = dsel.ptr, .rows = R, .k = K, .table = v.pt, .pts = @intCast(v.pts), .rslot = dslot.ptr, .psh = psh, .base = v.cv, .row_bytes = 584, .world = 2, .send = send.ptr, .tok = dtok.ptr };
        const pa = split.PackArgs.of(a, r.comm.rank(), dlens.ptr);
        var execs: [3]cuda.graph.Exec = undefined;
        for (&execs, 0..) |*e, m| {
            try cuda.graph.beginCapture(r.stream, .global);
            switch (m) {
                0 => try xd.window(pa, recv.ptr, r.stream.handle),
                1 => try r.x.dense(a, recv.ptr, r.stream.handle),
                else => try xo.window(pa, recv.ptr, r.stream.handle),
            }
            var gr = try cuda.graph.endCapture(r.stream);
            defer gr.deinit();
            e.* = try gr.instantiate();
        }
        defer for (&execs) |*e| e.deinit();
        for (0..4) |round| {
            r.verifyLike(rnd, fi, R, K, @intCast(round * 5), sel, rslot);
            try dsel.upload(0, std.mem.sliceAsBytes(sel));
            try dslot.upload(0, std.mem.sliceAsBytes(rslot));
            try recv.fill8(0xEE, r.stream.handle);
            if (round == 0) try xd.window(pa, recv.ptr, r.stream.handle) else try execs[0].launchOn(r.stream);
            try r.stream.synchronize();
            try dtok.download(0, std.mem.sliceAsBytes(tok));
            try recv.download(0, rh);
            var lens: [2]i32 = undefined;
            try dlens.download(0, std.mem.asBytes(&lens));
            var seen = std.AutoHashMap(u64, i32).init(g);
            defer seen.deinit();
            var unique: [2]u32 = .{ 0, 0 };
            for (sel, tok, 0..) |t, o, i| {
                r.checks += 1;
                if (t < 0 or o < 0) {
                    r.bad += @intFromBool((t < 0) != (o < 0));
                    continue;
                }
                const gp = try seen.getOrPut((@as(u64, @intCast(rslot[i / K])) << 32) | @as(u64, @intCast(t)));
                if (gp.found_existing) r.bad += @intFromBool(gp.value_ptr.* != o) else {
                    gp.value_ptr.* = o;
                    unique[split.ownerOf(@intCast(t), psh, 2)] += 1;
                }
                const before = r.bad;
                const y = std.mem.bytesAsSlice(u64, rh[@as(usize, @intCast(o)) * 584 ..][0..584]);
                for (y, 0..) |q, j| r.bad += @intFromBool(q != word(fi, @intCast(rslot[i / K]), @intCast(t), j));
                if (r.bad != before and r.shown < 6) {
                    r.shown += 1;
                    std.debug.print("DEDUP MISMATCH fam {d} R {d} round {d} entry {d} t {d} token {d}\n", .{ fi, R, round, i, t, o });
                }
            }
            for (0..2) |o| r.bad += @intFromBool(lens[o] != @as(i32, @intCast(unique[o] * 584)));
            const map = try g.alloc(i32, map_rows);
            defer g.free(map);
            try scratch.download(0, std.mem.sliceAsBytes(map));
            for (map) |m| r.bad += @intFromBool(m != std.math.maxInt(i32));
            if (round == 1) std.debug.print("dedup fam {d} R {d}: {d} + {d} unique rows of {d} entries ({s})\n", .{ fi, R, unique[0], unique[1], n, @tagName(xd.compact) });
        }
        r.verifyLike(rnd, fi, R, K, 3, sel, rslot);
        try dsel.upload(0, std.mem.sliceAsBytes(sel));
        try dslot.upload(0, std.mem.sliceAsBytes(rslot));
        var us: [3]f64 = undefined;
        for (execs, &us) |e, *u| {
            try e.launchOn(r.stream);
            try r.stream.synchronize();
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            for (0..50) |_| try e.launchOn(r.stream);
            try r.stream.synchronize();
            u.* = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 50e3;
        }
        std.debug.print("exchange fam {d} R {d} (3 in 16 fresh): dedup {d:.1} us, dense {d:.1} us, packed {d:.1} us a call (packed moves {s})\n", .{ fi, R, us[0], us[1], us[2], if (xo.packs(@as(u64, n) * 584)) "lengths" else "whole strides" });
    }

    /// A 4096-row segment's history union (rows by token == the pool's), timed vs unionOf's planned path.
    fn historyVsPlanned(r: *Rank, rnd: std.Random, io: std.Io, fi: u32, si: u32, rank: u32) !void {
        const K = 512;
        const S = 4096;
        const g, const n = .{ r.gpa, S * K };
        const sel = try g.alloc(i32, n);
        defer g.free(sel);
        for (sel) |*t| t.* = if (rnd.uintLessThan(u32, 16) == 0) -1 else @intCast(rnd.uintLessThan(u32, r.rowsOf(fi, si)));
        const v, const slot = .{ r.dp.view(fi, si), r.pool.slots.items[si] };
        const psh: u32 = @intCast(v.psh);
        const m = split.historyRows(slot.len, psh, 2);
        var dsel = try cuda.DeviceBuffer.fromHost(r.d, std.mem.sliceAsBytes(sel));
        defer dsel.free();
        var tok = try cuda.DeviceBuffer.alloc(r.d, 4 * n);
        defer tok.free();
        const cap: u64 = 128 << 20;
        var send = try cuda.DeviceBuffer.alloc(r.d, cap / 3);
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * cap / 3);
        defer recv.free();
        const h: split.HistArgs = .{ .sel = dsel.ptr, .rows = S, .k = K, .table = v.pt, .pages = slot.len, .psh = psh, .rank = rank, .world = 2, .m = m, .row_bytes = 584, .base = v.cv, .send = send.ptr, .tok = tok.ptr };
        try r.x.history(h, recv.ptr, r.stream.handle);
        try r.stream.synchronize();
        const ht = try g.alloc(i32, n);
        defer g.free(ht);
        try tok.download(0, std.mem.sliceAsBytes(ht)); // the tokens, then the rows behind them
        const hr = try g.alloc(u8, 2 * m * 584);
        defer g.free(hr);
        try recv.download(0, hr);
        for (sel, ht) |t, o| {
            r.checks += 1;
            if (t < 0) {
                r.bad += @intFromBool(o != -1);
                continue;
            }
            const row = std.mem.bytesAsSlice(u64, hr[@as(usize, @intCast(o)) * 584 ..][0..584]);
            for (row, 0..) |w, j| r.bad += @intFromBool(w != word(fi, si, @intCast(t), j));
        }
        const table = try g.alloc(u32, slot.max_pages);
        defer g.free(table);
        var um: u32, var us: [2]f64 = .{ 0, undefined };
        for (&us, 0..) |*u, path| {
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            for (0..10) |_| {
                if (path == 1) {
                    try r.x.history(h, recv.ptr, r.stream.handle);
                    continue;
                }
                try r.stream.synchronize();
                try dsel.download(0, std.mem.sliceAsBytes(sel));
                for (table, 0..) |*t, k| t.* = slot.localTableAt(@intCast(k));
                var blocks: std.ArrayList(split.Block) = .empty;
                defer {
                    for (blocks.items) |*b| b.deinit(g);
                    blocks.deinit(g);
                }
                try split.planUnion(g, .{ .sel = sel, .k = K, .table = table, .psh = psh, .world = 2, .rank = rank, .row_bytes = 584, .cap = cap }, &blocks);
                const b = blocks.items[0];
                um = b.m;
                var phys = try cuda.DeviceBuffer.fromHost(r.d, std.mem.sliceAsBytes(b.phys));
                defer phys.free();
                try tok.upload(0, std.mem.sliceAsBytes(b.tokens));
                try r.x.unionBlock(v.cv, 584, phys.ptr, b.m, send.ptr, recv.ptr, r.stream.handle);
            }
            try r.stream.synchronize();
            u.* = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 10e6;
        }
        std.debug.print("prefill union fam {d} slot {d} ({d} pages, 4096 x 512): history {d:.2} ms ({d} rows an owner), planned {d:.2} ms ({d} rows an owner) a call\n", .{ fi, si, slot.len, us[1], m, us[0], um });
    }

    fn unions(r: *Rank, rnd: std.Random, fi: u32, rank: u32) !void {
        const K = 512;
        const S = 256; // a prefill segment's rows
        const g = r.gpa;
        const sel = try g.alloc(i32, S * K);
        defer g.free(sel);
        for (sel) |*t| t.* = if (rnd.uintLessThan(u32, 16) == 0) -1 else @intCast(rnd.uintLessThan(u32, r.rowsOf(fi, 2)));
        const v = r.dp.view(fi, 2);
        const table = try g.alloc(u32, r.dp.pts);
        defer g.free(table);
        for (table, 0..) |*t, k| t.* = r.pool.slots.items[2].localTableAt(@intCast(k));
        var blocks: std.ArrayList(split.Block) = .empty;
        defer {
            for (blocks.items) |*b| b.deinit(g);
            blocks.deinit(g);
        }
        try split.planUnion(g, .{ .sel = sel, .k = K, .table = table, .psh = @intCast(v.psh), .world = 2, .rank = rank, .row_bytes = 584, .cap = 24 << 20 }, &blocks);
        for (blocks.items) |blk| {
            var phys = try cuda.DeviceBuffer.fromHost(r.d, std.mem.sliceAsBytes(blk.phys));
            defer phys.free();
            var send = try cuda.DeviceBuffer.alloc(r.d, blk.m * 584);
            defer send.free();
            var recv = try cuda.DeviceBuffer.alloc(r.d, 2 * blk.m * 584);
            defer recv.free();
            try r.x.unionBlock(v.cv, 584, phys.ptr, blk.m, send.ptr, recv.ptr, r.stream.handle);
            try r.stream.synchronize();
            const h = try g.alloc(u8, 2 * blk.m * 584);
            defer g.free(h);
            try recv.download(0, h);
            for (sel[blk.a * K .. blk.b * K], blk.tokens) |t, o| {
                r.checks += 1;
                if (t < 0) continue;
                const row = std.mem.bytesAsSlice(u64, h[@as(usize, @intCast(o)) * 584 ..][0..584]);
                for (row, 0..) |w, j| r.bad += @intFromBool(w != word(fi, 2, @intCast(t), j));
            }
        }
        r.blocks += @intCast(blocks.items.len);
    }
};

/// TP_FATBIN (else TF_TP_KERNEL_IMAGE): tp opens RoCE / mailbox as the server does; without it NCCL only.
pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-kv-split-test KVSPLIT_FATBIN [TP_FATBIN] (tp's TF_TP_* settings; TP_FATBIN: RoCE / mailbox)\n", .{});
        return 2;
    }
    const image = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], gpa, .limited(64 << 20));
    defer gpa.free(image);
    // TP_FATBIN, else TF_TP_KERNEL_IMAGE (the server's variable), else none (NCCL only)
    const tp_path: ?[]const u8 = if (args.len > 2) args[2] else if (std.c.getenv("TF_TP_KERNEL_IMAGE")) |v| std.mem.span(v) else null;
    const tp_image: []align(16) u8 = if (tp_path) |p| try std.Io.Dir.cwd().readFileAllocOptions(init.io, p, gpa, .limited(16 << 20), comptime .fromByteUnits(16), null) else &.{};
    defer gpa.free(tp_image);
    const cfg = try tp.Config.fromEnv();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    var s: tp.Session = undefined;
    try s.open(&driver, cfg, tp_image);
    std.debug.print("rank {d}: tp backend {t}\n", .{ cfg.rank, s.comm().kind() });
    var open = true;
    defer if (open) s.close();
    var module = try cuda.Module.load(&driver, image);
    defer module.unload();
    var dk: DeviceKernels = .{ .d = &driver, .dense_fn = try module.function("_ZN13dsv41_kvsplit12dense_kernelENS_9DenseArgsE"), .gather_fn = try module.function("_ZN13dsv41_kvsplit13gather_kernelEPKhjPKjjPh"), .pack_fn = try module.function("_ZN13dsv41_kvsplit11pack_kernelENS_8PackArgsE") };
    dk.dedup_fns = .{
        try module.function("_ZN13dsv41_kvsplit17dedup_mark_kernelENS_9DedupArgsE"),
        try module.function("_ZN13dsv41_kvsplit18dedup_flags_kernelENS_9DedupArgsE"),
        try module.function("_ZN13dsv41_kvsplit17dedup_pack_kernelENS_9DedupArgsE"),
        try module.function("_ZN13dsv41_kvsplit16dedup_tok_kernelENS_9DedupArgsE"),
    };
    dk.hist_fns = .{ try module.function("_ZN13dsv41_kvsplit18hist_gather_kernelENS_8HistArgsE"), try module.function("_ZN13dsv41_kvsplit15hist_tok_kernelENS_8HistArgsE") };
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 8192 * page, 2, cfg.rank);
    defer pool.deinit();
    const lens = [3]u64{ 300_000 + 77, 41_000, 1_000_000 + 5 };
    for (lens) |n| try (try pool.newSlot(1 << 20)).ensure(n);
    var dp = try DevicePool.init(gpa, &driver, &ctx, &pool, stream, 3, (1 << 20) / page);
    defer dp.deinit();
    try dp.syncTables();
    // each rank writes the rows of the pages it owns (kv_store's writes; the other rank's land on its discard page)
    for (fams, 0..) |f, fi| {
        const pb = f.pageBytes(page);
        const buf = try gpa.alloc(u8, pb);
        defer gpa.free(buf);
        const per = page / f.ratio;
        for (pool.slots.items, 0..) |sl, si| for (sl.mapped(), 0..) |pg, k| {
            if (!pool.owns(pg)) continue;
            const w = std.mem.bytesAsSlice(u64, buf);
            for (0..per) |row| for (0..73) |j| {
                w[row * 73 + j] = word(fi, si, k * per + row, j);
            };
            try dp.store().write(@intCast(fi), &.{pool.localPage(pg)}, buf);
        };
    }
    try stream.synchronize();
    var r: Rank = .{ .gpa = gpa, .d = &driver, .comm = s.comm(), .stream = stream, .pool = &pool, .dp = &dp, .x = .{ .comm = s.comm(), .kernels = dk.kernels() }, .xp = try split.Exchange.init(s.comm(), dk.kernels(), .force), .lens = lens };
    var prng = std.Random.DefaultPrng.init(20261006);
    const rnd = prng.random();
    for ([_]u32{ 0, 1 }) |fi| for ([_]u32{ 1, 6, 16 }) |R| {
        try r.run(.dense_eager, fi, Rank.dense, .{ rnd, fi, R, false });
        try r.run(.dense_graph, fi, Rank.dense, .{ rnd, fi, R, true });
    };
    for ([_]u32{ 0, 1 }) |fi| for ([_]u32{ 1, 6, 16, 64 }) |R| try r.run(.packed_vs_dense, fi, Rank.packedVsDense, .{ rnd, fi, R });
    for ([_]u32{ 0, 1 }) |fi| for ([_]u32{ 1, 4, 8, 24, 32 }) |R| try r.run(.dedup, fi, Rank.dedupVsDense, .{ rnd, init.io, fi, R });
    var blocks: u32 = 0;
    for ([_]u32{ 0, 1 }) |fi| {
        const b0 = r.blocks;
        try r.run(.unions, fi, Rank.unions, .{ rnd, fi, cfg.rank });
        blocks += r.blocks - b0;
    }
    for ([_][2]u32{ .{ 0, 1 }, .{ 0, 0 }, .{ 1, 1 } }) |c| try r.run(.history, c[0], Rank.historyVsPlanned, .{ rnd, init.io, c[0], c[1], cfg.rank }); // fits the cap
    // the time of one dense exchange (pack + all-gather), graph-replayed, R = 16, K = 512
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    for (0..20) |_| try r.dense(rnd, 1, 16, false);
    const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 20e6;
    std.debug.print("rank {d}: {d} entries checked over dense (R 1/6/16, eager + 3 graph replays, ratio 2 and 1), packed vs dense (R 1/6/16/64, eager + 3 graph replays) and {d} union blocks; mismatches {d}; {d:.2} ms a dense round trip incl. host checks\n{s}\n", .{ cfg.rank, r.checks, blocks, r.bad, dt, if (r.bad == 0) "PASS" else "FAIL" });
    for (r.tally, 0..) |t, si| for (t, 0..) |fam, fi| if (fam[1] > 0) {
        std.debug.print("rank {d}: section {s} fam {d} (ratio {d}): mismatches {d} of {d} checks\n", .{ cfg.rank, @tagName(@as(Rank.Section, @fromBackingInt(@intCast(si)))), fi, fams[fi].ratio, fam[0], fam[1] });
    };
    // both ranks finish, then close the session (its fate channel) at once, before the slower teardown below
    try r.comm.barrier();
    s.close();
    open = false;
    return if (r.bad == 0) 0 else 1;
}

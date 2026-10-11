//! TF_DSV41_X3LD_EPI micro-gate: the fused decode expert chain against the unfused one, byte for byte, and timed.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const exl3 = dsv41.exl3;
const Gpu = check.Gpu;

const D = 5120;
const I = 1152;
const SK_GU = 4;
const widths = [_]u32{ 4, 6, 8, 10 };
const limit: f32 = 10.0;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return r.s;
    }
    fn unit(r: *Rng) f32 {
        return @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
    }
};

fn halves(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, lo: f32, hi: f32) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f16, n);
    defer gpa.free(h);
    for (h) |*v| v.* = @floatCast(lo + (hi - lo) * rng.unit());
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

fn floats(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, n: usize, lo: f32, hi: f32) !cuda.DeviceBuffer {
    const h = try gpa.alloc(f32, n);
    defer gpa.free(h);
    for (h) |*v| v.* = lo + (hi - lo) * rng.unit();
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
}

/// One matrix kind's words for every expert (expert e at its own width): the int64 pointer table and the K2 table.
const Mats = struct {
    words: cuda.DeviceBuffer,
    table: cuda.DeviceBuffer,
    k2: cuda.DeviceBuffer,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, k2e: []const u32, K: usize, N: usize) !Mats {
        var total: usize = 0;
        for (k2e) |k2| total += (K / 16) * (N / 16) * 4 * k2;
        const host = try gpa.alloc(u32, total);
        defer gpa.free(host);
        for (host) |*w| w.* = @truncate(rng.next());
        const words = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        const tab = try gpa.alloc(i64, k2e.len);
        defer gpa.free(tab);
        var at: usize = 0;
        for (k2e, tab) |k2, *t| {
            t.* = @intCast(words.ptr + 4 * at);
            at += (K / 16) * (N / 16) * 4 * k2;
        }
        const k2s = try gpa.alloc(i32, k2e.len);
        defer gpa.free(k2s);
        for (k2e, k2s) |x, *y| y.* = @intCast(x);
        return .{ .words = words, .table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(tab)), .k2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2s)) };
    }

    fn deinit(m: *Mats) void {
        m.words.free();
        m.table.free();
        m.k2.free();
    }
};

/// A model's expert table: Et experts (the last the shared one), `topk` routed slots + the shared.
const Model = struct {
    name: []const u8,
    Et: usize,
    topk: usize,
    gate: Mats,
    up: Mats,
    down: Mats,
    suh_g: cuda.DeviceBuffer,
    suh_u: cuda.DeviceBuffer,
    svh_g: cuda.DeviceBuffer,
    svh_u: cuda.DeviceBuffer,
    suh_d: cuda.DeviceBuffer,
    svh_d: cuda.DeviceBuffer,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: *Rng, name: []const u8, Et: usize, topk: usize) !Model {
        const k2g = try gpa.alloc(u32, Et);
        defer gpa.free(k2g);
        const k2d = try gpa.alloc(u32, Et);
        defer gpa.free(k2d);
        for (0..Et) |e| {
            k2g[e] = widths[e % 4];
            k2d[e] = widths[(e / 4) % 4];
        }
        return .{
            .name = name,
            .Et = Et,
            .topk = topk,
            .gate = try Mats.init(gpa, d, rng, k2g, D, I),
            .up = try Mats.init(gpa, d, rng, k2g, D, I),
            .down = try Mats.init(gpa, d, rng, k2d, I, D),
            .suh_g = try halves(gpa, d, rng, Et * D, -1.5, 1.5),
            .suh_u = try halves(gpa, d, rng, Et * D, -1.5, 1.5),
            .svh_g = try halves(gpa, d, rng, Et * I, 0.01, 0.03),
            .svh_u = try halves(gpa, d, rng, Et * I, 0.01, 0.03),
            .suh_d = try halves(gpa, d, rng, Et * I, -1.5, 1.5),
            .svh_d = try halves(gpa, d, rng, Et * D, 0.01, 0.03),
        };
    }

    fn deinit(m: *Model) void {
        m.gate.deinit();
        m.up.deinit();
        m.down.deinit();
        for ([_]*cuda.DeviceBuffer{ &m.suh_g, &m.suh_u, &m.svh_g, &m.svh_u, &m.suh_d, &m.svh_d }) |b| b.free();
    }
};

/// One path's outputs.
const Outs = struct {
    z: cuda.DeviceBuffer,
    xd: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, P: usize, R: usize, y0: []const u8) !Outs {
        const zlen = 4 * P * @max(2 * SK_GU * I, D);
        const o: Outs = .{ .z = try cuda.DeviceBuffer.alloc(d, zlen), .xd = try cuda.DeviceBuffer.alloc(d, 2 * P * I), .y = try cuda.DeviceBuffer.fromHost(d, y0), .out = try cuda.DeviceBuffer.alloc(d, 4 * R * D) };
        try o.z.fill8(0, null);
        try o.xd.fill8(0, null);
        try o.out.fill8(0, null);
        return o;
    }

    fn deinit(o: *Outs) void {
        for ([_]*cuda.DeviceBuffer{ &o.z, &o.xd, &o.y, &o.out }) |b| b.free();
    }
};

const Run = struct {
    o: exl3.Ops,
    m: *const Model,
    R: usize,
    slots: usize,
    pick: u64,
    wts: u64,
    uids: u64,
    ucount: u64,
    members: u64,
    xg: u64,
    xu: u64,
    ticket: u64,
    bf16: bool,

    fn grouped(r: *const Run, gu: bool, x: u64, z: u64) exl3.Grouped {
        const P = r.R * r.slots;
        const m = r.m;
        return .{
            .x0 = if (gu) r.xg else x,
            .x1 = if (gu) r.xu else x,
            .tp0 = if (gu) m.gate.table.ptr else m.down.table.ptr,
            .tp1 = if (gu) m.up.table.ptr else m.down.table.ptr,
            .k2_0 = if (gu) m.gate.k2.ptr else m.down.k2.ptr,
            .k2_1 = if (gu) m.up.k2.ptr else m.down.k2.ptr,
            .uids = r.uids,
            .ucount = r.ucount,
            .members = r.members,
            .z = z,
            .mats = if (gu) 2 else 1,
            .K = if (gu) D else I,
            .N = if (gu) I else D,
            .P = P,
            .SK = if (gu) SK_GU else 1,
            .slots = r.slots,
            .maxm = r.R,
            .nexp = P,
            .lo = 4,
            .hi = 10,
        };
    }

    /// the unfused four launches (block_moe.moe with the knob off)
    fn unfused(r: *const Run, t: *const Outs) !void {
        const P = r.R * r.slots;
        const m = r.m;
        try r.o.x3ld(r.grouped(true, 0, t.z.ptr), 8, 1, 0, false);
        try r.o.gateupEpilogue(t.z.ptr, r.pick, m.svh_g.ptr, m.svh_u.ptr, m.suh_d.ptr, t.xd.ptr, r.R, P, I, SK_GU, r.slots, m.Et, limit, 1);
        try r.o.x3ld(r.grouped(false, t.xd.ptr, t.z.ptr), 8, 1, 0, false);
        try r.o.downCombine(t.z.ptr, r.pick, m.svh_d.ptr, t.y.ptr, r.wts, t.out.ptr, r.R, P, D, 1, r.slots, m.Et, r.bf16);
    }

    /// TF_DSV41_X3LD_EPI's two
    fn fused(r: *const Run, t: *const Outs) !void {
        const m = r.m;
        try r.o.x3ldEpi(r.grouped(true, 0, t.z.ptr), .gu, 1, .{ .pick = r.pick, .sv0 = m.svh_g.ptr, .sv1 = m.svh_u.ptr, .sd = m.suh_d.ptr, .xd = t.xd.ptr, .ticket = r.ticket, .E = @intCast(m.Et), .limit = limit, .act_mode = 1 });
        try r.o.x3ldEpi(r.grouped(false, t.xd.ptr, t.z.ptr), if (r.bf16) .dnb else .dn, 1, .{ .pick = r.pick, .sv0 = m.svh_d.ptr, .y = t.y.ptr, .wts = r.wts, .out = t.out.ptr, .ticket = r.ticket, .E = @intCast(m.Et) });
    }
};

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

fn timeIt(stream: cuda.Stream, d: *const cuda.Driver, r: *const Run, t: *const Outs, fused: bool, reps: usize) !f32 {
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    var ms: [256]f32 = undefined;
    const n = @min(reps, ms.len);
    for (0..n) |i| {
        try t0.record(stream);
        if (fused) try r.fused(t) else try r.unfused(t);
        try t1.record(stream);
        try t1.synchronize();
        ms[i] = try cuda.Event.elapsedMs(t0, t1);
    }
    return median(ms[0..n]);
}

fn sameDevice(gpa: std.mem.Allocator, what: []const u8, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer, n: usize) !void {
    const x = try gpa.alloc(u8, n);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, n);
    defer gpa.free(y);
    try a.download(0, x);
    try b.download(0, y);
    try check.sameBytes(what, x, y);
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, rows_arg: ?[]const u8, reps_arg: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    var rows_list: [8]usize = undefined;
    var nrows: usize = 0;
    var it = std.mem.tokenizeScalar(u8, rows_arg orelse "1,4,16,24", ',');
    while (it.next()) |t| : (nrows += 1) {
        if (nrows == rows_list.len) return error.TooManyRowCounts;
        rows_list[nrows] = try std.fmt.parseInt(usize, t, 10);
        if (rows_list[nrows] < 1 or rows_list[nrows] > 64) return error.Rows;
    }
    const reps: usize = if (reps_arg) |r| try std.fmt.parseInt(usize, r, 10) else 50;
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    const o = k.experts(stream);
    var rng: Rng = .{ .s = 0x2545f4914f6cdd1d };

    var backbone = try Model.init(gpa, d, &rng, "backbone", 385, 6);
    defer backbone.deinit();
    var draft = try Model.init(gpa, d, &rng, "dspark", 129, 3);
    defer draft.deinit();
    std.debug.print("x3ld-epi: D {d}, I {d}, widths 4/6/8/10, gate/up SK {d}, {d} reps (median), {d} SMs\n", .{ D, I, SK_GU, reps, k.sms });

    var total_u: f64 = 0;
    var total_f: f64 = 0;
    const Case = struct { m: *const Model, bf16: bool };
    const cases = [_]Case{ .{ .m = &backbone, .bf16 = true }, .{ .m = &backbone, .bf16 = false }, .{ .m = &draft, .bf16 = false } };
    for (cases) |cs| for (rows_list[0..nrows]) |R| {
        const m = cs.m;
        const slots = m.topk + 1;
        const P = R * slots;
        // routing: distinct routed experts a row, the shared expert last; some routed slots pruned (-1 or the drop id)
        const pick = try gpa.alloc(i32, P);
        defer gpa.free(pick);
        var pruned: usize = 0;
        for (0..R) |r| {
            var n: usize = 0;
            while (n < m.topk) {
                const e: i32 = @intCast(rng.next() % (m.Et - 1));
                if (std.mem.indexOfScalar(i32, pick[r * slots ..][0..n], e) != null) continue;
                pick[r * slots + n] = e;
                n += 1;
            }
            pick[r * slots + m.topk] = @intCast(m.Et - 1);
            // keep at least one routed slot; prune the others with probability 1/5
            for (1..m.topk) |s| if (rng.next() % 5 == 0) {
                pick[r * slots + s] = if (rng.next() % 2 == 0) -1 else @intCast(m.Et);
                pruned += 1;
            };
        }
        var pick_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pick));
        defer pick_d.free();
        var wts = try floats(gpa, d, &rng, P, 0.0, 0.5);
        defer wts.free();
        var x = blk: {
            const h = try gpa.alloc(u16, R * D);
            defer gpa.free(h);
            for (h) |*v| v.* = @truncate(@as(u32, @bitCast(-1.0 + 2.0 * rng.unit())) >> 16);
            break :blk try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(h));
        };
        defer x.free();
        var uids = try cuda.DeviceBuffer.alloc(d, 4 * P);
        defer uids.free();
        var ucount = try cuda.DeviceBuffer.alloc(d, 4);
        defer ucount.free();
        var members = try cuda.DeviceBuffer.alloc(d, 4 * P * R);
        defer members.free();
        var xg = try cuda.DeviceBuffer.alloc(d, 2 * P * D);
        defer xg.free();
        var xu = try cuda.DeviceBuffer.alloc(d, 2 * P * D);
        defer xu.free();
        try o.group(pick_d.ptr, uids.ptr, ucount.ptr, members.ptr, R, slots, m.Et, R);
        try o.rotIn(.bf16, x.ptr, D, pick_d.ptr, m.suh_g.ptr, m.suh_u.ptr, xg.ptr, xu.ptr, R, D, slots, m.Et);
        const tw = @max(exl3.epiTicketWords(.{ .x0 = 0, .x1 = 0, .tp0 = 0, .tp1 = 0, .k2_0 = 0, .k2_1 = 0, .uids = 0, .ucount = 0, .members = 0, .z = 0, .mats = 2, .K = D, .N = I, .P = P, .SK = SK_GU, .slots = slots, .maxm = R, .nexp = P, .lo = 4, .hi = 10 }, .gu), R * (D / 128));
        var ticket = try cuda.DeviceBuffer.alloc(d, 4 * tw);
        defer ticket.free();
        try ticket.fill8(0, null);
        // y starts with the same random words on both paths (a pruned slot's y is read, never written)
        const y0 = blk: {
            const h = try gpa.alloc(f32, P * D);
            for (h) |*v| v.* = rng.unit() - 0.5;
            break :blk h;
        };
        defer gpa.free(y0);
        var ref = try Outs.init(d, P, R, std.mem.sliceAsBytes(y0));
        defer ref.deinit();
        var got = try Outs.init(d, P, R, std.mem.sliceAsBytes(y0));
        defer got.deinit();
        const r: Run = .{ .o = o, .m = m, .R = R, .slots = slots, .pick = pick_d.ptr, .wts = wts.ptr, .uids = uids.ptr, .ucount = ucount.ptr, .members = members.ptr, .xg = xg.ptr, .xu = xu.ptr, .ticket = ticket.ptr, .bf16 = cs.bf16 };
        for (0..2) |_| {
            try r.unfused(&ref);
            try r.fused(&got);
        }
        try stream.synchronize();
        const label = try std.fmt.allocPrint(gpa, "x3ld-epi {s} {s} out, {d} rows", .{ m.name, if (cs.bf16) "bf16" else "fp32", R });
        defer gpa.free(label);
        std.debug.print("{s}: {d} pairs, {d} pruned\n", .{ label, P, pruned });
        // Xd, y, out (Z is scratch: the unfused down overwrites gate/up's, the fused down does not store its own)
        try sameDevice(gpa, "Xd", got.xd, ref.xd, 2 * P * I);
        try sameDevice(gpa, "y", got.y, ref.y, 4 * P * D);
        try sameDevice(gpa, "out", got.out, ref.out, (if (cs.bf16) @as(usize, 2) else 4) * R * D);
        {
            const tk = try gpa.alloc(i32, tw);
            defer gpa.free(tk);
            try ticket.download(0, std.mem.sliceAsBytes(tk));
            for (tk, 0..) |v, i| if (v != 0) {
                std.debug.print("FAIL {s}: ticket word {d} is {d} after the launches\n", .{ label, i, v });
                return error.TicketNotReset;
            };
        }
        const tu = try timeIt(stream, d, &r, &ref, false, reps);
        const tf = try timeIt(stream, d, &r, &got, true, reps);
        // the timed runs must leave the same bits too
        try stream.synchronize();
        try sameDevice(gpa, "out after the timed runs", got.out, ref.out, (if (cs.bf16) @as(usize, 2) else 4) * R * D);
        std.debug.print("  BITEXACT; unfused {d:.3} ms (4 launches), fused {d:.3} ms (2 launches): {d:.1} us ({d:.1}%)\n", .{ tu, tf, 1000.0 * (tu - tf), 100.0 * (tf - tu) / tu });
        if (cs.bf16 and m == &backbone) {
            total_u += tu;
            total_f += tf;
        }
    };
    check.pass("BITEXACT x3ld-epi: fused == unfused (Z, Xd, y, out) at {d} row counts x 3 cases; backbone bf16 sum unfused {d:.3} ms, fused {d:.3} ms", .{ nrows, total_u, total_f });
}

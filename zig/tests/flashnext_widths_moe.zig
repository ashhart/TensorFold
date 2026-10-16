//! tf-flashnext-widths, experts, embedding and n-gram tables: the recorded any-width kernels at each oQ width.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lib = @import("flashnext_widths_lib.zig");
const fz = tf.flashnext_replay;
const variants = fz.variants;
const Buf = fz.Buf;
const Run = fz.Run;
const a = lib.a;

const D = 2560;
const N = 640; // an expert's intermediate width
const E = 16; // experts held: the logits pick among these only
const R = 16;
const counts = [_]usize{ 1, 2, 3, 8, 16 };

fn up(device: mtl.Device, m: lib.Mat) ![3]Buf {
    return .{ .{ .b = try lib.upload(device, m.words) }, .{ .b = try lib.upload(device, std.mem.sliceAsBytes(m.scales)) }, .{ .b = try lib.upload(device, std.mem.sliceAsBytes(m.biases)) } };
}

/// Codes below 64, so every matrix also exists at 6/32 with the same values; `full`: codes over their whole range.
fn mat(n: usize, k: usize, w: [2]usize, seed: u64, full: bool) !lib.Mat {
    return lib.Mat.initBelow(n, k, @intCast(w[0]), w[1], seed, @min(@as(u32, if (full) 256 else 64), @as(u32, 1) << @intCast(w[0])));
}

/// Gate, up, shared gate, shared up, down and shared down at the routed and shared widths, and their 6/32 twins.
const Set = struct {
    e: [18]Buf,
    e6: [18]Buf,
    m: [6]lib.Mat, // gate, up, shared gate, shared up, down, shared down

    fn deinit(s: Set) void {
        for (s.e) |b| b.b.deinit();
        if (s.e6[0].b.id != s.e[0].b.id) for (s.e6) |b| b.b.deinit();
        for (s.m) |m| {
            m.deinit();
            a.free(m.codes);
        }
    }
};

fn set(device: mtl.Device, routed: [2]usize, shared: [2]usize, seed: u64, full: bool) !Set {
    const shapes = [_][3]usize{ .{ E * N, D, 0 }, .{ E * N, D, 0 }, .{ N, D, 1 }, .{ N, D, 1 }, .{ E * D, N, 0 }, .{ D, N, 1 } };
    var s: Set = undefined;
    for (shapes, 0..) |sh, j| {
        const m = try mat(sh[0], sh[1], if (sh[2] == 1) shared else routed, seed + j, full);
        const b = try up(device, m);
        const m6 = if (full) m else try m.recode(6, 32); // full-range 8-bit codes have no 6/32 twin
        defer if (!full) m6.deinit();
        const b6 = if (full) b else try up(device, m6);
        for (0..3) |k| {
            s.e[j * 3 + k] = b[k];
            s.e6[j * 3 + k] = b6[k];
        }
        s.m[j] = m;
    }
    return s;
}

const Io = struct { x: lib.Inputs, lg: mtl.Buffer };

fn inputs(device: mtl.Device) !Io {
    const lg = try device.buffer(R * 513 * 4, lib.opts);
    const v = lg.slice(f32, R * 513);
    for (0..R) |r| for (0..513) |e| {
        v[r * 513 + e] = if (e < E) 4.0 - @as(f32, @floatFromInt((e * 7 + r * 3) % E)) * 0.25 else if (e == 512) 0.5 else -100;
    };
    return .{ .x = try lib.Inputs.init(device, R, D, 0x3e), .lg = lg };
}

const Out = struct { act: mtl.Buffer, pick: mtl.Buffer, wts: mtl.Buffer, y: mtl.Buffer };

fn outputs(device: mtl.Device) !Out {
    return .{ .act = try lib.poisoned(device, R * 11 * N * 2), .pick = try lib.poisoned(device, R * 10 * 4), .wts = try lib.poisoned(device, R * 10 * 4), .y = try lib.poisoned(device, R * 11 * D * 2) };
}

/// Every window of `rows` rows through gate/up then down, at `w` (null: the recorded 6/32 kernels on `e`).
fn experts(r: *Run, io: Io, e: [18]Buf, rows: usize, o: Out, w: ?variants.Experts) !void {
    const cb = r.queue.commandBuffer();
    r.enc = cb.compute(.serial);
    var start: usize = 0;
    while (start + rows <= R) : (start += rows) {
        const rb = try r.buffer(16);
        rb.slice(i32, 1)[0] = @intCast(rows);
        r.rows = rows;
        const x: Buf = .{ .b = io.x.buf, .off = start * D * 2 };
        const lg: Buf = .{ .b = io.lg, .off = start * 513 * 4 };
        const act: Buf = .{ .b = o.act, .off = start * 11 * N * 2 };
        const pick: Buf = .{ .b = o.pick, .off = start * 40 };
        const wts: Buf = .{ .b = o.wts, .off = start * 40 };
        const y: Buf = .{ .b = o.y, .off = start * 11 * D * 2 };
        if (w) |ww| try variants.experts(r, "qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down", x, lg, e, act, pick, wts, .{ .b = rb }, y, ww) else {
            try r.call("qa_expert_gateup@moe.gate", &.{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11] }, &.{ act, pick, wts });
            try r.call("qa_expert_down_y@moe.down", &.{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], .{ .b = rb } }, &.{y});
        }
    }
    try lib.commit(cb, r.enc);
}

fn same4(t: *lib.Tally, label: []const u8, n: usize, want: Out, got: Out) bool {
    const rows = R / n * n;
    var ok = lib.sameRows(t, label, n, want.act.slice(u8, R * 11 * N * 2), got.act.slice(u8, R * 11 * N * 2), rows, 11 * N * 2);
    ok = lib.sameRows(t, label, n, want.pick.slice(u8, R * 40), got.pick.slice(u8, R * 40), rows, 40) and ok;
    ok = lib.sameRows(t, label, n, want.wts.slice(u8, R * 40), got.wts.slice(u8, R * 40), rows, 40) and ok;
    return lib.sameRows(t, label, n, want.y.slice(u8, R * 11 * D * 2), got.y.slice(u8, R * 11 * D * 2), rows, 11 * D * 2) and ok;
}

fn expertCheck(t: *lib.Tally, r: *Run, io: Io, routed: [2]usize, shared: [2]usize, seed: u64, full: bool) !void {
    var label_buf: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "experts routed q{d}/g{d} shared q{d}/g{d}{s}", .{ routed[0], routed[1], shared[0], shared[1], if (full) " full-range codes" else "" });
    const s = try set(r.device, routed, shared, seed, full);
    defer s.deinit();
    const w: variants.Experts = .{ .gu = .{ routed[0], routed[1], shared[0], shared[1] }, .down = .{ routed[0], routed[1], shared[0], shared[1] } };
    const one = try outputs(r.device);
    try experts(r, io, s.e, 1, one, w);
    // 6 and 8 bits read 8 codes a lane step, other widths 16 (0.6.6's lane_values): only the former sum as 6/32 does
    const as_recorded = !full and (routed[0] == 6 or routed[0] == 8) and (shared[0] == 6 or shared[0] == 8);
    var same = true;
    for (counts) |n| {
        const got = try outputs(r.device);
        try experts(r, io, s.e, n, got, w);
        if (as_recorded) {
            const rec = try outputs(r.device);
            try experts(r, io, s.e6, n, rec, null);
            same = same4(t, label, n, rec, got) and same;
        }
        same = same4(t, label, n, one, got) and same;
    }
    if (same) std.debug.print("RESULT {s}: every row's bits at 1-16 rows{s}\n", .{ label, if (as_recorded) ", and the recorded 6/32 kernels' bits on the same values" else "" });
    try gateUp(t, label, s, io, one);
    // the down products against float64 from the activations the gate/up launch wrote
    var worst = [2]lib.Worst{ .{}, .{} }; // against the right scales, and the negative control's 1% off
    for ([_]f64{ 1, 1.01 }, &worst) |skew, *wo| {
        const routed_terms = try s.m[4].skewed(skew).dense();
        const shared_terms = try s.m[5].skewed(skew).dense();
        defer for (routed_terms ++ shared_terms) |v| a.free(v);
        const act = one.act.slice(u16, R * 11 * N);
        const pick = one.pick.slice(u32, R * 10);
        const y = one.y.slice(u16, R * 11 * D);
        const xa = try a.alloc(f64, N);
        defer a.free(xa);
        for (0..R) |row| for (0..11) |slot| {
            for (xa, 0..) |*v, i| v.* = lib.fromBf(act[(row * 11 + slot) * N + i]);
            for (0..D) |d| {
                const ref = if (slot < 10) lib.dot(xa, routed_terms, pick[row * 10 + slot] * D + d) else lib.dot(xa, shared_terms, d);
                const got = lib.fromBf(y[(row * 11 + slot) * D + d]);
                const g = if (slot < 10) routed[1] else shared[1];
                wo.add(d, got, ref[0], lib.ratio(got, ref[0], ref[1], 32 + N / g + 32, true));
            }
        };
    }
    var down_label: [160]u8 = undefined;
    const dl = try std.fmt.bufPrint(&down_label, "{s} down", .{label});
    worst[0].report(t, dl);
    worst[1].control(t, dl);
}

/// Four rows' activations silu(x Wg) * (x Wu) against float64: sums' bounds through the activation and roundings.
fn gateUp(t: *lib.Tally, label: []const u8, s: Set, io: Io, one: Out) !void {
    var worst = [2]lib.Worst{ .{}, .{} }; // against the right scales, and the negative control's 1% off
    const act = one.act.slice(u16, R * 11 * N);
    const pick = one.pick.slice(u32, R * 10);
    for ([_]f64{ 1, 1.01 }, &worst) |skew, *w| for (0..4) |row| for (0..11) |slot| for (0..N) |n| {
        const x = io.x.row(row);
        const base: usize = if (slot < 10) pick[row * 10 + slot] * N else 0;
        const gm = (if (slot < 10) s.m[0] else s.m[2]).skewed(skew);
        const um = (if (slot < 10) s.m[1] else s.m[3]).skewed(skew);
        var gv: [2]f64 = .{ 0, 0 }; // the product and its magnitude sum |x| (s q + |b|)
        var uv: [2]f64 = .{ 0, 0 };
        for (0..D) |i| {
            const gq = gm.skew * lib.fromBf(gm.scales[(base + n) * (D / gm.group) + i / gm.group]) * @as(f64, @floatFromInt(gm.code(base + n, i)));
            const uq = um.skew * lib.fromBf(um.scales[(base + n) * (D / um.group) + i / um.group]) * @as(f64, @floatFromInt(um.code(base + n, i)));
            gv[0] += x[i] * gm.deq(base + n, i);
            gv[1] += @abs(x[i]) * (gq + @abs(lib.fromBf(gm.biases[(base + n) * (D / gm.group) + i / gm.group])));
            uv[0] += x[i] * um.deq(base + n, i);
            uv[1] += @abs(x[i]) * (uq + @abs(lib.fromBf(um.biases[(base + n) * (D / um.group) + i / um.group])));
        }
        const sig = 1 / (1 + @exp(-gv[0]));
        const silu = gv[0] * sig;
        const want = silu * uv[0];
        const steps: f64 = @floatFromInt(32 + D / gm.group + 16);
        const dg = steps * lib.unit * gv[1] + 0x1p-8 * @abs(gv[0]);
        const du = steps * lib.unit * uv[1] + 0x1p-8 * @abs(uv[0]);
        const tol = 1.1 * @abs(uv[0]) * dg + @abs(silu) * du + 0x1p-7 * @abs(want) + 1e-6;
        const got = lib.fromBf(act[(row * 11 + slot) * N + n]);
        w.add(n, got, want, @abs(got - want) / tol);
    };
    var gu_label: [160]u8 = undefined;
    const gl = try std.fmt.bufPrint(&gu_label, "{s} gate/up", .{label});
    worst[0].report(t, gl);
    worst[1].control(t, gl);
}

/// The embedding rows (4 stream copies, and the MTP head's one) and the n-gram lookups at width `w`.
fn lookups(t: *lib.Tally, r: *Run, w: [2]usize, seed: u64, full: bool) !void {
    var label_buf: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "embed and n-gram tables q{d}/g{d}{s}", .{ w[0], w[1], if (full) " full-range codes" else "" });
    var c: [2]variants.Const = undefined;
    const k = variants.width(&c, "BITS", "GS", w);
    const emb = try mat(64, D, w, seed, full);
    const e = try up(r.device, emb);
    const e6 = if (full) e else try up(r.device, try emb.recode(6, 32));
    const ids = try r.buffer(R * 16 * 4);
    for (ids.slice(u32, R * 16), 0..) |*v, i| v.* = @intCast((i * 37 + 5) % 64);
    var tabs: [26]Buf = undefined;
    var tabs6: [26]Buf = undefined;
    tabs[0] = .{ .b = ids };
    const starts = try r.buffer(32);
    for (starts.slice(u32, 8), 0..) |*v, g| v.* = @intCast(8 * g);
    tabs[1] = .{ .b = starts };
    var ple_mats: [8]lib.Mat = undefined;
    const tables: usize = if (160 % w[1] == 0) 3 else 2; // an n-gram row is 160 wide: groups of 32 only
    for (0..8) |g| {
        ple_mats[g] = try mat(8, 160, .{ w[0], if (tables == 3) w[1] else 32 }, seed + 1 + g, full);
        const b = try up(r.device, ple_mats[g]);
        const b6 = if (full) b else try up(r.device, try ple_mats[g].recode(6, 32));
        for (0..3) |j| {
            tabs[2 + 3 * g + j] = b[j];
            tabs6[2 + 3 * g + j] = b6[j];
        }
    }
    tabs6[0] = tabs[0];
    tabs6[1] = tabs[1];
    defer {
        for (e) |b| b.b.deinit();
        if (!full) for (e6) |b| b.b.deinit();
    }
    const roles = [_][]const u8{ "qa_embed_rows@embed", "mtp:qa_embed_rows@embed", "qa_ple_lookup@ple" };
    const sizes = [_]usize{ 4 * D * 2, D * 2, 16 * 160 * 2 };
    var same = true;
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    var ones: [3]mtl.Buffer = undefined; // each role's one-row outputs: a full-range pass's rows compare with them
    for (roles[0..tables], sizes[0..tables], 0..) |role, size, which| for (counts) |n| {
        const got = try lib.poisoned(r.device, R * size);
        const rec = try lib.poisoned(r.device, R * size);
        const cb = r.queue.commandBuffer();
        r.enc = cb.compute(.serial);
        var start: usize = 0;
        while (start + n <= R) : (start += n) {
            r.rows = n;
            const id: Buf = .{ .b = ids, .off = start * @as(usize, if (which == 2) 16 * 4 else 4) };
            const out: Buf = .{ .b = got, .off = start * size };
            const out6: Buf = .{ .b = rec, .off = start * size };
            if (which < 2) {
                try variants.callAs(r, role, n, &.{ id, e[0], e[1], e[2] }, &.{out}, k);
                try r.callAs(role, n, &.{ id, e6[0], e6[1], e6[2] }, &.{out6});
            } else {
                var tw = tabs;
                var tw6 = tabs6;
                tw[0] = id;
                tw6[0] = id;
                try variants.callAs(r, role, n, &tw, &.{out}, k);
                try r.callAs(role, n, &tw6, &.{out6});
            }
        }
        try lib.commit(cb, r.enc);
        if (n == 1) ones[which] = got;
        const want_rows = if (full) ones[which] else rec;
        same = lib.sameRows(t, label, n, want_rows.slice(u8, R * size), got.slice(u8, R * size), R / n * n, size) and same;
        if (n != 1) continue;
        // the values against s q + b in float64: two bf16 roundings at most
        const v = got.slice(u16, R * size / 2);
        for (0..R) |row| for (0..size / 2) |i| for ([_]f64{ 1, 1.01 }) |skew| {
            const want = if (which == 2) blk: {
                const head = i / 160;
                const id = ids.slice(u32, R * 16)[row * 16 + head];
                break :blk ple_mats[id / 8].skewed(skew).deq(id % 8, i % 160);
            } else emb.skewed(skew).deq(ids.slice(u32, R * 16)[row], i % D);
            const g = lib.fromBf(v[row * size / 2 + i]);
            (if (skew == 1) &worst else &ctrl).add(i, g, want, @abs(g - want) / (@abs(want) * 0x1p-7 + 1e-6));
        };
    };
    if (same) std.debug.print("RESULT {s}: {s} at 1-16 rows\n", .{ label, if (full) "every row's bits" else "the recorded 6/32 kernels' bits on the same values" });
    worst.report(t, label);
    ctrl.control(t, label);
}

pub fn run(t: *lib.Tally, r: *Run) !void {
    const io = try inputs(r.device);
    var seed: u64 = 0xe000;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        seed += 8;
        try expertCheck(t, r, io, .{ bits, group }, .{ 8, 128 }, seed, false);
        try lookups(t, r, .{ bits, group }, seed + 4, false);
    };
    try expertCheck(t, r, io, .{ 4, 32 }, .{ 4, 32 }, 0xe801, false); // the MTP head's experts in oQ2
    // 8-bit codes past 63 (the embedding and the shared expert are 8-bit in every oQ checkpoint): float64 and rows
    for ([_]usize{ 32, 64, 128 }, 0..) |group, i| {
        try expertCheck(t, r, io, .{ 8, group }, .{ 8, 128 }, 0xe900 + 8 * i, true);
        try lookups(t, r, .{ 8, group }, 0xe904 + 8 * i, true);
    }
}

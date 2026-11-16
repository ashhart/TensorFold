//! DeepSeek's release projections beyond gates 1-4: prompt == decode bytes, row independence, a slice cancellation.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kr = @import("dsv41_release");
const check = @import("check.zig");
const c = core.contract.spec;

const Form = kr.Form;
/// The launch shapes one tile set takes: a CTA a tile, one CTA looping all, warp tiles a warp each or in one CTA.
const Launch = enum { served, one_cta, warps, one_warp_cta };
const Case = struct { name: []const u8, form: Form, k: usize, n: usize, f32_out: bool };

/// One served projection a split: S = 8, 2, 1 for FP8, 8 and 1 for BF16.
const cases = [_]Case{
    .{ .name = "fp8 wq_a", .form = .fp8, .k = 5120, .n = 1280, .f32_out = false },
    .{ .name = "fp8 wo_b", .form = .fp8, .k = 2048, .n = 5120, .f32_out = false },
    .{ .name = "fp8 shared w2", .form = .fp8, .k = 576, .n = 5120, .f32_out = true },
    .{ .name = "bf16 compressor", .form = .bf16, .k = 5120, .n = 512, .f32_out = true },
    .{ .name = "bf16 ix_wk", .form = .bf16, .k = 512, .n = 128, .f32_out = false },
};

const prompt_rows = 2048;
const probe_rows = 100;

/// Device buffers for one case: activations of `rows` rows, the weight, partial scratch and one output.
const Bufs = struct {
    d: *const cuda.Driver,
    case: Case,
    act: cuda.DeviceBuffer,
    scales: cuda.DeviceBuffer,
    w: cuda.DeviceBuffer,
    ws: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,

    fn rowBytes(b: Bufs) usize {
        return b.case.k * @as(usize, if (b.case.form == .fp8) 1 else 2);
    }

    fn outBytes(b: Bufs) usize {
        return b.case.n * @as(usize, if (b.case.f32_out) 4 else 2);
    }

    fn init(gpu: check.Gpu, case: Case, rows: usize, rng: std.Random) !Bufs {
        const a = gpu.gpa;
        const es: usize = if (case.form == .fp8) 1 else 2;
        const w = try a.alloc(u8, case.n * case.k * es);
        defer a.free(w);
        randomize(case.form, w, rng);
        const ws = try a.alloc(u8, case.n / 32 * case.k / 32);
        defer a.free(ws);
        for (ws) |*v| v.* = 116 + rng.uintLessThan(u8, 12);
        var b: Bufs = undefined;
        b.d = gpu.d;
        b.case = case;
        b.act = try cuda.DeviceBuffer.alloc(gpu.d, rows * case.k * es);
        b.scales = try cuda.DeviceBuffer.alloc(gpu.d, rows * case.k / 32);
        b.w = try cuda.DeviceBuffer.fromHost(gpu.d, w);
        b.ws = try cuda.DeviceBuffer.fromHost(gpu.d, ws);
        b.part = try cuda.DeviceBuffer.alloc(gpu.d, @max(kr.partBytes(rows, case.n, case.k), 4));
        b.y = try cuda.DeviceBuffer.alloc(gpu.d, rows * b.outBytes());
        return b;
    }

    fn deinit(b: *Bufs) void {
        inline for (.{ "act", "scales", "w", "ws", "part", "y" }) |f| @field(b, f).free();
    }

    /// Activation rows `rows` of `act`/`scales` starting at row `first`, through the dispatching launch.
    fn launch(b: Bufs, ops: kr.Ops, first: usize, rows: usize) !void {
        return b.launchAs(ops, first, rows, .served);
    }

    fn launchAs(b: Bufs, ops: kr.Ops, first: usize, rows: usize, how: Launch) !void {
        const k = b.case.k;
        var p: kr.Proj = .{ .a = b.act.ptr, .a_off = first * b.rowBytes(), .a_ld = @intCast(k), .sa = b.scales.ptr, .sa_off = first * k / 32, .sa_ld = @intCast(k / 32), .w = b.w.ptr, .ws = b.ws.ptr, .y = b.y.ptr, .y_ld = @intCast(b.case.n), .part = b.part.ptr, .m = @intCast(rows), .n = @intCast(b.case.n), .k = @intCast(k), .f32 = @intFromBool(b.case.f32_out) };
        p.s = @intCast(kr.split(k, b.case.n));
        const warps: u32 = @intCast((rows + 15) / 16 * (b.case.n / 32) * @as(usize, @intCast(p.s)));
        // every launch starts on NaN outputs and partials: a launch that skips a tile cannot pass on old bytes
        try b.y.fill8(0xff, ops.s.handle);
        try b.part.fill8(0xff, ops.s.handle);
        try switch (how) {
            .served => ops.project(p, b.case.form, null),
            .one_cta => ops.project(p, b.case.form, 1),
            .warps => ops.projectWarps(p, b.case.form, (warps + 3) / 4),
            .one_warp_cta => ops.projectWarps(p, b.case.form, 1),
        };
        try ops.s.synchronize();
    }

    fn output(b: Bufs, out: []u8) !void {
        try b.y.download(0, out);
    }
};

/// Random finite codes (FP8) or bf16 values in [-2, 2) (BF16).
fn randomize(form: Form, bytes: []u8, rng: std.Random) void {
    if (form == .fp8) {
        for (bytes) |*v| {
            const r = rng.int(u8);
            v.* = if (r & 0x7f == 0x7f) r ^ 1 else r;
        }
    } else for (std.mem.bytesAsSlice(u16, @as([]align(1) u8, bytes))) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
}

/// Activations whose row i is probe row i % probe_rows, scales in [2^-9, 2^6].
fn upload(gpu: check.Gpu, b: Bufs, rows: usize, rng: std.Random) !void {
    const a = gpu.gpa;
    const rb = b.rowBytes();
    const gk = b.case.k / 32;
    const probe = try a.alloc(u8, probe_rows * rb);
    defer a.free(probe);
    randomize(b.case.form, probe, rng);
    const ps = try a.alloc(u8, probe_rows * gk);
    defer a.free(ps);
    for (ps) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
    const all = try a.alloc(u8, rows * rb);
    defer a.free(all);
    const sc = try a.alloc(u8, rows * gk);
    defer a.free(sc);
    for (0..rows) |r| {
        @memcpy(all[r * rb ..][0..rb], probe[(r % probe_rows) * rb ..][0..rb]);
        @memcpy(sc[r * gk ..][0..gk], ps[(r % probe_rows) * gk ..][0..gk]);
    }
    try b.act.upload(0, all);
    try b.scales.upload(0, sc);
}

/// The same rows at 1, 5, 16, 17 and 100 rows, and single rows at offsets 0, 15, 16 and 2047 of a 2048-row launch.
fn promptDecode(gpu: check.Gpu, ops: kr.Ops, case: Case, rng: std.Random) !void {
    const a = gpu.gpa;
    var b = try Bufs.init(gpu, case, prompt_rows, rng);
    defer b.deinit();
    try upload(gpu, b, prompt_rows, rng);
    const ob = b.outBytes();
    const big = try a.alloc(u8, prompt_rows * ob);
    defer a.free(big);
    try b.launch(ops, 0, prompt_rows);
    try b.output(big);
    var bad: usize = 0;
    for (probe_rows..prompt_rows) |r| {
        if (!std.mem.eql(u8, big[r * ob ..][0..ob], big[(r % probe_rows) * ob ..][0..ob])) bad += 1;
    }
    const small = try a.alloc(u8, probe_rows * ob);
    defer a.free(small);
    for ([_]usize{ 1, 5, 16, 17, 100 }) |m| {
        try b.launch(ops, 0, m);
        try b.output(small[0 .. m * ob]);
        if (!std.mem.eql(u8, small[0 .. m * ob], big[0 .. m * ob])) bad += 1;
    }
    for ([_]usize{ 0, 15, 16, 2047 }) |o| {
        try b.launch(ops, o % probe_rows, 1);
        try b.output(small[0..ob]);
        if (!std.mem.eql(u8, small[0..ob], big[o * ob ..][0..ob])) bad += 1;
    }
    try check.expect(bad == 0, "dsv41_rel prompt == decode {s} (K {d} N {d} S {d}): {d} launches differ", .{ case.name, case.k, case.n, kr.split(case.k, case.n), bad });
    check.pass("dsv41_rel prompt == decode {s} (K {d} N {d} S {d}): 2048-row launch, rows at 1/5/16/17/100 and offsets 0/15/16/2047 byte-identical", .{ case.name, case.k, case.n, kr.split(case.k, case.n) });
}

/// One row in every position of a 16-, 64- and 128-row launch (each tile size) beside 5 neighbour sets.
fn independence(gpu: check.Gpu, ops: kr.Ops, case: Case, rng: std.Random) !void {
    const a = gpu.gpa;
    var b = try Bufs.init(gpu, case, 128, rng);
    defer b.deinit();
    const rb = b.rowBytes();
    const gk = case.k / 32;
    const ob = b.outBytes();
    const row = try a.alloc(u8, rb);
    defer a.free(row);
    randomize(case.form, row, rng);
    const row_s: u8 = 121;
    const acts = try a.alloc(u8, 128 * rb);
    defer a.free(acts);
    const sc = try a.alloc(u8, 128 * gk);
    defer a.free(sc);
    const first = try a.alloc(u8, ob);
    defer a.free(first);
    const got = try a.alloc(u8, 128 * ob);
    defer a.free(got);
    var runs: usize = 0;
    var bad: usize = 0;
    for ([_]usize{ 16, 64, 128 }) |m| for (0..m) |p| for (0..5) |_| {
        randomize(case.form, acts[0 .. m * rb], rng);
        for (sc[0 .. m * gk]) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
        @memcpy(acts[p * rb ..][0..rb], row);
        @memset(sc[p * gk ..][0..gk], row_s);
        try b.act.upload(0, acts[0 .. m * rb]);
        try b.scales.upload(0, sc[0 .. m * gk]);
        try b.launch(ops, 0, m);
        try b.output(got[0 .. m * ob]);
        if (runs == 0) @memcpy(first, got[p * ob ..][0..ob]);
        if (!std.mem.eql(u8, first, got[p * ob ..][0..ob])) bad += 1;
        runs += 1;
    };
    try check.expect(bad == 0, "dsv41_rel independence {s}: {d} of {d} placements differ", .{ case.name, bad, runs });
    check.pass("dsv41_rel independence {s}: one row at all positions of a 16-, 64- and 128-row launch (decode, mid, prompt tiles), 5 neighbour sets each, {d} launches byte-identical", .{ case.name, runs });
}

/// Partials [2^25, 0, -2^25, 1] in four 32-wide groups of one column, summed at 1, 2 and 4 slices, FP32 out.
fn cancellation(gpu: check.Gpu, ops: kr.Ops) !void {
    const k = 128;
    const n = 32;
    var codes: [k]u8 = @splat(0);
    var scales = [4]u8{ 139, 127, 139, 127 };
    var w: [n * k]u8 = @splat(0);
    var ws = [4]u8{ 140, 127, 140, 127 };
    codes[0] = 0x38;
    codes[64] = 0xb8;
    codes[96] = 0x38;
    for ([_]usize{ 0, 64, 96 }) |i| w[i] = 0x38;
    var bufs: [5]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*x| x.free();
    for ([_][]const u8{ &codes, &scales, &w, &ws }, 0..) |bytes, i| {
        bufs[i] = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
        made += 1;
    }
    bufs[4] = try cuda.DeviceBuffer.alloc(gpu.d, 4 * n * 4 + n * 4);
    made += 1;
    const mag: f64 = 0x1p26 + 1;
    var bad: usize = 0;
    // the second pass negates every partial: a rounding toward zero and one toward +inf part there
    for ([_]f64{ 1, -1 }) |sign| for ([_]c_int{ 1, 2, 4 }) |s| {
        const fp64 = sign;
        var signed = codes;
        if (sign < 0) for ([_]usize{ 0, 64, 96 }) |i| {
            signed[i] ^= 0x80;
        };
        try bufs[0].upload(0, &signed);
        try bufs[4].fill8(0xff, ops.s.handle);
        const p: kr.Proj = .{ .a = bufs[0].ptr, .a_ld = k, .sa = bufs[1].ptr, .sa_ld = 4, .w = bufs[2].ptr, .ws = bufs[3].ptr, .y = bufs[4].ptr, .y_ld = n, .part = bufs[4].ptr + n * 4, .m = 1, .n = n, .k = k, .s = s, .f32 = 1 };
        try ops.project(p, .fp8, null);
        try ops.s.synchronize();
        var y: [n]f32 = undefined;
        try bufs[4].download(0, std.mem.asBytes(&y));
        const depth: f64 = @floatFromInt(@as(u32, 2 * (4 + 32)) + std.math.log2_int(u32, @intCast(s)));
        const bound = c.gamma(depth, 0x1p-24) * mag;
        const within = @abs(@as(f64, y[0]) - fp64) <= bound;
        for (y[1..]) |v| if (v != 0) {
            bad += 1;
        };
        if (!within) bad += 1;
        std.debug.print("INFO dsv41_rel cancellation {s}[2^25, 0, -2^25, 1] at {d} slices: {d} (float64 {d}, bound {d:.1}){s}\n", .{ if (sign < 0) "-" else "", s, y[0], fp64, bound, if (within) "" else " OUTSIDE" });
    };
    try check.expect(bad == 0, "dsv41_rel cancellation: {d} outputs leave the float64 bound", .{bad});
    check.pass("dsv41_rel cancellation: every slice count within the float64 bound of its depth", .{});
}

/// wo_a's two local groups as views of one [m, 8192] o row's codes and scales, each equal to a contiguous copy's.
fn groupViews(gpu: check.Gpu, ops: kr.Ops, rng: std.Random) !void {
    const a = gpu.gpa;
    const k = 4096;
    const n = 1024;
    const wide = 2 * k;
    const m = 17;
    const host = try a.alloc(u8, m * wide + m * wide / 32 + n * k + n * k / 1024);
    defer a.free(host);
    randomize(.fp8, host[0 .. m * wide], rng);
    for (host[m * wide ..][0 .. m * wide / 32]) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
    const w = host[m * wide + m * wide / 32 ..][0 .. n * k];
    randomize(.fp8, w, rng);
    for (host[host.len - n * k / 1024 ..]) |*v| v.* = 116 + rng.uintLessThan(u8, 12);
    var all = try cuda.DeviceBuffer.fromHost(gpu.d, host);
    defer all.free();
    const one = try a.alloc(u8, m * k + m * k / 32);
    defer a.free(one);
    var copy = try cuda.DeviceBuffer.alloc(gpu.d, one.len);
    defer copy.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * m * n * 2 + kr.partBytes(m, n, k));
    defer out.free();
    const got = try a.alloc(u8, 2 * m * n * 2);
    defer a.free(got);
    const part: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = out.ptr + 2 * m * n * 2, .len = kr.partBytes(m, n, k) };
    var bad: usize = 0;
    var flat: kr.Proj = undefined;
    for ([_]usize{ 5, 17 }) |rows| for (0..2) |grp| {
        for (0..rows) |r| {
            @memcpy(one[r * k ..][0..k], host[r * wide + grp * k ..][0..k]);
            @memcpy(one[m * k + r * k / 32 ..][0 .. k / 32], host[m * wide + r * wide / 32 + grp * k / 32 ..][0 .. k / 32]);
        }
        try copy.upload(0, one);
        try out.fill8(0xff, ops.s.handle);
        const base: kr.Proj = .{ .a = all.ptr, .a_off = grp * k, .a_ld = wide, .sa = all.ptr + m * wide, .sa_off = grp * k / 32, .sa_ld = wide / 32, .w = all.ptr + m * wide + m * wide / 32, .ws = all.ptr + host.len - n * k / 1024, .y = out.ptr, .y_ld = n, .part = out.ptr + 2 * m * n * 2, .m = @intCast(rows), .n = n, .k = k };
        try ops.fp8(base);
        flat = base;
        flat.a = copy.ptr;
        flat.a_off = 0;
        flat.a_ld = k;
        flat.sa = copy.ptr + m * k;
        flat.sa_off = 0;
        flat.sa_ld = k / 32;
        flat.y = out.ptr + m * n * 2;
        // base's partials are correct for flat too: poison them, so flat's sum reads only partials it wrote
        try part.fill8(0xff, ops.s.handle);
        try ops.fp8(flat);
        try ops.s.synchronize();
        try out.download(0, got);
        if (!std.mem.eql(u8, got[0 .. rows * n * 2], got[m * n * 2 ..][0 .. rows * n * 2])) bad += 1;
    };
    // control: the last flat launch's sum alone over poisoned partials (as if no slice were written) must differ
    try part.fill8(0xff, ops.s.handle);
    try ops.sum(.{ .part = flat.part, .y = flat.y, .y_ld = flat.y_ld, .m = flat.m, .n = flat.n, .s = @intCast(kr.split(k, n)) });
    try ops.s.synchronize();
    try out.download(0, got);
    const rows: usize = @intCast(flat.m);
    const caught = !std.mem.eql(u8, got[0 .. rows * n * 2], got[m * n * 2 ..][0 .. rows * n * 2]);
    if (!caught) std.debug.print("CONTROL MISSED dsv41_rel wo_a group views: a sum of unwritten partials matched\n", .{});
    try check.expect(bad == 0 and caught, "dsv41_rel wo_a group views: {d} of 4 launches differ from their contiguous copies, control {s}", .{ bad, if (caught) "red" else "missed" });
    check.pass("dsv41_rel wo_a group views: both local groups of an [m, 8192] o row at m 5 and 17 equal their contiguous launches on poisoned partials; a sum of unwritten partials differs", .{});
}

/// The served launch against one CTA looping every tile and the stage-free warp tiles: a persistent kernel's bytes.
fn launchShapes(gpu: check.Gpu, ops: kr.Ops, case: Case, rng: std.Random) !void {
    const a = gpu.gpa;
    var b = try Bufs.init(gpu, case, probe_rows, rng);
    defer b.deinit();
    try upload(gpu, b, probe_rows, rng);
    const ob = b.outBytes();
    const want = try a.alloc(u8, probe_rows * ob);
    defer a.free(want);
    const got = try a.alloc(u8, probe_rows * ob);
    defer a.free(got);
    var bad: usize = 0;
    for ([_]usize{ 5, 40, 100 }) |m| {
        try b.launchAs(ops, 0, m, .served);
        try b.output(want[0 .. m * ob]);
        for ([_]Launch{ .one_cta, .warps, .one_warp_cta }) |how| {
            try b.launchAs(ops, 0, m, how);
            try b.output(got[0 .. m * ob]);
            if (!std.mem.eql(u8, want[0 .. m * ob], got[0 .. m * ob])) {
                std.debug.print("FAIL dsv41_rel launch shapes {s} at {d} rows: {t} differs\n", .{ case.name, m, how });
                bad += 1;
            }
        }
    }
    // control: a launch without the last row tile must differ from the served bytes, else the poison proves nothing
    try b.launchAs(ops, 0, 84, .served);
    try b.output(got[0 .. 100 * ob]);
    const caught = !std.mem.eql(u8, want[0 .. 100 * ob], got[0 .. 100 * ob]);
    if (!caught) std.debug.print("CONTROL MISSED dsv41_rel launch shapes {s}: an 84-row launch matched 100 rows\n", .{case.name});
    try check.expect(bad == 0 and caught, "dsv41_rel launch shapes {s}: {d} launches differ, control {s}", .{ case.name, bad, if (caught) "red" else "missed" });
    check.pass("dsv41_rel launch shapes {s}: one CTA over every tile and the warp tiles (one CTA and a warp a tile) equal the served bytes at 5, 40 and 100 rows on poisoned outputs; a launch missing a row tile differs", .{case.name});
}

/// Every check, on sm_121 (the release images hold no other SM).
pub fn run(gpu: check.Gpu, ops: kr.Ops) !void {
    var prng = std.Random.DefaultPrng.init(0x7e1);
    const rng = prng.random();
    for (cases) |case| try promptDecode(gpu, ops, case, rng);
    for ([_]usize{ 0, 2, 3 }) |i| try independence(gpu, ops, cases[i], rng);
    for (cases) |case| try launchShapes(gpu, ops, case, rng);
    try groupViews(gpu, ops, rng);
    try cancellation(gpu, ops);
}

/// kernel-contract's release checks: on a full run, or when a word of `only` is part of "dsv41_rel".
pub fn suite(gpu: check.Gpu, only: []const []const u8) !void {
    const wanted = only.len == 0 or for (only) |w| {
        if (std.mem.indexOf(u8, "dsv41_rel", w) != null) break true;
    } else false;
    if (!wanted) return;
    if (try gpu.ctx.capability() != 121) return std.debug.print("N/A  dsv41_rel checks: the release images hold sm_121a only\n", .{});
    var mods: [2]cuda.Module = undefined;
    var f = try @import("contract_release.zig").load(gpu.d, &mods) orelse return error.ReleaseImagesNotBuilt;
    defer for (&mods) |*m| m.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try run(gpu, .{ .f = &f, .s = s });
}

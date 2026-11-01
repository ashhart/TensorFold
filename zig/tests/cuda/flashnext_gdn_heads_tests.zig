//! tf-cuda-test flashnext-gdn-heads: each DeltaNet twin on a rank's heads gives the full kernel's bytes for them.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("flashnext_rig.zig");
const lay = @import("flashnext_gdn_heads.zig");

const Rig = rig_.Rig;
const Heads = lay.Heads;
const Part = lay.Part;

/// Token rows: one, a few, a stage of 16, the TPR seam's either side, and the default chunk.
const counts = [_]usize{ 1, 4, 16, 63, 64, 1536 };

/// A pass's inputs, host bytes in one layout: projection rows, conv window and taps, gates, norm weight, state.
const In = struct { p: []const u8, cs: []const u8, cw: []const u8, a_log: []const u8, dt: []const u8, nw: []const u8, st: []const u8 };

const Out = struct { what: []const u8, part: Part, elem: usize = 4 };

/// Front's rows, then each form's readouts, state, gated-norm rows and their sums (plain, _to, TPR, fused).
const outs = blk: {
    var o: [4 + 4 * 4]Out = undefined;
    o[0..4].* = .{ .{ .what = "front q", .part = .keys }, .{ .what = "front k", .part = .keys }, .{ .what = "front v", .part = .values }, .{ .what = "front decay and beta", .part = .gb } };
    for ([_][]const u8{ "plain", "_to", "TPR", "fused" }, 0..) |form, i| {
        o[4 + 4 * i] = .{ .what = form ++ " readouts", .part = .values };
        o[5 + 4 * i] = .{ .what = form ++ " state", .part = .state };
        o[6 + 4 * i] = .{ .what = form ++ " gated-norm rows", .part = .values, .elem = 2 };
        o[7 + 4 * i] = .{ .what = form ++ " sums", .part = .sums };
    }
    break :blk o;
};

fn upload(r: *Rig, ptr: u64, bytes: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

/// Every form's launches for the heads h over m rows, as the pass makes them; the outputs in `outs`' order.
fn pass(r: *Rig, f: *const lay.Fns, h: Heads, m: usize, in: In) ![outs.len]u64 {
    const s = r.s;
    var o: [outs.len]u64 = undefined;
    for (outs, &o) |x, *p| p.* = try r.out(lay.bytes(x.part, x.elem, m, h));
    const p = try r.dev(in.p);
    const cs = try r.dev(in.cs);
    const cw = try r.dev(in.cw);
    const a_log = try r.dev(in.a_log);
    const dt = try r.dev(in.dt);
    const nw = try r.dev(in.nw);
    const st = try r.dev(in.st);
    try lay.front(f.of(.front, h), s, h, m, p, cs, cw, a_log, dt, o[0], o[1], o[2], o[3]);
    for ([_]lay.Kind{ .rows, .rows_to, .tpr }, 0..) |kind, i| {
        const b = 4 + 4 * i;
        if (kind == .rows) try upload(r, o[b + 1], in.st); // in place: the state starts where it lies
        const from = if (kind == .rows) o[b + 1] else st;
        try lay.chain(f.of(kind, h), s, kind, h, m, o[0], o[1], o[2], o[3], from, o[b + 1], o[b]);
        try lay.norm(f.of(.norm, h), s, h, m, o[b], p, nw, o[b + 2], o[b + 3]);
    }
    try upload(r, o[17], in.st);
    try lay.fused(f.of(.fused, h), s, h, m, p, cs, cw, a_log, dt, o[17], o[17], nw, o[18], o[19], o[16]);
    try s.synchronize();
    return o;
}

/// Rank r's inputs gathered from the full layout (the norm weight is whole on both ranks).
fn gathered(a: std.mem.Allocator, in: In, r: usize) !In {
    return .{
        .p = try lay.gather(a, in.p, 2, .proj, .half, r),
        .cs = try lay.gather(a, in.cs, 2, .conv, .half, r),
        .cw = try lay.gather(a, in.cw, 2, .taps, .half, r),
        .a_log = try lay.gather(a, in.a_log, 4, .gates, .half, r),
        .dt = try lay.gather(a, in.dt, 4, .gates, .half, r),
        .nw = in.nw,
        .st = try lay.gather(a, in.st, 4, .state, .half, r),
    };
}

/// Directed non-finites affect separate heads of both ranks; other heads retain the finite random inputs.
fn mixed(a: std.mem.Allocator, in: In, m: usize) !In {
    const p = try a.dupe(u8, in.p);
    const st = try a.dupe(u8, in.st);
    const h = Heads.half;
    for (0..2) |rank| {
        const q = lay.fullIndex(.proj, h, rank, 3);
        const v = (m / 2) * Heads.full.projWidth() + lay.fullIndex(.proj, h, rank, 2 * h.nk * lay.dk + 4 * lay.dv + 7);
        const z = (m - 1) * Heads.full.projWidth() + lay.fullIndex(.proj, h, rank, h.convDim() + 7 * lay.dv + 5);
        std.mem.writeInt(u16, p[q * 2 ..][0..2], 0x7fc1, .little);
        std.mem.writeInt(u16, p[v * 2 ..][0..2], 0x7f80, .little);
        std.mem.writeInt(u16, p[z * 2 ..][0..2], 0xff80, .little);
        const state = lay.fullIndex(.state, h, rank, 15 * lay.dv * lay.dk + 7);
        std.mem.writeInt(u32, st[state * 4 ..][0..4], 0xff800000, .little);
    }
    var out = in;
    out.p = p;
    out.st = st;
    return out;
}

/// Each rank must retain finite witnesses as well as observe the planted non-finites.
fn expectMixed(bytes: []const u8, elem: usize) !void {
    var finite: usize = 0;
    var nonfinite: usize = 0;
    var at: usize = 0;
    while (at < bytes.len) : (at += elem) {
        const bad = if (elem == 2)
            std.mem.readInt(u16, bytes[at..][0..2], .little) & 0x7f80 == 0x7f80
        else
            std.mem.readInt(u32, bytes[at..][0..4], .little) & 0x7f800000 == 0x7f800000;
        if (bad) nonfinite += 1 else finite += 1;
    }
    try check.expect(finite > 0 and nonfinite > 0, "mixed GDN fixture needs finite and non-finite outputs on each rank", .{});
}

fn compare(r: *Rig, f: *const lay.Fns, oracle: *const lay.Fns, m: usize, in: In, frozen: bool, bad: bool) !usize {
    const a = r.a;
    const want = try pass(r, oracle, .full, m, in);
    var checks: usize = 0;
    if (frozen) {
        const got = try pass(r, f, .full, m, in);
        for (outs, want, got) |x, w, mine| {
            const len = lay.bytes(x.part, x.elem, m, .full);
            const what = try std.fmt.allocPrint(a, "flashnext-gdn-heads frozen {s} {s}, full, {d} rows", .{ if (bad) "mixed" else "finite", x.what, m });
            try check.sameBytes(what, try r.back(u8, mine, len), try r.back(u8, w, len));
            checks += 1;
        }
    }
    for (0..2) |rank| {
        const got = try pass(r, f, .half, m, try gathered(a, in, rank));
        for (outs, want, got, 0..) |x, w, mine, i| {
            const whole = try r.back(u8, w, lay.bytes(x.part, x.elem, m, .full));
            const theirs = try lay.gather(a, whole, x.elem, x.part, .half, rank);
            if (bad and (i == 0 or i == 6)) try expectMixed(theirs, x.elem);
            const what = try std.fmt.allocPrint(a, "flashnext-gdn-heads {s}{s}, rank {d}, {d} rows", .{ if (bad) "mixed " else "", x.what, rank, m });
            try check.sameBytes(what, try r.back(u8, mine, theirs.len), theirs);
            checks += 1;
        }
    }
    return checks;
}

/// The existing embedded full-versus-half check; its finite inputs and 240 comparisons are unchanged.
pub fn run(gpu: check.Gpu) !void {
    return runWith(gpu, null);
}

/// Freeze the full-head oracle to a separate image while retaining the embedded candidate handles.
pub fn runBaseline(gpu: check.Gpu, path: []const u8) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(64 << 20));
    defer gpu.gpa.free(raw);
    const image = try gpu.gpa.alignedAlloc(u8, .@"8", raw.len);
    defer gpu.gpa.free(image);
    @memcpy(image, raw);
    var baseline = try cuda.Module.load(gpu.d, image);
    defer baseline.unload();
    try runWith(gpu, baseline);
}

fn runWith(gpu: check.Gpu, baseline: ?cuda.Module) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(61);
    var r = try Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const f = try lay.Fns.load(batch);
    const oracle = if (baseline) |b| try f.withFull(b) else f;
    const full = Heads.full;
    var checks: usize = 0;
    for (counts) |m| {
        const in: In = .{
            .p = std.mem.sliceAsBytes(r.bfs(m * full.projWidth(), 1)),
            .cs = std.mem.sliceAsBytes(r.bfs((lay.taps - 1) * full.convDim(), 1)),
            .cw = std.mem.sliceAsBytes(r.bfs(full.convDim() * lay.taps, 0.5)),
            .a_log = std.mem.sliceAsBytes(r.f32s(full.nv, -1, 1)),
            .dt = std.mem.sliceAsBytes(r.f32s(full.nv, -1, 1)),
            .nw = std.mem.sliceAsBytes(r.bfs(lay.dv, 1.5)),
            .st = std.mem.sliceAsBytes(r.f32s(full.nv * lay.dv * lay.dk, -0.5, 0.5)),
        };
        checks += try compare(&r, &f, &oracle, m, in, baseline != null, false);
        if (baseline != null) checks += try compare(&r, &f, &oracle, m, try mixed(a, in, m), true, true);
    }
    if (baseline != null) return check.pass("flashnext-gdn-heads frozen baseline: {d} finite and directed mixed outputs " ++
        "of the candidate full kernels and both half-head layouts equal the external image's full kernels at 1 to 1,536 rows", .{checks});
    check.pass("flashnext-gdn-heads: {d} outputs of the _h8 front, rows, _to, TPR, norm and fused kernels on each rank's heads " ++
        "equal the full kernels' bytes for those heads at 1 to 1,536 rows", .{checks});
}

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

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(61);
    var r = try Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const f = try lay.Fns.load(batch);
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
        const want = try pass(&r, &f, .full, m, in);
        for (0..2) |rank| {
            const got = try pass(&r, &f, .half, m, try gathered(a, in, rank));
            for (outs, want, got) |x, w, mine| {
                const whole = try r.back(u8, w, lay.bytes(x.part, x.elem, m, .full));
                const theirs = try lay.gather(a, whole, x.elem, x.part, .half, rank);
                const what = try std.fmt.allocPrint(a, "flashnext-gdn-heads {s}, rank {d}, {d} rows", .{ x.what, rank, m });
                try check.sameBytes(what, try r.back(u8, mine, theirs.len), theirs);
                checks += 1;
            }
        }
    }
    check.pass("flashnext-gdn-heads: {d} outputs of the _h8 front, rows, _to, TPR, norm and fused kernels on each rank's heads " ++
        "equal the full kernels' bytes for those heads at 1 to 1,536 rows", .{checks});
}

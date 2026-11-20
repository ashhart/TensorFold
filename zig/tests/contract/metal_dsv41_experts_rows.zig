//! DeepSeek V4.1's decode experts: a launch's rows equal each row launched alone, bit for bit, at every routing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_experts_ref.zig");

const ex = tf.dsv41_metal.experts;
const topk = ref.topk;
/// Experts the table holds: enough for 16 rows on disjoint experts.
const experts = 96;
/// Bytes between two experts' codes inside one buffer (their scales: a sixteenth of it), so 96 experts fit in few MB.
const stride = 4096;

fn at(r: rig_.Ref) ex.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

fn address(r: rig_.Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}

/// Random MXFP4 codes and scales for w1, w3 [width][hidden] and w2 [hidden][width], each expert `stride` further on.
const World = struct {
    refs: [6]rig_.Ref,
    table: rig_.Ref,

    fn init(a: std.mem.Allocator, rig: *rig_.Rig, rng: std.Random) !World {
        var w: World = undefined;
        for (0..3) |i| {
            const codes = try a.alloc(u8, ex.width * ex.hidden / 2 + (experts - 1) * stride);
            defer a.free(codes);
            const scales = try a.alloc(u8, ex.width * ex.hidden / 32 + (experts - 1) * stride / 16);
            defer a.free(scales);
            rng.bytes(codes);
            for (scales) |*s| s.* = 119 + rng.uintLessThan(u8, 7);
            w.refs[2 * i] = try rig.input(codes);
            w.refs[2 * i + 1] = try rig.input(scales);
        }
        var table: [experts]ex.ExpertPtrs = undefined;
        for (&table, 0..) |*t, e| {
            var v: [6]u64 = undefined;
            for (&v, w.refs, 0..) |*y, r, i| y.* = address(r) + e * (if (i % 2 == 0) @as(usize, stride) else stride / 16);
            t.* = .{ .w1p = v[0], .w1s = v[1], .w3p = v[2], .w3s = v[3], .w2p = v[4], .w2s = v[5] };
        }
        w.table = try rig.input(std.mem.sliceAsBytes(&table));
        return w;
    }

    fn backing(w: *const World) [6]mtl.Buffer {
        var b: [6]mtl.Buffer = undefined;
        for (&b, w.refs) |*x, r| x.* = r.buf;
        return b;
    }
};

/// One launch's results read back: act and out by pair position (bf16), y by row (fp32), where by pair.
const Out = struct {
    act: []u16,
    out: []u16,
    y: []u32,
    where: []u32,

    fn deinit(o: Out, a: std.mem.Allocator) void {
        a.free(o.act);
        a.free(o.out);
        a.free(o.y);
        a.free(o.where);
    }
};

/// Plan, act_quant, gate/up, down and combine of `rows` rows, as the decode layer encodes them.
fn launch(a: std.mem.Allocator, rig: *rig_.Rig, pipes: *const ex.Pipelines, w: *const World, x: []const u16, ids: []const u32, wts: []const f32) !Out {
    const rows: u32 = @intCast(ids.len / topk);
    const np = ids.len;
    const call: ex.Call = .{ .rows = rows, .experts = experts };
    const b = ex.bytes(call);
    var sc: [6]rig_.Ref = undefined;
    for (&sc, b) |*s, len| s.* = try rig.scratch(len);
    const ones = try a.alloc(f32, np);
    defer a.free(ones);
    @memset(ones, 1);
    const route: ex.Route = .{ .ids = at(try rig.input(std.mem.sliceAsBytes(ids))), .weights = at(try rig.input(std.mem.sliceAsBytes(wts))), .slots = at(sc[0]), .nslots = at(sc[1]), .pairs = at(sc[2]), .where = at(sc[3]) };
    const xq = try rig.scratch(2 * rows * ex.hidden);
    const xqs = try rig.scratch(rows * ex.hidden / 8);
    const y = try rig.scratch(4 * rows * ex.hidden);
    const unit = at(try rig.input(std.mem.sliceAsBytes(ones)));
    const back = w.backing();
    const pass = rig.begin();
    ex.use(pass.e, &back);
    ex.plan(pipes, pass.e, route, call);
    ex.quant(pipes, pass.e, at(try rig.input(std.mem.sliceAsBytes(x))), at(xq), at(xqs), call);
    pass.e.barrier();
    ex.upCoded(pipes, pass.e, at(xq), at(xqs), at(w.table), route, at(sc[4]), call);
    pass.e.barrier();
    ex.down(pipes, pass.e, at(sc[4]), at(w.table), route, at(sc[5]), call);
    pass.e.barrier();
    ex.combine(pipes, pass.e, at(sc[5]), route, unit, at(y), true, call);
    try rig.finish(pass);
    return .{
        .act = try a.dupe(u16, @alignCast(std.mem.bytesAsSlice(u16, rig.bytes(sc[4], 2 * np * ex.width)))),
        .out = try a.dupe(u16, @alignCast(std.mem.bytesAsSlice(u16, rig.bytes(sc[5], 2 * np * ex.hidden)))),
        .y = try a.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, rig.bytes(y, 4 * rows * ex.hidden)))),
        .where = try a.dupe(u32, @alignCast(std.mem.bytesAsSlice(u32, rig.bytes(sc[3], 4 * np)))),
    };
}

/// Values (bf16 or fp32 bits) that differ, two NaNs counting as equal.
fn differ(comptime T: type, x: []const T, y: []const T) usize {
    const nan_at: T = if (T == u16) 0x7f80 else 0x7f800000;
    const mag: T = if (T == u16) 0x7fff else 0x7fffffff;
    var n: usize = 0;
    for (x, y) |p, q| {
        if (p != q and !((p & mag) > nan_at and (q & mag) > nan_at)) n += 1;
    }
    return n;
}

const Case = struct { how: ref.Routing, rows: u32, nonfinite: bool };

/// Differing values between one launch of every row and each row launched alone.
fn case(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipes: *const ex.Pipelines, cs: Case, rng: std.Random) !usize {
    var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const w = try World.init(a, &rig, rng);
    const plan = try ref.Plan.routed(a, cs.rows, experts, rng, cs.how);
    defer plan.deinit();
    const H = ex.hidden;
    const x = try a.alloc(u16, cs.rows * H);
    defer a.free(x);
    ref.fillX(x, rng);
    if (cs.nonfinite) ref.plantNonfinite(x, cs.rows, H);
    const whole = try launch(a, &rig, pipes, &w, x, plan.ids, plan.weights);
    defer whole.deinit(a);
    var bad: usize = 0;
    for (0..cs.rows) |r| {
        const one = try launch(a, &rig, pipes, &w, x[r * H ..][0..H], plan.ids[r * topk ..][0..topk], plan.weights[r * topk ..][0..topk]);
        defer one.deinit(a);
        for (0..topk) |i| {
            const pw = whole.where[r * topk + i];
            const po = one.where[i];
            bad += differ(u16, whole.act[pw * ex.width ..][0..ex.width], one.act[po * ex.width ..][0..ex.width]);
            bad += differ(u16, whole.out[pw * H ..][0..H], one.out[po * H ..][0..H]);
        }
        bad += differ(u32, whole.y[r * H ..][0..H], one.y[0..H]);
    }
    return bad;
}

const rows_list = [_]u32{ 1, 2, 3, 4, 5, 7, 8, 9, 16 };

/// Every case at every routing; `planted`: the source with TF_CTL `planted`, which must differ somewhere.
fn sweep(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64, planted: u8) !usize {
    const head = if (planted != 0) try std.fmt.allocPrint(a, "#define TF_CTL {d}\n", .{planted}) else "";
    defer if (planted != 0) a.free(head);
    const text = try ex.source(a, head, &.{});
    defer a.free(text);
    var pipes = try ex.Pipelines.loadSource(device, text);
    defer pipes.deinit();
    var prng = std.Random.DefaultPrng.init(seed ^ 0x0e0e);
    const rng = prng.random();
    var bad: usize = 0;
    var cases: usize = 0;
    for ([_]ref.Routing{ .same, .disjoint, .random }) |how| for (rows_list) |rows| for ([_]bool{ false, true }) |nonfinite| {
        if (planted != 0 and nonfinite) continue;
        const n = try case(a, device, queue, &pipes, .{ .how = how, .rows = rows, .nonfinite = nonfinite }, rng);
        cases += 1;
        if (n > 0 and planted == 0) std.debug.print("FAIL dsv41 experts rows: {t} routing, {d} rows{s}: {d} values differ from one row a launch\n", .{ how, rows, if (nonfinite) " (non-finite planted)" else "", n });
        bad += n;
    };
    if (planted == 0) std.debug.print("{s} dsv41 experts rows: same/disjoint/random routing at 1-16 rows, finite and NaN/Inf planted, {d} cases: act, out and y equal one row a launch ({d} values differ)\n", .{ if (bad == 0) "PASS" else "FAIL", cases, bad });
    return bad;
}

/// The equality sweep, then its control: rows reading their pass's first row must differ (TF_CTL 6).
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64) !usize {
    const bad = try sweep(a, device, queue, seed, 0);
    const planted = try sweep(a, device, queue, seed, 6);
    if (planted > 0) std.debug.print("CONTROL dsv41 experts rows control: rows read their pass's first row: red ({d} values differ), as planted\n", .{planted}) else std.debug.print("CONTROL MISSED dsv41 experts rows control: rows read their pass's first row stayed equal\n", .{});
    return bad + @intFromBool(planted == 0);
}

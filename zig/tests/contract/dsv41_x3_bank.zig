//! DeepSeek V4.1 EXL3 routed experts for the contract: a bank of mixed-width experts on host and GPU, the serial chain.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
const eref = @import("dsv41_experts_ref.zig");

pub const x3 = tf.dsv41_metal.experts_exl3;
pub const ref = x3.ref;
const xl = tf.exl3_projection;
const fmt = xl.format;
pub const topk = eref.topk;
pub const limit = eref.limit;

pub const Kind = enum { gate, up, down };
/// Widths an expert's matrices take: `mul1` the 2.8 bpw pack's (2 to 5 bits by expert), `books` each codebook in turn.
pub const Mix = enum { mul1, books };
/// `random`: trellis words and fp16 scales at random; `stress`: every state 0 (|w| 0.7 to 3.5) and unit scales.
pub const Fill = enum { random, stress };

fn form(mix: Mix, kind: Kind, e: usize) struct { fmt.Width, fmt.Codebook } {
    const i = e + @intFromBool(kind == .down);
    return switch (mix) {
        .mul1 => .{ .{ .halves = @intCast(4 + 2 * (i % 4)) }, .mul1 },
        .books => switch (i % 3) {
            0 => .{ .{ .halves = 5 }, .mul1 },
            1 => .{ .{ .halves = 4 }, .mcg },
            else => .{ .{ .halves = 6 }, .inst3 },
        },
    };
}

/// `count` experts' matrices of the kinds asked for; words and scales of a kind back to back, expert by expert.
pub const Bank = struct {
    a: std.mem.Allocator,
    count: usize,
    hidden: usize,
    width: usize,
    mix: Mix,
    words: [3][]i16 = .{ &.{}, &.{}, &.{} },
    su: [3][]f16 = .{ &.{}, &.{}, &.{} },
    sv: [3][]f16 = .{ &.{}, &.{}, &.{} },

    pub fn dims(b: *const Bank, kind: Kind) [2]usize {
        return if (kind == .down) .{ b.width, b.hidden } else .{ b.hidden, b.width };
    }

    fn wordsOf(b: *const Bank, kind: Kind, e: usize) usize {
        const d = b.dims(kind);
        return (d[0] / 16) * (d[1] / 16) * form(b.mix, kind, e)[0].tileWords();
    }

    fn start(b: *const Bank, kind: Kind, e: usize) usize {
        var s: usize = 0;
        for (0..e) |i| s += b.wordsOf(kind, i);
        return s;
    }

    pub fn init(a: std.mem.Allocator, count: usize, hidden: usize, width: usize, mix: Mix, kinds: []const Kind, fill: Fill, rng: std.Random) !Bank {
        var b: Bank = .{ .a = a, .count = count, .hidden = hidden, .width = width, .mix = mix };
        errdefer b.deinit();
        for (kinds) |kind| {
            const i = @backingInt(kind);
            const d = b.dims(kind);
            b.words[i] = try a.alloc(i16, b.start(kind, count));
            b.su[i] = try a.alloc(f16, count * d[0]);
            b.sv[i] = try a.alloc(f16, count * d[1]);
            if (fill == .stress) {
                @memset(b.words[i], 0);
                @memset(b.su[i], 1);
                @memset(b.sv[i], 1);
                continue;
            }
            for (b.words[i]) |*w| w.* = rng.int(i16);
            for (b.su[i]) |*s| s.* = @floatCast(rng.float(f32) * 2 - 1);
            for (b.sv[i]) |*s| s.* = @floatCast(rng.float(f32) * 2 - 1);
        }
        return b;
    }

    pub fn deinit(b: *Bank) void {
        for (0..3) |i| {
            b.a.free(b.words[i]);
            b.a.free(b.su[i]);
            b.a.free(b.sv[i]);
        }
    }

    pub fn matrix(b: *const Bank, kind: Kind, e: usize) ref.Matrix {
        const i = @backingInt(kind);
        const d = b.dims(kind);
        const f = form(b.mix, kind, e);
        return .{ .width = f[0], .book = f[1], .trellis = b.words[i][b.start(kind, e)..][0..b.wordsOf(kind, e)], .su = b.su[i][e * d[0] ..][0..d[0]], .sv = b.sv[i][e * d[1] ..][0..d[1]] };
    }

    /// Expert `e`'s matrix of a kind as exl3_projection views into the uploaded words and scales.
    pub fn view(b: *const Bank, u: Upload, kind: Kind, e: usize) xl.Matrix {
        const r = u.refs[@backingInt(kind)].?;
        const d = b.dims(kind);
        const f = form(b.mix, kind, e);
        return .{
            .trellis = .{ .buffer = r[0].buf, .offset = r[0].off + 2 * b.start(kind, e) },
            .suh = .{ .buffer = r[1].buf, .offset = r[1].off + 2 * e * d[0] },
            .svh = .{ .buffer = r[2].buf, .offset = r[2].off + 2 * e * d[1] },
            .k = d[0],
            .n = d[1],
            .width = f[0],
            .codebook = f[1],
        };
    }

    /// The kinds held, uploaded: their descriptor tables, and every buffer the tables point into (for useResource).
    pub fn upload(b: *const Bank, rig: *rig_.Rig, backing: *std.ArrayList(mtl.Buffer)) !Upload {
        var u: Upload = .{ .tables = undefined };
        var tables: [3]?rig_.Ref = .{ null, null, null };
        for (0..3) |i| {
            if (b.words[i].len == 0) continue;
            const kind: Kind = @fromBackingInt(@intCast(i));
            const r = [3]rig_.Ref{ try rig.input(std.mem.sliceAsBytes(b.words[i])), try rig.input(std.mem.sliceAsBytes(b.su[i])), try rig.input(std.mem.sliceAsBytes(b.sv[i])) };
            for (r) |x| try backing.append(b.a, x.buf);
            u.refs[i] = r;
            const table = try b.a.alloc(xl.ExpertDescriptor, b.count);
            defer b.a.free(table);
            for (table, 0..) |*t, e| t.* = try xl.descriptor(b.view(u, kind, e));
            tables[i] = try rig.input(std.mem.sliceAsBytes(table));
        }
        const any = tables[0] orelse tables[2].?;
        u.tables = .{ .gate = at(any), .up = at(tables[1] orelse any), .down = at(tables[2] orelse any), .count = @intCast(b.count) };
        return u;
    }
};

/// A bank on the GPU: descriptor tables, and each kind's words, su and sv buffers.
pub const Upload = struct { tables: x3.Tables, refs: [3]?[3]rig_.Ref = .{ null, null, null } };

pub fn at(r: rig_.Ref) x3.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

/// The plan's routing buffers uploaded as the kernels read them.
pub fn route(rig: *rig_.Rig, plan: eref.Plan) !x3.Route {
    return .{
        .ids = at(try rig.input(std.mem.sliceAsBytes(plan.ids))),
        .weights = at(try rig.input(std.mem.sliceAsBytes(plan.weights))),
        .slots = at(try rig.input(std.mem.sliceAsBytes(plan.slots))),
        .nslots = at(try rig.input(std.mem.sliceAsBytes(&plan.nslots))),
        .pairs = at(try rig.input(std.mem.sliceAsBytes(plan.pairs))),
        .where = at(try rig.input(std.mem.sliceAsBytes(plan.where))),
    };
}

/// tf_dsv41_x3_silu's value at every bf16 gate, read back once.
pub fn silu(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !*ref.Silu {
    const p = try rig_.pipeline(device, x3.text, "tf_dsv41_x3_silu");
    defer p.deinit();
    const buf = try device.buffer(65536 * 4, mtl.ResourceOptions.shared);
    defer buf.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(p);
    e.setBuffer(buf, 0, 0);
    e.dispatchGroups(.{ .width = 256 }, .{ .width = 256 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    const t = try a.create(ref.Silu);
    @memcpy(t, buf.slice(f32, 65536));
    return t;
}

/// The serial up of every pair (by pair index): the slots' sums [2][pairs][width] and act [pairs][width].
pub fn upAll(a: std.mem.Allocator, b: *const Bank, plan: eref.Plan, x: []const u16, t: *const ref.Silu, sums: []f32, act: []u16) !void {
    var w = try ref.Work.init(a, b.hidden, b.width);
    defer w.deinit(a);
    const np = plan.ids.len;
    for (0..np) |p| {
        const s: [2][]f32 = .{ sums[p * b.width ..][0..b.width], sums[(np + p) * b.width ..][0..b.width] };
        const e = plan.ids[p];
        ref.upPair(b.matrix(.gate, e), b.matrix(.up, e), x[p / topk * b.hidden ..][0..b.hidden], plan.weights[p], limit, t, s, act[p * b.width ..][0..b.width], &w);
    }
}

/// The serial down of act [pairs][width]: sums [pairs][hidden] by pair index, out [position][hidden].
pub fn downAll(a: std.mem.Allocator, b: *const Bank, plan: eref.Plan, act: []const u16, sums: []f32, out: []u16) !void {
    var w = try ref.Work.init(a, b.hidden, b.width);
    defer w.deinit(a);
    for (plan.ids, 0..) |e, p| ref.downPair(b.matrix(.down, e), act[p * b.width ..][0..b.width], sums[p * b.hidden ..][0..b.hidden], out[plan.where[p] * b.hidden ..][0..b.hidden], &w);
}

/// main's combine (experts_kimi's order, the route weights already in act) over out [position][hidden].
pub fn combineAll(a: std.mem.Allocator, plan: eref.Plan, out: []const u16, hidden: usize, per: usize, y: []f32) !void {
    const f = try a.alloc(f32, out.len);
    defer a.free(f);
    for (f, out) |*v, h| v.* = ref.fromBf16(h);
    const ones = try a.alloc(f32, plan.ids.len);
    defer a.free(ones);
    @memset(ones, 1);
    for (0..plan.rows) |r| for (0..hidden) |j| {
        y[r * hidden + j] = eref.combineOne(plan, ones, f, hidden, r, j, per);
    };
}

/// Where two bit patterns of a kind disagree: equal bits, or both NaN; null when every element agrees.
pub fn differ(comptime T: type, got: []const T, want: []const T) ?usize {
    const F = if (T == u32) f32 else void;
    for (got, want, 0..) |g, w, i| {
        if (g == w) continue;
        const nan = if (T == u32) std.math.isNan(@as(F, @bitCast(g))) and std.math.isNan(@as(F, @bitCast(w))) else std.math.isNan(ref.fromBf16(g)) and std.math.isNan(ref.fromBf16(w));
        if (!nan) return i;
    }
    return null;
}

/// W_q of `m` (k x n) decoded into `wq`, then transposed into `wt` so column j is wt[j * k ..][0..k].
pub fn columns(m: ref.Matrix, k: usize, n: usize, wq: []f16, wt: []f16) void {
    fmt.unpack(m.width, m.book, m.trellis, k / 16, n / 16, wq[0 .. k * n]);
    for (0..k) |i| for (0..n) |j| {
        wt[j * k + i] = wq[i * n + j];
    };
}

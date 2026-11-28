//! GLM expert-parallel reductions: guarded send counts, rank sums, and full-width receive epochs.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rows = contract.rows;
const c = contract.spec;
const g = contract.gates;
const rig = @import("rig.zig");
const ep = @import("metal_glm_ep.zig");
const max_rows = @import("tensorfold").glm.ep.PROMPT_ROWS;

const width = 4096;
const picks = 8;
const threads = 256;
pub const Kind = enum {
    ep_rcombine,
    ep_rfinal,
    ep_rsum,

    fn combine(k: Kind) bool {
        return k == .ep_rcombine;
    }

    fn elem(k: Kind) c.Elem {
        return if (k.combine()) .f32 else .bf16;
    }

    fn spec(k: Kind, label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = if (k.combine()) combineDepth else receiveDepth,
            .out = k.elem(),
            .nan = .{ .stated = "NaN and infinity classes match the reference; a stale epoch blocks transfer data" },
            .served = if (k.combine()) &combine_shapes else &receive_shapes,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
        };
    }
};

fn shapes(comptime k: usize) [7]c.Shape {
    var set: [7]c.Shape = undefined;
    for (&set, [_]usize{ 5, 2, 32, 63, 64, max_rows - 1, max_rows }) |*s, n|
        s.* = .{ .rows = n, .n = width, .k = k };
    return set;
}
const combine_shapes = shapes(picks);
const receive_shapes = shapes(2);
fn combineDepth(_: c.Shape) usize {
    return picks * 2 - 1;
}
fn receiveDepth(_: c.Shape) usize {
    return 1;
}

fn combineInput(i: usize, salt: u64) f32 {
    const h = std.hash.int(i +% salt);
    return @as(f32, @floatFromInt(@as(i32, @as(u16, @truncate(h))) - 32768)) / 8192;
}

pub const Probe = struct {
    const vt = g.hooks(Probe);

    rig: rig.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    seed: u64 = 0,
    data: rig.Ref = undefined,
    extra: rig.Ref = undefined,
    shared: rig.Ref = undefined,
    ready: rig.Ref = undefined,
    expired: rig.Ref = undefined,
    out: rig.Ref = undefined,
    count_ref: rig.Ref = undefined,

    fn rank(p: *const Probe) u32 {
        const parity: u32 = @intCast(p.shape.rows % 2);
        return parity ^ @intFromBool(p.kind == .ep_rfinal);
    }

    pub fn kernel(p: *Probe, name: []const u8, control: bool) g.Kernel {
        return .{ .spec = p.kind.spec(name), .ctx = p, .vt = &vt, .control = if (control) .shapes else null };
    }

    fn part(p: *const Probe, slot: usize, i: usize) f32 {
        if (p.fill == .stress) {
            const stress = [_]f32{ 0x1p20, 1 - 0x1p20, 1, -1, 2, -2, 3, -3 };
            return stress[slot] * if (p.kind.combine()) @as(f32, picks) else 1;
        }
        if (p.fill == .nonfinite) {
            const inf = std.math.inf(f32);
            if (slot == 0 and i < 4) return ([_]f32{ std.math.nan(f32), inf, -inf, inf })[i];
            if (slot == 1 and i == 3) return -inf;
        }
        if (p.kind == .ep_rfinal and i == 4) return if (slot == 0) 1 else 0x1p-8;
        return combineInput(i, p.seed +% (slot + 1) * 0x9e3779b9);
    }

    fn weight(p: *const Probe, row: usize, slot: usize) f32 {
        if (p.fill == .stress) return 1.0 / @as(f32, picks);
        return (4 + combineInput(row * picks + slot, p.seed)) / 32 + 0x1p-8;
    }

    fn sharedAt(p: *const Probe, i: usize) u16 {
        if (p.fill == .stress) return 0;
        if (p.fill == .nonfinite and i == 5) return c.toBf16(std.math.inf(f32));
        if (i == 4) return c.toBf16(-1);
        return c.toBf16(combineInput(i, p.seed) / 4);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > max_rows or shape.n != width) return false;
        p.rig.reset();
        p.shape = shape;
        p.fill = fill;
        p.seed = seed;
        if (p.kind.combine()) {
            const data = try p.rig.a.alloc(f32, p.count() * picks);
            defer p.rig.a.free(data);
            for (data, 0..) |*v, i| v.* = p.part(i / width % picks, i / (width * picks) * width + i % width);
            p.data = try p.rig.input(std.mem.sliceAsBytes(data));
            const wts = try p.rig.a.alloc(f32, shape.rows * picks);
            defer p.rig.a.free(wts);
            for (wts, 0..) |*v, i| v.* = p.weight(i / picks, i % picks);
            p.extra = try p.rig.input(std.mem.sliceAsBytes(wts));
            p.count_ref = try p.rig.scratch(4);
        } else {
            const data = try p.rig.a.alloc(f32, p.count());
            defer p.rig.a.free(data);
            const own = p.rank();
            for (data, 0..) |*v, i| v.* = p.part(own, i);
            p.data = try p.rig.input(std.mem.sliceAsBytes(data));
            for (data, 0..) |*v, i| v.* = p.part(1 - own, i);
            p.extra = try p.rig.input(std.mem.sliceAsBytes(data));
            if (p.kind == .ep_rfinal) {
                const shared = try p.rig.a.alloc(u16, p.count());
                defer p.rig.a.free(shared);
                for (shared, 0..) |*v, i| v.* = p.sharedAt(i);
                p.shared = try p.rig.input(std.mem.sliceAsBytes(shared));
            }
        }
        p.ready = try p.rig.input(std.mem.asBytes(&ep.epoch));
        p.expired = try p.rig.input(std.mem.asBytes(&ep.stale));
        p.out = try p.rig.output(p.outLen());
        return true;
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * width;
    }

    fn outLen(p: *const Probe) usize {
        return p.count() * c.size(p.kind.elem());
    }

    fn dispatch(p: *Probe, ready: rig.Ref) !void {
        @memset(p.rig.bytes(p.out, p.outLen()), 0xff);
        const pass = p.rig.begin();
        const e = pass.e;
        const n: i32 = @intCast(p.shape.rows);
        e.setPipeline(p.pipe);
        if (p.kind.combine()) {
            @memset(p.rig.bytes(p.count_ref, 4), 0xff);
            e.setBuffer(p.data.buf, p.data.off, 0);
            e.setBuffer(p.extra.buf, p.extra.off, 1);
            e.setValue(n, 2);
            e.setBuffer(p.out.buf, p.out.off, 3);
            e.setBuffer(p.count_ref.buf, p.count_ref.off, 4);
            e.setBuffer(ready.buf, ready.off, 5);
            e.setValue(ep.epoch, 6);
            e.dispatchThreads(mtl.Size.of(p.count(), 1, 1), mtl.Size.of(threads, 1, 1));
        } else {
            e.setBuffer(ready.buf, ready.off, 0);
            e.setValue(ep.epoch, 1);
            e.setBuffer(p.data.buf, p.data.off, 3);
            e.setBuffer(p.extra.buf, p.extra.off, 4);
            e.setValue(p.rank(), 5);
            const out_binding: u32 = if (p.kind == .ep_rfinal) 7 else 6;
            if (p.kind == .ep_rfinal) e.setBuffer(p.shared.buf, p.shared.off, 6);
            e.setBuffer(p.out.buf, p.out.off, out_binding);
            e.setValue(n, out_binding + 1);
            e.dispatchGroups(mtl.Size.of(p.count() / threads, 1, 1), mtl.Size.of(threads, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.dispatch(p.ready);
        if (p.kind.combine() and p.sent() != p.shape.rows * 2) return error.WrongExchangeCount;
    }

    fn sent(p: *const Probe) u32 {
        return std.mem.readInt(u32, p.rig.bytes(p.count_ref, 4)[0..4], .little);
    }

    fn terms(p: *const Probe, i: usize, out: *[picks]f64) []const f64 {
        if (p.kind.combine()) {
            for (out, 0..) |*v, slot| v.* = @as(f64, p.weight(i / width, slot)) * p.part(slot, i);
            return out;
        }
        out[0] = p.part(0, i);
        out[1] = p.part(1, i);
        return out[0..2];
    }

    fn finish(p: *const Probe, i: usize, value: f64) f64 {
        if (p.kind.combine()) return value;
        const rounded = contract.bf16_round.bf16(value);
        if (p.kind == .ep_rfinal) return contract.bf16_round.bf16(rounded + c.fromBf16(p.sharedAt(i)));
        return rounded;
    }

    fn ordered(p: *const Probe, i: usize, ready: bool) f32 {
        if (!ready) {
            const local = p.part(p.rank(), i) + 0.0;
            return @floatCast(p.finish(i, local));
        }
        var values: [picks]f64 = undefined;
        const v = p.terms(i, &values);
        return @floatCast(p.finish(i, rows.sumIn(.f32, .sequential, v)));
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            var values: [picks]f64 = undefined;
            const terms_ = p.terms(i, &values);
            w.* = p.finish(i, rows.sumIn(.none, .sequential, terms_));
            m.* = 0;
            for (terms_) |v| m.* += @abs(v);
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| {
            var values: [picks]f64 = undefined;
            v.* = p.finish(i, rows.sumIn(acc, order, p.terms(i, &values)));
        }
    }

    fn compare(p: *Probe, gate: g.Gate, ready: bool) !?[]const u8 {
        if (!try p.rig.intact()) return "a launch wrote outside its buffers";
        const out = p.rig.bytes(p.out, p.outLen());
        if (p.kind.combine() and !ready) {
            if (p.sent() != std.math.maxInt(u32) or !std.mem.allEqual(u8, out, 0xff)) return "stale previous epoch changed the send payload or count";
            return null;
        }
        var planted: usize = 0;
        for (0..p.count()) |i| {
            const got = c.read(p.kind.elem(), out, i);
            if (gate == .nonfinite) {
                var values: [picks]f64 = undefined;
                const want = if (ready) p.finish(i, rows.sumIn(.none, .sequential, p.terms(i, &values))) else p.ordered(i, false);
                if (c.classOf(want) != .finite) planted += 1;
                if (c.classOf(got) != c.classOf(want)) {
                    p.report(ready, i, got, want);
                    return "non-finite classes differ from float64";
                }
            } else {
                const want = p.ordered(i, ready);
                if (@as(u32, @bitCast(@as(f32, @floatCast(got)))) != @as(u32, @bitCast(want))) {
                    p.report(ready, i, got, want);
                    return "output bits differ from the ordered fp32 and bf16 rounding reference";
                }
            }
        }
        if (gate == .nonfinite and ready and (planted == 0 or planted == p.count())) return "the fixture needs both finite and non-finite outputs";
        return null;
    }

    fn report(p: *const Probe, ready: bool, i: usize, got: f64, want: f64) void {
        std.debug.print("{t} rank {d} ready {} at row {d} column {d}: got {e}, want {e}\n", .{
            p.kind, p.rank(), ready, i / width, i % width, got, want,
        });
    }

    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (try p.compare(gate, true)) |why| return why;
        try p.dispatch(p.expired);
        return p.compare(gate, false);
    }
};

pub fn caseName(comptime kind: Kind, control: bool) []const u8 {
    const normal = comptime ep.name(kind);
    return if (control) normal ++ " low-word epoch control" else normal;
}

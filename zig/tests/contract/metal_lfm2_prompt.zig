//! The served NAX prompt selector: original finite bytes and independent binary64 visibility, including staged tails.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig = @import("rig.zig");
const data = @import("masked_value_data.zig");
const mask = tf.lfm2_moe.prompt_mask;
const Case = enum { finite, inf, nan, tail_inf, tail_nan, bypass };
const names = [_][]const u8{
    "metal lfm2 prompt selector finite bytes", "metal lfm2 prompt masked Inf",      "metal lfm2 prompt masked NaN",
    "metal lfm2 prompt staged tail Inf",       "metal lfm2 prompt staged tail NaN", "control: metal lfm2 prompt hidden Inf without selector",
};
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2048, .k = 15 },    .{ .rows = 8, .n = 2048, .k = 16 },
    .{ .rows = 16, .n = 2048, .k = 31 },   .{ .rows = 17, .n = 2048, .k = 63 },
    .{ .rows = 32, .n = 2048, .k = 4095 }, .{ .rows = 65, .n = 2048, .k = 0 },
    .{ .rows = 2048, .n = 2048, .k = 0 },
};
fn depth(_: c.Shape) usize {
    return 8192;
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig.Rig,
    pipes: mask.Pipelines,
    original: mtl.Pipeline,
    merge: mtl.Pipeline,
    case: Case,
    arena: ?std.heap.ArenaAllocator = null,
    fixture: data.Data = undefined,
    input: [3]rig.Ref = undefined,
    partial: [3]rig.Ref = undefined,
    pick: rig.Ref = undefined,
    out: rig.Ref = undefined,
    args: mask.Args = undefined,
    baseline: bool = false,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, case: Case) !Probe {
        const body = try tf.frags.source(device, a, sources.lfm2_attn_nax);
        defer a.free(body);
        const text = try std.mem.concat(a, u8, &.{ sources.lfm2, "\n", body });
        defer a.free(text);
        const original = try rig.pipeline(device, text, "lfm2_attn_nax");
        errdefer original.deinit();
        const merge = try rig.pipeline(device, text, "lfm2_attn_merge");
        errdefer merge.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try mask.Pipelines.load(device), .original = original, .merge = merge, .case = case };
    }
    fn deinit(p: *Probe) void {
        p.rig.deinit();
        if (p.arena) |*a| a.deinit();
        p.pipes.deinit();
        p.original.deinit();
        p.merge.deinit();
    }
    const vt = g.hooks(Probe);
    fn kernel(p: *Probe) g.Kernel {
        const finite = p.case == .finite;
        return .{ .ctx = p, .vt = &vt, .control = if (p.case == .bypass) .nonfinite else null, .spec = .{
            .name = names[@backingInt(p.case)],
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .mids = 2,
            .served = if (finite) &shapes else shapes[1..5],
            .replaces = if (finite) "unchanged direct lfm2_attn_nax and merge" else null,
            .nan = .{ .stated = "All classes and finite values match binary64 over the causal key set." },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "selector/visibility panel; existing LFM2 arithmetic entries own accumulator discrimination" },
        } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n != 2048 or shape.k > 4095) return false;
        p.rig.reset();
        if (p.arena) |*a| a.deinit();
        p.arena = .init(p.a);
        const a = p.arena.?.allocator();
        const cap = (shape.k + shape.rows + 63) / 32 * 32;
        p.fixture = try data.Data.init(a, .{ .heads = 32, .kv = 8, .dim = 64, .cache = .head, .cap = cap, .first = shape.k }, shape.rows, false, .inf);
        const d = p.fixture;
        if (fill == .nonfinite) {
            const tail = p.case == .tail_inf or p.case == .tail_nan;
            const key = if (tail) @as(usize, 0) else shape.k + 1;
            d.v[d.cacheAt(key, 0, 7)] = if (p.case == .nan or p.case == .tail_nan) 0x7fc0 else 0x7f80;
            if (tail) {
                const end = shape.k + shape.rows;
                // Tail cases deliberately have a partial final tile. The key-zero poison cannot select the twin.
                if (end % 32 != 0) d.v[d.cacheAt(end, 7, 63)] = if (p.case == .tail_nan) 0x7fc0 else 0xff80;
            } else {
                d.v[d.cacheAt(shape.k + shape.rows - 1, 7, 63)] = 0xff80;
            }
        } else {
            for (d.v, 0..) |*v, i| {
                v.* = c.toBf16(@as(f32, @floatFromInt((i *% 31 +% @as(usize, @intCast(seed & 255))) % 127)) / 128);
            }
        }
        for ([_][]const u16{ d.q, d.k, d.v }, &p.input) |v, *r| r.* = try p.rig.input(std.mem.sliceAsBytes(v));
        const n = shape.rows * 32 * 32;
        for (&p.partial, 0..) |*r, i| r.* = try p.rig.scratch(n * 4 * (if (i == 2) @as(usize, 64) else 1));
        p.out = try p.rig.output(p.count() * 2);
        p.pick = try p.rig.scratch(64);
        @memset(p.rig.bytes(p.pick, 64), 0);
        p.args = .{ .len = @intCast(shape.k), .cap = @intCast(cap), .scale = 0.125, .subs = 1 };
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows = p.fixture.rows;
        if (p.baseline or p.case == .bypass) {
            e.setPipeline(p.original);
            for (p.input, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(p.args, 3);
            for (p.partial, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(@as(u32, @intCast(rows)), 7);
            e.dispatchGroups(mtl.Size.of((rows + 15) / 16, 8, (p.args.len + rows + 4095) / 4096), mtl.Size.of(128, 1, 1));
        } else {
            var inputs: [3]mask.Ref = undefined;
            var partials: [3]mask.Ref = undefined;
            for (p.input, &inputs) |r, *v| v.* = .{ .buf = r.buf, .off = r.off };
            for (p.partial, &partials) |r, *v| v.* = .{ .buf = r.buf, .off = r.off };
            mask.launch(e, p.pipes, p.original, inputs, partials, .{ .buf = p.pick.buf, .off = p.pick.off }, p.args, rows);
        }
        e.barrier();
        e.setPipeline(p.merge);
        for (p.partial, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(p.args, 3);
        e.setBuffer(p.out.buf, p.out.off, 4);
        e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        p.baseline = true;
        defer p.baseline = false;
        try p.run();
        return p.rig.bytes(p.out, p.count() * 2);
    }
    pub fn count(p: *Probe) usize {
        return p.fixture.count();
    }
    /// Uniform zero-score attention, independently accumulated per cache head and column before each query row.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const d = p.fixture;
        for (0..8) |kh| for (0..64) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..d.geo.first + d.rows) |key| {
                const x: f64 = c.fromBf16(d.v[d.cacheAt(key, kh, col)]);
                sum += x;
                magnitude += @abs(x);
                if (key < d.geo.first) continue;
                for (0..4) |h| {
                    const at = d.outputAt(key - d.geo.first, kh * 4 + h, col);
                    const n: f64 = @floatFromInt(key + 1);
                    want[at] = sum / n;
                    mag[at] = magnitude / n;
                }
            }
        };
    }
    pub fn simulate(p: *Probe, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        try p.reference(out, mag);
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Case), names) |case, name| {
            if (only) |o| if (std.mem.indexOf(u8, name, o) == null) continue;
            const p = try s.a.create(Probe);
            p.* = try .init(s.a, s.device, s.queue, case);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};

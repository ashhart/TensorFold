//! Pre-M5 verify attention bytes against the unchanged one-row partial and merge.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const attention = @import("metal_nemotron_attention.zig");
const rig_ = @import("rig.zig");
const d = tf.nemotron.decode_attention;
const legacy = @import("nemotron_decode_legacy.zig");
const pages = @import("nemotron_decode_pages.zig");
const c = contract.spec;
const g = contract.gates;
const D = attention.D;
const heads = attention.heads;
const kvh = attention.kvh;
const qkv_n = (heads + 2 * kvh) * D;

const cases = @import("nemotron_decode_cases.zig");
const served = cases.served;

fn depth(s: c.Shape) usize {
    return D + s.k + s.rows;
}
const name = cases.name;

fn ref(r: rig_.Ref) d.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

const stressValue = cases.stressValue;

pub const Probe = struct {
    a: std.mem.Allocator,
    inner: attention.Probe,
    rig: *rig_.Rig = undefined,
    pipes: d.Pipelines,
    control: bool,
    old_window: bool = false,
    paged: pages.State = .{},
    partials: [16]?mtl.Pipeline = @splat(null),
    qkv: rig_.Ref = undefined,
    qp: rig_.Ref = undefined,
    att: rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        var inner = try attention.Probe.init(a, device, queue, .decode, 1, false);
        errdefer inner.deinit();
        var text: []const u8 = sources.nemotron_glue;
        var changed: ?[]u8 = null;
        defer if (changed) |x| a.free(x);
        if (control) {
            const from = "qkv[size_t(t) * dims.x + (hk * dims.y + g) * hd + d]";
            if (std.mem.count(u8, text, from) != 1) return error.ControlEditMissing;
            changed = try std.mem.replaceOwned(u8, a, text, from, "qkv[size_t(t) * dims.x + (hk * dims.y + g) * hd + (d + 1) % hd]");
            text = changed.?;
        }
        const q = try rig_.pipeline(device, text, "tf_attn_q");
        errdefer q.deinit();
        const out = try rig_.pipeline(device, sources.nemotron_glue, "tf_attn_out");
        return .{ .a = a, .inner = inner, .control = control, .pipes = .{ .q = q, .partial = inner.pipes[0].?, .merge = inner.pipes[1].?, .out = out } };
    }

    pub fn deinit(p: *Probe) void {
        p.paged.deinit();
        for (p.partials) |pipe| if (pipe) |x| x.deinit();
        p.pipes.q.deinit();
        p.pipes.out.deinit();
        p.inner.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .input_group = "nemotron verify attention identical inputs", .control = if (p.control) .shapes else null, .spec = .{
            .name = name(p.old_window, p.paged.enabled, p.control),
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = .bf16,
            .served = &served,
            .replaces = "unchanged token-at-a-time partial_1 + merge",
        } };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig = &p.inner.rig;
        if (!try p.inner.prepare(s, if (fill == .stress) .random else fill, seed)) return false;
        const t = p.inner.rows;
        if (fill == .stress) {
            const v = &p.inner;
            v.stress = true;
            @memset(v.q, 0);
            @memset(p.rig.bytes(v.qref, v.q.len * 2), 0);
            for (0..kvh) |h| for (0..v.cap) |j| for (0..D) |col| {
                v.host[v.at(false, h, j) + col] = 0;
                v.host[v.at(true, h, j) + col] = c.toBf16(stressValue(j));
            };
            const half = kvh * v.cap * D;
            for (v.kv, 0..) |b, i| @memcpy(p.rig.bytes(b, half * 2), std.mem.sliceAsBytes(v.host[i * half ..][0..half]));
        }
        const qkv = try p.a.alloc(u16, t * qkv_n);
        defer p.a.free(qkv);
        @memset(qkv, 0);
        for (0..t) |r| for (0..kvh) |h| {
            const src = (h * 16 * t + r * 16) * D;
            const dst = r * qkv_n + h * 16 * D;
            @memcpy(qkv[dst..][0 .. 16 * D], p.inner.q[src..][0 .. 16 * D]);
        };
        p.qkv = try p.rig.input(std.mem.sliceAsBytes(qkv));
        p.qp = try p.rig.scratch(heads * D * 2 * t);
        p.att = try p.rig.scratch(heads * D * 2 * t);
        p.baseline = try p.rig.scratch(p.inner.out_bytes);
        const tiles = if (p.old_window) @min(t, 16) else 1;
        if (p.paged.enabled) {
            try p.paged.prepare(p);
            _ = try p.paged.pipeline(p.a, p.rig.device, tiles);
        } else if (tiles > 1 and p.partials[tiles - 1] == null) p.partials[tiles - 1] = try @import("nemotron_decode_pipelines.zig").partial(p.a, p.rig.device, tiles);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.inner.count();
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        var e: tf.nemotron.forward.Enc = .{ .e = pass.e, .concurrent = true };
        const v = &p.inner;
        var pipes = p.pipes;
        const tiles = if (p.old_window) @min(v.rows, 16) else 1;
        if (p.paged.enabled) pipes.partial = try p.paged.pipeline(p.a, p.rig.device, tiles) else if (tiles > 1) pipes.partial = p.partials[tiles - 1].?;
        const call: d.Call = .{
            .head_dim = D,
            .heads = heads,
            .kv_heads = kvh,
            .qkv_n = qkv_n,
            .capacity = v.cap,
            .len = v.len,
            .first = 0,
            .rows = v.rows,
            .qkv = ref(p.qkv),
            .qp = ref(p.qp),
            .k = if (p.paged.enabled) ref(p.paged.arena) else ref(v.kv[0]),
            .v = if (p.paged.enabled) .{ .buf = p.paged.arena.buf, .off = p.paged.arena.off + @as(usize, @intCast(pages.page_bytes / 2)) } else ref(v.kv[1]),
            .table = if (p.paged.enabled) ref(p.paged.table) else null,
            .page_bytes = if (p.paged.enabled) pages.page_bytes else 0,
            .po = ref(v.parts[0]),
            .pm = ref(v.parts[1]),
            .pl = ref(v.parts[2]),
            .att = ref(p.att),
            .out = ref(v.out),
        };
        if (p.old_window) legacy.encode(&e, pipes, call) else d.encode(&e, pipes, call);
        try p.rig.finish(pass);
    }

    /// Plain attend(): Q reads the stacked projection directly, one-row partial_1, then merge directly to its row.
    pub fn replaced(p: *Probe) ![]const u8 {
        const v = &p.inner;
        const pass = p.rig.begin();
        const e = pass.e;
        const strides = [4]i64{ @intCast(kvh * v.cap * D), @intCast(v.cap * D), D, 1 };
        for (0..v.rows) |r| {
            const keys = v.len + r + 1;
            const chunks = (keys + 511) / 512;
            const dims = [5]i32{ @intCast(keys), @intCast(chunks), 1, 1, 1 };
            e.setPipeline(if (p.paged.enabled) try p.paged.pipeline(p.a, p.rig.device, 1) else v.pipes[0].?);
            e.setBuffer(p.qkv.buf, p.qkv.off + r * qkv_n * 2, 0);
            const k = if (p.paged.enabled) p.paged.arena else v.kv[0];
            const val = if (p.paged.enabled) rig_.Ref{ .buf = p.paged.arena.buf, .off = p.paged.arena.off + @as(usize, @intCast(pages.page_bytes / 2)) } else v.kv[1];
            e.setBuffer(k.buf, k.off, 1);
            e.setValue(strides, 2);
            e.setBuffer(val.buf, val.off, 3);
            e.setValue(strides, 4);
            e.setValue(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, D)))), 5);
            e.setValue(dims, 6);
            for (v.parts[0..3], 7..) |b, i| e.setBuffer(b.buf, b.off, i);
            if (p.paged.enabled) {
                e.setBuffer(p.paged.table.buf, p.paged.table.off, 10);
                e.setValue(pages.page_bytes, 11);
            }
            e.dispatchThreads(mtl.Size.of(kvh * 32, chunks, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
            e.setPipeline(v.pipes[1].?);
            for (v.parts[0..3], 0..) |b, i| e.setBuffer(b.buf, b.off, i);
            e.setValue(dims, 3);
            e.setBuffer(p.baseline.buf, p.baseline.off + r * heads * D * 2, 4);
            e.dispatchThreads(mtl.Size.of(kvh * 32, 16, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
        }
        try p.rig.finish(pass);
        return p.rig.bytes(p.baseline, v.out_bytes);
    }

    fn reorder(p: *Probe, src: []const f64, dst: []f64) void {
        for (0..p.inner.rows) |r| for (0..heads) |h| {
            @memcpy(dst[(r * heads + h) * D ..][0..D], src[(h * p.inner.rows + r) * D ..][0..D]);
        };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.inner.stress) {
            for (0..p.inner.rows) |r| {
                const n = p.inner.len + r + 1;
                var sum: f64 = 0;
                var magnitude: f64 = 0;
                for (0..n) |j| {
                    const v = c.fromBf16(c.toBf16(stressValue(j)));
                    sum += v;
                    magnitude += @abs(v);
                }
                @memset(want[r * heads * D ..][0 .. heads * D], sum / @as(f64, @floatFromInt(n)));
                @memset(mag[r * heads * D ..][0 .. heads * D], magnitude / @as(f64, @floatFromInt(n)));
            }
            return;
        }
        const tmp = try p.a.alloc(f64, 2 * want.len);
        defer p.a.free(tmp);
        const stress = p.inner.stress;
        p.inner.stress = false;
        defer p.inner.stress = stress;
        try p.inner.reference(tmp[0..want.len], tmp[want.len..]);
        p.reorder(tmp[0..want.len], want);
        p.reorder(tmp[want.len..], mag);
    }

    pub fn bounds(p: *Probe, want: []const f64, tol: []const f64, allow: []f64) !void {
        try p.inner.bounds(want, tol, allow);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
        if (!p.inner.stress) return error.StressRequiresEqualScores;
        const terms = try p.a.alloc(f64, p.inner.len + p.inner.rows);
        defer p.a.free(terms);
        for (terms, 0..) |*v, j| v.* = c.fromBf16(c.toBf16(stressValue(j)));
        for (0..p.inner.rows) |r| {
            const n = p.inner.len + r + 1;
            const sum = contract.rows.sumIn(acc, order, terms[0..n]) / @as(f64, @floatFromInt(n));
            @memset(out[r * heads * D ..][0 .. heads * D], sum);
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (s.device.tensorUnits()) return;
        try cases.visit(s, only);
    }
    pub fn addDecode(s: *Suite, old: bool, paged: bool, control: bool, _: []const u8) !void {
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, control);
            errdefer p.deinit();
            p.old_window = old;
            p.paged.enabled = paged;
            p.rig = &p.inner.rig;
            try contract.register(s.a, &s.probes, s.kernels, p, p.kernel());
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};

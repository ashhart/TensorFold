//! Fused cache/QA placement keeps the norm/RoPE numeric gates and checks every destination byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const at = tf.qwen27.attention;
const base = @import("metal_q27_qk.zig");
const D = 256;
const g = ct.gates;
const c = ct.spec;
pub fn label(pages: bool, bad: bool) []const u8 {
    return if (pages) (if (bad) "control: q27 cache paged missing padding" else "metal q27 cache paged norm rope") else (if (bad) "control: q27 cache flat missing padding" else "metal q27 cache flat norm rope");
}
pub const padding_write = "QA[(hk * uint(meta[5]) + pad) * a.width + at] = bfloat(value);";
/// Skip exactly one repeated slot, never a live query row.
pub fn control(a: std.mem.Allocator, text: []const u8) ![]u8 {
    if (std.mem.count(u8, text, padding_write) != 1) return error.QkCacheControlEdit;
    return std.mem.replaceOwned(u8, a, text, padding_write, "if (pad != first_slot + live) " ++ padding_write);
}
pub const Probe = struct {
    a: std.mem.Allocator,
    inner: *base.Probe,
    rig: *rr.Rig,
    pages: bool,
    bad: bool,
    pipe: mtl.Pipeline,
    plan: ?at.Plan = null,
    binding: ?at.Binding = null,
    cache: rr.Ref = undefined,
    qa: rr.Ref = undefined,
    nodes: rr.Ref = undefined,
    qr: rr.Ref = undefined,
    table: rr.Ref = undefined,
    bytes: usize = 0,
    cache_bytes: usize = 0,
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pages: bool, bad: bool) !Probe {
        const inner = try a.create(base.Probe);
        errdefer a.destroy(inner);
        inner.* = try base.Probe.init(a, device, queue, false);
        errdefer inner.deinit();
        const raw = try tf.qwen27.simd_attention.grouped(a, device.tensorUnits(), true, 6, 4, pages);
        defer a.free(raw);
        const text = if (bad) try control(a, raw) else try a.dupe(u8, raw);
        defer a.free(text);
        return .{ .a = a, .inner = inner, .rig = &inner.rig, .pages = pages, .bad = bad, .pipe = try rr.pipeline(device, text, "q27_qk_cache_rope") };
    }
    fn release(p: *Probe) void {
        if (p.plan) |*v| v.deinit();
        if (p.binding) |*v| v.deinit();
        p.plan = null;
        p.binding = null;
    }
    pub fn deinit(p: *Probe) void {
        p.release();
        p.pipe.deinit();
        p.inner.deinit();
        p.a.destroy(p.inner);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var k = p.inner.kernel();
        k.ctx = p;
        k.vt = &vt;
        k.control = if (p.bad) .shapes else null;
        k.spec.name = label(p.pages, p.bad);
        k.spec.replaces = "981c59cb2c norm/RoPE, append and packed Q";
        k.spec.nan = .{ .stated = "T625 norm/RoPE classes and every cache/QA bit, including native BF16 V payloads" };
        return k;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.release();
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const a = p.inner.args;
        const rows = a.rows;
        const hk = a.kv_heads;
        const nq: usize = rows * a.heads * D;
        const nk: usize = rows * hk * D;
        // V is a native payload copy, including signed zeros, infinities and NaN payloads.
        const mixed = std.mem.bytesAsSlice(u16, p.rig.bytes(p.inner.buffers[1], nk * 4));
        var rng = std.Random.DefaultPrng.init(seed);
        for (0..rows) |row| for (0..hk * D) |col| {
            mixed[(row * 2 * hk + hk) * D + col] = rng.random().int(u16);
        };
        try p.inner.run(); // T625's unchanged 2a kernel is the byte oracle for all norm and rotary outputs.
        p.inner.before.clearRetainingCapacity();
        try p.inner.before.appendSlice(p.a, try p.rig.outputBytes());
        const start = 255;
        const cap = start + rows + 3;
        const part = hk * 256 * D * 2;
        const np = at.pageCount(cap);
        const page_bytes = 2 * part + 64;
        const total = if (p.pages) np * page_bytes else @as(usize, cap) * hk * D * 4;
        const values = if (p.pages) part else total / 2;
        const ids = try p.a.alloc(u32, np);
        defer p.a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(np - i - 1);
        p.table = try p.rig.input(std.mem.sliceAsBytes(ids));
        p.cache_bytes = total;
        p.cache = try p.rig.output(total);
        const cache = at.Cache{ .keys = ref(p.cache), .values = .{ .buffer = p.cache.buf, .offset = p.cache.off + values }, .capacity = cap, .pages = if (p.pages) .{ .table = ref(p.table), .page_bytes = page_bytes } else null };
        const parents = try p.a.alloc(i32, rows);
        defer p.a.free(parents);
        for (parents, 0..) |*v, i| v.* = @as(i32, @intCast(i)) - 1;
        const geometry = at.planning.Geometry{ .heads = a.heads, .kv_heads = hk };
        p.plan = try at.Plan.init(p.a, &.{.{ .parents = parents, .start = start, .capacity = cap }}, geometry);
        const plan = p.plan.?;
        p.binding = try at.Binding.init(p.rig.device, .{ .rows = rows, .streams = 1, .keys = cap, .geometry = geometry });
        try p.binding.?.upload(plan, &.{cache});
        p.nodes = try p.rig.input(std.mem.sliceAsBytes(plan.nodes));
        p.qr = try p.rig.input(std.mem.sliceAsBytes(plan.q_rows));
        p.bytes = @as(usize, plan.tiles) * 16 * hk * D * 2;
        p.qa = try p.rig.output(p.bytes);
        const before = p.inner.before.items;
        const queries = try p.a.dupe(u8, before[(nq + nk) * 2 ..][0 .. nq * 2]);
        defer p.a.free(queries);
        const keys = try p.a.dupe(u8, before[(2 * nq + nk) * 2 ..][0 .. nk * 2]);
        defer p.a.free(keys);
        const expected = try p.a.alloc(u8, total + p.bytes);
        defer p.a.free(expected);
        @memset(expected, 0xff);
        for (0..rows) |row| for (0..hk) |h| {
            const physical = start + row;
            const offset = if (p.pages) @as(usize, ids[physical / 256]) * page_bytes + (h * 256 + physical % 256) * D * 2 else (h * cap + physical) * D * 2;
            @memcpy(expected[offset..][0 .. D * 2], keys[(row * hk + h) * D * 2 ..][0 .. D * 2]);
            @memcpy(expected[values + offset ..][0 .. D * 2], std.mem.sliceAsBytes(mixed[(row * 2 * hk + hk + h) * D ..][0..D]));
        };
        for (0..hk) |h| for (plan.q_rows, 0..) |mapped, slot| {
            const node: usize = @intCast(@divTrunc(mapped, @as(i32, @intCast(geometry.group()))));
            const head = h * geometry.group() + @as(usize, @intCast(@mod(mapped, @as(i32, @intCast(geometry.group())))));
            @memcpy(expected[total + (h * plan.q_rows.len + slot) * D * 2 ..][0 .. D * 2], queries[(node * a.heads + head) * D * 2 ..][0 .. D * 2]);
        };
        try p.inner.before.appendSlice(p.a, expected);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const nq: usize = p.inner.args.rows * p.inner.args.heads * D * 2;
        const nk: usize = p.inner.args.rows * p.inner.args.kv_heads * D * 2;
        for (p.inner.buffers[5..], [_]usize{ nq, nk, nq, nk }) |b, bytes| @memset(p.rig.bytes(b, bytes), 0xff);
        @memset(p.rig.bytes(p.cache, p.cache_bytes), 0xff);
        @memset(p.rig.bytes(p.qa, p.bytes), 0xff);
        const pass = p.rig.begin();
        const e = pass.e;
        const a = p.inner.args;
        e.setPipeline(p.pipe);
        for (p.inner.buffers[0..5], 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(a, 5);
        for (p.inner.buffers[5..], 6..) |b, i| e.setBuffer(b.buf, b.off, i);
        inline for (.{ p.binding.?.get(.caches), p.binding.?.get(.meta), ref(p.nodes), ref(p.qr), ref(p.qa) }, 10..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.useResource(p.cache.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        if (p.pages) e.useResource(p.table.buf, mtl.ResourceUsage.read);
        e.dispatchGroups(.of(a.heads + 2 * a.kv_heads, a.rows, 1), .of(64, 1, 1));
        try p.rig.finish(pass);
        try p.inner.independent();
    }
    pub fn count(p: *Probe) usize {
        return p.inner.count();
    }
    pub fn reference(p: *Probe, out: []f64, mag: []f64) !void {
        try p.inner.reference(out, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.inner.simulate(acc, order, out);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.inner.before.items;
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        return if (std.mem.eql(u8, try p.rig.outputBytes(), p.inner.before.items)) null else "cache, QA padding or retained norm/RoPE bytes differ";
    }
};
fn ref(r: rr.Ref) tf.qwen27.projection.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for ([_]bool{ false, true }) |bad| for ([_]bool{ false, true }) |pages| {
            if (only) |f| if (std.mem.indexOf(u8, label(pages, bad), f) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, pages, bad);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        };
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};

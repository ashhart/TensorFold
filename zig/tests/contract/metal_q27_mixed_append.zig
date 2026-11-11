//! Mixed value rows copied into flat or permuted page caches, every byte checked including untouched slots.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const at = @import("metal_q27_attn.zig");
const c = ct.spec;
const g = ct.gates;
const q = tf.qwen27;
const D = 256;
fn depth(_: c.Shape) usize {
    return 0;
}
pub fn label(pages: bool, bad: bool) []const u8 {
    return if (pages) (if (bad) "control: q27 mixed paged append contiguous V stride" else "metal q27 mixed paged append") else (if (bad) "control: q27 mixed append contiguous V stride" else "metal q27 mixed append");
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rr.Rig,
    pages: bool,
    bad: bool,
    pipe: mtl.Pipeline,
    binding: ?q.attention.Binding = null,
    plan: ?q.attention.Plan = null,
    expected: []u8 = &.{},
    output: rr.Ref = undefined,
    keys: rr.Ref = undefined,
    mixed: rr.Ref = undefined,
    nodes: rr.Ref = undefined,
    table: rr.Ref = undefined,
    args: q.attention.PackArgs = undefined,
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pages: bool, bad: bool) !Probe {
        const text = try q.simd_attention.grouped(a, device.tensorUnits(), true, 6, 4, pages);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pages = pages, .bad = bad, .pipe = try rr.pipeline(device, text, "q27_attn_append") };
    }
    pub fn release(p: *Probe) void {
        if (p.binding) |*b| b.deinit();
        if (p.plan) |*b| b.deinit();
        p.binding = null;
        p.plan = null;
        p.a.free(p.expected);
        p.expected = &.{};
    }
    pub fn deinit(p: *Probe) void {
        p.release();
        p.pipe.deinit();
        p.rig.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .shapes else null, .spec = .{
            .name = label(p.pages, p.bad),
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "all BF16 payload bits copy unchanged, including NaN and both infinities" },
            .served = if (p.rig.device.tensorUnits()) &.{ .{ .rows = 16, .n = 4, .k = D }, .{ .rows = 1, .n = 2, .k = D }, .{ .rows = 128, .n = 4, .k = D }, .{ .rows = 2048, .n = 2, .k = D }, .{ .rows = 2048, .n = 4, .k = D } } else &.{ .{ .rows = 16, .n = 4, .k = D }, .{ .rows = 1, .n = 2, .k = D }, .{ .rows = 128, .n = 4, .k = D } },
            .tiles = .{ .{ .size = 16 }, .{ .step = 2, .min = 2 }, .{ .size = D, .step = D, .min = D } },
        } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.k != D or (shape.n != 2 and shape.n != 4)) return false;
        p.release();
        p.rig.reset();
        const rows: u32 = @intCast(shape.rows);
        const hk: u32 = @intCast(shape.n);
        const width = hk * D;
        const start = 255;
        const cap = start + rows + 3;
        const part = hk * 256 * D * 2;
        const np = q.attention.pageCount(cap);
        const page_bytes = 2 * part + 64;
        const cache_bytes = if (p.pages) np * page_bytes else @as(usize, cap) * width * 4;
        p.output = try p.rig.output(cache_bytes);
        p.expected = try p.a.alloc(u8, cache_bytes);
        @memset(p.expected, 0xff);
        const keys = try p.a.alloc(u16, rows * width);
        defer p.a.free(keys);
        const mixed = try p.a.alloc(u16, rows * width * 2);
        defer p.a.free(mixed);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (keys) |*v| v.* = random.int(u16);
        for (mixed) |*v| v.* = random.int(u16);
        if (fill == .nonfinite) {
            keys[0] = 0x7fc5;
            keys[1] = 0x7f80;
            keys[2] = 0xff80;
            mixed[width] = 0xffc7;
            mixed[width + 1] = 0x7f80;
            mixed[width + 2] = 0xff80;
        }
        p.keys = try p.rig.input(std.mem.sliceAsBytes(keys));
        p.mixed = try p.rig.input(std.mem.sliceAsBytes(mixed));
        const ids = try p.a.alloc(u32, np);
        defer p.a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(np - i - 1);
        const table = try p.rig.input(std.mem.sliceAsBytes(ids));
        p.table = table;
        const base = p.output.off;
        const value_part: usize = if (p.pages) part else @as(usize, cap) * width * 2;
        const cache = q.attention.Cache{ .keys = .{ .buffer = p.output.buf, .offset = base }, .values = .{ .buffer = p.output.buf, .offset = base + value_part }, .capacity = cap, .pages = if (p.pages) .{ .table = at.ref(table), .page_bytes = page_bytes } else null };
        const parents = try p.a.alloc(i32, rows);
        defer p.a.free(parents);
        for (parents, 0..) |*x, i| x.* = @as(i32, @intCast(i)) - 1;
        const geo = q.attention_plan.Geometry{ .heads = hk * 6, .kv_heads = hk, .shared_prefix = p.rig.device.tensorUnits() };
        var plan = try q.attention.Plan.init(p.a, &.{.{ .parents = parents, .start = start, .capacity = cap }}, geo);
        errdefer plan.deinit();
        var binding = try q.attention.Binding.init(p.rig.device, .{ .rows = rows, .streams = 1, .keys = cap, .shared_prefix = geo.shared_prefix, .geometry = geo });
        errdefer binding.deinit();
        try binding.upload(plan, &.{cache});
        p.nodes = try p.rig.input(std.mem.sliceAsBytes(plan.nodes));
        p.plan = plan;
        p.binding = binding;
        p.args = .{ .rows = rows, .packed_rows = plan.tiles * 16, .heads = geo.heads, .kv_heads = hk, .group = 6, .value_stride = if (p.bad) width else 2 * width };
        for (0..rows) |row| for (0..hk) |h| {
            const physical = start + row;
            const offset = if (p.pages) @as(usize, ids[physical / 256]) * page_bytes + (h * 256 + physical % 256) * D * 2 else (h * cap + physical) * D * 2;
            @memcpy(p.expected[offset..][0 .. D * 2], std.mem.sliceAsBytes(keys[(row * hk + h) * D ..][0..D]));
            @memcpy(p.expected[value_part + offset ..][0 .. D * 2], std.mem.sliceAsBytes(mixed[(row * 2 * hk + hk + h) * D ..][0..D]));
        };
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const b = &p.binding.?;
        e.setPipeline(p.pipe);
        e.setBuffer(p.keys.buf, p.keys.off, 0);
        e.setBuffer(p.mixed.buf, p.mixed.off + p.args.kv_heads * D * 2, 1);
        e.setBuffer(b.get(.caches).buffer, 0, 2);
        e.setBuffer(b.get(.meta).buffer, 0, 3);
        e.setBuffer(p.nodes.buf, p.nodes.off, 4);
        e.setValue(p.args, 5);
        e.useResource(p.output.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        if (p.pages) e.useResource(p.table.buf, mtl.ResourceUsage.read);
        e.dispatchThreads(.of(D, p.args.kv_heads, p.args.rows), .of(32, 4, 1));
        try p.rig.finish(pass);
    }
    pub fn count(_: *Probe) usize {
        return 0;
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {}
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        return if (std.mem.eql(u8, p.rig.bytes(p.output, p.expected.len), p.expected)) null else "mixed K/V cache payload or untouched slots differ";
    }
};

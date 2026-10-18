//! Every target and assistant embedding width uses the production specialization and guarded, strided token rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const family = tf.gemma4.contracts;
pub const entries = family.embed_entries;
const vocab = 5;
const scale: f32 = 0.75;
fn depth(_: c.Shape) usize {
    return 2;
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: family.Embed,
    pipe: mtl.Pipeline,
    shapes: [2]c.Shape,
    rows: usize = 0,
    ids: []u32 = &.{},
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    hidden: []u16 = &.{},
    args: [7]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: family.Embed) !Probe {
        const source = try v.source(a);
        defer a.free(source);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .pipe = try rig_.pipeline(device, source, v.symbol()), .shapes = .{ .{ .rows = 1, .n = v.geo.hidden, .k = v.geo.hidden }, .{ .rows = 17, .n = v.geo.hidden, .k = v.geo.hidden } } };
    }
    fn drop(p: *Probe) void {
        p.a.free(p.ids);
        p.a.free(p.codes);
        p.a.free(p.scales);
        p.a.free(p.biases);
        p.a.free(p.hidden);
        p.ids = &.{};
        p.codes = &.{};
        p.scales = &.{};
        p.biases = &.{};
        p.hidden = &.{};
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }
    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = p.v.name, .accum = .none, .depth = depth, .ops = 3, .mids = 1, .out = .bf16, .served = &p.shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "dequantization and hidden copies have no accumulation" } } };
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const d = p.v.geo.hidden;
        if (s.k != d or s.n != d or s.rows == 0 or s.rows > 129) return false;
        p.drop();
        p.rows = s.rows;
        p.ids = try p.a.alloc(u32, 2 * p.rows);
        for (p.ids, 0..) |*id, i| id.* = if (i % 2 != 0) 1 else if (p.v.assistant and i % 6 == 4) vocab + 10 else @intCast((i / 2) % vocab);
        p.codes = try p.a.alloc(u8, vocab * d);
        const storage = try p.a.alloc(u8, vocab * d * p.v.bits / 8);
        defer p.a.free(storage);
        @memset(storage, 0);
        for (p.codes, 0..) |*code, i| {
            code.* = @intCast((i *% 37 +% seed) % (@as(usize, 1) << @intCast(p.v.bits)));
            // Pack one bit at a time so the oracle never decodes with the shader's two-byte extraction.
            for (0..p.v.bits) |bit| {
                const at = i * p.v.bits + bit;
                storage[at / 8] |= @as(u8, @intCast((code.* >> @as(u3, @intCast(bit))) & 1)) << @as(u3, @intCast(at % 8));
            }
        }
        p.scales = try p.a.alloc(u16, vocab * d / 64);
        p.biases = try p.a.alloc(u16, p.scales.len);
        for (p.scales, p.biases, 0..) |*sc, *bias, i| {
            sc.* = c.toBf16(@as(f32, @floatFromInt(i % 7 + 1)) / 32);
            bias.* = c.toBf16(@as(f32, @floatFromInt(i % 5)) / 8 - 0.25);
        }
        if (fill == .nonfinite) p.scales[0] = 0x7fc0;
        p.hidden = try p.a.alloc(u16, p.rows * d);
        for (p.hidden, 0..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt(i % 31)) / 8 - 1);
        if (fill == .nonfinite and p.v.assistant) p.hidden[5] = 0x7f80;
        // The target has contiguous IDs; the assistant reads every second ID and clamps an out-of-vocabulary pick.
        const target_ids = try p.a.alloc(u32, p.rows);
        defer p.a.free(target_ids);
        for (target_ids, 0..) |*id, r| id.* = p.ids[2 * r];
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(if (p.v.assistant) p.ids else target_ids));
        const stride = [_]u32{ 2, vocab };
        p.args[1] = try p.rig.input(std.mem.asBytes(&stride));
        p.args[2] = try p.rig.input(storage);
        p.args[3] = try p.rig.input(std.mem.sliceAsBytes(p.scales));
        p.args[4] = try p.rig.input(std.mem.sliceAsBytes(p.biases));
        p.args[5] = try p.rig.input(std.mem.sliceAsBytes(p.hidden));
        p.args[6] = try p.rig.input(std.mem.asBytes(&scale));
        p.y = try p.rig.output(p.count() * 2);
        return true;
    }
    pub fn count(p: *Probe) usize {
        return p.rows * p.v.geo.hidden * (if (p.v.assistant) @as(usize, 2) else 1);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        if (p.v.assistant) {
            for (p.args[0..6], 0..) |arg, i| e.setBuffer(arg.buf, arg.off, i);
            e.setBuffer(p.y.buf, p.y.off, 6);
            e.setBuffer(p.args[6].buf, p.args[6].off, 7);
        } else {
            for ([_]rig_.Ref{ p.args[0], p.args[2], p.args[3], p.args[4], p.y, p.args[6] }, 0..) |arg, i| e.setBuffer(arg.buf, arg.off, i);
        }
        e.dispatchThreads(mtl.Size.of(p.v.geo.hidden / 2, p.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const d = p.v.geo.hidden;
        const stride: usize = if (p.v.assistant) 2 * d else d;
        for (0..p.rows) |r| {
            const id = @min(p.ids[2 * r], vocab - 1);
            for (0..d) |i| {
                const at = id * (d / 64) + i / 64;
                const weight = @import("gemma4_math.zig").roundedWeight(c.fromBf16(p.scales[at]), p.codes[id * d + i], c.fromBf16(p.biases[at]));
                const value = weight * scale;
                want[r * stride + i] = c.fromBf16(c.toBf16(@floatCast(value)));
                mag[r * stride + i] = @abs(value);
                if (p.v.assistant) {
                    want[r * stride + d + i] = c.fromBf16(p.hidden[r * d + i]);
                    mag[r * stride + d + i] = @abs(want[r * stride + d + i]);
                }
            }
        }
    }
};
pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    var problems: usize = 0;
    for (entries) |v| {
        if (only) |s| if (std.mem.indexOf(u8, v.name, s) == null) continue;
        const source = try v.source(a);
        defer a.free(source);
        problems += try @import("metal_limits.zig").onThisGpu(a, device, .{ .decl = .{ .name = v.name, .threads = 256, .tg_memory = 0, .offline = false }, .function = v.symbol(), .here = source, .pre_m5 = "" }, false);
    }
    return problems;
}

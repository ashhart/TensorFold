//! Three mixed-width runs write separate columns through the actual family encoder; all padding stays guarded.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const weights = tf.gemma4.weights;
const lane = tf.lane_projection;
const bits = [_]u8{ 2, 5, 8 };
const pitch = 128;
const col = 16;
const name = "gemma4 mixed q2 q5 q8 output pitch";
fn depth(_: c.Shape) usize {
    return 320;
}
fn ref(r: rig_.Ref) lane.Ref {
    return .{ .buf = r.buf, .off = r.off };
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    rows: usize = 0,
    shapes: [3]c.Shape = .{ .{ .rows = 1, .n = 96, .k = 256 }, .{ .rows = 17, .n = 96, .k = 256 }, .{ .rows = 33, .n = 96, .k = 256 } },
    f: [3]?contract.affine.Affine = @splat(null),
    projections: [3]?lane.Projection = @splat(null),
    runs: [3]weights.Linear = undefined,
    storage: [3][2]rig_.Ref = undefined,
    x: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    bytes: std.ArrayList(u8) = .empty,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue } };
    }
    fn drop(p: *Probe) void {
        for (&p.f, &p.projections) |*f, *projection| {
            if (f.*) |v| v.deinit();
            f.* = null;
            if (projection.*) |v| v.deinit();
            projection.* = null;
        }
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.bytes.deinit(p.a);
    }
    const vt = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        v.intact = intact;
        break :blk v;
    };
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = name, .accum = .f32, .depth = depth, .out = .bf16, .served = &p.shapes } };
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.bytes.clearRetainingCapacity();
        for (0..p.rows) |r| try p.bytes.appendSlice(p.a, p.rig.bytes(p.y, p.rows * pitch * 2)[(r * pitch + col) * 2 ..][0 .. 96 * 2]);
        return p.bytes.items;
    }
    fn intact(ctx: *anyopaque) !bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (!try p.rig.intact()) return false;
        const data = p.rig.bytes(p.y, p.rows * pitch * 2);
        for (0..p.rows) |r| for (0..pitch) |i| if (i < col or i >= col + 96) {
            if (data[(r * pitch + i) * 2] != 0xff or data[(r * pitch + i) * 2 + 1] != 0xff) return false;
        };
        return true;
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.k != 256 or s.n != 96 or s.rows == 0 or s.rows > 33) return false;
        p.drop();
        p.rows = s.rows;
        p.y = try p.rig.output(p.rows * pitch * 2);
        for (bits, 0..) |width, j| {
            var f = try contract.affine.Affine.init(p.a, .{ .rows = p.rows, .n = 32, .k = 256 }, width, 64);
            p.f[j] = f;
            try f.fill(fill, seed + j);
            if (j == 0) p.x = try p.rig.input(std.mem.sliceAsBytes(f.x)) else @memcpy(f.x, p.f[0].?.x);
            const x = weights.Linear{ .w = undefined, .sb = undefined, .n = 32, .k = 256, .format = .{ .bits = width, .group = 64 }, .col = j * 32, .y_stride = pitch };
            const l = weights.layout(tf.gemma4.config.unified12b, x);
            const words = try p.a.alloc(u32, l.sourceWords());
            defer p.a.free(words);
            f.words(words);
            const pw = try p.a.alloc(u32, l.weightWords());
            defer p.a.free(pw);
            const sb = try p.a.alloc(u16, l.metadataElements());
            defer p.a.free(sb);
            try lane.pack(l, words, f.scales, f.biases, pw, sb);
            const w = try p.rig.input(std.mem.sliceAsBytes(pw));
            const meta = try p.rig.input(std.mem.sliceAsBytes(sb));
            p.runs[j] = x;
            p.storage[j] = .{ w, meta };
            p.projections[j] = try lane.Projection.init(p.a, p.rig.device, l);
        }
        return true;
    }
    pub fn count(p: *Probe) usize {
        return p.rows * 96;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        for (p.runs, p.projections, p.storage) |x, projection, storage| try x.encodeRefsRows(projection.?, pass.e, ref(p.x), ref(storage[0]), ref(storage[1]), .{ .buf = p.y.buf, .off = p.y.off + (col + x.col) * 2 }, p.rows);
        try p.rig.finish(pass);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const tmp = try p.a.alloc(f64, p.rows * 32 * 2);
        defer p.a.free(tmp);
        for (p.f, 0..) |f, j| {
            try f.?.reference(tmp[0 .. p.rows * 32], tmp[p.rows * 32 ..]);
            for (0..p.rows) |r| {
                @memcpy(want[r * 96 + j * 32 ..][0..32], tmp[r * 32 ..][0..32]);
                @memcpy(mag[r * 96 + j * 32 ..][0..32], tmp[p.rows * 32 + r * 32 ..][0..32]);
            }
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const tmp = try p.a.alloc(f64, p.rows * 32);
        defer p.a.free(tmp);
        for (p.f, 0..) |f, j| {
            try f.?.simulate(acc, order, tmp);
            for (0..p.rows) |r| @memcpy(out[r * 96 + j * 32 ..][0..32], tmp[r * 32 ..][0..32]);
        }
    }
};
pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    if (only) |s| if (std.mem.indexOf(u8, name, s) == null) return 0;
    var problems: usize = 0;
    for (bits) |width| {
        const x = weights.Linear{ .w = undefined, .sb = undefined, .n = 32, .k = 256, .format = .{ .bits = width, .group = 64 }, .y_stride = pitch };
        const l = weights.layout(tf.gemma4.config.unified12b, x);
        const raw = try lane.source(a, l);
        defer a.free(raw);
        const source = try familySource(device, a, raw);
        defer a.free(source);
        const label = try std.fmt.allocPrint(a, "{s} q{d}", .{ name, width });
        defer a.free(label);
        problems += try @import("metal_limits.zig").onThisGpu(a, device, .{ .decl = .{ .name = label, .threads = @intCast(32 * l.sk), .tg_memory = @intCast(l.threadgroupBytes()), .offline = false }, .function = "tf_lane", .here = source, .pre_m5 = "" }, false);
    }
    return problems;
}
const familySource = tf.gemma4.contracts.adapt;

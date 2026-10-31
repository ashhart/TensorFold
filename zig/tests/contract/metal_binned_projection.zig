//! Exponent-bin group dots and compensated affine sums against independent FP64 weights and real failure controls.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const tf = @import("tensorfold");
const affine = @import("metal_affine.zig");
const rig_ = @import("rig.zig");
const bp = tf.lfm3.binned_projection;
const c = ct.spec;
const g = ct.gates;
pub const Fault = enum { none, rows, accum, nonfinite };

pub fn label(a: std.mem.Allocator, bits: u8, group: usize, fault: Fault) ![]u8 {
    return std.fmt.allocPrint(a, "{s}LFM3 binned q{d} g{d} {t}", .{ if (fault == .none) "metal " else "control: ", bits, group, fault });
}

fn depth(shape: c.Shape) usize {
    return 64 + 2 * (shape.k / 32) + 8;
}

pub const Probe = struct {
    inner: affine.Proj,
    pipelines: bp.Pipelines,
    fault: Fault,
    label: []u8,
    sums: rig_.Ref = undefined,
    meta: rig_.Ref = undefined,
    served: [3]c.Shape = .{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 33, .n = 512, .k = 2048 }, .{ .rows = 65, .n = 64, .k = 7680 } },
    const vt = blk: {
        var hooks = g.hooks(affine.Proj);
        hooks.prepare = prepare;
        hooks.run = run;
        break :blk hooks;
    };
    fn parent(ctx: *anyopaque) *Probe {
        const inner: *affine.Proj = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, group: usize, fault: Fault) !Probe {
        const name = try label(a, bits, group, fault);
        errdefer a.free(name);
        var inner = try affine.Proj.init(a, device, queue, .{ .kind = .row, .bits = bits, .group = group }, name);
        errdefer inner.deinit();
        const text = try bp.source(device, a);
        defer a.free(text);
        const source = try std.mem.concat(a, u8, &.{ switch (fault) {
            .none => "",
            .rows => "#define TF_BP_BAD_ROWS 1\n",
            .accum => "#define TF_BP_BAD_ACC 1\n",
            .nonfinite => "#define TF_BP_BAD_NONFINITE 1\n",
        }, text });
        defer a.free(source);
        return .{ .inner = inner, .pipelines = try bp.Pipelines.loadSource(device, source), .fault = fault, .label = name };
    }
    pub fn deinit(p: *Probe) void {
        p.pipelines.deinit();
        p.inner.deinit();
        p.inner.a.free(p.label);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var spec = affine.spec(p.inner.v, p.label);
        spec.served = &p.served;
        spec.depth = depth;
        spec.tiles = .{ .{ .size = 32 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 64, .step = 64, .min = 64 } };
        return .{ .ctx = &p.inner, .vt = &vt, .spec = spec, .control = switch (p.fault) {
            .none => null,
            .rows => .shapes,
            .accum => .sums,
            .nonfinite => .nonfinite,
        } };
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (shape.n % 32 != 0 or shape.k % 64 != 0) return false;
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        if (fill == .nonfinite and shape.rows >= 5 and shape.n >= 6) {
            const f = p.inner.f.?;
            const groups = shape.k / p.inner.v.group;
            // A finite row with positive sum tests infinity signs without a spurious FMA residual.

            @memset(f.x[(shape.rows - 1) * shape.k ..][0..shape.k], c.toBf16(1));
            const values = [_]f32{ std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
            for (values, 0..) |value, i| {
                f.biases[i * groups] = c.toBf16(value);
                f.scales[(i + 3) * groups] = c.toBf16(value);
            }
            if (p.inner.v.bits >= 6) {
                // Nibble/bin accumulation leaves (0, 2); infinite scale must multiply that low-component sum.

                const input = f.x[(shape.rows - 1) * shape.k ..][0..p.inner.v.group];
                @memset(input, c.toBf16(0));
                input[0] = c.toBf16(0x1p26);
                input[1] = c.toBf16(-0x1p22);
                input[2] = c.toBf16(2);
                for (3..6) |col| {
                    const codes = f.codes[col * shape.k ..][0..p.inner.v.group];
                    @memset(codes, 0);
                    codes[0] = 1;
                    codes[1] = 16;
                    codes[2] = 1;
                }
                const words = try p.inner.a.alloc(u32, shape.n * shape.k * p.inner.v.bits / 32);
                defer p.inner.a.free(words);
                f.words(words);
                @memcpy(p.inner.rig.bytes(p.inner.w, words.len * 4), std.mem.sliceAsBytes(words));
            }
            for ([_]struct { ref: rig_.Ref, data: []const u8 }{
                .{ .ref = p.inner.x, .data = std.mem.sliceAsBytes(f.x) },
                .{ .ref = p.inner.s, .data = std.mem.sliceAsBytes(f.scales) },
                .{ .ref = p.inner.b, .data = std.mem.sliceAsBytes(f.biases) },
            }) |input| @memcpy(p.inner.rig.bytes(input.ref, input.data.len), input.data);
        }
        const bytes = try bp.scratchBytes(shape.k, p.inner.v.group, shape.rows);
        p.sums = try p.inner.rig.scratch(bytes);
        p.meta = try p.inner.rig.scratch(bytes);
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const base = &p.inner;
        const shape = base.f.?.shape;
        const w = tf.row_projection.Weights{ .w = base.w.buf, .w_off = base.w.off, .scales = base.s.buf, .s_off = base.s.off, .biases = base.b.buf, .b_off = base.b.off, .n = shape.n, .k = shape.k, .group = base.v.group, .bits = base.v.bits };
        const pass = base.rig.begin();
        try bp.encode(pass.e, &p.pipelines, w, .{ .buf = base.x.buf, .off = base.x.off }, .{ .buf = p.sums.buf, .off = p.sums.off }, .{ .buf = p.meta.buf, .off = p.meta.off }, .{ .buf = base.out.buf, .off = base.out.off }, shape.rows);
        try base.rig.finish(pass);
    }
};

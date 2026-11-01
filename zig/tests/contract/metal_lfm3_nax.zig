//! The tensor-unit prompt projection at every served LFM3 width; model FP32/BF16 envelopes remain separate gates.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const affine = @import("metal_affine.zig");
const rig = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
const nax = tf.lfm3.prompt_nax;
pub const Fault = enum { none, bias, nonfinite };
pub const formats = [_]struct { bits: u8, group: u16 }{ .{ .bits = 4, .group = 64 }, .{ .bits = 6, .group = 64 }, .{ .bits = 8, .group = 64 }, .{ .bits = 4, .group = 32 } };

pub fn name(a: std.mem.Allocator, bits: u8, group: u16, fault: Fault) ![]const u8 {
    const lead = switch (fault) {
        .none => "metal",
        .bias => "control: bad-bias",
        .nonfinite => "control: bad-nonfinite",
    };
    return std.fmt.allocPrint(a, "{s} LFM3 NAX prompt q{d}g{d}", .{ lead, bits, group });
}
/// The fp32 sum's k terms, plus each dequantized weight's bf16 rounding (2^-9 relative) in units of 2^-24.
fn depth(s: c.Shape) usize {
    return s.k + 8 + (1 << 15);
}
const needle_bias = " - 1u)) + b));";
const needle_store = "  if (col + tn < N && live > 0)\n    tfqw::store<bfloat16_t, 2>(acc, Y + long(row + tm) * LD";
const sanitize = "  TF_UNROLL\n  for (short i = 0; i < 2; i++) for (short j = 0; j < 2; j++) for (short e = 0; e < 8; e++)\n    if (!isfinite(acc[i][j][e])) acc[i][j][e] = 0.0f;\n";

pub const Probe = struct {
    a: std.mem.Allocator,
    inner: affine.Proj,
    pipe: mtl.Pipeline,
    label: []const u8,
    fault: Fault,
    group: u16,
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
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, group: u16, fault: Fault) !Probe {
        if (!device.tensorUnits()) return error.NaxPromptUnsupported;
        const label = try name(a, bits, group, fault);
        errdefer a.free(label);
        var inner = try affine.Proj.init(a, device, queue, .{ .kind = .simd_bits, .bits = bits, .group = group }, label);
        errdefer inner.deinit();
        const clean = try nax.source(device, a, bits, group);
        defer a.free(clean);
        const text = switch (fault) {
            .none => try a.dupe(u8, clean),
            .bias => try std.mem.replaceOwned(u8, a, clean, needle_bias, " - 1u)) + 0.0f));"),
            .nonfinite => try std.mem.replaceOwned(u8, a, clean, needle_store, sanitize ++ needle_store),
        };
        defer a.free(text);
        if (fault != .none and std.mem.eql(u8, text, clean)) return error.MissingControlSite;
        const pipe = try rig.pipeline(device, text, "tf_qmmw_t_nax_win");
        errdefer pipe.deinit();
        if (pipe.simdWidth() != 32 or pipe.maxThreads() < 128 or rig.tgMemory(pipe) != 11264) return error.NaxPromptLimits;
        return .{ .a = a, .inner = inner, .pipe = pipe, .label = label, .fault = fault, .group = group };
    }
    pub fn deinit(p: *Probe) void {
        p.pipe.deinit();
        p.inner.deinit();
        p.a.free(p.label);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        const window: usize = 8 * @as(usize, p.group);
        return .{ .ctx = &p.inner, .vt = &vt, .control = switch (p.fault) {
            .none => null,
            .bias => .shapes,
            .nonfinite => .nonfinite,
        }, .spec = .{ .name = p.label, .accum = .f32, .depth = depth, .out = .bf16, .tiles = .{ .{ .size = 64, .min = 1 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 64, .step = window, .min = window } }, .served = &.{ .{ .rows = 128, .n = 7680, .k = 2048 }, .{ .rows = 128, .n = 2048, .k = 7680 }, .{ .rows = 65, .n = 6144, .k = 2048 }, .{ .rows = 33, .n = 512, .k = 2048 }, .{ .rows = 5, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 2048 } } } };
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (shape.rows == 0 or shape.n % 64 != 0 or shape.k % (8 * @as(usize, p.group)) != 0) return false;
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        if (p.inner.f) |*f| f.form = .dequantized;
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        const s = b.f.?.shape;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for ([_]rig.Ref{ b.w, b.s, b.b, b.x }, 0..) |r, at| e.setBuffer(r.buf, r.off, at);
        e.setValue(extern struct { k: i32, n: i32, m: i32, ld: i32 }{ .k = @intCast(s.k), .n = @intCast(s.n), .m = @intCast(s.rows), .ld = 0 }, 4);
        e.setBuffer(b.out.buf, b.out.off, 5);
        e.dispatchGroups(.of(s.n / 64, (s.rows + 63) / 64, 1), .of(128, 1, 1));
        try b.rig.finish(pass);
    }
};

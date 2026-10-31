//! Residual FFN launches through guarded buffers; model FP32/BF16 envelopes remain separate.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const affine = @import("metal_affine.zig");
const rig = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
pub const Fault = enum { none, pack, nonfinite };
pub fn name(fault: Fault) []const u8 {
    return switch (fault) {
        .none => "metal LFM3 residual FFN",
        .pack => "control: LFM3 residual FFN pack",
        .nonfinite => "control: LFM3 residual FFN nonfinite",
    };
}
fn depth(s: c.Shape) usize {
    return s.k + 4;
}
const Args = extern struct { k: i32, n: i32, rows: i32, one: f32 = 1 };
pub const Probe = struct {
    inner: affine.Proj,
    pipes: [2]mtl.Pipeline,
    flags: rig.Ref = undefined,
    fault: Fault,
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
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, fault: Fault) !Probe {
        var inner = try affine.Proj.init(a, device, queue, .{ .kind = .row }, name(fault));
        errdefer inner.deinit();
        const text = try std.mem.concat(a, u8, &.{ switch (fault) {
            .none => "",
            .pack => "#define L3_NAX_BAD_PACK 1\n",
            .nonfinite => "#define L3_NAX_BAD_NF 1\n",
        }, sources.core_row_projection, "\n", sources.nax, "\n", sources.lfm3_residual_ffn });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const first = try mtl.Pipeline.init(device, lib, "lfm3_nax_prompt_q4", false);
        errdefer first.deinit();
        const second = try mtl.Pipeline.init(device, lib, "lfm3_nax_nf_repair_q4", false);
        errdefer second.deinit();
        if (first.simdWidth() != 32 or second.simdWidth() != 32 or first.maxThreads() < 128 or
            second.maxThreads() < 64 or rig.tgMemory(first) != 18448 or rig.tgMemory(second) != 0)
            return error.ResidualLaunchLimits;
        return .{ .inner = inner, .pipes = .{ first, second }, .fault = fault };
    }
    pub fn deinit(p: *Probe) void {
        for (p.pipes) |pipe| pipe.deinit();
        p.inner.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = &p.inner, .vt = &vt, .control = switch (p.fault) {
            .none => null,
            .pack => .shapes,
            .nonfinite => .nonfinite,
        }, .spec = .{ .name = name(p.fault), .accum = .f32, .depth = depth, .out = .bf16, .tiles = .{ .{ .size = 64, .min = 33 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 64, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 128, .n = 7680, .k = 2048 }, .{ .rows = 128, .n = 2048, .k = 7680 }, .{ .rows = 33, .n = 128, .k = 64 } } } };
    }
    fn prepare(ctx: *anyopaque, requested: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        var shape = requested;
        if ((fill == .nonfinite or fill == .stress) and shape.rows < 33) {
            shape.rows = 33;
            std.debug.print("FIXTURE {s}: {t} rows {d} expanded to 33 for the enabled route\n", .{ name(p.fault), fill, requested.rows });
        }
        if (shape.rows < 33 or shape.rows > 128 or shape.n % 64 != 0 or shape.k % 64 != 0) return false;
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        p.flags = try p.inner.rig.scratch(((shape.rows + 63) / 64) * (shape.n / 64) * 4);
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        const s = b.f.?.shape;
        const pass = b.rig.begin();
        const e = pass.e;
        for (p.pipes, 0..) |pipe, i| {
            e.setPipeline(pipe);
            for ([_]rig.Ref{ b.x, b.w, b.s, b.b }, 0..) |r, at| e.setBuffer(r.buf, r.off, at);
            e.setValue(Args{ .k = @intCast(s.k), .n = @intCast(s.n), .rows = @intCast(s.rows) }, 4);
            e.setBuffer(b.out.buf, b.out.off, 5);
            e.setBuffer(p.flags.buf, p.flags.off, 6);
            if (i == 0) {
                e.setBuffer(p.flags.buf, p.flags.off, 7);
                e.dispatchGroups(.of(s.n / 64, (s.rows + 63) / 64, 1), .of(128, 1, 1));
            } else e.dispatchGroups(.of((s.rows + 63) / 64, s.n / 64, 1), .of(64, 1, 1));
            e.barrier();
        }
        try b.rig.finish(pass);
    }
};

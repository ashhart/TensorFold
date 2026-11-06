//! Unlimited-OCR-2's image-encoder kernels under the contract on CUDA: what the map and attention groups share.
const std = @import("std");
const core = @import("core");
const hub = @import("contract_uocr.zig");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
pub const rows = core.contract.rows;
pub const steps = hub.kernels.vision;

pub const S = hub.sam_width;
/// SAM's heads and head width, and its window side.
pub const heads = 12;
pub const dh = 64;
pub const window = 14;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

/// A planted control: a kernel with one fault (contract_uocr_vision.cuh), which one gate must report.
pub const Control = struct {
    /// The control's function in the controls image.
    function: [:0]const u8,
    name: []const u8,
    gate: g.Gate,
};

/// A kernel's spec renamed and marked for `control`.
pub fn mark(k: g.Kernel, control: ?Control) g.Kernel {
    var out = k;
    if (control) |ctl| {
        out.spec.name = ctl.name;
        out.control = ctl.gate;
    }
    return out;
}

/// bf16 bits of random values in [lo, hi).
pub fn fill(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// The least side whose square holds `n`.
pub fn sideOf(n: usize) usize {
    var s: usize = 1;
    while (s * s < n) s += 1;
    return s;
}

/// Every `M.Kind` as a `T` registration (`init`, `kernel`, `deinit`), then `M.controls`; `M.function` names the kernel.
pub fn register(comptime M: type, comptime T: type, ctx: hub.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !hub.Held {
    const Set = struct {
        a: std.mem.Allocator,
        made: std.ArrayList(*T) = .empty,

        fn add(o: *@This(), cx: hub.Ctx, l: *std.ArrayList(g.Kernel), kind: M.Kind, control: ?Control) !void {
            const k = if (control) |ctl| switch (kind) {
                inline else => |kd| try cx.with(comptime M.function(kd), ctl.function),
            } else cx.k.*;
            const x = try o.a.create(T);
            errdefer o.a.destroy(x);
            x.* = T.init(cx.rig(), k, kind, control);
            try o.made.append(o.a, x);
            try l.append(o.a, x.kernel());
        }

        fn free(p: *anyopaque) void {
            const o: *@This() = @ptrCast(@alignCast(p));
            for (o.made.items) |x| {
                x.deinit();
                o.a.destroy(x);
            }
            o.made.deinit(o.a);
            o.a.destroy(o);
        }
    };
    const set = try a.create(Set);
    set.* = .{ .a = a };
    errdefer Set.free(set);
    for (std.enums.values(M.Kind)) |kind| try set.add(ctx, list, kind, null);
    for (M.controls) |ctl| try set.add(ctx, list, ctl[0], ctl[1]);
    return .{ .ptr = set, .free = Set.free };
}

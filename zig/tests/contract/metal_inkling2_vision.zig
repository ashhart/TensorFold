//! Independently referenced Mini image-tower gates; controls compile a private copy of the production source.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = @import("contract").gates;
const sources = @import("inkling2_metal");
const controls = @import("metal_inkling2_vision_controls.zig");
const gm = tf.inkling2_metal_host.vision.gemm;
const gemm = @import("metal_inkling2_vision_gemm.zig");
const rows = @import("metal_inkling2_vision_rows.zig");
const glue = @import("metal_inkling2_vision_glue.zig");

const Held = struct {
    ptr: *anyopaque,
    drop: *const fn (std.mem.Allocator, *anyopaque) void,
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, name, only orelse return true) != null;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(Held) = .empty,
    production_lib: ?mtl.Library = null,
    mm: ?*gm.Gemm = null,

    fn hold(s: *Suite, comptime T: type, value: T) !*T {
        var v = value;
        errdefer v.deinit();
        const p = try s.a.create(T);
        errdefer s.a.destroy(p);
        p.* = v;
        const drop = struct {
            fn release(a: std.mem.Allocator, ptr: *anyopaque) void {
                const t: *T = @ptrCast(@alignCast(ptr));
                t.deinit();
                a.destroy(t);
            }
        }.release;
        try s.held.append(s.a, .{ .ptr = p, .drop = drop });
        return p;
    }

    fn library(s: *Suite, text: []const u8) !mtl.Library {
        const src = try tf.frags.source(s.device, s.a, text);
        defer s.a.free(src);
        return mtl.Library.fromSource(s.device, src, mtl.CompileOptions.mlx());
    }

    fn production(s: *Suite) !mtl.Library {
        if (s.production_lib == null) s.production_lib = try s.library(sources.vision);
        return s.production_lib.?;
    }

    fn mutated(s: *Suite, kind: controls.Kind) !mtl.Library {
        const text = try controls.source(s.a, sources.vision, kind);
        defer s.a.free(text);
        return s.library(text);
    }

    fn plain(s: *Suite, comptime M: type, only: ?[]const u8) !void {
        inline for (std.enums.values(M.Kind)) |k| {
            const label = comptime M.name(k);
            if (wanted(only, label)) {
                const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, try s.production(), k, label));
                try s.kernels.append(s.a, p.kernel(null));
            }
            const control = comptime @field(controls.Kind, @tagName(k));
            const e = comptime controls.edit(control);
            if (wanted(only, e.label)) {
                const lib = try s.mutated(control);
                defer lib.deinit();
                const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, lib, k, e.label));
                try s.kernels.append(s.a, p.kernel(e.gate));
            }
        }
    }

    fn gemmControl(s: *Suite, only: ?[]const u8, kind: controls.Kind, run: gm.Kind) !void {
        const e = controls.edit(kind);
        if (!wanted(only, e.label)) return;
        const lib = try s.mutated(kind);
        defer lib.deinit();
        const mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, lib, &.{run}));
        const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, mm, run, e.label));
        try s.kernels.append(s.a, p.kernel(e.gate));
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(gm.Kind)) |k| if (wanted(only, gemm.name(k))) {
            if (s.mm == null) s.mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, try s.production(), std.enums.values(gm.Kind)));
            const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, s.mm.?, k, gemm.name(k)));
            try s.kernels.append(s.a, p.kernel(null));
        };
        try s.gemmControl(only, .gemm_tail, .bf16f_nt);
        // Without tensor units a few-row GEMM runs the 8x8 simdgroup kernel; its fault is only in that text.
        if (!s.device.tensorUnits()) try s.gemmControl(only, .gemm_rows, .bf16_nt);
        try s.plain(rows, only);
        try s.plain(glue, only);
    }

    pub fn deinit(s: *Suite) void {
        var i = s.held.items.len;
        while (i > 0) {
            i -= 1;
            const p = s.held.items[i];
            p.drop(s.a, p.ptr);
        }
        s.held.deinit(s.a);
        if (s.production_lib) |lib| lib.deinit();
    }
};

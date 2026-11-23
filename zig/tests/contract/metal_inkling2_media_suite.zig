//! Independently referenced Mini image/draft gates; controls compile a private copy of the production source.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ks = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const gm = tf.inkling2_metal.vision.gemm;
const gemm = @import("metal_inkling2_media_gemm.zig");
const rows = @import("metal_inkling2_media_rows.zig");
const attn = @import("metal_inkling2_media_attn.zig");
const dops = @import("metal_inkling2_media_draft.zig");
const vision = @import("metal_inkling2_media_vision.zig");
const rotary = @import("metal_inkling2_media_rotary.zig");
const step = @import("metal_inkling2_media_step.zig");

const Held = struct {
    ptr: *anyopaque,
    drop: *const fn (std.mem.Allocator, *anyopaque) void,
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, name, only orelse return true) != null;
}

fn raw(vis: bool) []const u8 {
    return if (vis) ks.inkling2_vision else ks.inkling2_draft;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(u.g.Kernel),
    held: std.ArrayList(Held) = .empty,
    libs: [2]?mtl.Library = @splat(null),
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

    fn production(s: *Suite, vis: bool) !mtl.Library {
        const i: usize = @intFromBool(vis);
        if (s.libs[i] == null) s.libs[i] = try s.library(raw(vis));
        return s.libs[i].?;
    }

    fn mutate(s: *Suite, vis: bool, from: []const u8, to: []const u8) !mtl.Library {
        const text = try u.mutate(s.a, raw(vis), from, to);
        defer s.a.free(text);
        return s.library(text);
    }

    fn plain(s: *Suite, comptime M: type, only: ?[]const u8) !void {
        inline for (std.enums.values(M.Kind)) |k| {
            const label = comptime M.name(k);
            if (wanted(only, label)) {
                const lib = try s.production(M.vision(k));
                const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, lib, k, label));
                try s.kernels.append(s.a, p.kernel(null));
            }
        }
        inline for (M.controls) |m| {
            if (wanted(only, m[1])) {
                const lib = try s.mutate(M.vision(m[0]), m[2], m[3]);
                defer lib.deinit();
                const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, lib, m[0], m[1]));
                try s.kernels.append(s.a, p.kernel(m[4]));
            }
        }
    }

    fn composite(s: *Suite, comptime M: type, only: ?[]const u8) !void {
        if (wanted(only, M.name)) {
            const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, try s.production(false), M.name));
            try s.kernels.append(s.a, p.kernel(null));
        }
        if (wanted(only, M.control.label)) {
            const lib = try s.mutate(false, M.control.from, M.control.to);
            defer lib.deinit();
            const p = try s.hold(M.Probe, try M.Probe.init(s.a, s.device, s.queue, lib, M.control.label));
            try s.kernels.append(s.a, p.kernel(.shapes));
        }
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(gm.Kind)) |k| if (wanted(only, gemm.name(k))) {
            if (s.mm == null) s.mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, try s.production(true), std.enums.values(gm.Kind)));
            const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, s.mm.?, k, gemm.name(k)));
            try s.kernels.append(s.a, p.kernel(null));
        };
        if (wanted(only, gemm.control.label)) {
            const lib = try s.mutate(true, gemm.control.from, gemm.control.to);
            defer lib.deinit();
            const mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, lib, &.{.bf16f_nt}));
            const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, mm, .bf16f_nt, gemm.control.label));
            try s.kernels.append(s.a, p.kernel(.shapes));
        }
        if (wanted(only, gemm.control_score.label)) {
            const lib = try s.mutate(true, gemm.control_score.from, gemm.control_score.to);
            defer lib.deinit();
            const mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, lib, &.{.score_nt}));
            const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, mm, .score_nt, gemm.control_score.label));
            try s.kernels.append(s.a, p.kernel(.shapes));
        }
        if (!s.device.tensorUnits() and wanted(only, gemm.control_rows.label)) {
            const lib = try s.mutate(true, gemm.control_rows.from, gemm.control_rows.to);
            defer lib.deinit();
            const mm = try s.hold(gm.Gemm, try gm.Gemm.init(s.device, lib, &.{.bf16_nt}));
            const p = try s.hold(gemm.Probe, gemm.Probe.init(s.a, s.device, s.queue, mm, .bf16_nt, gemm.control_rows.label));
            try s.kernels.append(s.a, p.kernel(.shapes));
        }
        try s.plain(rows, only);
        try s.plain(dops, only);
        try s.plain(vision, only);
        try s.plain(rotary, only);
        try s.composite(attn, only);
        try s.composite(step, only);
    }

    pub fn deinit(s: *Suite) void {
        var i = s.held.items.len;
        while (i > 0) {
            i -= 1;
            const p = s.held.items[i];
            p.drop(s.a, p.ptr);
        }
        s.held.deinit(s.a);
        for (s.libs) |l| if (l) |lib| lib.deinit();
    }
};

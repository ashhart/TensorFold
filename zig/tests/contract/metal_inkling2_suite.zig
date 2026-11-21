//! Backbone probes (kernels/metal/inkling2), each planted source fault in its own `Ops` so a control cannot leak.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const u = @import("inkling2_util.zig");
const conv = @import("metal_inkling2_conv.zig");
const norm = @import("metal_inkling2_norm.zig");
const attn_res = @import("metal_inkling2_attn_res.zig");
const route = @import("metal_inkling2_route.zig");
const glue = @import("metal_inkling2_glue.zig");
const prep = @import("metal_inkling2_prep.zig");
const linear = @import("metal_inkling2_linear.zig");
const attention = @import("metal_inkling2_attention.zig");
const g = contract.gates;
const Ops = u.ops.Ops;

fn wanted(only: ?[]const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, name, only orelse return true) != null;
}

/// A heap-held probe or Ops and how to release it.
const Held = struct {
    ptr: *anyopaque,
    drop: *const fn (std.mem.Allocator, *anyopaque) void,
};

fn Payload(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .error_union => |e| e.payload,
        else => T,
    };
}

fn unwrap(x: anytype) !Payload(@TypeOf(x)) {
    return if (@typeInfo(@TypeOf(x)) == .error_union) try x else x;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(Held) = .empty,
    production: ?*Ops = null,

    fn hold(s: *Suite, comptime T: type, value: anytype) !*T {
        const ptr = try s.a.create(T);
        errdefer s.a.destroy(ptr);
        ptr.* = try unwrap(value);
        const drop = struct {
            fn run(a: std.mem.Allocator, raw: *anyopaque) void {
                const t: *T = @ptrCast(@alignCast(raw));
                t.deinit();
                a.destroy(t);
            }
        }.run;
        try s.held.append(s.a, .{ .ptr = ptr, .drop = drop });
        return ptr;
    }

    fn production_ops(s: *Suite) !*Ops {
        if (s.production) |o| return o;
        const o = try s.hold(Ops, Ops.load(s.device, .{}));
        s.production = o;
        return o;
    }

    fn controlOps(s: *Suite, text: []const u8) !*Ops {
        return s.hold(Ops, Ops.loadSource(s.device, .{ .scratch_bytes = 16 << 20 }, text));
    }

    fn plain(s: *Suite, comptime M: type, only: ?[]const u8, kinds: []const M.Kind) !void {
        for (kinds) |kind| {
            const label = M.name(kind);
            if (!wanted(only, label)) continue;
            const p = try s.hold(M.Probe, M.Probe.init(s.a, s.device, s.queue, try s.production_ops(), kind, label));
            try s.kernels.append(s.a, p.kernel());
        }
    }

    fn controls(s: *Suite, comptime M: type, only: ?[]const u8) !void {
        for (M.mutations) |m| {
            if (!wanted(only, m.label)) continue;
            const text = try M.controlSource(s.a, m);
            defer s.a.free(text);
            const o = try s.controlOps(text);
            const p = try s.hold(M.Probe, M.Probe.init(s.a, s.device, s.queue, o, m.kind, m.label));
            const k = if (comptime @typeInfo(@TypeOf(M.controlKernel)).@"fn".param_types.len == 2) M.controlKernel(p, m.gate) else M.controlKernel(p);
            try s.kernels.append(s.a, k);
        }
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.plain(conv, only, std.enums.values(conv.Kind));
        try s.controls(conv, only);
        try s.plain(norm, only, std.enums.values(norm.Kind));
        try s.controls(norm, only);
        try s.plain(attn_res, only, std.enums.values(attn_res.Kind));
        try s.controls(attn_res, only);
        try s.plain(route, only, std.enums.values(route.Kind));
        try s.controls(route, only);
        try s.plain(glue, only, std.enums.values(glue.Kind));
        try s.controls(glue, only);
        try s.plain(prep, only, &prep.kinds);
        try s.controls(prep, only);
        try s.plain(attention, only, std.enums.values(attention.Kind));
        try s.controls(attention, only);
        try s.plain(linear, only, &linear.kinds);
        // The dense projection's faults are edits of the shared matvec and tile sources the probe builds itself.
        for (linear.controls) |ctl| {
            if (!wanted(only, ctl.label)) continue;
            const p = try s.hold(linear.Probe, linear.Probe.init(s.a, s.device, s.queue, try s.production_ops(), ctl.kind, ctl.label));
            try s.kernels.append(s.a, linear.controlKernel(p));
        }
    }

    pub fn deinit(s: *Suite) void {
        var i = s.held.items.len;
        while (i > 0) {
            i -= 1;
            s.held.items[i].drop(s.a, s.held.items[i].ptr);
        }
        s.held.deinit(s.a);
    }
};

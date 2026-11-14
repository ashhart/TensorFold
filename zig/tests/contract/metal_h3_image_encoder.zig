//! MiniMax-H3-Image's edit encoder kernels (h3_image_encoder.metal) under the contract, each against float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const common = @import("metal_h3_image_common.zig");
const rig_ = @import("rig.zig");
const cv = @import("metal_h3_image_encoder_conv.zig");
const nm = @import("metal_h3_image_encoder_norms.zig");
const g = contract.gates;

/// The test an entry runs: a convolution form, a GroupNorm pass, or the norm's map.
const Test = enum { conv, part, part_centered, finish, norm };

/// One registration: its name, its kernel function, its test and (a convolution) its form.
const Entry = struct { name: []const u8, function: []const u8, test_: Test, form: usize = 0 };

fn conv(comptime form: usize) Entry {
    return .{ .name = common.name(.encoder, "hie_conv") ++ " " ++ cv.forms[form].label, .function = "hie_conv", .test_ = .conv, .form = form };
}

const entries = [_]Entry{
    conv(0),
    conv(1),
    conv(2),
    conv(3),
    .{ .name = common.name(.encoder, "hie_group_part"), .function = "hie_group_part", .test_ = .part },
    .{ .name = common.name(.encoder, "hie_group_part") ++ " centered", .function = "hie_group_part", .test_ = .part_centered },
    .{ .name = common.name(.encoder, "hie_group_finish"), .function = "hie_group_finish", .test_ = .finish },
    .{ .name = common.name(.encoder, "hie_norm_silu"), .function = "hie_norm_silu", .test_ = .norm },
};

/// A planted control: the entry it breaks, the gate that must catch it, and its one edit to the source.
const Control = struct { entry: usize, gate: g.Gate, what: []const u8, old: []const u8, new: []const u8 };

const controls = [_]Control{
    .{ .entry = 0, .gate = .shapes, .what = "drops its last input channel", .old = "(live && c < s.cin) ? src[c] : 0.0f", .new = "(live && c + 1 < s.cin) ? src[c] : 0.0f" },
    .{ .entry = 0, .gate = .sums, .what = "rounds its sums to half after each 8-deep step", .old = "simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);", .new = "{ simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]); thread auto& q = acc[i][j].thread_elements(); for (uint z = 0; z < sizeof(q) / sizeof(float); z++) q[z] = float(half(q[z])); }" },
    .{ .entry = 4, .gate = .sums, .what = "sums in half", .old = "sum += g.centered != 0 ? v * v : v;", .new = "sum = float(half(sum) + half(g.centered != 0 ? v * v : v));" },
    .{ .entry = 5, .gate = .shapes, .what = "reads the next group's mean", .old = "stats[c / (g.channels / 32)]", .new = "stats[(c / (g.channels / 32) + 1) % 32]" },
    .{ .entry = 6, .gate = .sums, .what = "sums in half", .old = "sum += part[(i / per) * g.channels + sg * per + i % per];", .new = "sum = float(half(sum) + half(part[(i / per) * g.channels + sg * per + i % per]));" },
    .{ .entry = 6, .gate = .shapes, .what = "reads the next group's parts", .old = "sg * per + i % per", .new = "((sg + 1) % 32) * per + i % per" },
    .{ .entry = 7, .gate = .shapes, .what = "SiLU without its sigmoid", .old = "y[at] = v / (1.0f + exp(-v));", .new = "y[at] = v;" },
    .{ .entry = 7, .gate = .shapes, .what = "reads the next group's statistics", .old = "const uint g = i.x / (n.channels / 32);", .new = "const uint g = (i.x / (n.channels / 32) + 1) % 32;" },
    .{ .entry = 7, .gate = .nonfinite, .what = "writes 0 for a NaN", .old = "y[at] = v / (1.0f + exp(-v));", .new = "y[at] = isnan(v) ? 0.0f : v / (1.0f + exp(-v));" },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    /// The served source's library, once an entry needs it.
    served: ?mtl.Library = null,
    libs: std.ArrayList(mtl.Library) = .empty,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,
    names: std.ArrayList([]u8) = .empty,
    convs: std.ArrayList(*cv.Conv) = .empty,
    parts: std.ArrayList(*nm.GroupPart) = .empty,
    finishes: std.ArrayList(*nm.GroupFinish) = .empty,
    norms: std.ArrayList(*nm.NormSilu) = .empty,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (entries) |e| try s.add(device, queue, e, null, kernels, only);
        for (controls) |x| try s.add(device, queue, entries[x.entry], x, kernels, only);
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        inline for (.{ &s.convs, &s.parts, &s.finishes, &s.norms }) |list| {
            for (list.items) |t| {
                t.deinit();
                a.destroy(t);
            }
            list.deinit(a);
        }
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(a);
        for (s.libs.items) |l| l.deinit();
        s.libs.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }

    /// The served source's library, built once; a control's own copy of the source with its edit.
    fn library(s: *Suite, device: mtl.Device, ctl: ?Control) !mtl.Library {
        if (ctl == null) if (s.served) |lib| return lib;
        const text = if (ctl) |x| try common.edited(s.a, .encoder, x.old, x.new) else try s.a.dupe(u8, common.source(.encoder));
        defer s.a.free(text);
        const lib = try common.library(device, .encoder, text);
        errdefer lib.deinit();
        try s.libs.append(s.a, lib);
        if (ctl == null) s.served = lib;
        return lib;
    }

    fn make(s: *Suite, comptime T: type, list: *std.ArrayList(*T), value: T) !*T {
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = value;
        try list.append(s.a, t);
        return t;
    }

    /// Entry `e` on the served pipeline, or on a control's (named for what it breaks, red at its gate).
    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, e: Entry, ctl: ?Control, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        const name = if (ctl) |x| try std.fmt.allocPrint(s.a, "{s} control: {s}", .{ e.name, x.what }) else try s.a.dupe(u8, e.name);
        s.names.append(s.a, name) catch |err| {
            s.a.free(name);
            return err;
        };
        if (!common.keep(only, name)) return;
        const lib = try s.library(device, ctl);
        const pipe = try mtl.Pipeline.init(device, lib, e.function, false);
        s.pipes.append(s.a, pipe) catch |err| {
            pipe.deinit();
            return err;
        };
        const rig: rig_.Rig = .{ .a = s.a, .device = device, .queue = queue };
        const gate: ?g.Gate = if (ctl) |x| x.gate else null;
        const k = switch (e.test_) {
            .conv => (try s.make(cv.Conv, &s.convs, .{ .a = s.a, .rig = rig, .pipe = pipe, .form = &cv.forms[e.form], .name = name })).kernel(gate),
            .part, .part_centered => (try s.make(nm.GroupPart, &s.parts, .{ .a = s.a, .rig = rig, .pipe = pipe, .centered = e.test_ == .part_centered, .name = name })).kernel(gate),
            .finish => (try s.make(nm.GroupFinish, &s.finishes, .{ .a = s.a, .rig = rig, .pipe = pipe, .name = name, .ends = .{ .a = s.a } })).kernel(gate),
            .norm => (try s.make(nm.NormSilu, &s.norms, .{ .a = s.a, .rig = rig, .pipe = pipe, .name = name, .ends = .{ .a = s.a } })).kernel(gate),
        };
        try out.append(s.a, k);
    }
};

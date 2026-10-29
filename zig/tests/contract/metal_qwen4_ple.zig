//! Native lexical probes preserve the production command-session copies and both shader launches.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const ct = @import("contract");
const common = family.ple_contract;
const Api = family.metal_ple.Kernels(metal);
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 256, .k = 2 }, .{ .rows = 16, .n = 256, .k = 2 }, .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 128, .n = 5120, .k = 4 } };
    a: std.mem.Allocator,
    device: metal.Device,
    c: ?family.Config = null,
    kernels: ?Api = null,
    session: ?metal.CommandSession = null,
    views: [11]Api.View = undefined,
    original: @import("rig.zig").Ref = undefined,
    control: common.Control = .production,
    pub fn deinit(b: *Backend) void {
        if (b.session) |*s| s.deinit() catch return;
        if (b.kernels) |*k| k.deinit();
        if (b.c) |*c| c.deinit();
    }
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (b.session == null) b.session = .init(b.a, r.queue);
        if (b.c) |*c| if (c.hidden != f.p.hidden or c.streams != f.p.streams or b.control != control) {
            b.kernels.?.deinit();
            b.kernels = null;
            c.deinit();
            b.c = null;
        };
        if (b.c == null) {
            b.c = try common.configuration(b.a, f.p);
            errdefer {
                b.c.?.deinit();
                b.c = null;
            }
            b.kernels = try Api.loadControl(b.a, b.device, &b.c.?, control);
        }
        b.control = control;
        for (f.inputs, [_]usize{ 0, 1, 2, 3, 4, 8, 9 }, 0..) |bytes, i, from| {
            const ref = try r.input(bytes);
            if (from == 1) {
                b.original = ref;
                continue;
            }
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = bytes.len };
        }
        const w = f.p.hidden * f.p.streams;
        const tail = (f.p.taps - 1) * f.p.dilation * w * 2;
        const selected = ([_]usize{ 5, 6, 1, 10 })[@backingInt(f.part)];
        const lengths = [_]usize{ f.p.rows * w * 2, f.p.rows * w * 2, f.p.rows * w * 2, tail + f.p.rows * w * 2, f.p.rows * tail };
        for ([_]usize{ 1, 5, 6, 7, 10 }, lengths) |i, n| {
            const ref = if (i == selected) try r.output(n) else try r.scratch(n);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = n };
        }
    }
    pub fn launch(b: *Backend, _: *Rig, f: common.Fixture, repeat: u32) !void {
        const session = &b.session.?;
        try session.begin();
        errdefer session.discard();
        const h = b.views[1];
        try session.copy(h.buffer, h.offset, b.original.buf, b.original.off, h.len);
        const tail = (f.p.taps - 1) * f.p.dilation * f.p.hidden * f.p.streams * 2;
        try b.kernels.?.testEncode(session, f.p.rows, b.views, tail, repeat);
        try session.submit();
        try session.drain();
    }
};
pub const Case = common.Case(ct, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    cases: [20]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(ct.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (std.enums.values(common.Part)) |part| for (std.enums.values(common.Control)) |control| {
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .a = a, .device = device }, .part = part, .control = control };
            s.made += 1;
            const kernel = case.kernel();
            if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
            try out.append(a, kernel);
        };
        return s;
    }
    pub fn destroy(s: *Suite) void {
        for (s.cases[0..s.made]) |*case| case.deinit();
        const a = s.a;
        a.destroy(s);
    }
};

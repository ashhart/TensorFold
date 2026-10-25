//! Selection Cases call the generated serving entry point with exact integer output and source faults.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const ct = @import("contract");
const common = family.qsa_select_contract;
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 8, .n = 129, .k = 1 }, .{ .rows = 1, .n = 0, .k = 2 }, .{ .rows = 128, .n = 128, .k = 1 }, .{ .rows = 1, .n = 127, .k = 8 }, .{ .rows = 8, .n = 513, .k = 8 }, .{ .rows = 3, .n = 4097, .k = 8 }, .{ .rows = 1, .n = 8192, .k = 8 } };
    pipe: metal.Pipeline,
    input: [3]@import("rig.zig").Ref = undefined,
    output: @import("rig.zig").Ref = undefined,
    control: common.Control,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, _: common.Control) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.scores), std.mem.sliceAsBytes(f.complete), std.mem.sliceAsBytes(f.ends) }, 0..) |bytes, i| b.input[i] = try r.input(bytes);
        b.output = try r.output(f.want.len * 4);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, repeat: u32) !void {
        const pass = r.begin();
        const e = pass.e;
        e.setPipeline(b.pipe);
        if (b.control == .repeat) family.metal_control.bind(e, repeat);
        for (b.input, [_]usize{ 0, 2, 3 }) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        const shape = [_]i32{ @intCast(f.rows), @intCast(f.stride) };
        e.setBytes(std.mem.sliceAsBytes(&shape), 1);
        e.setBuffer(b.output.buf, b.output.off, 4);
        e.dispatchGroups(metal.Size.of(f.rows, 1, 1), metal.Size.of(1024, 1, 1));
        e.barrier();
        try r.finish(pass);
    }
};
pub const Case = common.Case(ct, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: [4]metal.Pipeline = undefined,
    opened: usize = 0,
    cases: [4]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(ct.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const source = try family.metal_attention.source(a, family.qsa_norm_contract.reference.geometry);
        defer a.free(source);
        for (std.enums.values(common.Control)[0..4]) |control| {
            const text = try family.qsa_control_source.selection(a, source, control);
            defer a.free(text);
            const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try metal.Pipeline.init(device, lib, "qwen4_idx_select", false);
            s.pipes[s.opened] = pipe;
            s.opened += 1;
            if (pipe.maxThreads() < 1024 or pipe.simdWidth() != 32) return error.Qwen4AttentionThreads;
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .pipe = pipe, .control = control }, .control = control };
            s.made += 1;
            const kernel = case.kernel();
            if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
            try out.append(a, kernel);
        }
        return s;
    }
    pub fn destroy(s: *Suite) void {
        for (s.cases[0..s.made]) |*case| case.deinit();
        for (s.pipes[0..s.opened]) |pipe| pipe.deinit();
        const a = s.a;
        a.destroy(s);
    }
};

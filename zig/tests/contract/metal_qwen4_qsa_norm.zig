//! QSA frontend Cases use the serving source, geometry, thread reservations and actual shader entry points.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const ct = @import("contract");
const common = family.qsa_norm_contract;
const Pipes = struct {
    pipes: [2]metal.Pipeline = undefined,
    made: usize = 0,
    control: common.Control,
    fn open(a: std.mem.Allocator, device: metal.Device, control: common.Control) !Pipes {
        const source = try family.metal_attention.source(a, common.reference.geometry);
        defer a.free(source);
        const text = try family.qsa_control_source.frontend(a, source, control);
        defer a.free(text);
        const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipes = .{ .control = control };
        errdefer out.deinit();
        for ([_][]const u8{ "qwen4_attn_prep", "qwen4_idx_pool" }, [_]usize{ 256, 128 }, 0..) |name, threads, i| {
            out.pipes[i] = try metal.Pipeline.init(device, lib, name, false);
            out.made += 1;
            if (out.pipes[i].maxThreads() < threads or out.pipes[i].simdWidth() != 32) return error.Qwen4AttentionThreads;
        }
        return out;
    }
    fn deinit(p: *Pipes) void {
        for (p.pipes[0..p.made]) |pipe| pipe.deinit();
    }
};
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 24, .k = 4 }, .{ .rows = 128, .n = 24, .k = 4 } };
    pipes: *Pipes,
    input: [6]@import("rig.zig").Ref = undefined,
    output: [3]@import("rig.zig").Ref = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, _: common.Control) !void {
        for (f.inputs[0..f.count], 0..) |bytes, i| b.input[i] = try r.input(bytes);
        if (f.part == .pool) {
            b.output[0] = try r.output(f.want.len * 2);
            return;
        }
        for ([_]usize{ f.rows * 24 * 256 * 2, f.rows * 4 * 256 * 2, f.rows * 4 * 128 * 2 }, 0..) |n, i| b.output[i] = if (i == @backingInt(f.part)) try r.output(n) else try r.scratch(n);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, repeat: u32) !void {
        const pass = r.begin();
        const e = pass.e;
        const pool = f.part == .pool;
        e.setPipeline(b.pipes.pipes[if (pool) 1 else 0]);
        if (b.pipes.control == .repeat) e.setBytes(std.mem.asBytes(&repeat), 31);
        const eps: f32 = 1e-6;
        if (pool) {
            for ([_]usize{ 0, 2, 1, 3 }, [_]usize{ 0, 1, 2, 4 }) |from, to| e.setBuffer(b.input[from].buf, b.input[from].off, to);
            e.setBytes(std.mem.asBytes(&eps), 3);
            e.setBuffer(b.output[0].buf, b.output[0].off, 5);
        } else {
            for ([_]usize{ 0, 4, 1, 2, 3, 5 }, [_]usize{ 0, 1, 2, 3, 4, 6 }) |from, to| e.setBuffer(b.input[from].buf, b.input[from].off, to);
            e.setBytes(std.mem.asBytes(&eps), 5);
            for (b.output, 7..) |ref, to| e.setBuffer(ref.buf, ref.off, to);
        }
        e.dispatchGroups(if (pool) metal.Size.of(1, f.rows, 1) else metal.Size.of(1, 32, f.rows), metal.Size.of(if (pool) 128 else 256, 1, 1));
        e.barrier();
        try r.finish(pass);
    }
};
pub const Case = common.Case(ct, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: [5]Pipes = undefined,
    opened: usize = 0,
    cases: [20]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(ct.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (std.enums.values(common.Control)) |control| {
            s.pipes[s.opened] = try Pipes.open(a, device, control);
            const pipe = &s.pipes[s.opened];
            s.opened += 1;
            for (std.enums.values(common.Part)) |part| {
                const case = &s.cases[s.made];
                case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .pipes = pipe }, .part = part, .control = control };
                s.made += 1;
                const kernel = case.kernel();
                if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
                try out.append(a, kernel);
            }
        }
        return s;
    }
    pub fn destroy(s: *Suite) void {
        for (s.cases[0..s.made]) |*case| case.deinit();
        for (s.pipes[0..s.opened]) |*pipe| pipe.deinit();
        const a = s.a;
        a.destroy(s);
    }
};

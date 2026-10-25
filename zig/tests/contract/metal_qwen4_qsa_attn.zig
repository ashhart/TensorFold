//! Native QSA parts and merge run together against the shared binary64 attention contract.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const ct = @import("contract");
const common = family.qsa_attn_contract;
const Ref = @import("rig.zig").Ref;
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 1024 }, .{ .rows = 1, .n = 2, .k = 17 }, .{ .rows = 128, .n = 2, .k = 33 }, .{ .rows = 1, .n = 2, .k = 1025 } };
    pipes: [2]metal.Pipeline,
    input: [6]Ref = undefined,
    partials: [2]Ref = undefined,
    output: Ref = undefined,
    control: common.Control,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, _: common.Control) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.query), std.mem.sliceAsBytes(f.key), std.mem.sliceAsBytes(f.value), std.mem.sliceAsBytes(f.ids), std.mem.sliceAsBytes(f.nk), std.mem.sliceAsBytes(f.spr) }, 0..) |bytes, i| b.input[i] = try r.input(bytes);
        b.partials = .{ try r.scratch(f.rows * 24 * 16 * 256 * 4), try r.scratch(f.rows * 24 * 16 * 2 * 4) };
        b.output = try r.output(f.rows * 24 * 256 * 2);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, repeat: u32) !void {
        const pass = r.begin();
        const e = pass.e;
        e.setPipeline(b.pipes[0]);
        for (b.input, [_]usize{ 0, 1, 3, 4, 6, 7 }) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        const ks = [_]i32{ 1, 4, @intCast(f.keys) };
        const ids = [_]i32{ @intCast(f.rows), @intCast(f.keys) };
        e.setBytes(std.mem.asBytes(&ks), 2);
        e.setBytes(std.mem.asBytes(&ids), 5);
        e.setBytes(std.mem.asBytes(&common.scale), 8);
        for (b.partials, 9..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.dispatchGroups(metal.Size.of(24, f.rows, 16), metal.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(b.pipes[1]);
        if (b.control == .repeat) family.metal_control.bind(e, repeat);
        for (b.partials, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(b.input[0].buf, b.input[0].off, 2);
        e.setBuffer(b.output.buf, b.output.off, 3);
        e.dispatchGroups(metal.Size.of(1, 24, f.rows), metal.Size.of(256, 1, 1));
        e.barrier();
        try r.finish(pass);
    }
};
pub const Case = common.Case(ct, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: [10]metal.Pipeline = undefined,
    opened: usize = 0,
    cases: [5]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(ct.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const source = try family.metal_attention.source(a, family.qsa_norm_contract.reference.geometry);
        defer a.free(source);
        for (std.enums.values(common.Control)) |control| {
            const text = try family.qsa_control_source.attention(a, source, control);
            defer a.free(text);
            const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
            defer lib.deinit();
            var pair: [2]metal.Pipeline = undefined;
            for ([_][]const u8{ "qwen4_attn_parts", "qwen4_attn_merge" }, 0..) |name, i| {
                pair[i] = try metal.Pipeline.init(device, lib, name, false);
                s.pipes[s.opened] = pair[i];
                s.opened += 1;
                if (pair[i].maxThreads() < 256 or pair[i].simdWidth() != 32) return error.Qwen4AttentionThreads;
            }
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .pipes = pair, .control = control }, .control = control };
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

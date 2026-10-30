//! Score Cases launch the generated serving shader at short, tile-edge and sampled-selection block counts.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const ct = @import("contract");
const common = family.qsa_scores_contract;
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const ct.spec.Shape = &.{ .{ .rows = 1, .n = 129, .k = 128 }, .{ .rows = 128, .n = 129, .k = 128 }, .{ .rows = 8, .n = 513, .k = 128 }, .{ .rows = 1, .n = 4097, .k = 128 } };
    pipe: metal.Pipeline,
    input: [3]@import("rig.zig").Ref = undefined,
    output: @import("rig.zig").Ref = undefined,
    control: common.Control,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, _: common.Control) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.query), std.mem.sliceAsBytes(f.pooled), std.mem.sliceAsBytes(f.complete) }, 0..) |bytes, i| b.input[i] = try r.input(bytes);
        b.output = try r.output(f.rows * f.blocks * 4);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, repeat: u32) !void {
        const pass = r.begin();
        const e = pass.e;
        e.setPipeline(b.pipe);
        if (b.control == .repeat) family.metal_control.bind(e, repeat);
        for (b.input, [_]usize{ 0, 2, 4 }) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        const rows: i32 = @intCast(f.rows);
        const blocks: i32 = @intCast(f.blocks);
        e.setBytes(std.mem.asBytes(&rows), 1);
        e.setBytes(std.mem.asBytes(&blocks), 3);
        e.setBuffer(b.output.buf, b.output.off, 5);
        e.dispatchGroups(metal.Size.of((f.blocks + 7) / 8, (f.rows + 7) / 8, 1), metal.Size.of(256, 1, 1));
        e.barrier();
        try r.finish(pass);
    }
};
pub const Case = common.Case(ct, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: [5]metal.Pipeline = undefined,
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
            const text = try family.qsa_control_source.scores(a, source, control);
            defer a.free(text);
            const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try metal.Pipeline.init(device, lib, "qwen4_idx_scores", false);
            s.pipes[s.opened] = pipe;
            s.opened += 1;
            if (pipe.maxThreads() < 256 or pipe.simdWidth() != 32) return error.Qwen4AttentionThreads;
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

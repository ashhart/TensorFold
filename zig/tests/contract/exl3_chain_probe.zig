//! Direct launches of the production chain bodies over guarded buffers and independently owned host fixtures.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const source = @import("exl3_chain_source.zig");
const Fixture = @import("exl3_chain_fixture.zig").Fixture;
const c = contract.spec;
const g = contract.gates;
const Ref = rig_.Ref;

pub fn For(comptime ex: type, comptime Old: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        pipeline: mtl.Pipeline,
        old: *const Old,
        spec: c.Spec,
        kind: source.Kind,
        fault: ?source.Fault,
        f: ?Fixture = null,
        sums: Ref = undefined,
        x: Ref = undefined,
        up: Ref = undefined,
        su: Ref = undefined,
        sv: Ref = undefined,
        bias: Ref = undefined,
        weights: Ref = undefined,
        norm_weight: Ref = undefined,
        factors: Ref = undefined,
        routes: Ref = undefined,
        ids: Ref = undefined,
        destinations: Ref = undefined,
        kv_weight: Ref = undefined,
        descriptors: Ref = undefined,
        route_work: Ref = undefined,
        y: Ref = undefined,
        rotated: Ref = undefined,
        key_pack: Ref = undefined,
        work: Ref = undefined,
        activation: Ref = undefined,
        result: std.ArrayList(f32) = .empty,
        baseline: std.ArrayList(f32) = .empty,
        why: [160]u8 = undefined,

        const hooks = blk: {
            var h = g.hooks(Self);
            h.output = output;
            h.intact = intact;
            h.release = release;
            break :blk h;
        };

        pub fn kernel(p: *Self) g.Kernel {
            return .{ .spec = p.spec, .ctx = p, .vt = &hooks, .control = if (p.fault) |fault| switch (fault) {
                .clip => .nonfinite,
                .input_write => .repeat,
                .tail => .shapes,
                .half_sum => .sums,
                .half_norm => .shapes,
            } else null };
        }

        pub fn drop(p: *Self) void {
            if (p.f) |*f| f.deinit();
            p.f = null;
            p.rig.reset();
            p.result.clearRetainingCapacity();
            p.baseline.clearRetainingCapacity();
        }

        fn release(ctx: *anyopaque) void {
            const p: *Self = @ptrCast(@alignCast(ctx));
            p.drop();
            p.result.deinit(p.rig.a);
            p.result = .empty;
            p.baseline.deinit(p.rig.a);
            p.baseline = .empty;
            p.rig.seen.deinit(p.rig.a);
            p.rig.seen = .empty;
            p.rig.outs.deinit(p.rig.a);
            p.rig.outs = .empty;
        }

        pub fn deinit(p: *Self) void {
            p.drop();
            p.rig.deinit();
            p.result.deinit(p.rig.a);
            p.baseline.deinit(p.rig.a);
        }

        pub fn count(p: *Self) usize {
            return p.f.?.outputCount();
        }

        fn intact(ctx: *anyopaque) !bool {
            const p: *Self = @ptrCast(@alignCast(ctx));
            if (!try p.rig.intact()) return false;
            const f = p.f.?;
            const k = f.shape.k;
            if (p.kind == .rms or p.kind == .rope_in) {
                const original = std.mem.sliceAsBytes(f.x);
                const work = p.rig.bytes(p.work, original.len);
                for (0..f.shape.rows) |row| {
                    const first = (row * f.stride + k) * 2;
                    const end = (row + 1) * f.stride * 2;
                    if (!std.mem.eql(u8, original[first..end], work[first..end])) return false;
                }
            }
            if (p.kind == .qkv) {
                const y = p.rig.bytes(p.y, 2 * f.shape.rows * k * 2);
                for (0..f.shape.rows) |row| {
                    for (y[((2 * row + 1) * k + 512) * 2 .. (2 * row + 2) * k * 2]) |v|
                        if (v != 0xff) return false;
                }
            }
            return true;
        }

        pub fn prepare(p: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            p.drop();
            if (shape.rows == 0 or shape.rows > 64 or shape.k < 128 or shape.k > Fixture.max_width or
                shape.k % 128 != 0 or shape.n != shape.k or
                (p.kind == .rms and shape.k > 4096) or
                (p.kind == .qkv and shape.k != 1280)) return false;
            p.f = try Fixture.init(p.rig.a, p.kind, shape, fill, seed);
            const f = &p.f.?;
            const bytes = std.mem.sliceAsBytes;
            p.sums = try p.rig.input(bytes(f.sums));
            p.x = try p.rig.input(bytes(f.x));
            p.up = try p.rig.input(bytes(f.up));
            p.su = try p.rig.input(bytes(f.su));
            p.sv = try p.rig.input(bytes(f.sv));
            p.bias = try p.rig.input(bytes(f.bias));
            p.weights = try p.rig.input(bytes(f.weights));
            p.norm_weight = try p.rig.input(bytes(f.norm_weight));
            p.factors = try p.rig.input(bytes(f.factors));
            p.routes = try p.rig.input(bytes(f.routes));
            p.ids = try p.rig.input(bytes(f.ids));
            p.destinations = try p.rig.input(bytes(f.destinations));
            const kv_ones: [512]u16 = @splat(c.toBf16(1));
            p.kv_weight = try p.rig.input(bytes(&kv_ones));
            const desc = try p.rig.a.alloc(ex.ExpertDescriptor, f.descriptors);
            defer p.rig.a.free(desc);
            for (desc, 0..) |*d, id| d.* = .{
                .trellis = 0,
                .suh = try ex.address(view(p.su, id * shape.k * 2)),
                .svh = try ex.address(view(p.sv, id * shape.k * 2)),
                .bias = try ex.address(view(p.bias, id * shape.k * 2)),
                .tile_k_stride = 0,
                .tile_n_stride = 0,
                .halves = 6,
                .codebook = 2,
                .suh_signs = @intFromBool(f.signs),
                .svh_signs = @intFromBool(f.signs),
                .has_bias = 1,
                .reserved = @intCast(if (p.kind == .qkv and id == 1) 512 else shape.k),
            };
            p.descriptors = try p.rig.input(bytes(desc));
            p.route_work = try p.rig.scratch(bytes(f.routes).len);
            p.work = try p.rig.scratch(bytes(f.x).len);
            const y_rows = shape.rows * (switch (p.kind) {
                .gate_up => @as(usize, 7),
                .qkv => 2,
                else => 1,
            });
            p.y = try p.rig.output(y_rows * shape.k * 2);
            p.rotated = try p.rig.output((if (p.kind == .gate_up) @as(usize, 7) else 1) * shape.rows * shape.k * 2);
            p.key_pack = try p.rig.output(shape.rows * 528);
            p.activation = try p.rig.scratch(f.physical_rows * shape.k * 2);
            try p.result.resize(p.rig.a, f.outputCount());
            try p.baseline.resize(p.rig.a, f.outputCount());
            return true;
        }

        pub fn view(ref: Ref, extra: usize) ex.View {
            return .{ .buffer = ref.buf, .offset = ref.off + extra };
        }
        pub fn bind(e: mtl.ComputeEncoder, ref: Ref, extra: usize, at: usize) void {
            e.setBuffer(ref.buf, ref.off + extra, at);
        }

        pub fn args(p: *Self, rows: usize) ex.Args {
            const f = p.f.?;
            return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.k), .rows = @intCast(rows), .halves = 6, .codebook = 2, .su_signs = @intFromBool(f.signs), .sv_signs = @intFromBool(f.signs), .has_bias = 1, .output_f32 = 0, .input_ids = 0, .output_ids = 0, .routed = 0, .expert_id = 0, .repeat = 1, .indexed = 0, .experts = @intCast(f.descriptors), .x_stride = f.stride, .y_stride = f.shape.k, .tile_k_stride = 0, .tile_n_stride = 0 };
        }

        pub fn resetWork(p: *Self) void {
            const f = p.f.?;
            @memcpy(p.rig.bytes(p.work, f.x.len * 2), std.mem.sliceAsBytes(f.x));
            @memcpy(p.rig.bytes(p.route_work, f.routes.len * 4), std.mem.sliceAsBytes(f.routes));
        }

        pub fn run(p: *Self) !void {
            p.resetWork();
            const pass = p.rig.begin();
            p.encode(pass.e);
            try p.rig.finish(pass);
        }

        pub fn encode(p: *Self, e: mtl.ComputeEncoder) void {
            const f = p.f.?;
            const k = f.shape.k;
            const n = f.shape.rows;
            e.useResource(p.su.buf, mtl.ResourceUsage.read);
            e.useResource(p.sv.buf, mtl.ResourceUsage.read);
            e.useResource(p.bias.buf, mtl.ResourceUsage.read);
            e.setPipeline(p.pipeline);
            switch (p.kind) {
                .swiglu => {
                    bind(e, p.x, 0, 0);
                    bind(e, p.up, 0, 1);
                    bind(e, p.weights, 0, 2);
                    bind(e, p.routes, 0, 3);
                    bind(e, p.destinations, 0, 4);
                    bind(e, p.descriptors, 0, 5);
                    bind(e, p.rotated, 0, 6);
                    e.setValue([8]u32{ @intCast(n), @intCast(k), @intCast(f.stride), @intCast(k), @intFromBool(f.weighted), 1, @intCast(f.expert_count), 0 }, 7);
                    e.dispatchGroups(.{ .width = (k + 511) / 512, .height = n }, .{ .width = 128 });
                },
                .out_in, .out_rope => {
                    var a = p.args(n);
                    a.indexed = @intFromBool(p.kind == .out_in);
                    a.output_ids = @intFromBool(p.kind == .out_in);
                    bind(e, p.sums, 0, 0);
                    bind(e, p.sv, 0, 1);
                    bind(e, p.bias, 0, 2);
                    bind(e, p.y, 0, 3);
                    e.setValue(a, 4);
                    bind(e, p.destinations, 0, 5);
                    bind(e, p.routes, 0, 6);
                    bind(e, p.descriptors, 0, 7);
                    if (p.kind == .out_in) {
                        bind(e, p.su, 0, 8);
                        e.setValue([4]u32{ @intCast(n * k), @intFromBool(f.signs), 0, 0 }, 9);
                        bind(e, p.rotated, 0, 10);
                    } else {
                        bind(e, p.factors, 0, 8);
                        e.setValue([8]u32{ @intCast(f.head_width), @intCast(f.rotary), @intCast(f.position), 1, @intFromBool(f.inverse), 0, 0, 0 }, 9);
                    }
                    e.dispatchGroups(.{ .width = (k + 511) / 512, .height = n }, .{ .width = 128 });
                },
                .rms => {
                    bind(e, p.work, 0, 0);
                    bind(e, p.norm_weight, 0, 1);
                    bind(e, p.work, 0, 2);
                    e.setValue(extern struct {
                        rows: u32,
                        width: u32,
                        x_stride: u32,
                        y_stride: u32,
                        eps: f32,
                        weight_f32: u32,
                        su_signs: u32,
                        reserved: u32,
                    }{ .rows = @intCast(n), .width = @intCast(k), .x_stride = @intCast(f.stride), .y_stride = @intCast(f.stride), .eps = f.eps, .weight_f32 = 1, .su_signs = @intFromBool(f.signs), .reserved = 0 }, 3);
                    bind(e, p.su, 0, 4);
                    bind(e, p.rotated, 0, 5);
                    e.dispatchGroups(.{ .width = n }, .{ .width = 32 * @max(8, k / 128) });
                },
                .rope_in => {
                    var a = p.args(n);
                    a.indexed = 1;
                    bind(e, p.work, 0, 0);
                    bind(e, p.su, 0, 1);
                    bind(e, p.rotated, 0, 2);
                    e.setValue(a, 3);
                    bind(e, p.ids, 0, 4);
                    bind(e, p.routes, 0, 5);
                    bind(e, p.descriptors, 0, 6);
                    bind(e, p.factors, 0, 7);
                    e.setValue([8]u32{ @intCast(f.head_width), @intCast(f.rotary), @intCast(8 * f.stride), @intCast(f.position), 1, @intFromBool(f.inverse), 0, 0 }, 8);
                    e.dispatchGroups(.{ .width = (k + 511) / 512, .height = n }, .{ .width = 128 });
                },
                .gate_up => {
                    bind(e, p.sums, 0, 0);
                    bind(e, p.ids, 0, 1);
                    bind(e, p.weights, 0, 2);
                    bind(e, p.route_work, 0, 3);
                    bind(e, p.descriptors, 0, 4);
                    bind(e, p.descriptors, 0, 5);
                    bind(e, p.descriptors, f.expert_count * @sizeOf(ex.ExpertDescriptor), 6);
                    bind(e, p.rotated, 0, 7);
                    e.setValue([9]u32{ @intCast(6 * n), @intCast(n), @intCast(k), @intCast(k), @intCast(f.expert_count), @intCast(f.descriptors), @intCast(6 * n), @intFromBool(f.weighted), @intCast(f.expert_count) }, 8);
                    e.dispatchGroups(.{ .width = (k + 511) / 512, .height = 7 * n }, .{ .width = 128 });
                },
                .down => {
                    bind(e, p.sums, 0, 0);
                    bind(e, p.routes, 0, 1);
                    bind(e, p.descriptors, 0, 2);
                    bind(e, p.y, 0, 3);
                    e.setValue([8]u32{ @intCast(n), 6, @intCast(k), @intCast(k), @intCast(f.expert_count), @intCast(k), @intCast(6 * n), 0 }, 4);
                    e.dispatchGroups(.{ .width = k / 128, .height = n }, .{ .width = 256 });
                },
                .qkv => {
                    for (0..n) |row| {
                        var a = p.args(2);
                        a.indexed = 1;
                        bind(e, p.sums, 2 * row * k * 4, 0);
                        bind(e, p.y, 2 * row * k * 2, 1);
                        e.setValue(a, 2);
                        bind(e, p.destinations, 0, 3);
                        bind(e, p.ids, 0, 4);
                        bind(e, p.descriptors, 0, 5);
                        bind(e, p.norm_weight, 0, 6);
                        bind(e, p.su, 0, 7);
                        bind(e, p.rotated, row * k * 2, 8);
                        bind(e, p.kv_weight, 0, 9);
                        bind(e, p.factors, 0, 10);
                        bind(e, p.key_pack, row * 528, 11);
                        e.setValue(extern struct {
                            q_width: u32,
                            q_su_signs: u32,
                            q_weight_f32: u32,
                            kv_width: u32,
                            position: u32,
                            rotary: u32,
                            packed_stride: u32,
                            block: u32,
                            eps: f32,
                            reserved0: u32 = 0,
                            reserved1: u32 = 0,
                            reserved2: u32 = 0,
                        }{ .q_width = @intCast(k), .q_su_signs = @intFromBool(f.signs), .q_weight_f32 = 1, .kv_width = 512, .position = @intCast(f.position + row), .rotary = @intCast(f.rotary), .packed_stride = 528, .block = 32, .eps = f.eps }, 12);
                        e.dispatchGroups(.{ .width = 2 }, .{ .width = 320 });
                    }
                },
            }
        }

        fn pushHalf(p: *Self, ref: Ref, first: usize, len: usize, bf: bool, dest: usize) void {
            const bytes = p.rig.bytes(ref, (first + len) * 2);
            for (p.result.items[dest..][0..len], 0..) |*v, i| v.* = @floatCast(c.read(if (bf) .bf16 else .f16, bytes, first + i));
        }

        fn gather(p: *Self) void {
            const f = p.f.?;
            const n = f.shape.rows;
            const k = f.shape.k;
            switch (p.kind) {
                .swiglu => p.pushHalf(p.rotated, 0, n * k, false, 0),
                .out_rope, .down => p.pushHalf(p.y, 0, n * k, true, 0),
                .out_in => {
                    p.pushHalf(p.y, 0, n * k, true, 0);
                    p.pushHalf(p.rotated, 0, n * k, false, n * k);
                },
                .rope_in, .rms => {
                    for (0..n) |row| p.pushHalf(p.work, row * f.stride, k, true, row * k);
                    p.pushHalf(p.rotated, 0, n * k, false, n * k);
                },
                .gate_up => {
                    p.pushHalf(p.rotated, 0, 7 * n * k, false, 0);
                    for (p.result.items[7 * n * k ..], 0..) |*v, row| v.* = @floatFromInt(std.mem.readInt(u32, p.rig.bytes(p.route_work, (6 * n + row + 1) * 4)[(6 * n + row) * 4 ..][0..4], .little));
                },
                .qkv => for (0..n) |row| {
                    const dest = row * (2 * k + 1040);
                    p.pushHalf(p.y, 2 * row * k, k, true, dest);
                    p.pushHalf(p.rotated, row * k, k, false, dest + k);
                    p.pushHalf(p.y, (2 * row + 1) * k, 512, true, dest + 2 * k);
                    for (p.result.items[dest + 2 * k + 512 ..][0..528], p.rig.bytes(p.key_pack, n * 528)[row * 528 ..][0..528]) |*v, byte| v.* = @floatFromInt(byte);
                },
            }
        }

        fn output(ctx: *anyopaque) ![]const u8 {
            const p: *Self = @ptrCast(@alignCast(ctx));
            p.gather();
            return std.mem.sliceAsBytes(p.result.items);
        }

        pub fn reference(p: *Self, want: []f64, mag: []f64) !void {
            try p.f.?.evaluate(.none, .sequential, want);
            for (mag, want) |*v, w| v.* = @abs(w);
        }

        pub fn simulate(p: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            try p.f.?.evaluate(acc, order, out);
        }

        pub fn replaced(p: *Self) ![]const u8 {
            p.resetWork();
            const pass = p.rig.begin();
            p.old.encode(p, pass.e);
            try p.rig.finish(pass);
            std.mem.swap(std.ArrayList(f32), &p.result, &p.baseline);
            defer std.mem.swap(std.ArrayList(f32), &p.result, &p.baseline);
            p.gather();
            return std.mem.sliceAsBytes(p.result.items);
        }
    };
}

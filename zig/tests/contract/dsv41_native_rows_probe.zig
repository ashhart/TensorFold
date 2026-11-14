//! Guarded native row launches, compared to per-slot GPU execution with separately owned output buffers.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const source = @import("dsv41_native_rows_source.zig");
const ref = @import("dsv41_native_rows_ref.zig");
const c = contract.spec;
const g = contract.gates;
const Ref = rig_.Ref;

fn bind(e: mtl.ComputeEncoder, r: Ref, extra: usize, at: usize) void {
    e.setBuffer(r.buf, r.off + extra, at);
}

pub fn For(comptime native: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        spec: c.Spec,
        kind: source.Kind,
        inverse: bool,
        fault: ?source.Fault,
        pipeline: mtl.Pipeline,
        old: mtl.Pipeline,
        f: ?ref.Fixture = null,
        pos: Ref = undefined,
        x: Ref = undefined,
        weight: Ref = undefined,
        factors: Ref = undefined,
        chunk: Ref = undefined,
        addr: Ref = undefined,
        work: Ref = undefined,
        base_work: Ref = undefined,
        live: [8]Ref = undefined,
        base_live: [8]Ref = undefined,
        result: std.ArrayList(u8) = .empty,
        baseline: std.ArrayList(u8) = .empty,
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
                .position => .shapes,
                .half_sum => .sums,
            } else null };
        }

        fn drop(p: *Self) void {
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
            return p.f.?.count();
        }

        pub fn prepare(p: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            p.drop();
            if (shape.rows == 0 or shape.rows > 8) return false;
            switch (p.kind) {
                .rope => if (shape.k != 512 or (shape.n != 8192 and shape.n != 16384 and shape.n != 32768)) return false,
                .glue => if (shape.k != 512 or shape.n != 512) return false,
                .publish => if (shape.k != 528 or shape.n != 528) return false,
            }
            p.f = try ref.Fixture.init(p.rig.a, p.kind, shape, p.inverse, fill, seed);
            const f = p.f.?;
            const b = std.mem.sliceAsBytes;
            p.pos = try p.rig.input(b(f.pos));
            switch (p.kind) {
                .rope => {
                    p.factors = try p.rig.input(b(f.factors));
                    p.work = try p.rig.output(f.x.len * 2);
                    p.base_work = try p.rig.output(f.x.len * 2);
                },
                .glue => {
                    p.factors = try p.rig.input(b(f.factors));
                    p.x = try p.rig.input(b(f.x));
                    p.weight = try p.rig.input(b(f.weight));
                    p.work = try p.rig.output(shape.rows * 528);
                    p.base_work = try p.rig.output(shape.rows * 528);
                },
                .publish => {
                    p.chunk = try p.rig.input(f.chunk);
                    var addresses: [32]u64 = @splat(0xffffffffffffffff);
                    for (0..shape.rows) |slot| {
                        p.live[slot] = try p.rig.output(ref.live_bytes);
                        p.base_live[slot] = try p.rig.output(ref.live_bytes);
                    }
                    for (0..shape.rows) |row| {
                        const live = p.live[shape.rows - row - 1];
                        addresses[4 * row] = live.buf.gpuAddress() + live.off;
                    }
                    p.addr = try p.rig.input(b(addresses[0 .. shape.rows * 4]));
                },
            }
            const size = c.size(p.spec.out);
            try p.result.resize(p.rig.a, p.count() * size);
            try p.baseline.resize(p.rig.a, p.count() * size);
            return true;
        }

        fn reset(p: *Self, old: bool) void {
            const f = p.f.?;
            if (p.kind == .rope) @memcpy(p.rig.bytes(if (old) p.base_work else p.work, f.x.len * 2), std.mem.sliceAsBytes(f.x));
            if (p.kind == .publish) for (0..f.shape.rows) |slot| {
                @memcpy(p.rig.bytes(if (old) p.base_live[slot] else p.live[slot], ref.live_bytes), f.images[slot * ref.live_bytes ..][0..ref.live_bytes]);
            };
        }

        pub fn run(p: *Self) !void {
            p.reset(false);
            const pass = p.rig.begin();
            const e = pass.e;
            const f = p.f.?;
            e.setPipeline(p.pipeline);
            switch (p.kind) {
                .rope => {
                    bind(e, p.work, 0, 0);
                    e.setValue(native.rope.RowsParams{ .rows = @intCast(f.shape.rows), .heads = @intCast(f.shape.n / 512), .width = 512, .rotary = 64, .stride = @intCast(f.stride), .inverse = @intFromBool(p.inverse) }, 1);
                    bind(e, p.factors, 0, 2);
                    bind(e, p.pos, 0, 3);
                    e.dispatchThreads(.{ .width = 32, .height = f.shape.n / 512, .depth = f.shape.rows }, .{ .width = 128 });
                },
                .glue => {
                    bind(e, p.x, 0, 0);
                    bind(e, p.weight, 0, 1);
                    bind(e, p.factors, 0, 2);
                    bind(e, p.work, 0, 3);
                    bind(e, p.work, 0, 4);
                    e.setValue(p.glueArgs(f.shape.rows, 0), 5);
                    bind(e, p.pos, 0, 6);
                    e.dispatchGroups(.{ .width = 16, .height = f.shape.rows }, .{ .width = 256 });
                },
                .publish => {
                    for (p.live[0..f.shape.rows]) |live| e.useResource(live.buf, mtl.ResourceUsage.write);
                    bind(e, p.chunk, 0, 0);
                    e.setValue([4]u32{ ref.ring_offset, 528, 528, 0 }, 1);
                    bind(e, p.addr, 0, 2);
                    bind(e, p.pos, 0, 3);
                    e.dispatchThreads(.{ .width = 528, .height = f.shape.rows }, .{ .width = 32 });
                },
            }
            try p.rig.finish(pass);
        }

        fn glueArgs(p: *Self, rows: usize, position: u32) native.kv.Glue {
            return .{ .rows = @intCast(rows), .width = 512, .stride = @intCast(p.f.?.stride), .packed_stride = 528, .block = 32, .position = position, .rotary = 64, .write_y = 0, .eps = 1e-6 };
        }

        fn gather(p: *Self, old: bool) []const u8 {
            const f = p.f.?;
            const dest = if (old) p.baseline.items else p.result.items;
            switch (p.kind) {
                .rope => {
                    const src = p.rig.bytes(if (old) p.base_work else p.work, f.x.len * 2);
                    for (0..f.shape.rows) |row| @memcpy(dest[row * f.shape.n * 2 ..][0 .. f.shape.n * 2], src[row * f.stride * 2 ..][0 .. f.shape.n * 2]);
                },
                .glue => @memcpy(dest, p.rig.bytes(if (old) p.base_work else p.work, dest.len)),
                .publish => for (0..f.shape.rows) |slot| {
                    @memcpy(dest[slot * ref.live_bytes ..][0..ref.live_bytes], p.rig.bytes(if (old) p.base_live[slot] else p.live[slot], ref.live_bytes));
                },
            }
            return dest;
        }

        fn output(ctx: *anyopaque) ![]const u8 {
            const p: *Self = @ptrCast(@alignCast(ctx));
            return p.gather(false);
        }

        fn intact(ctx: *anyopaque) !bool {
            const p: *Self = @ptrCast(@alignCast(ctx));
            if (!try p.rig.intact()) return false;
            if (p.kind != .rope) return true;
            const f = p.f.?;
            const original = std.mem.sliceAsBytes(f.x);
            const work = p.rig.bytes(p.work, original.len);
            for (0..f.shape.rows) |row| {
                const first = (row * f.stride + f.shape.n) * 2;
                const end = (row + 1) * f.stride * 2;
                if (!std.mem.eql(u8, original[first..end], work[first..end])) return false;
            }
            return true;
        }

        pub fn replaced(p: *Self) ![]const u8 {
            p.reset(true);
            const pass = p.rig.begin();
            const e = pass.e;
            const f = p.f.?;
            for (0..f.shape.rows) |row| {
                e.setPipeline(p.old);
                switch (p.kind) {
                    .rope => {
                        bind(e, p.base_work, row * f.stride * 2, 0);
                        e.setValue(native.rope.Params{ .rows = 1, .heads = @intCast(f.shape.n / 512), .width = 512, .rotary = 64, .stride = @intCast(f.stride), .position = f.pos[row], .inverse = @intFromBool(p.inverse) }, 1);
                        bind(e, p.factors, 0, 2);
                        e.dispatchThreads(.{ .width = 32, .height = f.shape.n / 512 }, .{ .width = 128 });
                    },
                    .glue => {
                        bind(e, p.x, row * f.stride * 2, 0);
                        bind(e, p.weight, 0, 1);
                        bind(e, p.factors, 0, 2);
                        bind(e, p.base_work, row * 528, 3);
                        bind(e, p.base_work, row * 528, 4);
                        e.setValue(p.glueArgs(1, f.pos[row]), 5);
                        e.dispatchGroups(.{ .width = 16 }, .{ .width = 256 });
                    },
                    .publish => {
                        bind(e, p.chunk, row * 528, 0);
                        bind(e, p.base_live[f.shape.rows - row - 1], ref.ring_offset, 1);
                        e.setValue([4]u32{ 1, f.pos[row], 528, 528 }, 2);
                        e.dispatchThreads(.{ .width = 528 }, .{ .width = 32 });
                    },
                }
            }
            try p.rig.finish(pass);
            return p.gather(true);
        }

        pub fn reference(p: *Self, want: []f64, mag: []f64) !void {
            p.f.?.evaluate(.none, .sequential, want);
            for (mag, want) |*m, w| m.* = @abs(w);
        }

        pub fn simulate(p: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            p.f.?.evaluate(acc, order, out);
        }

        pub fn check(p: *Self, gate: g.Gate) !?[]const u8 {
            if (gate != .nonfinite) return null;
            const out = p.gather(false);
            if (p.kind != .rope) {
                const old = try p.replaced();
                if (!std.mem.eql(u8, out, old)) return "non-finite packed bytes or untouched ring state differ from per-slot execution";
                return null;
            }
            const want = try p.rig.a.alloc(f64, p.count());
            defer p.rig.a.free(want);
            p.f.?.evaluate(.none, .sequential, want);
            var finite: usize = 0;
            var nonfinite: usize = 0;
            for (want, 0..) |w, i| {
                if (c.classOf(w) != c.classOf(c.read(.bf16, out, i))) return "row rotation changed a float64 finite, NaN or infinity class";
                if (std.math.isFinite(w)) finite += 1 else nonfinite += 1;
            }
            if (finite == 0 or nonfinite == 0) return error.NoNativeRowsClassMixture;
            return null;
        }
    };
}

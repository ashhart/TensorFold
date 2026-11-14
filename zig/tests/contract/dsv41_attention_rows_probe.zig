//! Real native dense/scan launches and independent per-slot GPU baselines, including full-sized source KV views.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const source = @import("dsv41_attention_rows_source.zig");
const ref = @import("dsv41_attention_rows_ref.zig");
const c = contract.spec;
const g = contract.gates;
const Ref = rig_.Ref;
fn bind(e: mtl.ComputeEncoder, r: Ref, offset: usize, at: usize) void {
    e.setBuffer(r.buf, r.off + offset, at);
}

pub const Old = struct {
    dense: mtl.Pipeline,
    scan: mtl.Pipeline,
    merge: mtl.Pipeline,
    merge_split: mtl.Pipeline,
    pub fn load(device: mtl.Device, lib: mtl.Library) !Old {
        const dense = try mtl.Pipeline.init(device, lib, "dsv41_dense_kv", false);
        errdefer dense.deinit();
        const scan = try mtl.Pipeline.init(device, lib, "dsv41_sparse_nax", false);
        errdefer scan.deinit();
        const merge = try mtl.Pipeline.init(device, lib, "dsv41_sparse_merge", false);
        errdefer merge.deinit();
        return .{ .dense = dense, .scan = scan, .merge = merge, .merge_split = try mtl.Pipeline.init(device, lib, "dsv41_sparse_merge_split", false) };
    }
    pub fn deinit(p: Old) void {
        p.dense.deinit();
        p.scan.deinit();
        p.merge.deinit();
        p.merge_split.deinit();
    }
};

pub fn For(comptime native: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        spec: c.Spec,
        kind: source.Kind,
        config: ref.Config,
        fault: ?source.Fault,
        pipeline: mtl.Pipeline,
        dense_pipeline: mtl.Pipeline,
        merge: mtl.Pipeline,
        old: *const Old,
        f: ?ref.Fixture = null,
        pos: Ref = undefined,
        sink: Ref = undefined,
        chunk: Ref = undefined,
        picks: Ref = undefined,
        addr: Ref = undefined,
        mains_table: Ref = undefined,
        decoy: Ref = undefined,
        rings: [8]Ref = undefined,
        mains: [8]Ref = undefined,
        work: Ref = undefined,
        base_work: Ref = undefined,
        dense: Ref = undefined,
        base_dense: Ref = undefined,
        parts: Ref = undefined,
        base_parts: Ref = undefined,
        result: std.ArrayList(u8) = .empty,
        baseline: std.ArrayList(u8) = .empty,
        const hooks = blk: {
            var h = g.hooks(Self);
            h.output = output;
            h.intact = intact;
            h.release = release;
            break :blk h;
        };
        pub fn kernel(p: *Self) g.Kernel {
            return .{ .spec = p.spec, .ctx = p, .vt = &hooks, .control = if (p.fault) |f| switch (f) {
                .clip => .nonfinite,
                .input_write => .repeat,
                .position, .half_score => .shapes,
                .half_value, .split_tail => .sums,
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
            return p.f.?.shape.rows * (if (p.kind == .dense) ref.dense_rows * 512 else p.f.?.shape.n);
        }
        pub fn prepare(p: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            p.drop();
            if (shape.rows == 0 or shape.rows > 8 or shape.k != 512) return false;
            if (p.kind == .dense) {
                if (shape.n != 512) return false;
            } else if (shape.n != 8192 and shape.n != 16384 and shape.n != 32768) return false;
            p.f = try ref.Fixture.init(p.rig.a, shape, p.config, fill, seed, p.kind == .dense);
            const f = p.f.?;
            const b = std.mem.sliceAsBytes;
            p.pos = try p.rig.input(b(f.pos));
            p.chunk = try p.rig.input(f.chunk);
            p.picks = try p.rig.input(b(f.picks));
            p.decoy = try p.rig.input(f.decoy);
            var addresses: [32]u64 = @splat(0);
            var mains: [32]u64 = undefined;
            for (0..shape.rows) |slot| {
                p.rings[slot] = try p.rig.input(f.rings[slot * ref.live_bytes ..][0..ref.live_bytes]);
                p.mains[slot] = try p.rig.input(f.mains[slot]);
            }
            var max_main: usize = 0;
            for (0..shape.rows) |row| {
                const ring = p.rings[shape.rows - row - 1];
                addresses[4 * row] = ring.buf.gpuAddress() + ring.off;
                addresses[4 * row + 2] = p.picks.buf.gpuAddress() + p.picks.off + row * 512 * 4;
                for (0..4) |owner| {
                    const main = if (owner == p.config.owner) p.mains[row] else p.decoy;
                    mains[4 * row + owner] = main.buf.gpuAddress() + main.off;
                }
                max_main = @max(max_main, f.compressedRows(row));
            }
            p.addr = try p.rig.input(b(addresses[0 .. shape.rows * 4]));
            p.mains_table = try p.rig.input(b(mains[0 .. shape.rows * 4]));
            p.dense = try p.rig.output(shape.rows * ref.dense_rows * 512 * 2);
            if (p.kind == .dense) p.base_dense = try p.rig.output(shape.rows * ref.dense_rows * 512 * 2) else {
                p.sink = try p.rig.input(b(f.sink));
                p.work = try p.rig.output(f.q.len * 2);
                p.base_work = try p.rig.output(f.q.len * 2);
                // The parent addresses selected rows by absolute compressed index, not by compacted pick index.
                p.base_dense = try p.rig.scratch((129 + max_main) * 512 * 2);
                const one = @as(usize, p.config.splits) * (shape.n / 512 / 16) * 4 * 32 * 68 * 4;
                p.parts = try p.rig.scratch(shape.rows * one);
                p.base_parts = try p.rig.scratch(one);
            }
            try p.result.resize(p.rig.a, p.count() * 2);
            try p.baseline.resize(p.rig.a, p.count() * 2);
            return true;
        }
        fn args(p: *Self) native.csa2.RowsParams {
            const f = p.f.?;
            return .{ .rows = @intCast(f.shape.rows), .heads = @intCast(if (p.kind == .dense) 16 else f.shape.n / 512), .q_stride = @intCast(if (p.kind == .dense) 8192 + 16 else f.q_stride), .ring_off = ref.ring_offset, .ratio = p.config.ratio, .owner = p.config.owner, .splits = p.config.splits };
        }
        fn mergeArgs(p: *Self, rows: usize, position: u32, compressed: usize) native.csa2.Params {
            return .{ .rows = @intCast(rows), .heads = @intCast(p.f.?.shape.n / 512), .width = 512, .q_stride = @intCast(p.f.?.q_stride), .position = position, .chunk_rows = 1, .ratio = p.config.ratio, .compressed_rows = @intCast(compressed), .splits = p.config.splits };
        }
        pub fn run(p: *Self) !void {
            const f = p.f.?;
            if (p.kind == .scan) @memcpy(p.rig.bytes(p.work, f.q.len * 2), std.mem.sliceAsBytes(f.q));
            const pass = p.rig.begin();
            const e = pass.e;
            for (p.rings[0..f.shape.rows]) |r| e.useResource(r.buf, mtl.ResourceUsage.read);
            for (p.mains[0..f.shape.rows]) |r| e.useResource(r.buf, mtl.ResourceUsage.read);
            e.useResource(p.decoy.buf, mtl.ResourceUsage.read);
            e.useResource(p.picks.buf, mtl.ResourceUsage.read);
            e.setPipeline(if (p.kind == .dense) p.pipeline else p.dense_pipeline);
            bind(e, p.addr, 0, 0);
            bind(e, p.pos, 0, 1);
            bind(e, p.chunk, 0, 2);
            bind(e, p.dense, 0, 3);
            e.setValue(p.args(), 4);
            bind(e, p.mains_table, 0, 5);
            e.dispatchThreads(.{ .width = ref.dense_rows, .height = 128, .depth = f.shape.rows }, .{ .width = 32, .height = 4 });
            if (p.kind == .scan) {
                e.barrier();
                e.setPipeline(p.pipeline);
                bind(e, p.work, 0, 0);
                bind(e, p.addr, 0, 1);
                bind(e, p.pos, 0, 2);
                bind(e, p.sink, 0, 3);
                bind(e, p.work, 0, 4);
                e.setValue(p.args(), 5);
                bind(e, p.dense, 0, 6);
                bind(e, p.parts, 0, 7);
                e.dispatchGroups(.{ .width = f.shape.n / 512 / 16, .height = f.shape.rows, .depth = p.config.splits }, .{ .width = 32, .height = 4 });
                if (p.config.splits > 1) {
                    e.barrier();
                    e.setPipeline(p.merge);
                    bind(e, p.parts, 0, 0);
                    bind(e, p.work, 0, 1);
                    bind(e, p.sink, 0, 2);
                    e.setValue(p.mergeArgs(f.shape.rows, 0, 0), 3);
                    e.dispatchGroups(.{ .width = f.shape.n / 512 / 16, .height = f.shape.rows, .depth = if (p.config.merge_split) 4 else 1 }, .{ .width = 32, .height = 4 });
                }
            }
            try p.rig.finish(pass);
        }
        fn gather(p: *Self, old: bool) []const u8 {
            const f = p.f.?;
            const dest = if (old) p.baseline.items else p.result.items;
            if (p.kind == .dense) @memcpy(dest, p.rig.bytes(if (old) p.base_dense else p.dense, dest.len)) else {
                const data = p.rig.bytes(if (old) p.base_work else p.work, f.q.len * 2);
                for (0..f.shape.rows) |row| @memcpy(dest[row * f.shape.n * 2 ..][0 .. f.shape.n * 2], data[row * f.q_stride * 2 ..][0 .. f.shape.n * 2]);
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
            if (p.kind == .dense) return true;
            const f = p.f.?;
            const original = std.mem.sliceAsBytes(f.q);
            const work = p.rig.bytes(p.work, original.len);
            for (0..f.shape.rows) |row| {
                const start = (row * f.q_stride + f.shape.n) * 2;
                const end = (row + 1) * f.q_stride * 2;
                if (!std.mem.eql(u8, original[start..end], work[start..end])) return false;
            }
            return true;
        }
        fn oldDecode(p: *Self, e: mtl.ComputeEncoder, slot: usize, compact: bool) void {
            const f = p.f.?;
            const dst_offset = if (compact) slot * ref.dense_rows * 512 * 2 else 0;
            e.setPipeline(p.old.dense);
            const ring = p.rings[f.shape.rows - slot - 1];
            const window_rows = @min(f.pos[slot], 128);
            if (window_rows != 0) {
                bind(e, ring, ref.ring_offset, 0);
                bind(e, p.base_dense, dst_offset, 1);
                e.setValue(native.csa2.DenseKv{ .rows = window_rows, .width = 512, .stride = 528, .first_row = 0, .source = 0 }, 2);
                e.dispatchThreads(.{ .width = window_rows, .height = 128 }, .{ .width = 32, .height = 4 });
            }
            bind(e, p.chunk, slot * 528, 0);
            bind(e, p.base_dense, dst_offset, 1);
            e.setValue(native.csa2.DenseKv{ .rows = 1, .width = 512, .stride = 528, .first_row = 128, .source = 0 }, 2);
            e.dispatchThreads(.{ .width = 1, .height = 128 }, .{ .width = 32, .height = 4 });
            // Invoke the old decoder only for visible picks; the new contract also tests ignored future IDs.
            for (0..@min(f.compressedRows(slot), 512)) |i| {
                const pick = f.picks[slot * 512 + i];
                if (pick < 0 or @as(usize, @intCast(pick)) >= f.compressedRows(slot)) continue;
                bind(e, p.mains[slot], @as(usize, @intCast(pick)) * 288, 0);
                e.setValue(native.csa2.DenseKv{ .rows = 1, .width = 512, .stride = 288, .first_row = @intCast(129 + (if (compact) i else @as(usize, @intCast(pick)))), .source = 1 }, 2);
                e.dispatchThreads(.{ .width = 1, .height = 128 }, .{ .width = 32, .height = 4 });
            }
        }
        pub fn replaced(p: *Self) ![]const u8 {
            const f = p.f.?;
            if (p.kind == .scan) @memcpy(p.rig.bytes(p.base_work, f.q.len * 2), std.mem.sliceAsBytes(f.q));
            for (0..f.shape.rows) |slot| {
                const pass = p.rig.begin();
                const e = pass.e;
                p.oldDecode(e, slot, p.kind == .dense);
                if (p.kind == .scan) {
                    const offset = slot * f.q_stride * 2;
                    const ring = p.rings[f.shape.rows - slot - 1];
                    e.barrier();
                    e.setPipeline(p.old.scan);
                    bind(e, p.base_work, offset, 0);
                    bind(e, ring, ref.ring_offset, 1);
                    bind(e, p.chunk, slot * 528, 2);
                    bind(e, p.mains[slot], 0, 3);
                    bind(e, p.picks, slot * 512 * 4, 4);
                    bind(e, p.sink, 0, 5);
                    bind(e, p.base_work, offset, 6);
                    e.setValue(p.mergeArgs(1, f.pos[slot], f.compressedRows(slot)), 7);
                    bind(e, p.base_dense, 0, 8);
                    bind(e, p.base_parts, 0, 9);
                    e.dispatchGroups(.{ .width = f.shape.n / 512 / 16, .depth = p.config.splits }, .{ .width = 32, .height = 4 });
                    if (p.config.splits > 1) {
                        e.barrier();
                        e.setPipeline(if (p.config.merge_split) p.old.merge_split else p.old.merge);
                        bind(e, p.base_parts, 0, 0);
                        bind(e, p.base_work, offset, 1);
                        bind(e, p.sink, 0, 2);
                        e.setValue(p.mergeArgs(1, f.pos[slot], f.compressedRows(slot)), 3);
                        e.dispatchGroups(.{ .width = f.shape.n / 512 / 16, .depth = if (p.config.merge_split) 4 else 1 }, .{ .width = 32, .height = 4 });
                    }
                }
                try p.rig.finish(pass);
            }
            return p.gather(true);
        }
        pub fn reference(p: *Self, want: []f64, mag: []f64) !void {
            if (p.kind == .dense) p.f.?.denseReference(want) else {
                if (p.f.?.fill != .stress) return error.ExpectedAttentionStressInputs;
                try p.simulate(.none, .sequential, want);
            }
            for (mag, want) |*m, w| m.* = @abs(w);
        }
        pub fn simulate(p: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const f = p.f.?;
            if (p.kind == .dense) {
                f.denseReference(out);
                return;
            }
            for (0..f.shape.rows) |slot| {
                const keys = 128 + @min(f.compressedRows(slot), 512);
                @memset(out[slot * f.shape.n ..][0..f.shape.n], ref.stressValue(keys, acc, order));
            }
        }
        pub fn check(p: *Self, gate: g.Gate) !?[]const u8 {
            if (gate != .nonfinite) return null;
            const out = p.gather(false);
            const f = p.f.?;
            if (p.kind == .dense) {
                const want = try p.rig.a.alloc(f64, p.count());
                defer p.rig.a.free(want);
                f.denseReference(want);
                var finite: usize = 0;
                var nonfinite: usize = 0;
                for (want, 0..) |w, i| {
                    const actual = c.read(.bf16, out, i);
                    if (c.classOf(w) != c.classOf(actual)) {
                        std.debug.print("dense class at element {d}: expected {d}, actual {d}, raw 0x{x}\n", .{ i, w, actual, std.mem.readInt(u16, out[i * 2 ..][0..2], .little) });
                        return "packed native decode changed a float64 value's class";
                    }
                    if (std.math.isFinite(w)) finite += 1 else nonfinite += 1;
                }
                if (finite == 0 or nonfinite == 0) return error.NoAttentionDecodeClassMixture;
            } else {
                for (0..f.shape.rows) |slot| for (0..f.shape.n) |i| {
                    const v = c.read(.bf16, out, slot * f.shape.n + i);
                    switch (i / 512) {
                        0, 2 => if (v != 0) return "NaN or negative-infinite scores were not masked to zero attention",
                        1 => if (!std.math.isNan(v)) return "positive-infinite scores lost their NaN probability class",
                        else => if (!std.math.isFinite(v) or v <= 0) return "finite positive keys did not retain finite positive attention",
                    }
                };
            }
            return null;
        }
    };
}

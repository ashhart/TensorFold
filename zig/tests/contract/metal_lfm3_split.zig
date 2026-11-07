//! Mechanical key-parallel attention checks; model FP32/BF16 fidelity remains a separate recorded gate.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
const Args = @import("tensorfold").lfm3.kernels.AttentionArgs;
const parts = @import("tensorfold").lfm3.attention_split.parts;
pub const Fault = enum { none, tail, nonfinite };

pub fn name(sliding: bool, fault: Fault) []const u8 {
    return switch (fault) {
        .none => if (sliding) "metal LFM3 split structural sliding" else "metal LFM3 split structural full",
        .tail => "metal LFM3 split structural bad-tail",
        .nonfinite => "metal LFM3 split structural bad-nonfinite",
    };
}
fn depth(s: c.Shape) usize {
    return s.k + s.rows + 2;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipelines: [2]mtl.Pipeline,
    scratch: [3]rig_.Ref = undefined,
    sliding: bool,
    fault: Fault,
    rows: usize = 0,
    history: usize = 0,
    cap: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    args: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    const vt = g.hooks(Probe);

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, sliding: bool, fault: Fault) !Probe {
        const source = try std.mem.concat(a, u8, &.{ sources.lfm3_attention, "\n", sources.lfm3_attention_split });
        defer a.free(source);
        const needle = "  const size_t target = (size_t(row) * 16 + head) * L3S_PARTS + part;";
        const drop = "  for (uint j = 0; j < 4; ++j) if (!isfinite(out[j])) out[j] = 0.0f;\n  if (!isfinite(sum)) sum = 1.0f;\n";
        const compiled = if (fault == .nonfinite) try std.mem.replaceOwned(u8, a, source, needle, drop ++ needle) else try a.dupe(u8, source);
        defer a.free(compiled);
        if (fault == .nonfinite and std.mem.eql(u8, source, compiled)) return error.MissingNonfiniteControl;
        const partial = try rig_.pipeline(device, compiled, "lfm3_attn_split");
        errdefer partial.deinit();
        const merge = try rig_.pipeline(device, compiled, "lfm3_attn_split_merge");
        errdefer merge.deinit();
        if (partial.simdWidth() != 32 or partial.maxThreads() < 128 or merge.maxThreads() < 32 or rig_.tgMemory(partial) != 2592 or rig_.tgMemory(merge) != 0)
            return error.SplitAttentionLimits;
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .pipelines = .{ partial, merge }, .sliding = sliding, .fault = fault };
    }
    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
        for (p.pipelines) |pipe| pipe.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = switch (p.fault) {
            .none => null,
            .tail => .shapes,
            .nonfinite => .nonfinite,
        }, .spec = .{
            .name = name(p.sliding, p.fault),
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .nan = .{ .stated = "visible-query/key/value nonfinite classes; masked future values must not contaminate" },
            .served = &.{
                .{ .rows = 1, .n = 2048, .k = 0 },     .{ .rows = 5, .n = 2048, .k = 2043 },
                .{ .rows = 8, .n = 2048, .k = 257 },   .{ .rows = 9, .n = 2048, .k = 0 },
                .{ .rows = 17, .n = 2048, .k = 31 },   .{ .rows = 31, .n = 2048, .k = 255 },
                .{ .rows = 32, .n = 2048, .k = 256 },  .{ .rows = 33, .n = 2048, .k = 257 },
                .{ .rows = 63, .n = 2048, .k = 2048 }, .{ .rows = 64, .n = 2048, .k = 8192 },
            },
            .tiles = .{ .{ .size = 1 }, .{}, .{ .size = 128, .step = 32, .min = 0 } },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "uniform-score structural fixture; no claimed 100x reduced-accumulator witness" },
        } };
    }
    pub fn prepare(p: *Probe, requested: c.Shape, fill: g.Fill, seed: u64) !bool {
        var s = requested;
        if (fill != .random and s.rows < 5) {
            std.debug.print("split structural {t}: expanding fixture rows {d} to eligible 5\n", .{ fill, s.rows });
            s.rows = 5;
        }
        if (fill == .nonfinite and p.sliding and s.k < 257) {
            std.debug.print("split sliding nonfinite: expanding history to 257 for a masked-past plant\n", .{});
            s.k = 257;
        }
        if (s.rows == 0 or s.rows > 64 or s.n != 2048 or s.k > 8192) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.rows = s.rows;
        p.history = s.k;
        p.cap = s.k + s.rows + 19;
        p.q = try a.alloc(u16, s.rows * 2048);
        p.k = try a.alloc(u16, p.cap * 512);
        const k = p.k;
        p.v = try a.alloc(u16, p.cap * 512);
        @memset(p.q, 0);
        @memset(k, 0);
        @memset(p.v, 0x7fc0);
        for (0..4) |head| for (0..s.k + s.rows) |key| for (0..128) |col| {
            const value: i32 = @as(i32, @intCast((key * 17 + col * 7 + head + seed % 31) % 31)) - 15;
            p.v[(head * p.cap + key) * 128 + col] = c.toBf16(@floatFromInt(value));
        };
        if (fill == .nonfinite) {
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |x, row| {
                if (row < s.rows) p.q[row * 2048] = x;
            }
            if (s.rows > 3) p.q[3 * 2048 + 1] = 0xff80;
            p.v[(p.cap + s.k + s.rows - 1) * 128] = 0x7fc0;
            // A future nonfinite key must remain masked until the final query.
            k[(2 * p.cap + s.k + s.rows - 1) * 128] = 0x7fc0;
            if (p.sliding) p.v[2 * p.cap * 128] = 0x7fc0;
            const visible = (3 * p.cap + s.k) * 128;
            p.v[visible + 1] = 0x7f80;
            p.v[visible + 2] = 0xff80;
            p.v[visible + 3] = 0x7f80;
            p.v[visible + 128 + 3] = 0xff80;
        }
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(k));
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(p.v));
        p.out = try p.rig.output(s.rows * 2048 * 2);
        const slots = s.rows * 16 * 64;
        p.scratch = .{ try p.rig.scratch(slots * 4), try p.rig.scratch(slots * 4), try p.rig.scratch(slots * 128 * 4) };
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows = p.rows - @intFromBool(p.fault == .tail and p.rows > 1);
        const args = Args{ .len = @intCast(p.history), .cap = @intCast(p.cap), .scale = 1.0 / @sqrt(@as(f32, 128)), .window = if (p.sliding) 256 else 0, .chunk_base = 0 };
        e.setPipeline(p.pipelines[0]);
        for (p.args, 0..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.setBytes(std.mem.asBytes(&args), 3);
        for (p.scratch, 4..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.dispatchGroups(.of(16, rows, parts(p.history + rows, args.window)), .of(128, 1, 1));
        e.barrier();
        e.setPipeline(p.pipelines[1]);
        for (p.scratch, 0..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(p.out.buf, p.out.off, 4);
        for (p.args, 5..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.dispatchGroups(.of(16, rows, 1), .of(32, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        return p.rows * 2048;
    }
    fn expected(p: *Probe, row: usize, head: usize, col: usize) f32 {
        if (!std.math.isFinite(c.fromBf16(p.q[row * 2048 + head * 128]))) return std.math.nan(f32);
        const end = p.history + row + 1;
        const first = if (p.sliding) end -| 256 else 0;
        var sum: f32 = 0;
        for (first..end) |key| {
            if (!std.math.isFinite(c.fromBf16(p.k[((head / 4) * p.cap + key) * 128]))) return std.math.nan(f32);
            sum += c.fromBf16(p.v[((head / 4) * p.cap + key) * 128 + col]);
        }
        return sum * (1.0 / @as(f32, @floatFromInt(end - first)));
    }
    pub fn check(p: *Probe, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite and at != .shapes) return null;
        const out = try p.rig.outputBytes();
        var nonfinite: usize = 0;
        for (0..p.rows) |row| for (0..16) |head| for (0..128) |col| {
            const want = p.expected(row, head, col);
            const index = (row * 16 + head) * 128 + col;
            const got = c.read(.bf16, out, index);
            if (!std.math.isFinite(want)) nonfinite += 1;
            if (c.classOf(got) != c.classOf(want)) {
                std.debug.print("split class mismatch row={d} head={d} col={d}: got={t} want={t}\n", .{ row, head, col, c.classOf(got), c.classOf(want) });
                return "visible/masked nonfinite class mismatch";
            }
            if (at == .shapes and got != c.fromBf16(c.toBf16(want)))
                return "uniform-score exact FP32 normalization differs after BF16 rounding";
        };
        if (at == .nonfinite and (nonfinite == 0 or nonfinite == p.count())) return "nonfinite plant is vacuous";
        return null;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, index| {
            w.* = p.expected(index / 2048, (index % 2048) / 128, index % 128);
            m.* = 15;
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.history + p.rows);
        defer p.a.free(terms);
        for (out, 0..) |*value, index| {
            const row = index / 2048;
            const end = p.history + row + 1;
            const first = if (p.sliding) end -| 256 else 0;
            const head = (index % 2048) / 128;
            for (first..end) |key| terms[key - first] = c.fromBf16(p.v[((head / 4) * p.cap + key) * 128 + index % 128]);
            value.* = ct.rows.sumIn(acc, order, terms[0 .. end - first]) *
                @as(f32, 1.0 / @as(f32, @floatFromInt(end - first)));
        }
    }
};

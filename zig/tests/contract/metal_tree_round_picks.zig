//! Chunked BF16 picks: every Pick byte equals the original row reduction, including non-finite masks.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
pub const name = "tree round picks";
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 248320, .k = 1 },
    .{ .rows = 4, .n = 248320, .k = 1 },
    .{ .rows = 16, .n = 248320, .k = 1 },
    .{ .rows = 64, .n = 248320, .k = 1 },
    .{ .rows = 4, .n = 8192, .k = 1 },
    .{ .rows = 4, .n = 8193, .k = 1 },
};
fn none(_: c.Shape) usize {
    return 0;
}
pub const Mutation = enum {
    none,
    last_tie,
    no_mask,
    fn label(m: Mutation) []const u8 {
        return switch (m) {
            .none => name,
            .last_tie => "control: tree round picks, higher id wins a chunk tie",
            .no_mask => "control: tree round picks, chunk masks are lost",
        };
    }
};
fn source(a: std.mem.Allocator, mutation: Mutation) ![]u8 {
    const text = sources.core_tree_round;
    if (mutation == .none) return a.dupe(u8, text);
    const at = std.mem.indexOf(u8, text, "kernel void tf_round_pick_rows").?;
    const from, const to = if (mutation == .last_tie)
        .{ "const uint who = simd_min(best == top && id != 0xffffffffu ? id : 0xffffffffu);", "const uint who = simd_max(best == top && id != 0xffffffffu ? id : 0u);" }
    else
        .{ "mask |= other.mask;", "mask |= 0;" };
    if (std.mem.count(u8, text[at..], from) != 1) return error.ControlEditMissing;
    const tail = try std.mem.replaceOwned(u8, a, text[at..], from, to);
    defer a.free(tail);
    return std.mem.concat(a, u8, &.{ text[0..at], tail });
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    chunks_pipe: mtl.Pipeline,
    rows_pipe: mtl.Pipeline,
    old_pipe: mtl.Pipeline,
    mutation: Mutation,
    rows: usize = 0,
    vocab: u32 = 0,
    stride: u32 = 0,
    x: []u16 = &.{},
    input: rig_.Ref = undefined,
    scratch: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    original: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: Mutation) !Probe {
        const text = try source(a, mutation);
        defer a.free(text);
        const chunks = try rig_.pipeline(device, text, "tf_round_pick_chunks");
        errdefer chunks.deinit();
        const rows = try rig_.pipeline(device, text, "tf_round_pick_rows");
        errdefer rows.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .chunks_pipe = chunks, .rows_pipe = rows, .old_pipe = try rig_.pipeline(device, sources.core_tree_round, "tf_round_bf16_argmax"), .mutation = mutation };
    }
    fn drop(p: *Probe) void {
        p.a.free(p.x);
        p.x = &.{};
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.chunks_pipe.deinit();
        p.rows_pipe.deinit();
        p.old_pipe.deinit();
    }
    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = p.mutation.label(), .accum = .none, .depth = none, .out = .raw, .served = &shapes, .replaces = "tf_round_bf16_argmax", .nan = .{ .stated = "any NaN or infinity gives token 0xffffffff and ORed NaN/+Inf/-Inf bits 1/2/4" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "comparison only, no sums" } }, .control = switch (p.mutation) {
            .none => null,
            .last_tie => .shapes,
            .no_mask => .nonfinite,
        } };
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (s.rows == 0 or s.rows > 64 or s.n < 8192 or s.k != 1) return false;
        p.rows = s.rows;
        p.vocab = @intCast(s.n);
        p.stride = p.vocab + 37;
        p.x = try p.a.alloc(u16, p.rows * p.stride);
        @memset(p.x, 0x7fc0); // padding is never a logit
        for (0..p.rows) |r| {
            const row = p.x[r * p.stride ..][0..p.vocab];
            for (row, 0..) |*v, i| {
                const h = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed +% r) >> 40;
                v.* = c.toBf16(@as(f32, @floatFromInt(h % 61)) / 8 - 4);
            }
            if (r % 4 == 1) {
                row[p.vocab - 1] = c.toBf16(5); // maximum in the partial last chunk
            } else if (r % 4 == 2) {
                for (row, 0..) |*v, i| v.* = if (i % 2 == 0) 0x8000 else 0;
            } else {
                for ([_]usize{ 31, 32, 1023, 1024, 32799, 32800 }) |i| if (i < p.vocab) {
                    row[i] = c.toBf16(5);
                };
            }
            if (fill == .nonfinite) switch (r % 4) {
                0 => {
                    row[17] = 0x7fc0;
                    row[1024] = 0x7f80;
                    row[p.vocab - 1] = 0xff80;
                },
                1 => row[p.vocab - 1] = 0x7fc0,
                2 => row[p.vocab - 1] = 0x7f80,
                3 => row[p.vocab - 1] = 0xff80,
                else => unreachable,
            };
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.scratch = try p.rig.scratch(p.rows * ((p.vocab + 1023) / 1024) * 12);
        p.out = try p.rig.output(p.count());
        p.original = try p.rig.scratch(p.count());
        return true;
    }
    pub fn count(p: *Probe) usize {
        return p.rows * 8;
    }
    fn args(p: *Probe, e: mtl.ComputeEncoder) void {
        e.setValue([3]u32{ @intCast(p.rows), p.vocab, p.stride }, 2);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.chunks_pipe);
        e.setBuffer(p.input.buf, p.input.off, 0);
        e.setBuffer(p.scratch.buf, p.scratch.off, 1);
        p.args(e);
        e.dispatchGroups(mtl.Size.of((p.vocab + 1023) / 1024, p.rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        e.setPipeline(p.rows_pipe);
        e.setBuffer(p.scratch.buf, p.scratch.off, 0);
        e.setBuffer(p.out.buf, p.out.off, 1);
        p.args(e);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.old_pipe);
        e.setBuffer(p.input.buf, p.input.off, 0);
        e.setBuffer(p.original.buf, p.original.off, 1);
        p.args(e);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.original, p.count());
    }
    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const out = p.rig.bytes(p.out, p.count());
        for (0..p.rows) |r| {
            var mask: u32 = 0;
            var best: u32 = std.math.maxInt(u32);
            var value: f64 = -std.math.inf(f64);
            for (p.x[r * p.stride ..][0..p.vocab], 0..) |v, id| {
                const x = c.fromBf16(v);
                if (!std.math.isFinite(x)) {
                    mask |= if (std.math.isNan(x)) @as(u32, 1) else if (x > 0) @as(u32, 2) else 4;
                } else if (best == std.math.maxInt(u32) or x > value) {
                    best = @intCast(id);
                    value = x;
                }
            }
            const got = std.mem.readInt(u32, out[r * 8 ..][0..4], .little);
            const flags = std.mem.readInt(u32, out[r * 8 + 4 ..][0..4], .little);
            if (flags != mask) return "non-finite mask differs from the scalar rule";
            if (got != (if (mask != 0) std.math.maxInt(u32) else best)) return "lowest-id pick differs from the scalar rule";
        }
        return null;
    }
};

pub const Suite = struct {
    probes: std.ArrayList(Probe) = .empty,
    pub fn collect(s: *Suite, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        try s.probes.ensureTotalCapacity(a, 3);
        for (std.enums.values(Mutation)) |m| {
            if (only) |o| if (std.mem.indexOf(u8, m.label(), o) == null) continue;
            s.probes.appendAssumeCapacity(try Probe.init(a, device, queue, m));
            try kernels.append(a, s.probes.items[s.probes.items.len - 1].kernel());
        }
    }
    pub fn deinit(s: *Suite, a: std.mem.Allocator) void {
        for (s.probes.items) |*p| p.deinit();
        s.probes.deinit(a);
    }
};

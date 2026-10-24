//! LFM2's tap copy for an external drafter (lfm2_tap): a layer's rows into their tap's columns, the others kept.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");

const hidden = 2048;
const taps = 5;
pub const name = "metal lfm2 tap copy";

/// rows: one decode row, a verify window, a shared round's 32 and a 2,048-row prompt chunk.
const shapes = [_]c.Shape{ .{ .rows = 1, .n = taps * hidden, .k = 1 }, .{ .rows = 10, .n = taps * hidden, .k = 1 }, .{ .rows = 32, .n = taps * hidden, .k = 1 }, .{ .rows = 2048, .n = taps * hidden, .k = 1 } };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec() c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .out = .bf16, .served = &shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "a copy moves values: no sums" } };
}

/// A bf16 value for element `i` over signs and binades; the nonfinite fill plants NaN and infinities.
fn value(i: usize, seed: u64, fill: g.Fill) u16 {
    if (fill == .nonfinite) switch ((@as(u64, i) *% 2654435761 +% seed) % 9973) {
        0 => return 0x7fc0,
        1 => return 0x7f80,
        2 => return 0xff80,
        else => {},
    };
    const h: u64 = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    const mant: f32 = @as(f32, @floatFromInt(h % 1024)) / 1024 + 1;
    const exp: i32 = @as(i32, @intCast((h >> 10) % 8)) - 4;
    const v = std.math.ldexp(mant, exp) * @as(f32, if (((h >> 15) & 1) == 0) 1 else -1);
    return c.toBf16(v);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    src: []u16 = &.{}, // H rows
    prior: []u16 = &.{}, // the taps rows as the call finds them (the other taps' columns must stay)
    args: [2]rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, sources.lfm2, "lfm2_tap") };
    }

    fn drop(p: *Probe) void {
        p.a.free(p.src);
        p.a.free(p.prior);
        p.src = &.{};
        p.prior = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = spec() };
    }

    fn fill(out: []u16, seed: u64, f: g.Fill) void {
        for (out, 0..) |*v, i| v.* = value(i, seed, f);
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        p.drop();
        if (s.n != taps * hidden or s.rows == 0) return false;
        p.rows = s.rows;
        p.src = try p.a.alloc(u16, s.rows * hidden);
        fill(p.src, seed, f);
        p.prior = try p.a.alloc(u16, s.rows * taps * hidden);
        fill(p.prior, seed +% 3, .random);
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.src));
        const at = [2]u32{ taps * hidden, 2 * hidden }; // the third tap's columns
        p.args[1] = try p.rig.input(std.mem.asBytes(&at));
        p.y = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.rows * taps * hidden;
    }

    pub fn run(p: *Probe) !void {
        @memcpy(p.rig.bytes(p.y, p.count() * 2), std.mem.sliceAsBytes(p.prior));
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.args[0].buf, p.args[0].off, 0);
        e.setBuffer(p.y.buf, p.y.off, 1);
        e.setBuffer(p.args[1].buf, p.args[1].off, 2);
        e.dispatchThreads(mtl.Size.of(hidden / 8, p.rows, 1), mtl.Size.of(256, 1, 1));
        return p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, p.prior) |*w, *m, v| { // the other taps' columns keep what the call found
            w.* = c.fromBf16(v);
            m.* = @abs(w.*);
        }
        for (0..p.rows) |r| for (0..hidden) |i| {
            const v = c.fromBf16(p.src[r * hidden + i]);
            want[r * taps * hidden + 2 * hidden + i] = v;
            mag[r * taps * hidden + 2 * hidden + i] = @abs(v);
        };
    }

    /// The copy is exact: the third tap's columns hold H's bits, the others what the call found.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const out = std.mem.bytesAsSlice(u16, p.rig.bytes(p.y, p.count() * 2));
        for (out, p.prior, 0..) |got, before, i| {
            const r = i / (taps * hidden);
            const col = i % (taps * hidden);
            const want = if (col >= 2 * hidden and col < 3 * hidden) p.src[r * hidden + col - 2 * hidden] else before;
            if (got != want) return "an output differs from the copy's bits";
        }
        return null;
    }
};

/// Gate 5 for the tap's launch of 256 threads and no threadgroup memory: every Apple chip we serve, then this GPU.
pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    if (only) |s| if (std.mem.indexOf(u8, name, s) == null) return 0;
    const d = contract.limits.MetalLaunch{ .name = name, .threads = 256, .tg_memory = 0, .offline = false };
    var notes: std.ArrayList(u8) = .empty;
    defer notes.deinit(a);
    var problems: usize = 0;
    for (contract.limits.apple) |chip| problems += try contract.limits.metal(d, chip, &notes, a);
    std.debug.print("{s}", .{notes.items});
    return problems + try @import("metal_limits.zig").onThisGpu(a, device, .{ .decl = d, .function = "lfm2_tap", .here = sources.lfm2, .pre_m5 = "" }, false);
}

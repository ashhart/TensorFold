//! In-place row penalties retain every untouched logit and round one fp32 subtraction to bf16.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const lower = @import("tensorfold").lanes.penalty_rows.lower;
const vocab = tf.flashnext_replay.VOCAB;
const max_rows = tf.flashnext_replay.MAXR;

const Buffers = struct {
    base: rig_.Rig,
    logits: rig_.Ref = undefined,
    bytes: usize = 0,
    pub fn reset(b: *Buffers) void {
        b.base.reset();
    }
    pub fn outputBytes(b: *Buffers) ![]const u8 {
        return b.base.bytes(b.logits, b.bytes);
    }
    pub fn inputBytes(b: *Buffers) ![]const u8 {
        return b.base.inputBytes();
    }
    pub fn intact(b: *Buffers) !bool {
        return b.base.intact();
    }
    pub fn overlap(_: *Buffers) bool {
        return true;
    }
};

fn depth(_: c.Shape) usize {
    return 0;
}
fn sub(row: usize, token: usize) f32 {
    return @as(f32, @floatFromInt(1 + (row + token) % 13)) * 0.0713 + 0.00037;
}
fn selected(k: usize, t: usize) bool {
    return t < @min(k - 1, 256) or t == k - 1;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: Buffers,
    control: bool,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    words: std.ArrayList(u16) = .empty,
    rows: rig_.Ref = undefined,
    tokens: rig_.Ref = undefined,
    amounts: rig_.Ref = undefined,
    entries: usize = 0,
    nonfinite: bool = false,
    msg: [180]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        const source = if (control)
            try std.mem.replaceOwned(u8, a, tf.metal_penalties.source, "ulong(row[i]) * c.x", "ulong(row[i]) * (c.x - 1)")
        else
            tf.metal_penalties.source;
        defer if (control) a.free(source);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .a = a, .rig = .{ .base = .{ .a = a, .device = device, .queue = queue } }, .control = control, .pipe = try mtl.Pipeline.init(device, lib, "fz_penalize_rows", false) };
    }

    pub fn deinit(p: *Probe) void {
        p.pipe.deinit();
        p.rig.base.deinit();
        p.words.deinit(p.a);
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .control = if (p.control) .shapes else null, .vt = &vt, .spec = .{
            .name = if (p.control) "control: flashnext penalties wrong row stride" else "flashnext penalties fz_penalize_rows",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .in_place = true,
            .nan = .{ .stated = "each logit's class is unchanged by its finite penalty; finite words equal core lower" },
            .tiles = .{ .{ .size = 2 }, .{}, .{ .size = 256, .min = 4 } },
            .served = &.{ .{ .rows = 1, .n = 1, .k = vocab }, .{ .rows = 2, .n = 1, .k = vocab }, .{ .rows = max_rows, .n = 1, .k = vocab } },
            .sums = .{ .why = "no sum: one fp32 subtraction, rounded to bf16 once" },
        } };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > max_rows or shape.n != 1 or shape.k < 4) return false;
        p.rig.reset();
        p.shape = shape;
        p.nonfinite = fill == .nonfinite;
        try p.words.resize(p.a, shape.rows * shape.k);
        for (p.words.items, 0..) |*w, i| w.* = c.toBf16(@as(f32, @floatFromInt((i +% seed) % 67)) / 16 - 2);
        if (p.nonfinite) for (0..shape.rows) |r| {
            p.words.items[r * shape.k] = 0x7fc1;
            p.words.items[r * shape.k + 1] = 0x7f80;
            p.words.items[(r + 1) * shape.k - 1] = 0xff80;
        };
        p.entries = shape.rows * @min(shape.k, 257);
        const rs = try p.a.alloc(u32, p.entries);
        defer p.a.free(rs);
        const ts = try p.a.alloc(u32, p.entries);
        defer p.a.free(ts);
        const amounts = try p.a.alloc(f32, p.entries);
        defer p.a.free(amounts);
        var at: usize = 0;
        for (0..shape.rows) |r| for (0..@min(shape.k, 257)) |j| {
            const t = if (j == @min(shape.k, 257) - 1) shape.k - 1 else j;
            rs[at] = @intCast(r);
            ts[at] = @intCast(t);
            amounts[at] = sub(r, t);
            at += 1;
        };
        p.rows = try p.rig.base.input(std.mem.sliceAsBytes(rs));
        p.tokens = try p.rig.base.input(std.mem.sliceAsBytes(ts));
        p.amounts = try p.rig.base.input(std.mem.sliceAsBytes(amounts));
        p.rig.bytes = p.words.items.len * 2;
        p.rig.logits = try p.rig.base.scratch(p.rig.bytes);
        return true;
    }

    pub fn run(p: *Probe) !void {
        @memcpy(p.rig.base.bytes(p.rig.logits, p.rig.bytes), std.mem.sliceAsBytes(p.words.items));
        const pass = p.rig.base.begin();
        pass.e.setPipeline(p.pipe);
        for ([_]rig_.Ref{ p.rig.logits, p.rows, p.tokens, p.amounts }, 0..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
        const sizes = [2]u32{ @intCast(p.shape.k), @intCast(p.entries) };
        pass.e.setBytes(std.mem.asBytes(&sizes), 4);
        pass.e.dispatchThreads(mtl.Size.of(p.entries, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.base.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        return p.rig.bytes;
    }
    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        const bytes = try p.rig.outputBytes();
        for (p.words.items, 0..) |w, i| {
            const row = i / p.shape.k;
            const t = i % p.shape.k;
            const want = if (selected(p.shape.k, t)) lower(w, sub(row, t)) else w;
            const got = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
            const same_class = c.classOf(c.fromBf16(got)) == c.classOf(c.fromBf16(want));
            const nan = std.math.isNan(c.fromBf16(want));
            if (!same_class or (gate != .nonfinite and !nan and got != want))
                return try std.fmt.bufPrint(&p.msg, "row {d} token {d}: 0x{x} expected 0x{x}", .{ row, t, got, want });
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [2]?*Probe = .{ null, null },
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (0..2) |i| {
            const name = if (i == 1) "control: flashnext penalties wrong row stride" else "flashnext penalties fz_penalize_rows";
            if (only) |part| if (std.mem.indexOf(u8, name, part) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, i == 1);
            s.probes[i] = p;
            try s.kernels.append(s.a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes) |p| if (p) |x| {
            x.deinit();
            s.a.destroy(x);
        };
    }
};

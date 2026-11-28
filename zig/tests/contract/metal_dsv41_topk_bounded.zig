//! DeepSeek-V4.1's indexer top-512 of long rows by chunk bounds on Metal: tf_topk512's bytes at every served shape.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const topk = @import("metal_dsv41_topk512.zig");

const c = contract.spec;
const g = contract.gates;
const tb = tf.dsv41_metal.topk_bounded;
const base = "dsv41 topk bounded ";
const rule = "tf_topk512's: NaN ranks as -inf, -0 as +0, ties go to the lower index; min(512, visible) indices, then -1";

fn noSums(_: c.Shape) usize {
    return 0;
}

// The first shape is the small gates' (8 rows, every bounded row kind); widths straddle tb.long and reach 2^20.
const served = blk: {
    const rows = [_]usize{ 1, 4, 16 };
    const widths = [_]usize{ 512, tb.long, tb.long + 1, tb.long + tb.chunk + 1, 32768, 131072 };
    var out: [rows.len * widths.len + 3]c.Shape = undefined;
    out[0] = .{ .rows = 8, .n = 512, .k = 32768 };
    var n: usize = 1;
    for (rows) |r| for (widths) |w| {
        out[n] = .{ .rows = r, .n = 512, .k = w };
        n += 1;
    };
    out[n] = .{ .rows = 1, .n = 512, .k = 1048576 };
    out[n + 1] = .{ .rows = 4, .n = 512, .k = 1048576 };
    break :blk out[0 .. n + 2].*;
};

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .replaces = "dsv41 topk512 tf_topk512",
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = tb.long, .step = tb.chunk, .min = tb.chunk } },
        .served = &served,
    };
}

fn ref(comptime T: type, r: rig_.Ref) T {
    return .{ .buf = r.buf, .off = r.off };
}

const vt = g.hooks(Bounded);
const plain_vt = blk: {
    var v = g.hooks(Bounded);
    v.replaced = null; // a control is proven at its own gate by the stable rank, not against the baseline
    break :blk v;
};

pub const Bounded = struct {
    /// the inputs, the CPU stable rank and tf_topk512 (no control) for the baseline
    t: topk.TopK,
    rig: *rig_.Rig = undefined,
    pipes: tb.Pipelines,
    ctl: u8,
    bounds: rig_.Ref = undefined,
    base: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8) !Bounded {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.topk512_bounded });
        defer a.free(text);
        var t = try topk.TopK.init(a, device, queue, 0);
        errdefer t.deinit();
        return .{ .t = t, .pipes = try tb.Pipelines.loadSource(device, text), .ctl = ctl };
    }

    pub fn deinit(p: *Bounded) void {
        p.pipes.deinit();
        p.t.deinit();
    }

    pub fn kernel(p: *Bounded, name: []const u8) g.Kernel {
        p.rig = &p.t.rig;
        var s = spec(name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null };
        s.replaces = null;
        return .{ .spec = s, .ctx = p, .vt = &plain_vt, .control = if (p.ctl == 1) .nonfinite else .shapes };
    }

    /// TopK's rows, with every fifth row from row 2 masked as a candidate source masks: 7 blocks of 8 in 8 at -inf.
    pub fn prepare(p: *Bounded, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try p.t.prepare(shape, fill, seed)) return false;
        const k = shape.k;
        const live = std.mem.bytesAsSlice(f32, p.rig.bytes(p.t.s, shape.rows * k * 4));
        var r: usize = 2;
        while (r < shape.rows) : (r += 5) {
            const vis: usize = @intCast(p.t.vis[r]);
            for (0..vis) |i| if ((i / 8) % 8 != 0) {
                p.t.scores[r * k + i] = -std.math.inf(f32);
                live[r * k + i] = -std.math.inf(f32);
            };
        }
        p.bounds = try p.rig.scratch(shape.rows * tb.chunksOf(@intCast(k)) * 4);
        p.base = try p.rig.scratch(p.t.count());
        return true;
    }

    pub fn run(p: *Bounded) !void {
        @memset(p.rig.bytes(p.t.out, p.t.count()), 0x55);
        const pass = p.rig.begin();
        tb.encode(&p.pipes, pass.e, ref(tb.Ref, p.t.s), ref(tb.Ref, p.t.v), ref(tb.Ref, p.t.out), ref(tb.Ref, p.bounds), @intCast(p.t.shape.rows), @intCast(p.t.shape.k));
        try p.rig.finish(pass);
    }

    /// tf_topk512's bytes on the same inputs.
    pub fn replaced(p: *Bounded) ![]const u8 {
        const n = p.t.count();
        @memset(p.rig.bytes(p.base, n), 0x55);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.t.pipe);
        e.setValue([1]u32{@intCast(p.t.shape.k)}, 0);
        for ([_]rig_.Ref{ p.t.s, p.t.v, p.base }, 1..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.dispatchGroups(mtl.Size.of(p.t.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base, n);
    }

    pub fn count(p: *Bounded) usize {
        return p.t.count();
    }

    pub fn reference(p: *Bounded, want: []f64, mag: []f64) !void {
        return p.t.reference(want, mag);
    }

    pub fn check(p: *Bounded, at: g.Gate) !?[]const u8 {
        return p.t.check(at);
    }
};

/// Every registered variant: the kernel, then its planted controls.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Bounded) = .empty,

    const list = [_]struct { ctl: u8, name: []const u8 }{
        .{ .ctl = 0, .name = base ++ "tf_topk512_bounds then tf_topk512_bounded" },
        .{ .ctl = 1, .name = base ++ "control: NaN ranked above +inf" },
        .{ .ctl = 5, .name = base ++ "control: a chunk's last key left out of its bound" },
        .{ .ctl = 6, .name = base ++ "control: the keys equal to the 512th bound left out" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Bounded);
            errdefer a.destroy(p);
            p.* = try Bounded.init(a, device, queue, entry.ctl);
            errdefer p.deinit();
            try contract.register(a, &s.probes, &kernels, p, p.kernel(entry.name));
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};

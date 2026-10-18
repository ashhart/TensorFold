//! Drama 3's RMS norm rows on Metal (d3_rmsnorm, plain gamma) under the contract, and its sources for gate 5.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// A drama3 source with d3_common.h inlined, by the file its gate 5 label names ("drama3 <file> <function>").
pub fn source(a: std.mem.Allocator, label: []const u8) ![]u8 {
    const d = sources.drama3;
    const files = .{ .{ "dense", d.dense }, .{ "gdn", d.gdn }, .{ "attn", d.attn }, .{ "moe", d.moe }, .{ "sample", d.sample }, .{ "codec", d.codec } };
    var it = std.mem.tokenizeScalar(u8, label, ' ');
    _ = it.next();
    const file = it.next() orelse return error.UnknownLaunch;
    inline for (files) |f| if (std.mem.eql(u8, file, f[0])) return std.mem.replaceOwned(u8, a, f[1], "#include \"d3_common.h\"", d.common);
    return error.UnknownLaunch;
}

/// ceil(k / 256) squares a thread, a simdgroup's tree, then the 8 simdgroups' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec() c.Spec {
    return .{ .name = "metal drama3 d3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } }, .sums = .{ .bf16 = 0, .why = why } };
}

const Args = extern struct { xs: u32, ys: u32, D: u32, zc: u32, eps: f32 };

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    r: ?rows.Rows = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Norm {
        const src = try source(a, "drama3 dense d3_rmsnorm");
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "d3_rmsnorm") };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const r = try rows.Rows.init(self.a, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(spec().out));
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k: u32 = @intCast(r.shape.k);
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        e.setBuffer(self.x.buf, self.x.off, 0);
        e.setBuffer(self.w.buf, self.w.off, 1);
        e.setBuffer(self.out.buf, self.out.off, 2);
        e.setValue(Args{ .xs = k, .ys = k, .D = k, .zc = 0, .eps = r.eps }, 3);
        e.dispatchGroups(mtl.Size.of(r.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try self.rig.finish(p);
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};

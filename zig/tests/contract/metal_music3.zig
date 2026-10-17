//! MiniMax Music 3's RMS norm rows on Metal (m3_rmsnorm, plain gamma) under the contract, and its sources for gate 5.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// A music3 source, includes inlined, by its gate 5 label ("music3 <file> <function>"); `simd`: the pre-M5 text.
pub fn source(a: std.mem.Allocator, label: []const u8, simd: bool) ![]u8 {
    const m = sources.music3;
    const files = .{ .{ "ar", m.ar }, .{ "attn", m.attn }, .{ "sample", m.sample }, .{ "gemm", m.gemm }, .{ "dit", m.dit }, .{ "dit_attn", m.dit_attn }, .{ "dav", m.dav } };
    var it = std.mem.tokenizeScalar(u8, label, ' ');
    _ = it.next();
    const file = it.next() orelse return error.UnknownLaunch;
    inline for (files) |f| if (std.mem.eql(u8, file, f[0])) {
        const t1 = try std.mem.replaceOwned(u8, a, f[1], "#include \"m3_common.h\"", m.common);
        defer a.free(t1);
        const t2 = try std.mem.replaceOwned(u8, a, t1, "#include \"../prefill/gemm_nax.metal\"", m.gemm_nax);
        defer a.free(t2);
        const t3 = try std.mem.replaceOwned(u8, a, t2, "#include \"../nax.h\"", sources.nax);
        defer a.free(t3);
        return std.mem.concat(a, u8, &.{ if (simd) "#define TF_SIMD_FRAGS 1\n" else "", t3 });
    };
    return error.UnknownLaunch;
}

/// ceil(k / 256) squares a thread, a simdgroup's tree, then the 8 simdgroups' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}

const why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec() c.Spec {
    return .{ .name = "metal music3 m3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 2, .n = 4096, .k = 4096 }, .{ .rows = 4, .n = 4096, .k = 4096 }, .{ .rows = 600, .n = 4096, .k = 4096 } }, .sums = .{ .bf16 = 0, .why = why } };
}

const Args = extern struct { xs: u32, ys: u32, dim: u32, eps: f32 };

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    r: ?rows.Rows = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Norm {
        const src = try source(a, "music3 ar m3_rmsnorm", false);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "m3_rmsnorm") };
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
        e.setValue(Args{ .xs = k, .ys = k, .dim = k, .eps = r.eps }, 3);
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

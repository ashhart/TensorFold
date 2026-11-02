//! The served attention product in both arithmetic modes, with the real cut path as a full-float control.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = contract.attention_pv;
const c = contract.spec;
const g = contract.gates;

pub const function = "tf_attention_pv_contract";
const wrapper =
    \\[[max_total_threads_per_threadgroup(32)]]
    \\kernel void tf_attention_pv_contract(const device float* P [[buffer(0)]], const device bfloat* V [[buffer(1)]],
    \\    device float* O [[buffer(2)]], constant int* dims [[buffer(3)]],
    \\    uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    \\  using namespace tfp;
    \\  const short2 home = frag_home(lane);
    \\  const int row = int(tg.y) * 16, col = int(tg.x) * 32;
    \\  frag<float> lo = frag<float>(0), hi = frag<float>(0);
    \\  for (int k = 0; k < dims[2]; k += 16) {
    \\    frag<float> p; frag<bfloat> v0, v1;
    \\    frag_get_in(p, P, dims[2], row, k, home, dims[0], dims[2]);
    \\    frag_get_in(v0, V, dims[1], k, col, home, dims[2], dims[1]);
    \\    frag_get_in(v1, V, dims[1], k, col + 16, home, dims[2], dims[1]);
    \\    attention_pv(lo, hi, p, v0, v1, lane);
    \\  }
    \\  frag_put_in(lo, O, dims[1], row, col, home, dims[0], dims[1]);
    \\  frag_put_in(hi, O, dims[1], row, col + 16, home, dims[0], dims[1]);
    \\}
;

pub fn text(a: std.mem.Allocator, device: mtl.Device, full: bool) ![]u8 {
    const attention = for (sources.prefill) |f| {
        if (std.mem.eql(u8, f.name, "attention_nax")) break f.text;
    } else return error.AttentionSource;
    const joined = try std.mem.concat(a, u8, &.{ attention, "\n", wrapper });
    defer a.free(joined);
    return tf.frags.sourceMode(device, a, joined, full);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mode: ref.Mode,
    pipe: mtl.Pipeline,
    fixture: ?ref.Fixture = null,
    p: rig_.Ref = undefined,
    v: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mode: ref.Mode) !Probe {
        if (!device.tensorUnits()) return error.TensorUnitsRequired;
        const source = try text(a, device, mode == .fp32);
        defer a.free(source);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mode = mode, .pipe = try rig_.pipeline(device, source, function) };
    }
    pub fn deinit(p: *Probe) void {
        if (p.fixture) |f| f.deinit();
        p.rig.deinit();
        p.pipe.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = ref.spec(p.mode), .ctx = p, .vt = &vt, .control = if (p.mode == .cut_control) .shapes else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (p.fixture) |f| f.deinit();
        p.fixture = null;
        p.rig.reset();
        const f = try ref.Fixture.init(p.a, s);
        p.fixture = f;
        try f.fill(fill, seed);
        p.p = try p.rig.input(std.mem.sliceAsBytes(f.p));
        p.v = try p.rig.input(std.mem.sliceAsBytes(f.v));
        p.out = try p.rig.output(s.rows * s.n * 4);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const s = p.fixture.?.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.p.buf, p.p.off, 0);
        e.setBuffer(p.v.buf, p.v.off, 1);
        e.setBuffer(p.out.buf, p.out.off, 2);
        e.setValue([3]i32{ @intCast(s.rows), @intCast(s.n), @intCast(s.k) }, 3);
        e.dispatchGroups(mtl.Size.of((s.n + 31) / 32, (s.rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        const s = p.fixture.?.shape;
        return s.rows * s.n;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.fixture.?.reference(p.mode, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.fixture.?.simulate(p.mode, acc, order, out);
    }
};

pub const Suite = struct {
    probes: [3]?Probe = @splat(null),
    pub fn deinit(s: *Suite) void {
        for (&s.probes) |*p| if (p.*) |*probe| probe.deinit();
    }
    pub fn collect(s: *Suite, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(ref.Mode), 0..) |mode, i| {
            if (only) |needle| if (std.mem.indexOf(u8, ref.name(mode), needle) == null) continue;
            if (!device.tensorUnits()) {
                std.debug.print("UNSUPPORTED {s}: this precision mode is specific to M5; older-chip arithmetic unchanged\n", .{ref.name(mode)});
                continue;
            }
            s.probes[i] = try Probe.init(a, device, queue, mode);
            try kernels.append(a, s.probes[i].?.kernel());
        }
    }
};

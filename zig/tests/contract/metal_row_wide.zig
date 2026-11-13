//! Wide row projections against one-row bytes, independent FP64 dots and a dispatched row-coupling control.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const ct = @import("contract");
const affine = @import("metal_affine.zig");
const rig = @import("rig.zig");
const row = tf.row_projection;
const c = ct.spec;
const g = ct.gates;
const Style = enum { plain, relu2, strided };
const Proof = enum { bytes, fp64, control };

fn shapes() [34]c.Shape {
    var out: [34]c.Shape = undefined;
    for (out[0..32], 1..) |*s, n| s.* = .{ .rows = n, .n = 16, .k = 576 };
    out[32] = .{ .rows = 8, .n = 2688, .k = 4096 };
    out[33] = .{ .rows = 16, .n = 4608, .k = 2688 };
    return out;
}

fn symbol(a: std.mem.Allocator, bits: u8, sum: row.Sum, style: Style) ![]u8 {
    return std.fmt.allocPrint(a, "tf_row_projection_wide{s}_q{d}_{t}", .{ switch (style) {
        .plain => "",
        .relu2 => "_relu2",
        .strided => "_strided",
    }, bits, sum });
}

const Probe = struct {
    inner: affine.Proj,
    pipes: *const row.Pipelines,
    pipe: mtl.Pipeline,
    sum: row.Sum,
    style: Style,
    proof: Proof,
    previous: rig.Ref = undefined,
    served: [34]c.Shape = shapes(),

    const vt = blk: {
        var v = g.hooks(affine.Proj);
        v.prepare = prepare;
        v.run = run;
        v.replaced = replaced;
        v.reference = reference;
        v.simulate = simulate;
        v.check = nonfinite;
        break :blk v;
    };
    const oracle_vt = blk: {
        var v = vt;
        v.check = null;
        break :blk v;
    };
    fn of(ctx: *anyopaque) *Probe {
        const inner: *affine.Proj = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }
    fn kernel(p: *Probe) g.Kernel {
        var s = affine.spec(p.inner.v, p.inner.name);
        s.served = &p.served;
        s.tiles[0].size = 8;
        s.replaces = if (p.proof == .fp64) null else "the narrow projection applied to each row alone";
        if (p.proof != .fp64) s.nan = .{ .stated = "the same nonfinite output bytes as each row alone" };
        return .{ .spec = s, .ctx = &p.inner, .vt = if (p.proof == .fp64) &oracle_vt else &vt, .control = if (p.proof == .control) .shapes else null };
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = of(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        if (fill == .stress and p.style == .relu2) {
            const f = p.inner.f.?;
            for (f.x) |*x| x.* ^= 0x8000;
            @memcpy(p.inner.rig.bytes(p.inner.x, f.x.len * 2), std.mem.sliceAsBytes(f.x));
        }
        p.previous = try p.inner.rig.scratch(p.inner.f.?.outputs() * 2);
        return true;
    }
    fn weights(p: *Probe) row.Weights {
        const b = &p.inner;
        const s = b.f.?.shape;
        return .{ .w = b.w.buf, .w_off = b.w.off, .scales = b.s.buf, .s_off = b.s.off, .biases = b.b.buf, .b_off = b.b.off, .n = s.n, .k = s.k, .group = b.v.group, .bits = b.v.bits, .sum = p.sum };
    }
    fn encode(p: *Probe, call: row.Call, target: rig.Ref, first: usize) !void {
        const b = &p.inner;
        const pass = b.rig.begin();
        pass.e.setPipeline(call.pipeline);
        pass.e.setBuffer(b.x.buf, b.x.off + first * b.f.?.shape.k * 2, 0);
        for ([_]rig.Ref{ b.w, b.s, b.b }, 1..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setValue(call.dims, 4);
        pass.e.setBuffer(target.buf, target.off + first * b.f.?.shape.n * 2, 5);
        pass.e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
        try b.rig.finish(pass);
    }
    fn run(ctx: *anyopaque) !void {
        const p = of(ctx);
        const w = p.weights();
        const n = p.inner.f.?.shape.rows;
        var call = if (p.style == .strided) try row.callStrided(p.pipes, w, n, w.n) else try row.call(p.pipes, w, n, p.style == .relu2);
        call.pipeline = p.pipe;
        try p.encode(call, p.inner.out, 0);
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = of(ctx);
        const call = try row.call(p.pipes, p.weights(), 1, p.style == .relu2);
        for (0..p.inner.f.?.shape.rows) |r| try p.encode(call, p.previous, r);
        return p.inner.rig.bytes(p.previous, p.inner.f.?.outputs() * 2);
    }
    fn nonfinite(ctx: *anyopaque, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const p = of(ctx);
        const want = try replaced(ctx);
        const got = p.inner.rig.bytes(p.inner.out, want.len);
        return if (std.mem.eql(u8, want, got)) null else "wide nonfinite bytes differ from the one-row path";
    }
    fn rounded(x: f64) f64 {
        return c.fromBf16(c.toBf16(@floatCast(x)));
    }
    fn epilogue(p: *Probe, x: f64) f64 {
        if (p.style != .relu2) return x;
        const h = @max(rounded(x), 0);
        return rounded(h * h);
    }
    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) !void {
        const p = of(ctx);
        const f = p.inner.f.?;
        const kg = f.shape.k / f.group;
        for (want, mag, 0..) |*out, *magnitude, index| {
            const r = index / f.shape.n;
            const col = index % f.shape.n;
            var total: f64 = 0;
            var terms: f64 = 0;
            var k: usize = 0;
            while (k < f.shape.k) : (k += 16) {
                var dot: f64 = 0;
                var xs: f64 = 0;
                for (0..16) |i| dot += c.fromBf16(f.x[r * f.shape.k + k + i]) * @as(f64, @floatFromInt(f.codes[col * f.shape.k + k + i]));
                for (0..4) |part| {
                    const x = f.x[r * f.shape.k + k + 4 * part ..][0..4];
                    var sum: f64 = @as(f64, c.fromBf16(x[0])) + c.fromBf16(x[1]);
                    if (p.sum == .bf16) sum = rounded(sum);
                    sum += c.fromBf16(x[2]);
                    if (p.sum == .bf16) sum = rounded(sum);
                    sum += c.fromBf16(x[3]);
                    xs += if (p.sum == .bf16) rounded(sum) else sum;
                }
                const scale = c.fromBf16(f.scales[col * kg + k / f.group]);
                const bias = c.fromBf16(f.biases[col * kg + k / f.group]);
                total += scale * dot + bias * xs;
                for (0..16) |i| terms += @abs(scale * c.fromBf16(f.x[r * f.shape.k + k + i]) * @as(f64, @floatFromInt(f.codes[col * f.shape.k + k + i])));
                terms += @abs(bias * xs);
            }
            out.* = p.epilogue(total);
            magnitude.* = terms;
        }
    }
    fn simulate(ctx: *anyopaque, acc: c.Accum, order: g.Order, out: []f64) !void {
        const p = of(ctx);
        try p.inner.simulate(acc, order, out);
        for (out) |*v| v.* = p.epilogue(v.*);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    pipes: ?row.Pipelines = null,
    control: ?mtl.Library = null,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (row.bit_widths) |bits| for ([_]row.Sum{ .f32, .bf16 }) |sum| for (std.enums.values(Style)) |style| {
            for (std.enums.values(Proof)) |proof| {
                if (proof == .fp64 and (sum != .f32 or style != .plain)) continue;
                const fun = try symbol(s.a, bits, sum, style);
                defer s.a.free(fun);
                const name = try std.fmt.allocPrint(s.a, "{s}row projection wide q{d} {t} {t} {s}", .{ if (proof == .control) "control: metal " else "metal ", bits, sum, proof, fun });
                defer s.a.free(name);
                if (only) |q| if (std.mem.indexOf(u8, name, q) == null) continue;
                if (s.pipes == null) s.pipes = try row.Pipelines.load(s.device);
                const i = std.mem.indexOfScalar(u8, &row.bit_widths, bits).? * 2 + @backingInt(sum);
                var pipeline = switch (style) {
                    .plain => s.pipes.?.wide_plain[i],
                    .relu2 => s.pipes.?.wide_relu2[i],
                    .strided => s.pipes.?.wide_strided[i],
                };
                if (proof == .control) {
                    if (s.control == null) {
                        const text = try std.mem.concat(s.a, u8, &.{ "#define TF_RP_BAD_ROWS 1\n", sources.core_row_projection });
                        defer s.a.free(text);
                        s.control = try mtl.Library.fromSource(s.device, text, mtl.CompileOptions.mlx());
                    }
                    pipeline = try mtl.Pipeline.init(s.device, s.control.?, fun, false);
                }
                const p = blk: {
                    errdefer if (proof == .control) pipeline.deinit();
                    const probe = try s.a.create(Probe);
                    errdefer s.a.destroy(probe);
                    const owned_name = try s.a.dupe(u8, name);
                    errdefer s.a.free(owned_name);
                    probe.* = .{ .inner = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .v = .{ .kind = .row, .bits = bits, .group = 64 }, .name = owned_name }, .pipes = &s.pipes.?, .pipe = pipeline, .sum = sum, .style = style, .proof = proof };
                    try s.probes.append(s.a, probe);
                    break :blk probe;
                };
                try s.kernels.append(s.a, p.kernel());
            }
        };
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            if (p.proof == .control) p.pipe.deinit();
            p.inner.deinit();
            s.a.free(p.inner.name);
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        if (s.control) |lib| lib.deinit();
        if (s.pipes) |*pipes| pipes.deinit();
    }
};

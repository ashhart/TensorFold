const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const name = "metal sam3 compensated normalization";
const control_name = "control: sam3 compensated normalization half variance";
const planes_name = "metal sam3 compensated normalization planes";
const wide_name = "metal sam3 compensated normalization wide";
const pair_name = "metal sam3 compensated normalization pair";

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const source = "#define TF_UNROLL _Pragma(\"clang loop unroll(full)\")\n" ++ sources.sam3_dense_small ++ sources.sam3_planes ++ sources.sam3_norm;
    const needle = "float2 variance = sam_norm_sum(squares);";
    if (std.mem.count(u8, source, needle) != 1) return error.ControlTextNotFoundOnce;
    return if (control) std.mem.replaceOwned(u8, a, source, needle, needle ++ " variance = float2(float(half(variance.x)), 0);") else a.dupe(u8, source);
}

fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    // Planes/pair output: clear, restore and copy pipelines read the buffer back as FP32.
    planes: ?[3]mtl.Pipeline = null,
    is_pair: bool = false,
    // Wide output: clear pipeline resets the 1-word domain flag before norm.
    wide_clear: ?mtl.Pipeline = null,
    rows: ?contract.rows.Rows = null,
    inputs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    emitted: rig_.Ref = undefined,

    fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .sums else null, .spec = .{
            .name = if (self.control) control_name else if (self.is_pair) pair_name else if (self.planes != null) planes_name else if (self.wide_clear != null) wide_name else name,
            .accum = .f32,
            .out = .f32,
            .depth = depth,
            .ops = 32,
            .tiles = .{ .{ .size = 4 }, .{}, .{ .size = 32 } },
            .served = &.{
                .{ .rows = 5184, .n = 1024, .k = 1024 },
                .{ .rows = 8, .n = 256, .k = 256 },
                .{ .rows = 3, .n = 31, .k = 31 },
                .{ .rows = 4, .n = 32, .k = 32 },
                .{ .rows = 5, .n = 33, .k = 33 },
                .{ .rows = 5, .n = 1023, .k = 1023 },
                .{ .rows = 5, .n = 1025, .k = 1025 },
            },
        } };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != shape.k or shape.k == 0 or shape.k > 4096) return false;
        if (self.rows) |r| r.deinit();
        self.rows = null;
        self.rig.reset();
        const rows = try contract.rows.Rows.init(self.a, .layer_norm, shape);
        self.rows = rows;
        rows.fill(fill, seed);
        for ([_][]const u16{ rows.x, rows.w, rows.b }, &self.inputs) |values, *input| {
            const floats = try self.a.alloc(f32, values.len);
            defer self.a.free(floats);
            for (values, floats) |value, *target| target.* = c.fromBf16(value);
            input.* = try self.rig.input(std.mem.sliceAsBytes(floats));
        }
        self.out = try self.rig.output(rows.outputs() * 4);
        if (self.is_pair) self.emitted = try self.rig.scratch(sources.sam3_dense_pair.pairCount(rows.outputs()) * 2) else if (self.planes != null) self.emitted = try self.rig.scratch(sources.sam3_dense_planes.planesCount(rows.outputs()) * 2);
        if (self.wide_clear != null) self.emitted = try self.rig.scratch(4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const rows = self.rows.?;
        const n: u32 = @intCast(rows.outputs());
        const target = if (self.planes != null) self.emitted else self.out;
        const pass = self.rig.begin();
        if (self.planes) |pipes| {
            pass.e.setPipeline(pipes[0]);
            pass.e.setBuffer(target.buf, target.off, 0);
            pass.e.setValue(n, 1);
            pass.e.dispatchThreads(mtl.Size.of(8, 1, 1), mtl.Size.of(8, 1, 1));
            pass.e.barrier();
        } else if (self.wide_clear) |clear| {
            pass.e.setPipeline(clear);
            pass.e.setBuffer(self.emitted.buf, self.emitted.off, 0);
            pass.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
            pass.e.barrier();
        }
        pass.e.setPipeline(self.pipe);
        for (self.inputs, 0..) |input, i| pass.e.setBuffer(input.buf, input.off, i);
        pass.e.setBuffer(target.buf, target.off, 3);
        if (self.wide_clear != null) {
            pass.e.setBuffer(self.emitted.buf, self.emitted.off, 4);
            pass.e.setValue([2]u32{ @intCast(rows.shape.rows), @intCast(rows.shape.k) }, 5);
            pass.e.setValue(rows.eps, 6);
            pass.e.setValue(@as(u32, 0), 7);
        } else {
            pass.e.setValue([2]u32{ @intCast(rows.shape.rows), @intCast(rows.shape.k) }, 4);
            pass.e.setValue(rows.eps, 5);
            if (self.planes != null) pass.e.setValue(@as(u32, 0), 6);
        }
        pass.e.dispatchGroups(mtl.Size.of((rows.shape.rows + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
        if (self.planes) |pipes| {
            pass.e.barrier();
            pass.e.setPipeline(pipes[1]);
            pass.e.setBuffer(target.buf, target.off, 0);
            if (self.is_pair) pass.e.setValue(n, 1) else pass.e.setValue([3]u32{ n, 0, 1 }, 1);
            pass.e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(128, 1, 1));
            pass.e.barrier();
            pass.e.setPipeline(pipes[2]);
            const restore_off = if (self.is_pair) sources.sam3_dense_pair.restoreOffset(n) else sources.sam3_dense_planes.restoreOffset(n);
            pass.e.setBuffer(target.buf, target.off + restore_off, 0);
            pass.e.setBuffer(self.out.buf, self.out.off, 1);
            pass.e.setValue(n, 2);
            pass.e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(128, 1, 1));
        }
        try self.rig.finish(pass);
    }

    pub fn count(self: *Probe) usize {
        return self.rows.?.outputs();
    }

    pub fn reference(self: *Probe, want: []f64, magnitude: []f64) !void {
        try self.rows.?.reference(want, magnitude);
    }

    pub fn simulate(self: *Probe, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        try self.rows.?.simulate(accumulator, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [5]?Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&self.probes, 0..) |*entry, i| {
            const label = if (i == 1) control_name else if (i == 2) planes_name else if (i == 3) wide_name else if (i == 4) pair_name else name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            if ((i == 2 or i == 4) and !device.tensorUnits()) continue;
            const source = try text(self.a, i == 1);
            defer self.a.free(source);
            const fn_name = if (i == 2) "sam_norm_compensated_planes" else if (i == 3) "sam_norm_compensated_wide" else if (i == 4) "sam_norm_compensated_pair" else "sam_norm_compensated";
            entry.* = .{ .a = self.a, .rig = .{ .a = self.a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, fn_name), .control = i == 1, .is_pair = i == 4 };
            if (i == 2 or i == 4) {
                const full = try sources.tensor_library.source(self.a, sources.sam3_source);
                defer self.a.free(full);
                const clear_name = if (i == 4) "sam_norm_pair_clear" else "sam_norm_planes_clear";
                const restore_name = if (i == 4) "sam_dense_pair_restore" else "sam_dense_planes_restore";
                entry.*.?.planes = .{ try rig_.pipeline(device, source, clear_name), try rig_.pipeline(device, full, restore_name), try rig_.pipeline(device, full, "sam3_copy") };
            } else if (i == 3) {
                entry.*.?.wide_clear = try rig_.pipeline(device, source, "sam_norm_wide_clear");
            }
            if (entry.*) |*probe| try kernels.append(self.a, probe.kernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            if (probe.rows) |rows| rows.deinit();
            probe.rig.deinit();
            probe.pipe.deinit();
            if (probe.planes) |pipes| for (pipes) |pipe| pipe.deinit();
            if (probe.wide_clear) |pipe| pipe.deinit();
        };
    }
};

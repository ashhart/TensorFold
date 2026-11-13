const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const name = "metal sam3 compensated normalization";
const control_name = "control: sam3 compensated normalization half variance";

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const source = "#define TF_UNROLL _Pragma(\"clang loop unroll(full)\")\n" ++ sources.sam3_dense_small ++ sources.sam3_norm;
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
    rows: ?contract.rows.Rows = null,
    inputs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .sums else null, .spec = .{
            .name = if (self.control) control_name else name,
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
        return true;
    }

    pub fn run(self: *Probe) !void {
        const rows = self.rows.?;
        const pass = self.rig.begin();
        pass.e.setPipeline(self.pipe);
        for (self.inputs, 0..) |input, i| pass.e.setBuffer(input.buf, input.off, i);
        pass.e.setBuffer(self.out.buf, self.out.off, 3);
        pass.e.setValue([2]u32{ @intCast(rows.shape.rows), @intCast(rows.shape.k) }, 4);
        pass.e.setValue(rows.eps, 5);
        pass.e.dispatchGroups(mtl.Size.of((rows.shape.rows + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
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
    probes: [2]?Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&self.probes, 0..) |*entry, i| {
            const label = if (i == 1) control_name else name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            const source = try text(self.a, i == 1);
            defer self.a.free(source);
            entry.* = .{ .a = self.a, .rig = .{ .a = self.a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "sam_norm_compensated"), .control = i == 1 };
            if (entry.*) |*probe| try kernels.append(self.a, probe.kernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            if (probe.rows) |rows| rows.deinit();
            probe.rig.deinit();
            probe.pipe.deinit();
        };
    }
};

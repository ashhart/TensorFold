const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const fixture = contract.sam3_mma;
const c = contract.spec;
const g = contract.gates;
const fma = @import("metal_sam3_fma.zig");

pub fn text(a: std.mem.Allocator, device: mtl.Device) ![]u8 {
    return tf.frags.source(device, a, sources.sam3_contract);
}

fn depth(_: c.Shape) usize {
    return 6 * fixture.k;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    ta: bool,
    tb: bool,
    control: bool,
    range: fixture.Range,
    panel: fixture.Panel = undefined,
    x: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .tiles = .{ .{ .size = 16 }, .{}, .{} },
                .served = &.{ .{ .rows = 16, .n = 32, .k = 16 }, .{ .rows = 7, .n = 32, .k = 16 }, .{ .rows = 1, .n = 32, .k = 16 } },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .shapes else null,
        };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n != 32 or shape.k != 16) return false;
        self.rig.reset();
        self.panel = .{ .count = shape.rows, .ta = self.ta, .tb = self.tb };
        self.panel.fill(fill, seed, self.range);
        self.x = try self.rig.input(std.mem.sliceAsBytes(&self.panel.x));
        self.y = try self.rig.input(std.mem.sliceAsBytes(&self.panel.y));
        self.out = try self.rig.output(16 * 32 * 4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        pass.e.setPipeline(self.pipe);
        pass.e.setBuffer(self.x.buf, self.x.off, 0);
        pass.e.setBuffer(self.y.buf, self.y.off, 1);
        pass.e.setBuffer(self.out.buf, self.out.off, 2);
        pass.e.dispatchThreads(mtl.Size.of(32, 1, 1), mtl.Size.of(32, 1, 1));
        try self.rig.finish(pass);
    }

    pub fn count(self: *Probe) usize {
        return self.panel.count * 32;
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        self.panel.reference(want, mag);
    }

    pub fn simulate(self: *Probe, accum: c.Accum, order: g.Order, out: []f64) !void {
        self.panel.simulate(accum, order, out);
    }

    fn deinit(self: *Probe) void {
        self.pipe.deinit();
        self.rig.deinit();
        self.a.free(self.name);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    scalar: [2]?*fma.Probe = @splat(null),

    pub fn collect(self: *Suite, only: ?[]const u8) !void {
        if (!self.device.tensorUnits()) return;
        for ([_]bool{ false, true }) |ta| for ([_]bool{ false, true }) |tb| for (0..8) |kind| {
            const layout = if (ta) (if (tb) "tt" else "tn") else (if (tb) "nt" else "nn");
            const control = kind == 1;
            const range: fixture.Range = switch (kind) {
                2 => .subnormal,
                3 => .extreme,
                4 => .half_left,
                5 => .half_right,
                6 => .half_both,
                7 => .mixed_tiny,
                else => .normal,
            };
            const suffix = switch (range) {
                .normal => if (control) " cut" else "",
                .subnormal => " subnormal",
                .extreme => " exponent range",
                .half_left => " exact half left",
                .half_right => " exact half right",
                .half_both => " exact half operands",
                .mixed_tiny => " mixed tiny operand",
            };
            const name = try std.fmt.allocPrint(self.a, "{s}sam3 fp32 mma {s}{s}", .{ if (control) "control: " else "metal ", layout, suffix });
            errdefer self.a.free(name);
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) {
                self.a.free(name);
                continue;
            };
            const source = try text(self.a, self.device);
            defer self.a.free(source);
            const symbol = try std.fmt.allocPrint(self.a, "tf_sam3_{s}_{s}", .{ if (control) "cut" else "mma", layout });
            defer self.a.free(symbol);
            const probe = try self.a.create(Probe);
            errdefer self.a.destroy(probe);
            try self.probes.ensureUnusedCapacity(self.a, 1);
            try self.kernels.ensureUnusedCapacity(self.a, 1);
            const pipe = try rig_.pipeline(self.device, source, symbol);
            probe.* = .{
                .a = self.a,
                .rig = .{ .a = self.a, .device = self.device, .queue = self.queue },
                .name = name,
                .ta = ta,
                .tb = tb,
                .control = control,
                .range = range,
                .pipe = pipe,
            };
            self.probes.appendAssumeCapacity(probe);
            self.kernels.appendAssumeCapacity(probe.kernel());
        };
        for (0..2) |i| {
            const name = if (i == 1) fma.control_name else fma.name;
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
            try self.kernels.ensureUnusedCapacity(self.a, 1);
            const source = try text(self.a, self.device);
            defer self.a.free(source);
            const probe = try self.a.create(fma.Probe);
            errdefer self.a.destroy(probe);
            probe.* = try fma.Probe.init(self.a, self.device, self.queue, source, i == 1);
            self.scalar[i] = probe;
            self.kernels.appendAssumeCapacity(probe.kernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (self.scalar) |maybe| if (maybe) |probe| {
            probe.deinit();
            self.a.destroy(probe);
        };
        for (self.probes.items) |probe| {
            probe.deinit();
            self.a.destroy(probe);
        }
        self.probes.deinit(self.a);
    }
};

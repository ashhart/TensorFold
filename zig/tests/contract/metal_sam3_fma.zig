const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const fixture = contract.sam3_fma;
const c = contract.spec;
const g = contract.gates;
pub const name = "metal sam3 fp32 mma IEEE fma";
pub const control_name = "control: sam3 fp32 mma IEEE fma flushing subnormals";

fn depth(_: c.Shape) usize {
    return 1;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    values: []fixture.Triple = &.{},
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    message: [160]u8 = undefined,
    control: bool,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, source: []const u8, control: bool) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, if (control) "tf_sam3_fma_flush" else "tf_sam3_fma"), .control = control };
    }

    pub fn deinit(self: *Probe) void {
        self.a.free(self.values);
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = if (self.control) control_name else name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .nan = .{ .stated = "IEEE binary32 fused multiply-add, nearest even, canonical positive quiet NaN" },
                .served = &.{ .{ .rows = 65536, .n = 1, .k = 2 }, .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 128, .n = 1, .k = 2 } },
                .tiles = .{ .{ .size = 32 }, .{}, .{} },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .shapes else null,
        };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 1 or shape.k != 2) return false;
        self.rig.reset();
        self.a.free(self.values);
        self.values = &.{};
        self.values = try self.a.alloc(fixture.Triple, shape.rows);
        fixture.fill(self.values, fill, seed);
        self.input = try self.rig.input(std.mem.sliceAsBytes(self.values));
        self.out = try self.rig.output(self.values.len * 4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        pass.e.setPipeline(self.pipe);
        pass.e.setBuffer(self.input.buf, self.input.off, 0);
        pass.e.setBuffer(self.out.buf, self.out.off, 1);
        pass.e.setValue(@as(u32, @intCast(self.values.len)), 2);
        pass.e.dispatchThreads(mtl.Size.of(self.values.len, 1, 1), mtl.Size.of(32, 1, 1));
        try self.rig.finish(pass);
    }

    pub fn count(self: *Probe) usize {
        return self.values.len;
    }

    pub fn check(self: *Probe, _: g.Gate) !?[]const u8 {
        const bytes = self.rig.bytes(self.out, self.values.len * 4);
        for (self.values, 0..) |value, i| {
            const got = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
            const want = fixture.expected(value);
            if (got != want) return try std.fmt.bufPrint(&self.message, "output {d}: got 0x{x}, CPU IEEE FMA gives 0x{x}", .{ i, got, want });
        }
        return null;
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        for (self.values, want, mag) |value, *w, *m| {
            w.* = @as(f32, @bitCast(fixture.expected(value)));
            m.* = @abs(@as(f64, value.a) * value.b) + @abs(@as(f64, value.c));
        }
    }

    pub fn simulate(self: *Probe, accum: c.Accum, order: g.Order, out: []f64) !void {
        for (self.values, out) |value, *result| {
            const terms = [_]f64{ @as(f64, value.a) * value.b, value.c };
            result.* = contract.rows.sumIn(accum, order, &terms);
        }
    }
};

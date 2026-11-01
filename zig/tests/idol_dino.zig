//! tf-idol-dino on Metal: idol_dino_run.zig's case driver for Idol's DINOv3 encoder on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const Metal = struct {
    pub const dn = tf.idol.dino;
    pub const st = tf.safetensors;
    pub const seconds = mtl.clock.seconds;

    pub fn stream(e: *dn.Encoder) ![]const u16 {
        return e.stream();
    }

    pub fn hidden(e: *dn.Encoder) ![]const u16 {
        return e.lastHidden();
    }

    pub const Device = struct {
        pool: mtl.objc.Pool,
        device: mtl.Device,

        pub fn init(g: *Device) !void {
            g.pool = mtl.objc.Pool.push();
            errdefer g.pool.pop();
            g.device = try mtl.Device.init();
        }

        pub fn deinit(g: *Device) void {
            g.pool.pop();
        }

        pub fn name(g: *Device) []const u8 {
            return std.mem.span(g.device.name());
        }

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*dn.Encoder {
            return dn.Encoder.load(gpa, io, g.device, dir);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_dino_run.zig").run(Metal, init);
}

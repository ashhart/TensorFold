//! tf-moss-image on Metal: moss_image_run.zig's case driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const Metal = struct {
    pub const mi = tf.moss_image;
    pub const st = tf.safetensors;
    pub const seconds = mtl.clock.seconds;

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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, text: usize, width: usize, height: usize) !*mi.Model {
            return mi.Model.load(gpa, io, g.device, dir, text, width, height);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_run.zig").run(Metal, init);
}

//! tf-idol-vae on Metal: idol_vae_run.zig's case driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const Metal = struct {
    pub const vae = tf.idol.vae;
    pub const ref = tf.idol.reference;
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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*vae.Encoder {
            return vae.Encoder.load(gpa, io, g.device, dir);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_vae_run.zig").run(Metal, init);
}

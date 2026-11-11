//! tf-idol on Metal: idol_run.zig's case driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const Metal = struct {
    pub const id = tf.idol;
    pub const st = tf.safetensors;
    pub const seconds = mtl.clock.seconds;

    pub fn stream(m: *id.Model, rows: usize) ![]const f16 {
        return m.stream(rows);
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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*id.Model {
            return id.Model.load(gpa, io, g.device, dir);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_run.zig").run(Metal, init);
}

//! tf-idol-generate on Metal: idol_generate_run.zig's driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const idol = tf.idol;

const Metal = struct {
    pub const id = idol;
    pub const pre = idol.preprocess;
    pub const photo = idol.photo;
    pub const st = tf.safetensors;
    pub const seconds = mtl.clock.seconds;
    /// The peak in the summary line and its JSON key: Metal's allocated bytes, sampled after each stage.
    pub const peak_name = "Metal peak";
    pub const peak_key = "metal_peak_allocated_gb";

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

        pub fn allocated(g: *Device) u64 {
            return g.device.allocated();
        }

        pub fn preprocess(g: *Device, a: std.mem.Allocator, io: std.Io, weights: [:0]const u8, path: []const u8, margin: f64, out: []u8, t: *idol.photo.Timing) !void {
            return idol.photo.preprocess(a, io, g.device, weights, path, margin, out, t);
        }

        pub fn loadDino(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.dino.Encoder {
            return idol.dino.Encoder.load(gpa, io, g.device, dir);
        }

        pub fn loadVae(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.vae.Encoder {
            return idol.vae.Encoder.load(gpa, io, g.device, dir);
        }

        pub fn loadModel(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.Model {
            return idol.Model.load(gpa, io, g.device, dir);
        }

        pub fn loadDecoder(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, anchors: usize) !*idol.Decoder {
            return idol.Decoder.load(gpa, io, g.device, dir, anchors);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_generate_run.zig").Driver(Metal).main(init);
}

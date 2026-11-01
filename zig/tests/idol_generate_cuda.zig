//! tf-idol-generate on CUDA: idol_generate_run.zig's driver on the first GPU (sm_120 or sm_121; others refused).
const std = @import("std");
const cuda = @import("cuda");
const e2e = @import("idol_e2e");
const idol = e2e.idol;
const br = e2e.birefnet;

const Cuda = struct {
    pub const id = idol;
    pub const pre = br.preprocess;
    pub const photo = br.photo;
    pub const st = e2e.safetensors;
    pub const seconds = idol.seconds;
    /// The peak in the summary line and its JSON key: the device bytes this process allocated at its peak.
    pub const peak_name = "CUDA peak";
    pub const peak_key = "cuda_peak_allocated_gb";

    pub const Device = struct {
        driver: cuda.Driver,
        ctx: cuda.Context,
        buf: [256]u8,

        pub fn init(g: *Device) !void {
            g.driver = try cuda.Driver.open();
            errdefer g.driver.close();
            g.ctx = try cuda.Context.init(&g.driver, 0);
        }

        pub fn deinit(g: *Device) void {
            g.ctx.deinit();
            g.driver.close();
        }

        pub fn name(g: *Device) []const u8 {
            return g.ctx.name(&g.buf) catch "a CUDA GPU";
        }

        /// The peak of the DeviceBuffer bytes held so far (the CUDA context and the libraries' own pools excluded).
        pub fn allocated(_: *Device) u64 {
            return cuda.usage(false).peak;
        }

        pub fn preprocess(g: *Device, a: std.mem.Allocator, io: std.Io, weights: [:0]const u8, path: []const u8, margin: f64, out: []u8, t: *br.photo.Timing) !void {
            return br.photo.preprocess(Photo{ .ctx = &g.ctx, .io = io }, a, io, weights, path, margin, out, t);
        }

        pub fn loadDino(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.dino.Encoder {
            return idol.dino.Encoder.load(gpa, io, &g.ctx, dir);
        }

        pub fn loadVae(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.vae.Encoder {
            return idol.vae.Encoder.load(gpa, io, &g.ctx, dir);
        }

        pub fn loadModel(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*idol.Model {
            return idol.Model.load(gpa, io, &g.ctx, dir);
        }

        pub fn loadDecoder(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, anchors: usize) !*idol.Decoder {
            return idol.Decoder.load(gpa, io, &g.ctx, dir, anchors);
        }
    };

    /// BiRefNet for photo_host.zig's preprocess: its clock, its loader on the context, the allocation peak.
    const Photo = struct {
        ctx: *const cuda.Context,
        io: std.Io,

        pub fn seconds(_: Photo) f64 {
            return br.seconds();
        }

        pub fn load(p: Photo, weights: [:0]const u8) !*br.Net {
            return br.Net.load(std.heap.page_allocator, p.io, p.ctx, weights);
        }

        pub fn allocated(_: Photo) u64 {
            return cuda.usage(false).peak;
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_generate_run.zig").Driver(Cuda).main(init);
}

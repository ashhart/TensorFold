//! tf-moss-image on CUDA: moss_image_run.zig's case driver on the first GPU.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tokenizer = @import("tokenizer");

const Cuda = struct {
    pub const mi = @import("moss_image");
    pub const st = core.safetensors;
    pub const seconds = mi.seconds;
    pub const Picture = struct { png: []const u8, load_s: f64, wall: f64, gpu: f64 };

    pub const Device = struct {
        driver: cuda.Driver,
        ctx: cuda.Context,
        buf: [256]u8,

        /// This build's last stage is its latents: it decodes only when --picture asks for a file.
        pub const decode_always = false;

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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, text: usize, width: usize, height: usize) !*mi.Model {
            return mi.Model.load(gpa, io, &g.ctx, dir, text, width, height);
        }

        /// Both prompts' rows from the checkpoint `dir`'s encoder, freed before this returns; the seconds it took.
        pub fn encode(_: *Device, a: std.mem.Allocator, io: std.Io, dir: []const u8, texts: [2][]const u8, rows: *[2][]u16) !f64 {
            const enc = try mi.Encoder.load(std.heap.page_allocator, io, dir);
            defer enc.deinit();
            const path = try std.fs.path.join(a, &.{ dir, "tokenizer", "tokenizer.json" });
            var tok = try tokenizer.loadTokenizer(io, a, path);
            defer tok.deinit();
            const encoding = mi.seconds();
            for (texts, rows) |text, *out| {
                const ids = try mi.ids(&tok, a, text);
                out.* = try a.alloc(u16, ids.len * mi.encoder_config.hidden);
                try enc.rows(ids, out.*, null);
            }
            return mi.seconds() - encoding;
        }

        /// The latents as a PNG by the checkpoint `dir`'s vae/ decoder, quantised as the serving host quantises one.
        pub fn decode(_: *Device, a: std.mem.Allocator, io: std.Io, dir: []const u8, latents: []const f32, width: usize, height: usize) !Picture {
            const loading = mi.seconds();
            const decoder = try mi.Decoder.load(std.heap.page_allocator, io, try std.fs.path.join(a, &.{ dir, "vae" }));
            defer decoder.deinit();
            const load_s = mi.seconds() - loading;
            const picture = try a.alloc(f32, width * height * 3);
            const t = try decoder.decode(latents, @intCast(width), @intCast(height), picture);
            const rgb = try a.alloc(u8, picture.len);
            mi.host.quantize(rgb, picture);
            return .{ .png = try mi.png.encode(a, rgb, width, height), .load_s = load_s, .wall = t.wall, .gpu = t.gpu };
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_run.zig").run(Cuda, init);
}

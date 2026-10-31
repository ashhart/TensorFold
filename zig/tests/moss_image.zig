//! tf-moss-image on Metal: moss_image_run.zig's case driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tokenizer = @import("tokenizer");

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

        /// Both prompts' rows from the checkpoint's encoder, freed before this returns; the seconds the encode took.
        pub fn encode(g: *Device, a: std.mem.Allocator, io: std.Io, checkpoint: []const u8, texts: [2][]const u8, rows: *[2][]u16) !f64 {
            const enc = try mi.Encoder.load(std.heap.page_allocator, io, g.device, checkpoint);
            defer enc.deinit();
            var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ checkpoint, "tokenizer", "tokenizer.json" }));
            defer tok.deinit();
            const encoding = mtl.clock.seconds();
            for (texts, rows) |t, *m| {
                const ids = try mi.encoder.ids(&tok, a, t);
                m.* = try a.alloc(u16, ids.len * mi.encoder_config.hidden);
                try enc.rows(ids, m.*, null);
            }
            return mtl.clock.seconds() - encoding;
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_run.zig").run(Metal, init);
}

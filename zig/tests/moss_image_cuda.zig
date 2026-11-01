//! tf-moss-image on CUDA: moss_image_run.zig's case driver on the first GPU.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Cuda = struct {
    pub const mi = @import("moss_image");
    pub const st = core.safetensors;
    pub const seconds = mi.seconds;

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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, text: usize, width: usize, height: usize) !*mi.Model {
            return mi.Model.load(gpa, io, &g.ctx, dir, text, width, height);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_run.zig").run(Cuda, init);
}

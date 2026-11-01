//! tf-idol-vae on CUDA: idol_vae_run.zig's case driver on the first GPU (sm_120 or sm_121; others are refused by name).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Cuda = struct {
    const id = @import("idol");
    pub const vae = id.vae;
    pub const ref = id.reference;
    pub const st = core.safetensors;
    pub const seconds = id.seconds;

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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*vae.Encoder {
            return vae.Encoder.load(gpa, io, &g.ctx, dir);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_vae_run.zig").run(Cuda, init);
}

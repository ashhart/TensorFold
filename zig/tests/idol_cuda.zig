//! tf-idol on CUDA: idol_run.zig's case driver on the first GPU (sm_120 or sm_121; others are refused by name).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Cuda = struct {
    pub const id = @import("idol");
    pub const st = core.safetensors;
    pub const seconds = id.seconds;

    pub fn stream(m: *id.Model, rows: usize) ![]const f16 {
        return m.residual(rows);
    }

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

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*id.Model {
            return id.Model.load(gpa, io, &g.ctx, dir);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return @import("idol_run.zig").run(Cuda, init);
}

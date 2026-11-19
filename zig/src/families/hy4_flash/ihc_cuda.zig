//! Native CUDA launches for Hy4-flash's independent hyper-connections.
const cuda = @import("cuda");
const ihc = @import("ihc.zig");

pub const Collapse = struct { hidden: u64, projection: u64, bias: u64, scale: u64, output: u64, post: u64 = 0 };
pub const Add = struct { residual: u64, branch: u64, post: u64, output: u64 };
pub const Kernels = struct {
    image: cuda.Module,
    pre: cuda.Function,
    mix: cuda.Function,

    pub fn load(driver: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var image = try cuda.Module.load(driver, cuda.kernels.hy4_ihc);
        errdefer image.unload();
        return .{ .image = image, .pre = try image.function("tf_hy4_ihc_collapse"), .mix = try image.function("tf_hy4_ihc_add") };
    }

    pub fn deinit(self: *Kernels) void {
        self.image.unload();
        self.* = undefined;
    }

    pub fn collapse(self: Kernels, s: cuda.Stream, b: Collapse, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != ihc.streams and b.post == 0) return error.MissingHy4PostGates;
        var args: cuda.Args = .{};
        inline for (.{ b.hidden, b.projection, b.bias, b.scale, b.output, b.post, shape }) |x| args.add(x);
        try cuda.launch.launch(self.pre, .{ .grid = .{ .x = shape.rows }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn add(self: Kernels, s: cuda.Stream, b: Add, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != 2 * ihc.streams or b.post == 0) return error.MissingHy4PostGates;
        var args: cuda.Args = .{};
        inline for (.{ b.residual, b.branch, b.post, b.output, shape }) |x| args.add(x);
        const cols = (ihc.streams * shape.width + 255) / 256;
        try cuda.launch.launch(self.mix, .{
            .grid = .{ .x = cols, .y = shape.rows },
            .block = .{ .x = 256 },
        }, s, &args);
    }
};

//! Native Metal launches for Hy4-flash's independent hyper-connections.
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ihc = @import("ihc.zig");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
pub const Collapse = struct { hidden: Ref, projection: Ref, bias: Ref, scale: Ref, output: Ref, post: ?Ref = null };
pub const Add = struct { residual: Ref, branch: Ref, post: Ref, output: Ref };
pub const Kernels = struct {
    pre: mtl.Pipeline,
    mix: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        const library = try mtl.Library.fromSource(device, sources.hy4_flash_ihc, mtl.CompileOptions.mlx());
        defer library.deinit();
        var pre = try mtl.Pipeline.init(device, library, "tf_hy4_ihc_collapse", false);
        errdefer pre.deinit();
        return .{ .pre = pre, .mix = try mtl.Pipeline.init(device, library, "tf_hy4_ihc_add", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.pre.deinit();
        self.mix.deinit();
        self.* = undefined;
    }

    pub fn collapse(self: Kernels, e: mtl.ComputeEncoder, b: Collapse, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != ihc.streams and b.post == null) return error.MissingHy4PostGates;
        e.setPipeline(self.pre);
        const args = [_]Ref{ b.hidden, b.projection, b.bias, b.scale, b.output, b.post orelse b.hidden };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, args.len);
        e.dispatchGroups(mtl.Size.of(shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn add(self: Kernels, e: mtl.ComputeEncoder, b: Add, shape: ihc.Shape) !void {
        try shape.check();
        if (shape.gates != 2 * ihc.streams) return error.MissingHy4PostGates;
        e.setPipeline(self.mix);
        const args = [_]Ref{ b.residual, b.branch, b.post, b.output };
        for (args, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(shape, args.len);
        e.dispatchThreads(mtl.Size.of(ihc.streams * shape.width, shape.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

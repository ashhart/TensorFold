//! Metal launches for Hy4-flash's closed-group scores and selection.
const mtl = @import("metal");
const sources = @import("kernel_sources");
const recall = @import("index_recall.zig");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
pub const Buffers = struct {
    query: Ref,
    keys: Ref,
    weights: Ref,
    rows: Ref,
    scores: Ref,
    tokens: Ref,
    counts: Ref,
};

pub const Kernels = struct {
    score: mtl.Pipeline,
    select: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        const lib = try mtl.Library.fromSource(device, sources.hy4_flash_index_recall, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var score = try mtl.Pipeline.init(device, lib, "tf_hy4_group_scores", false);
        errdefer score.deinit();
        return .{ .score = score, .select = try mtl.Pipeline.init(device, lib, "tf_hy4_recall_groups", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.score.deinit();
        self.select.deinit();
        self.* = undefined;
    }

    pub fn run(self: Kernels, e: mtl.ComputeEncoder, b: Buffers, shape: recall.Shape) !void {
        try shape.check();
        e.setPipeline(self.score);
        const input = [_]Ref{ b.query, b.keys, b.weights, b.rows, b.scores };
        for (input, 0..) |ref, i| e.setBuffer(ref.buffer, ref.offset, i);
        e.setValue(shape, input.len);
        e.setValue(recall.head_scale, input.len + 1);
        e.dispatchGroups(mtl.Size.of((shape.score_groups + 3) / 4, shape.rows, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        e.setPipeline(self.select);
        const selection = [_]Ref{ b.scores, b.rows, b.tokens, b.counts };
        for (selection, 0..) |ref, i| e.setBuffer(ref.buffer, ref.offset, i);
        e.setValue(shape, selection.len);
        e.dispatchGroups(mtl.Size.of(shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

//! CUDA launches for Hy4-flash's closed-group scores and selection.
const cuda = @import("cuda");
const recall = @import("index_recall.zig");

pub const Buffers = struct {
    query: u64,
    keys: u64,
    weights: u64,
    rows: u64,
    scores: u64,
    tokens: u64,
    counts: u64,
};

pub const Kernels = struct {
    image: cuda.Module,
    score: cuda.Function,
    select: cuda.Function,

    pub fn load(driver: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var image = try cuda.Module.load(driver, cuda.kernels.hy4_index_recall);
        errdefer image.unload();
        return .{ .image = image, .score = try image.function("tf_hy4_group_scores"), .select = try image.function("tf_hy4_recall_groups") };
    }

    pub fn deinit(self: *Kernels) void {
        self.image.unload();
        self.* = undefined;
    }

    pub fn run(self: Kernels, s: cuda.Stream, b: Buffers, shape: recall.Shape) !void {
        try shape.check();
        var score: cuda.Args = .{};
        inline for (.{ b.query, b.keys, b.weights, b.rows, b.scores, shape, recall.head_scale }) |x| score.add(x);
        try cuda.launch.launch(self.score, .{
            .grid = .{ .x = (shape.score_groups + 3) / 4, .y = shape.rows },
            .block = .{ .x = 128 },
        }, s, &score);
        var select: cuda.Args = .{};
        inline for (.{ b.scores, b.rows, b.tokens, b.counts, shape }) |x| select.add(x);
        try cuda.launch.launch(self.select, .{ .grid = .{ .x = shape.rows }, .block = .{ .x = 256 } }, s, &select);
    }
};

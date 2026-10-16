//! Model-independent DeltaNet pipelines. The caller supplies dimensions, storage flags and state ownership.
const metal = @import("metal");
const source = @import("kernel_sources");

/// Rows a prompt chain can store its state and conv window at without ending there.
pub const MARKS = 4;

/// A segment's marks as the kernels take them: rows after which to store, ascending, at most `rows`, 0 for unused.
pub fn markList(rows: u32, marks: []const u32) ![MARKS]u32 {
    if (marks.len > MARKS) return error.TooManyGdnMarks;
    var list: [MARKS]u32 = @splat(0);
    for (marks, 0..) |row, i| {
        if (row == 0 or row > rows or (i > 0 and row <= marks[i - 1])) return error.BadGdnMark;
        list[i] = row;
    }
    return list;
}

pub const Pipelines = struct { pre: metal.Pipeline, tree: metal.Pipeline, replay: metal.Pipeline, post: metal.Pipeline, conv_commit: metal.Pipeline };

pub const Kernels = struct {
    library: metal.Library,
    pipelines: Pipelines,
    publish: metal.Pipeline,
    clear: metal.Pipeline,
    chain: metal.Pipeline,
    chain_wide: metal.Pipeline,
    publish_conv: metal.Pipeline,
    chain_wide_marks: metal.Pipeline,
    mark_conv: metal.Pipeline,
    /// tf_delta_gdn_pre for one-row segments kept in place: the conv window written to history, no commit or publish.
    pre_step: metal.Pipeline,

    pub fn init(device: metal.Device) !Kernels {
        const library = try metal.Library.fromSource(device, source.core_deltanet, metal.CompileOptions.mlx());
        errdefer library.deinit();
        const names = [_][]const u8{
            "pre",
            "tree",
            "replay",
            "post",
            "conv_commit",
            "publish",
            "clear",
            "chain",
            "publish_conv",
            "chain_wide",
            "chain_wide_marks",
            "mark_conv",
            "pre_step",
        };
        var pipelines: [names.len]metal.Pipeline = undefined;
        var done: usize = 0;
        errdefer for (pipelines[0..done]) |p| p.deinit();
        inline for (names, 0..) |name, index| {
            pipelines[index] = try metal.Pipeline.init(device, library, "tf_delta_gdn_" ++ name, false);
            done += 1;
        }
        if (pipelines[7].simdWidth() != 32 or pipelines[7].maxThreads() < 128) return error.UnsupportedGdnPipeline;
        return .{
            .library = library,
            .pipelines = .{
                .pre = pipelines[0],
                .tree = pipelines[1],
                .replay = pipelines[2],
                .post = pipelines[3],
                .conv_commit = pipelines[4],
            },
            .publish = pipelines[5],
            .clear = pipelines[6],
            .chain = pipelines[7],
            .publish_conv = pipelines[8],
            .chain_wide = pipelines[9],
            .chain_wide_marks = pipelines[10],
            .mark_conv = pipelines[11],
            .pre_step = pipelines[12],
        };
    }

    pub fn deinit(k: Kernels) void {
        inline for (.{
            k.pipelines.pre,
            k.pipelines.tree,
            k.pipelines.replay,
            k.pipelines.post,
            k.pipelines.conv_commit,
            k.publish,
            k.clear,
            k.chain,
            k.publish_conv,
            k.chain_wide,
            k.chain_wide_marks,
            k.mark_conv,
            k.pre_step,
        }) |p| p.deinit();
        k.library.deinit();
    }
};

test "a mark list keeps rows in order and refuses an empty, late, repeated or fifth mark" {
    const std = @import("std");
    try std.testing.expectEqual([MARKS]u32{ 0, 0, 0, 0 }, try markList(8, &.{}));
    try std.testing.expectEqual([MARKS]u32{ 1, 3, 8, 0 }, try markList(8, &.{ 1, 3, 8 }));
    try std.testing.expectError(error.BadGdnMark, markList(8, &.{0}));
    try std.testing.expectError(error.BadGdnMark, markList(8, &.{9}));
    try std.testing.expectError(error.BadGdnMark, markList(8, &.{ 3, 3 }));
    try std.testing.expectError(error.TooManyGdnMarks, markList(8, &.{ 1, 2, 3, 4, 5 }));
}

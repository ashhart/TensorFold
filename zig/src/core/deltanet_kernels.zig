//! Model-independent DeltaNet pipelines. The caller supplies dimensions, storage flags and state ownership.
const metal = @import("metal");
const source = @import("kernel_sources");
pub const Pipelines = struct { pre: metal.Pipeline, tree: metal.Pipeline, replay: metal.Pipeline, post: metal.Pipeline, conv_commit: metal.Pipeline };

pub const Kernels = struct {
    library: metal.Library,
    pipelines: Pipelines,
    publish: metal.Pipeline,
    clear: metal.Pipeline,
    chain: metal.Pipeline,
    chain_wide: metal.Pipeline,
    publish_conv: metal.Pipeline,

    pub fn init(device: metal.Device) !Kernels {
        const library = try metal.Library.fromSource(device, source.core_deltanet, metal.CompileOptions.mlx());
        errdefer library.deinit();
        const names = [_][]const u8{ "pre", "tree", "replay", "post", "conv_commit", "publish", "clear", "chain", "publish_conv", "chain_wide" };
        var pipelines: [names.len]metal.Pipeline = undefined;
        var done: usize = 0;
        errdefer for (pipelines[0..done]) |p| p.deinit();
        inline for (names, 0..) |name, index| {
            pipelines[index] = try metal.Pipeline.init(device, library, "tf_delta_gdn_" ++ name, false);
            done += 1;
        }
        if (pipelines[7].simdWidth() != 32 or pipelines[7].maxThreads() < 128) return error.UnsupportedGdnPipeline;
        return .{ .library = library, .pipelines = .{ .pre = pipelines[0], .tree = pipelines[1], .replay = pipelines[2], .post = pipelines[3], .conv_commit = pipelines[4] }, .publish = pipelines[5], .clear = pipelines[6], .chain = pipelines[7], .publish_conv = pipelines[8], .chain_wide = pipelines[9] };
    }

    pub fn deinit(k: Kernels) void {
        inline for (.{ k.pipelines.pre, k.pipelines.tree, k.pipelines.replay, k.pipelines.post, k.pipelines.conv_commit, k.publish, k.clear, k.chain, k.publish_conv, k.chain_wide }) |p| p.deinit();
        k.library.deinit();
    }
};

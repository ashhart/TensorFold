const cuda = @import("cuda");

/// A driver whose only live entry points are the two synchronizations, which count themselves.
pub const Fake = struct {
    pub var sync_calls: usize = 0;

    fn sync() callconv(.c) c_int {
        sync_calls += 1;
        return 0;
    }

    fn streamSync(_: cuda.abi.Stream) callconv(.c) c_int {
        return sync();
    }

    pub fn driver() cuda.Driver {
        sync_calls = 0;
        var result: cuda.Driver = undefined;
        result.api.cuCtxSynchronize = sync;
        result.api.cuStreamSynchronize = streamSync;
        return result;
    }
};

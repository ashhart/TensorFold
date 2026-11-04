const cuda = @import("cuda");

/// A driver whose live entry points are the two counted synchronizations and device-to-host reads of host memory.
pub const Fake = struct {
    pub var sync_calls: usize = 0;

    fn sync() callconv(.c) c_int {
        sync_calls += 1;
        return 0;
    }

    fn streamSync(_: cuda.abi.Stream) callconv(.c) c_int {
        return sync();
    }

    fn download(destination: ?*anyopaque, source: u64, length: usize) callconv(.c) c_int {
        const target: [*]u8 = @ptrCast(destination.?);
        const origin: [*]const u8 = @ptrFromInt(source);
        @memcpy(target[0..length], origin[0..length]);
        return 0;
    }

    fn downloadAsync(destination: ?*anyopaque, source: u64, length: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return download(destination, source, length);
    }

    pub fn driver() cuda.Driver {
        sync_calls = 0;
        var result: cuda.Driver = undefined;
        result.api.cuCtxSynchronize = sync;
        result.api.cuStreamSynchronize = streamSync;
        result.api.cuMemcpyDtoH_v2 = download;
        result.api.cuMemcpyDtoHAsync_v2 = downloadAsync;
        return result;
    }
};

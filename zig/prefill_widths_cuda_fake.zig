//! Existing CUDA stream waits over an inert driver, with a planted wait failure.
const cuda = @import("cuda");
pub const Driver = struct {
    pub var fail = false;
    pub var calls: usize = 0;
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        calls += 1;
        return if (fail) cuda.abi.error_not_ready else 0;
    }
    fn name(_: c_int, out: *?[*:0]const u8) callconv(.c) c_int {
        out.* = "fixture";
        return 0;
    }
    pub fn init() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuStreamSynchronize = sync;
        d.api.cuGetErrorName = name;
        d.api.cuGetErrorString = name;
        fail = false;
        calls = 0;
        return d;
    }
};

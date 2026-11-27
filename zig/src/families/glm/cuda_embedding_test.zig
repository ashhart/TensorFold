//! Token IDs are checked against the model vocabulary before every CUDA embedding upload.
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");

const Fake = struct {
    var calls: usize = 0;

    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        calls += 1;
        return cuda.abi.error_not_ready;
    }
};

fn check(comptime mtp: bool, id: i32, expected: anyerror, expect_upload: bool) !void {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemHostAlloc = Fake.hostAlloc;
    var w: weights.Weights = undefined;
    w.config = .{ .vocab = 128 };
    w.mtp = null;
    var b: forward.Buffers = undefined;
    b.rows = 1;
    b.prefill = false;
    b.ids = 0;
    var st: state.State = undefined;
    st.capacity = 1;
    st.pos = 0;
    st.base = 0;
    st.ring_rows = 1;
    st.mtp_len = 0;
    st.mtp_base = 0;
    const ranks = [_]forward.Rank{.{ .w = &w, .st = &st, .b = &b }};
    var f: forward.Forward = undefined;
    f.d = &d;
    f.ranks = &ranks;
    if (mtp) {
        const m: weights.Mtp = undefined;
        w.mtp = m;
        try std.testing.expectError(expected, f.mtpMode(&.{id}, &.{0}, false));
    } else {
        try std.testing.expectError(expected, f.runMode(&.{id}, false));
    }
    if (expect_upload) try std.testing.expect(Fake.calls > 0) else try std.testing.expectEqual(0, Fake.calls);
}

test "GLM embedding: negative and out-of-vocabulary IDs refuse before CUDA upload" {
    inline for (.{ false, true }) |mtp| {
        try check(mtp, -1, error.BadTokenId, false);
        try check(mtp, 128, error.BadTokenId, false);
    }
}

test "GLM embedding: the greatest valid ID still reaches the CUDA upload" {
    inline for (.{ false, true }) |mtp| try check(mtp, 127, error.NotReady, true);
}

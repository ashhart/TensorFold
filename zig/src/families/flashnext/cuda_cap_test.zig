//! Flash Next on CUDA declares no `chunk_invariant`: the lane host never offers it --decode-share's row cap.
const std = @import("std");
const Cuda = @import("cuda_launch.zig").native.Cuda;

test "Flash Next CUDA: no chunk_invariant, so the flag's cap stays off (lane_host_decode_share_test.zig)" {
    var c: Cuda = undefined; // backend() reads only its head and ranks: no device
    c.head = null;
    c.ranks = null;
    try std.testing.expect(!c.backend().chunk_invariant); // declare it only after a device token-hash qualification
}

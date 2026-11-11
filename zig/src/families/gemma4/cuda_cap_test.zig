//! Gemma 4 on CUDA declares no `chunk_invariant`: the lane host never offers it --decode-share's row cap.
const std = @import("std");
const Gemma = @import("cuda_backend.zig").Gemma;

test "Gemma 4 CUDA: no chunk_invariant, so the flag's cap stays off (lane_host_decode_share_test.zig)" {
    var g: Gemma = undefined; // backend() takes only the pointer: no device
    try std.testing.expect(!g.backend().chunk_invariant); // declare it only after a device token-hash qualification
}

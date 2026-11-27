//! Where a tf-idol-generate run's VAE posterior noise came from, as its JSON record reports it.
const std = @import("std");

pub const VaeNoise = enum {
    /// the native seeded generator
    native,
    /// the --noise file's `noise` tensor
    given,
    /// --features skipped both encoders
    unused,

    /// A --noise file without `noise` (only `latent` and `camera`) leaves the posterior's noise native.
    pub fn of(features_given: bool, noise_tensor: bool) VaeNoise {
        if (features_given) return .unused;
        return if (noise_tensor) .given else .native;
    }
};

test "a --noise file reports given posterior noise only when it holds a noise tensor" {
    // --noise with `latent` and `camera` alone: the encoder drew native noise, and the record says so
    try std.testing.expectEqualStrings("native", @tagName(VaeNoise.of(false, false)));
    try std.testing.expectEqualStrings("given", @tagName(VaeNoise.of(false, true)));
    try std.testing.expectEqualStrings("unused", @tagName(VaeNoise.of(true, true)));
    try std.testing.expectEqualStrings("unused", @tagName(VaeNoise.of(true, false)));
}

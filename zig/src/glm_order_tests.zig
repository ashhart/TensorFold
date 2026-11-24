//! GLM Metal's prompt order on the lane host's Srpt: the default, the env's say, and the cases that keep main's order.
const std = @import("std");
const Srpt = @import("core/lane_host_srpt.zig").Srpt;
const order = @import("native/glm_order.zig");
const env = @import("sys").env;

const name = "TENSORFOLD_PREFILL_ORDER";

fn started(pair: bool, share: ?f64, prompt_share: f64) Srpt {
    return if (order.srptByDefault(pair, share, prompt_share)) Srpt.withDefault(true) else Srpt.fromEnv();
}

test "glm order: one Mac with no share starts with shortest-remaining-first" {
    _ = env.unsetenv(name);
    try std.testing.expect(started(false, null, 1).on);
}

test "glm order: an explicit order wins over the default, either way" {
    defer _ = env.unsetenv(name);
    _ = env.setenv(name, "fifo", 1);
    try std.testing.expect(!started(false, null, 1).on);
    _ = env.setenv(name, "srpt", 1);
    try std.testing.expect(started(false, null, 1).on);
    try std.testing.expect(started(true, null, 1).on); // opt-in still works on the pair and beside a share
    try std.testing.expect(started(false, 0.5, 1).on);
}

test "glm order: the pair, a decode share and a prompt share below 1 keep main's order" {
    _ = env.unsetenv(name);
    try std.testing.expect(!started(true, null, 1).on);
    try std.testing.expect(!started(false, 0.25, 1).on);
    try std.testing.expect(!started(false, null, 0.5).on);
}

test "glm order: a host nobody gave a default starts off, as every other family's" {
    _ = env.unsetenv(name);
    try std.testing.expect(!Srpt.fromEnv().on);
}

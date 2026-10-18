//! DeepSeek-V4.1 KV memory: families, split KV exchange, priced floor, device pool (neutral parts: zig/src/sessions).

pub const sessions = @import("sessions");
pub const layout = @import("layout.zig");
pub const Layout = layout.Layout;
pub const split = @import("split.zig");
pub const budget = @import("budget.zig");
pub const device = @import("device.zig");
pub const DevicePool = device.DevicePool;
pub const sess = @import("sess.zig");
pub const sched = @import("sched.zig");
pub const progress = @import("progress.zig");

test {
    _ = layout;
    _ = split;
    _ = budget;
    _ = device;
    _ = @import("split_test.zig");
    _ = @import("union_plan_test.zig");
    if (@import("builtin").os.tag == .linux) _ = @import("split_roce_test.zig");
    _ = sess;
    _ = @import("session_slots_test.zig");
    _ = @import("resend_test.zig");
    _ = sched;
    _ = progress;
}

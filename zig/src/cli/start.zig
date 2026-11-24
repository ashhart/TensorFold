//! The checkpoint registry and pull entry points shared by `tensorfold start`.
pub const Entry = @import("catalog.zig").Entry;
pub const entries = @import("catalog.zig").entries;
pub const Choice = @import("models.zig").StartChoice;
pub const choose = @import("models.zig").chooseForStart;
pub const pull = @import("pull.zig");

test {
    _ = @import("models.zig");
}

//! The HIP families' contract entries, one line a family: `tf-hip-kernel-contract image <group>` runs one.
pub const Profile = @import("contract_entry.zig").Profile;
pub const Entry = @import("contract_entry.zig").Entry;

pub const entries = [_]Entry{};

comptime { // one entry a group, so `image <group>` never has to choose
    for (entries, 0..) |e, i| for (entries[0..i]) |f| {
        if (e.group == f.group) @compileError("two contract entries for group " ++ @tagName(e.group));
    };
}

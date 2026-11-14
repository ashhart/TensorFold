//! The HIP families' contract entries, one line a family: `tf-hip-kernel-contract image <group>` runs one.
pub const Profile = @import("contract_entry.zig").Profile;
pub const Entry = @import("contract_entry.zig").Entry;

pub const entries = [_]Entry{
    .{ .group = .glm_ocr, .run = @import("contract_glm_ocr.zig").run },
    .{ .group = .joint_schema, .run = @import("contract_joint_schema.zig").run },
    .{ .group = .lighton_ocr, .run = @import("contract_lighton.zig").run },
    .{ .group = .qwen_vision, .run = @import("contract_qwen_vision.zig").run },
};

comptime { // one entry a group, so `image <group>` never has to choose
    for (entries, 0..) |e, i| for (entries[0..i]) |f| {
        if (e.group == f.group) @compileError("two contract entries for group " ++ @tagName(e.group));
    };
}

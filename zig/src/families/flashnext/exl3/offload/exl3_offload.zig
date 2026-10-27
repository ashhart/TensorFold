//! Flash Next EXL3 offload hub: re-exports the policy, VRAM, count, fill and scheduling modules under one name.

const policy = @import("exl3_offload_policy.zig");
const vram = @import("exl3_offload_vram.zig");
const counts = @import("exl3_offload_counts.zig");
const fill = @import("exl3_offload_fill.zig");
const used = @import("exl3_offload_used.zig");

pub const shape = policy.shape;
pub const shared_id = policy.shared_id;
pub const Decision = policy.Decision;
pub const Policy = policy.Policy;
pub const Cache = policy.Cache;
pub const init = policy.init;
pub const find = policy.find;
pub const planRound = policy.planRound;
pub const swap = policy.swap;
pub const installProfile = policy.installProfile;
pub const Line = vram.Line;
pub const Vram = vram.Vram;
pub const onDevice = vram.onDevice;
pub const residentAt = vram.residentAt;
pub const touch = vram.touch;
pub const stage = vram.stage;
pub const stageKeeping = vram.stageKeeping;
pub const schedule = vram.schedule;
pub const reapAt = vram.reapAt;
pub const reap = vram.reap;
pub const vramIndexAt = vram.vramIndexAt;
pub const touchAt = vram.touchAt;
pub const lineOf = vram.lineOf;
pub const Copy = vram.Copy;
pub const lineIds = vram.lineIds;
pub const stagedCopies = vram.stagedCopies;
pub const countAt = counts.countAt;
pub const observe = counts.observe;
pub const Owned = counts.Owned;
pub const allOwned = counts.allOwned;
pub const seedHottest = fill.seedHottest;
pub const fillOwned = fill.fillOwned;
pub const fillOwnedLayers = fill.fillOwnedLayers;
pub const stageUsed = used.stageUsed;
pub const stageUsedAt = used.stageUsedAt;
pub const scheduleUsed = used.scheduleUsed;
pub const scheduleUsedAt = used.scheduleUsedAt;

comptime {
    _ = @import("exl3_offload_policy_test.zig");
    _ = @import("exl3_offload_vram_test.zig");
    _ = @import("exl3_offload_used_test.zig");
    _ = @import("exl3_offload_fill_test.zig");
}

//! Generic wrapper for drafter opening in the nemotron family.
const std = @import("std");
const lanes = @import("lanes");
const native = @import("cuda_native.zig");
const cuda_lilicorr = @import("cuda_lilicorr.zig");
const cuda_draft_config = @import("cuda_draft_config.zig");

/// The opened drafter structure.
pub const Opened = struct {
    drafter: lanes.drafter.Drafter,
    ctx: *anyopaque,
    close: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    prepare: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } = null,
};

/// Finish the drafter opening process (no-op for nemotron).
pub fn finishDrafterOpen(p: *anyopaque) void {
    _ = p; // the nemotron Owned carries no startup-room callback to clear (qwen's room lives in its family Lanes)
}

/// Open a drafter for the nemotron family.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    if (bits != 4) return error.DrafterBitsUnsupported;

    // Cast p to *native.Owned
    const own: *native.Owned = @ptrCast(@alignCast(p));

    // Open the LiLiCorr drafter
    const d = try cuda_lilicorr.open(gpa, io, own.e, dir);
    errdefer cuda_lilicorr.deinit(d);

    return .{
        .drafter = cuda_lilicorr.drafter(d),
        .ctx = d,
        .close = closeDrafter,
        .reserve = reserveDrafter,
    };
}

/// Close the drafter.
fn closeDrafter(p: *anyopaque) void {
    const d: *cuda_lilicorr.Draft = @ptrCast(@alignCast(p));
    cuda_lilicorr.deinit(d);
    d.gpa.destroy(d);
}

/// Reserve space for the drafter.
fn reserveDrafter(p: *anyopaque, count: u32) anyerror!void {
    const d: *cuda_lilicorr.Draft = @ptrCast(@alignCast(p));
    for (0..count) |_| {
        if (d.pool.items.len > 0) continue;
        const ctx = try cuda_lilicorr.create(d);
        d.pool.append(d.gpa, ctx) catch cuda_lilicorr.drop(d, ctx);
    }
}

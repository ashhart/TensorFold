//! Refuse uncertain learned shards before a native engine maps them or opens a GPU.
const std = @import("std");
const guard = @import("engine_api").save_guard;
pub fn refused(io: std.Io, dir: []const u8, problem: *[]const u8) bool {
    guard.check(io, dir) catch |e| {
        if (e == error.CheckpointSaveUncertain) {
            problem.* = "this folder has an unresolved Living Weights save: finish that save or use tensorfold slide rollback before reopening";
            return true;
        }
    };
    return false;
}

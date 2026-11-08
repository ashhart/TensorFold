//! Dispatch a release-tree entry on its kind; a kind the listing cannot report is lstat'd.
const std = @import("std");

pub const Decision = enum { file, directory, sym_link, refuse };

/// A kind the directory listing cannot report (for example `.unknown` on a
/// filesystem without ftype) is lstat'd through `stat` and dispatched on the
/// result. A failed lstat refuses, so the check fails closed.
pub fn dispatch(kind: std.Io.File.Kind, ctx: *anyopaque, stat: fn (ctx: *anyopaque) ?std.Io.File.Kind) Decision {
    switch (kind) {
        .file => return .file,
        .directory => return .directory,
        .sym_link => return .sym_link,
        else => {
            const k = stat(ctx) orelse return .refuse;
            return switch (k) {
                .file => .file,
                .directory => .directory,
                .sym_link => .sym_link,
                else => .refuse,
            };
        },
    }
}

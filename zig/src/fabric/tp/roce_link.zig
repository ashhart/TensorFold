//! Diagnostics common to the unchanged two-rank wire and the versioned multirank gather wire.
const two = @import("roce.zig").Roce;
const many = @import("roce_multi.zig").Roce;
pub const Link = union(enum) {
    two: *two,
    many: *many,
    pub fn kept() u32 {
        return two.kept.load(.monotonic) + many.kept.load(.monotonic);
    }
    pub fn linkIdle(l: Link) bool {
        return switch (l) {
            inline else => |p| p.linkIdle(),
        };
    }
    pub fn describeLink(l: Link, buf: []u8) []const u8 {
        return switch (l) {
            inline else => |p| p.describeLink(buf),
        };
    }
    pub fn inject(l: Link) void {
        switch (l) {
            inline else => |p| p.inject(),
        }
    }
    pub fn region(l: Link) []const u8 {
        return switch (l) {
            inline else => |p| p.region,
        };
    }
};

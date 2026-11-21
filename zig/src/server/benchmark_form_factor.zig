//! Coarse Apple product families; raw product identifiers never become receipt values.
const std = @import("std");

pub const FormFactor = enum {
    mac_mini,
    mac_studio,
    macbook_pro,
    macbook_air,
    imac,
    mac_pro,

    pub fn label(f: FormFactor) []const u8 {
        return switch (f) {
            .mac_mini => "Mac mini",
            .mac_studio => "Mac Studio",
            .macbook_pro => "MacBook Pro",
            .macbook_air => "MacBook Air",
            .imac => "iMac",
            .mac_pro => "Mac Pro",
        };
    }

    pub fn fromLabel(value: []const u8) ?FormFactor {
        for (std.enums.values(FormFactor)) |f| if (std.mem.eql(u8, value, f.label())) return f;
        return null;
    }
};

const Group = struct { family: FormFactor, products: []const []const u8 };
const groups = [_]Group{
    // https://support.apple.com/102852
    .{ .family = .mac_mini, .products = &.{
        "Macmini9,1", "Mac14,3", "Mac14,12", "Mac16,10", "Mac16,11", "Mac17,16", "Mac18,5",
    } },
    // https://support.apple.com/102231
    .{ .family = .mac_studio, .products = &.{
        "Mac13,1", "Mac13,2", "Mac14,13", "Mac14,14", "Mac15,14", "Mac16,9", "Mac17,14", "Mac17,15",
    } },
    // https://support.apple.com/108052
    .{ .family = .macbook_pro, .products = &.{
        "MacBookPro17,1", "MacBookPro18,1", "MacBookPro18,2", "MacBookPro18,3", "MacBookPro18,4",
        "Mac14,5",        "Mac14,6",        "Mac14,7",        "Mac14,9",        "Mac14,10",
        "Mac15,3",        "Mac15,6",        "Mac15,7",        "Mac15,8",        "Mac15,9",
        "Mac15,10",       "Mac15,11",       "Mac16,1",        "Mac16,5",        "Mac16,6",
        "Mac16,7",        "Mac16,8",        "Mac17,2",        "Mac17,6",        "Mac17,7",
        "Mac17,8",        "Mac17,9",
    } },
    // https://support.apple.com/102869
    .{ .family = .macbook_air, .products = &.{
        "MacBookAir10,1", "Mac14,2", "Mac14,15", "Mac15,12", "Mac15,13", "Mac16,12", "Mac16,13", "Mac17,3", "Mac17,4",
    } },
    // https://support.apple.com/108054
    .{ .family = .imac, .products = &.{ "iMac21,1", "iMac21,2", "Mac15,4", "Mac15,5", "Mac16,2", "Mac16,3" } },
    // https://support.apple.com/102887
    .{ .family = .mac_pro, .products = &.{"Mac14,8"} },
};

/// Exact, documented Apple silicon product IDs only; an unlisted product remains unknown.
pub fn fromProduct(product: []const u8) ?FormFactor {
    for (groups) |g| for (g.products) |p| if (std.mem.eql(u8, product, p)) return g.family;
    return null;
}

comptime {
    @setEvalBranchQuota(100000);
    for (groups, 0..) |g, i| for (g.products, 0..) |p, j| {
        for (groups[0 .. i + 1], 0..) |before, k| {
            const limit = if (k == i) j else before.products.len;
            for (before.products[0..limit]) |old| if (std.mem.eql(u8, p, old)) {
                @compileError("A Mac product must map to exactly one family");
            };
        }
    };
}

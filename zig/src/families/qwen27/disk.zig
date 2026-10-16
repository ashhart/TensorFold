//! The 27B's kernel sources for kept states on disk: every text it compiles at run time, hashed into their identity.
const std = @import("std");

/// The 27B's own kernels and the shared sources beside them; a changed source moves every disk state's identity.
pub fn sourceHash() u64 {
    var h = std.hash.Wyhash.init(0x7127);
    texts(&h, @import("qwen_runtime_sources"));
    texts(&h, @import("kernel_sources"));
    return h.final();
}

/// Every source text a module embeds at its top level: strings, and lists of named files.
fn texts(h: *std.hash.Wyhash, comptime M: type) void {
    inline for (comptime std.meta.declarations(M)) |d| {
        const v = @field(M, d);
        const T = @TypeOf(v);
        if (comptime T == type) continue;
        switch (@typeInfo(T)) {
            .pointer => |p| if (comptime p.child == u8 or (@typeInfo(p.child) == .array and @typeInfo(p.child).array.child == u8)) {
                h.update(d);
                h.update(v);
            },
            .array => |arr| if (comptime @typeInfo(arr.child) == .@"struct" and @hasField(arr.child, "text")) for (v) |f| {
                h.update(f.name);
                h.update(f.text);
            },
            else => {},
        }
    }
}

test "a module's top-level texts and file lists are hashed, its types skipped" {
    const F = struct { name: []const u8, text: []const u8 };
    const M = struct {
        pub const a = "kernel a";
        pub const files = [_]F{.{ .name = "b", .text = "kernel b" }};
        pub const T = struct {};
        pub const n: u32 = 3;
    };
    const N = struct {
        pub const a = "kernel a";
        pub const files = [_]F{.{ .name = "b", .text = "kernel B" }};
    };
    var x = std.hash.Wyhash.init(0);
    texts(&x, M);
    var y = std.hash.Wyhash.init(0);
    texts(&y, N);
    try std.testing.expect(x.final() != y.final());
}

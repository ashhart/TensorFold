//! Compare fusion emission to the complete unfused program, including every unaffected call.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");

fn named(c: calls.Call, name: []const u8) calls.Arg {
    for (c.args) |a| if (std.mem.eql(u8, a.name, name)) return a.arg;
    @panic("missing call argument");
}
fn kvNorm(c: calls.Call) bool {
    if (!std.mem.eql(u8, c.name, "_rms")) return false;
    const w = named(c, "W");
    return w == .t and w.t.role == .weight and std.mem.endsWith(u8, w.t.role.weight, ".attn.kv_norm");
}

pub fn compare(plain: []const calls.Call, fused: []const calls.Call, count: usize) !void {
    try testing.expectEqual(plain.len - count, fused.len);
    var pi: usize = 0;
    var got: usize = 0;
    for (fused) |c| {
        while (pi < plain.len and kvNorm(plain[pi])) pi += 1;
        try testing.expect(pi < plain.len);
        if (std.mem.eql(u8, c.name, "tf_dsv41_kv_glue_v1.norm_store")) {
            const store = plain[pi];
            try testing.expectEqualStrings("_kv_store", store.name);
            try testing.expectEqual(@as(i64, 0), named(store, "RATIO").i);
            var ni = pi;
            while (ni > 0) {
                ni -= 1;
                if (kvNorm(plain[ni])) break;
            }
            const norm = plain[ni];
            try testing.expect(kvNorm(norm));
            try testing.expectEqualDeep(named(norm, "OUT"), named(store, "LAT"));
            const want = [_]calls.Arg{
                named(norm, "X"),         named(norm, "W"),     named(store, "CS"),  named(store, "V"),         named(store, "S"),
                named(store, "POS"),      named(store, "SL"),   named(norm, "x_rs"), named(store, "cs_stride"), named(store, "v_stride"),
                named(store, "s_stride"), named(norm, "inv_k"), named(norm, "eps"),  named(store, "RING"),      named(store, "ROWS"),
            };
            try testing.expectEqual(want.len, c.args.len);
            for (want, c.args) |arg, actual| try testing.expectEqualDeep(arg, actual.arg);
            try testing.expectEqual(store.side, c.side);
            try testing.expectEqual(norm.fork or store.fork, c.fork);
            try testing.expectEqual(norm.join or store.join, c.join);
            got += 1;
        } else try testing.expectEqualDeep(plain[pi], c);
        pi += 1;
    }
    try testing.expectEqual(plain.len, pi);
    try testing.expectEqual(count, got);
}

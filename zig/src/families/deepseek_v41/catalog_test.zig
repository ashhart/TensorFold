const std = @import("std");
const t = std.testing;
const catalog = @import("catalog.zig");
const registry = @import("core").contract.registry;
test "every resolved CUDA symbol has one explicit resource contract in the composition catalog" {
    try catalog.collect();
    for (catalog.entries[0..catalog.count]) |e| {
        var found: usize = 0;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.image, e.image) and std.mem.eql(u8, r.launch.symbol, e.name[0..e.len])) {
            found += 1;
            try t.expectEqual(r.launch.threads, e.threads);
            try t.expectEqual(r.launch.dynamic, e.shared);
            try t.expectEqual(r.launch.cluster, e.cluster);
            try t.expectEqual(r.optional_shared, e.optional_shared);
        };
        try t.expectEqual(@as(usize, 1), found);
    }
}
test "family calls the common EXL3 operator types" {
    const dk = @import("dsv41_kernels");
    const cuda = @import("cuda");
    try t.expect(dk.dense.Functions == cuda.qlinear.block.dense.Functions);
    try t.expect(dk.exl3.Functions == cuda.qlinear.block.experts.Functions);
}

test "real resolver agrees with per-target availability for every optional symbol" {
    const target = @import("core").contract.limits.sms[4];
    try catalog.collectWithLimit(target.smem_optin);
    var refused: usize = 0;
    for (catalog.entries[0..catalog.count]) |e| if (e.optional_shared) {
        const record = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.image, e.image) and std.mem.eql(u8, r.launch.symbol, e.name[0..e.len])) break r;
        } else return error.MissingRecord;
        try t.expectEqual(e.available, registry.targetDecision(record, target, record.usage[4]) == .applicable);
        if (!e.available) refused += 1;
    };
    try t.expectEqual(@as(usize, 7), refused);
}

//! Flash Next's top-k in parts and prompt router tile against tf_topk and the host's fma chain, byte for byte.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const kt = @import("ops.zig");
const act = kt.act;
const tensor = kt.tensor;

test "top-k in parts gives tf_topk's ids, values and order byte for byte (Flash Next draft and full vocabularies)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x7095 };
    const stride = 64;
    for ([_]usize{ 8192, 79591, 248320 }) |n| for ([_]u8{ 0, 1, 2, 3 }) |shape| {
        const rows = 4;
        const hl = try gpa.alloc(u16, rows * n);
        defer gpa.free(hl);
        for (hl) |*x| x.* = t.bits(rng.unit() * 8.0);
        switch (shape) {
            // a few distinct values: ties at the k-th across every part
            1 => for (hl, 0..) |*x, j| {
                x.* = t.bits(@floatFromInt((j * 7919) % 5));
            },
            // NaN, infinities and both zeros among the largest
            2 => for (0..rows) |r| {
                hl[r * n + 3] = 0x7fc0;
                hl[r * n + n - 1] = 0x7f80;
                hl[r * n + n / 2] = 0xff80;
                hl[r * n + 11] = 0x8000;
                hl[r * n + 12] = 0;
            },
            // one value everywhere: every entry ties
            3 => @memset(hl, t.bits(1.5)),
            else => {},
        }
        var logits = try t.upload(hl);
        defer logits.free();
        const ks = [_]i32{ 28, 1, 0, 64 };
        var dev_ks = try t.upload(&ks);
        defer dev_ks.free();
        var got: [2][2][]u8 = undefined;
        for (0..2) |v| {
            var ids = try t.alloc(rows * stride * 4);
            defer ids.free();
            var vals = try t.alloc(rows * stride * 2);
            defer vals.free();
            try ids.fill8(0xff);
            try vals.fill8(0xff);
            if (v == 0) {
                try t.on.tf_topk_rows(@ptrFromInt(at(logits)), rows, @intCast(n), @ptrFromInt(at(dev_ks)), stride, @ptrFromInt(at(ids)), @ptrFromInt(at(vals)), t.stream.handle);
            } else try o.topkRows(tensor(logits, act(t)), rows, n, at(dev_ks), stride, at(ids), at(vals));
            try t.stream.synchronize();
            got[v][0] = try rig.download(u8, ids);
            got[v][1] = try rig.download(u8, vals);
        }
        defer for (&got) |*g| for (g) |b| gpa.free(b);
        for (0..2) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
            std.debug.print("top-k parts n={d} shape={d}: buffer {d} differs\n", .{ n, shape, k });
            return error.TestUnexpectedResult;
        };
    };
}

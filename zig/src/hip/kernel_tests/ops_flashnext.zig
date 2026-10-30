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

test "the prompt router tile: every logit one fp32 fma chain over k in order, the bytes of the host's chain" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var o = t.ops(&t.on);
    o.prefill = true;
    var rng: rig.Rng = .{ .state = 0x7071E };
    const d: usize = 2560;
    const e: usize = 513;
    // the rows: two tiles and a part, and one row past a full tile of 128
    for ([_]usize{ 300, 129 }) |rows| {
        const hx = try gpa.alloc(u16, rows * d);
        defer gpa.free(hx);
        for (hx) |*v| v.* = t.bits(rng.unit() * 2.0);
        const hw = try gpa.alloc(f32, e * d);
        defer gpa.free(hw);
        for (hw) |*v| v.* = @bitCast(@as(u32, rig.bf16Bits(rng.unit() / 8.0)) << 16);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(rows * e * 4);
        defer out.free();
        try o.moeRouter(tensor(x, act(t)), at(w), at(out), rows, d, e);
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        // every expert of a sample of rows (each tile's first and last, every 37th, the last three): the host is slow
        for (0..rows) |r| if (r % 37 == 0 or r % 128 == 0 or r % 128 == 127 or r + 3 >= rows) for (0..e) |c| {
            var acc: f32 = 0;
            for (0..d) |k| acc = @mulAdd(f32, @floatCast(t.value(hx[r * d + k])), hw[c * d + k], acc);
            if (@as(u32, @bitCast(acc)) != @as(u32, @bitCast(got[r * e + c]))) {
                std.debug.print("router tile rows={d}: logit ({d}, {d}) {e} where the chain gives {e}\n", .{ rows, r, c, got[r * e + c], acc });
                return error.TestUnexpectedResult;
            }
        };
    }
}

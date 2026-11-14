//! Device check: layer 63's prompt tail gives the whole-layer path's last-row logits and keys, bit for bit.
const std = @import("std");
const q = @import("qwen27.zig");
const ze = @import("ze");

const lens = [_]usize{ 1, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 2049 };

fn read(d: *q.Dev, p: *anyopaque, n: usize, out: []u8) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, out.ptr, p, n, null, 0, null));
    try d.sync();
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const dir = if (argv.len > 1) argv[1] else if (std.c.getenv("TF_XPU_QWEN27")) |v| std.mem.span(v) else {
        std.debug.print("skipped: TF_XPU_QWEN27 not set\n", .{});
        return;
    };
    var probe = q.ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    probe.close();
    const r = try q.Runner.init(dir, 4096);
    defer r.deinit();
    if (q.kv_i8) {
        std.debug.print("skipped: tf-xpu-qwen27-tail-test does not compare XPU_KV_INT8 layout or scales\n", .{});
        return;
    }
    var ids: [2049]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x7a11);
    for (&ids) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 100000);
    const last = q.LAYERS - 1;
    const kv = 4 * 256 * 2; // layer 63: four KV heads of 256 values, bf16
    for (lens) |len| {
        var want_logits: []u8 = &.{};
        var want_k: []u8 = &.{};
        var want_v: []u8 = &.{};
        defer gpa.free(want_logits);
        defer gpa.free(want_k);
        defer gpa.free(want_v);
        for ([_]bool{ true, false }) |tail_on| {
            q.prompt_tail_on = tail_on;
            try r.reset();
            try r.prompt(ids[0..len], true);
            const l = try r.logits();
            const bytes = len * kv;
            if (tail_on) {
                want_logits = try gpa.dupe(u8, std.mem.sliceAsBytes(l));
                want_k = try gpa.alloc(u8, bytes);
                want_v = try gpa.alloc(u8, bytes);
                try read(&r.d, r.model.layers[last].kc.?, bytes, want_k);
                try read(&r.d, r.model.layers[last].vc.?, bytes, want_v);
            } else {
                const got_k = try gpa.alloc(u8, bytes);
                defer gpa.free(got_k);
                const got_v = try gpa.alloc(u8, bytes);
                defer gpa.free(got_v);
                try read(&r.d, r.model.layers[last].kc.?, bytes, got_k);
                try read(&r.d, r.model.layers[last].vc.?, bytes, got_v);
                if (!std.mem.eql(u8, want_logits, std.mem.sliceAsBytes(l))) return error.LogitsDiffer;
                if (!std.mem.eql(u8, want_k, got_k)) return error.KeysDiffer;
                if (!std.mem.eql(u8, want_v, got_v)) return error.ValuesDiffer;
            }
        }
        std.debug.print("len {d}: tail and whole layer agree\n", .{len});
    }
    q.prompt_tail_on = true;
    std.debug.print("ok\n", .{});
}

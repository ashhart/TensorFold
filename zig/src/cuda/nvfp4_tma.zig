//! TMA prompt products and a fused BF16-rounded gate/up/SwiGLU, using the existing NVFP4 weight pack.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const maps = @import("tensor_map.zig");
const qmmf = @import("qmmf.zig");

pub const Tma = struct {
    mod: Module,
    map: maps.Api,
    single: [2][2]Function,
    pair: [2]Function,
    packed_single: [2][2]Function,
    packed_pair: [2]Function,
    pack: Function,

    pub fn load(d: *const Driver) !Tma {
        var m = try Module.load(d, kernels.nvfp4_tma);
        errdefer m.unload();
        var t: Tma = .{ .mod = m, .map = try maps.Api.load(d), .single = undefined, .pair = undefined, .packed_single = undefined, .packed_pair = undefined, .pack = try m.function("tf_nvfp4_pack_mma") };
        inline for (.{ "128", "256" }, 0..) |name, i| {
            t.single[i] = .{ try m.function("tf_nvfp4_tma_" ++ name), try m.function("tf_nvfp4_tma_" ++ name ++ "_f32") };
            t.pair[i] = try m.function("tf_nvfp4_tma_swiglu_" ++ name);
            t.packed_single[i] = .{ try m.function("tf_nvfp4_tma_packed_" ++ name), try m.function("tf_nvfp4_tma_packed_" ++ name ++ "_f32") };
            t.packed_pair[i] = try m.function("tf_nvfp4_tma_packed_swiglu_" ++ name);
            for (t.single[i]) |f| try f.allowDynamicShared(@intCast(maps.sharedBytes(if (i == 0) 128 else 256)));
            try t.pair[i].allowDynamicShared(@intCast(maps.sharedBytes(if (i == 0) 128 else 256)));
            for (t.packed_single[i]) |f| try f.allowDynamicShared(@intCast(maps.sharedBytes(if (i == 0) 128 else 256)));
            try t.packed_pair[i].allowDynamicShared(@intCast(maps.sharedBytes(if (i == 0) 128 else 256)));
        }
        return t;
    }

    pub fn unload(t: *Tma) void {
        t.mod.unload();
    }

    pub fn supports(m: usize, k: usize) bool {
        return m >= 128 and k % 256 == 0;
    }

    fn tile(m: usize) usize {
        _ = m;
        return 128;
    }

    pub fn repack(t: *const Tma, s: Stream, source: u64, out: u64, bytes: usize) !void {
        var a: launch.Args = .{};
        a.add(source);
        a.add(out);
        a.add(@as(i32, @intCast(bytes / 8)));
        try launch.launch(t.pack, .{ .grid = .{ .x = @intCast((bytes / 8 + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn matmul(t: *const Tma, s: Stream, x: u64, xs: u64, m: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out, prepared: bool) !void {
        const rows = tile(m);
        const packet = try t.map.single(x, xs, w.codes, w.scales, m, w.npad, w.k, rows);
        var a: launch.Args = .{};
        a.add(packet);
        a.add(out);
        for ([_]usize{ m, w.n, w.k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(f32, w.act * w.scale));
        const fns = if (prepared) t.packed_single else t.single;
        try launch.launch(fns[@intFromBool(rows == 256)][@intFromBool(kind == .f32)], .{
            .grid = .{ .x = @intCast((m + rows - 1) / rows), .y = @intCast((w.n + 127) / 128) },
            .block = .{ .x = 288 },
            .shared = @intCast(maps.sharedBytes(rows)),
        }, s, &a);
    }

    pub fn swiglu(t: *const Tma, s: Stream, x: u64, xs: u64, m: usize, gate: qmmf.Weight, up: qmmf.Weight, out: u64, prepared: bool) !void {
        if (gate.n != up.n or gate.k != up.k or gate.npad != up.npad or gate.act != up.act)
            return error.IncompatibleFp4Pair;
        const rows = tile(m);
        const packet = try t.map.pair(x, xs, .{ gate.codes, up.codes }, .{ gate.scales, up.scales }, m, gate.npad, gate.k, rows);
        var a: launch.Args = .{};
        a.add(packet);
        a.add(out);
        for ([_]usize{ m, gate.n, gate.k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(f32, gate.act * gate.scale));
        a.add(@as(f32, up.act * up.scale));
        const fns = if (prepared) t.packed_pair else t.pair;
        try launch.launch(fns[@intFromBool(rows == 256)], .{
            .grid = .{ .x = @intCast((m + rows - 1) / rows), .y = @intCast((gate.n + 63) / 64) },
            .block = .{ .x = 288 },
            .shared = @intCast(maps.sharedBytes(rows)),
        }, s, &a);
    }
};

//! Eight packed PLE groups, populated with bounded positional reads and donated
//! slice updates. Never hold all source shards plus a second concatenated copy.
const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
pub const Resident = struct {
    format: @import("quantization.zig").Spec = .{ .bits = 4, .group_size = 32 },
    groups: [8][3]mx.Array = @splat(@splat(mx.empty)),
    starts: mx.Array = mx.empty,
    writes: usize = 0,
    pub fn deinit(r: *Resident) void {
        for (r.groups) |group| for (group) |array| mx.free(array);
        mx.free(r.starts);
        r.* = .{};
    }
    fn storage(rows: i32, part: usize, format: @import("quantization.zig").Spec) !mx.Array {
        var s = mx.Scope{};
        defer s.deinit();
        const array = try s.zeros(&.{ rows, if (part == 0) @divExact(160 * format.bits, 32) else @divExact(160, format.group_size) }, if (part == 0) mx.c.MLX_UINT32 else mx.bf16);
        try mx.eval(array);
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        return mx.retain(array);
    }
    fn address(array: mx.Array) usize {
        return if (mx.dtype(array) == mx.c.MLX_UINT32) @intFromPtr(mx.c.mlx_array_data_uint32(array)) else @import("kv_buffer.zig").address(array);
    }
    fn update(destination: *mx.Array, added: mx.Array, row: i32) !void {
        var s = mx.Scope{};
        defer s.deinit();
        const start = try s.ints(&.{row});
        const donor = destination.*;
        const before = address(donor);
        if (before == 0) return error.InvalidBufferAddress;
        const axis: c_int = 0;
        var output = mx.c.mlx_array_new();
        const rc = mx.c.mlx_slice_update_dynamic(&output, donor, added, start, &axis, 1, mx.stream);
        const updated = try s.result(rc, output);
        mx.free(donor);
        destination.* = mx.empty;
        try mx.eval(updated);
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        // Enforce the loading-memory contract rather than silently accepting
        // an extra multi-GB copy on each chunk.
        if (address(updated) != before) return error.ResidentBufferWasNotDonated;
        destination.* = try mx.retain(updated);
    }
    pub fn exercise() !void {
        var r = Resident{};
        defer r.deinit();
        const staging = try mx.allocator.alloc(u32, 40);
        defer mx.allocator.free(staging);
        for (0..3) |part| {
            @memset(staging, if (part == 0) 1 else 0x3f803f80);
            r.groups[0][part] = try storage(4, part, r.format);
            var s = mx.Scope{};
            defer s.deinit();
            const added = try s.data(staging.ptr, &.{ 2, if (part == 0) 20 else 5 }, if (part == 0) mx.c.MLX_UINT32 else mx.bf16);
            try update(&r.groups[0][part], added, 0);
            try update(&r.groups[0][part], added, 2);
            try @import("sampling_checks.zig").equal(&s, try s.cat(&.{ added, added }, 0), r.groups[0][part]);
        }
    }
    pub fn init(tables: *const @import("ple_tables.zig").Tables) !Resident {
        var r = Resident{ .format = tables.format };
        errdefer r.deinit();
        // At most 10 MiB of host payload, plus one MLX input copy per update.
        const chunk_rows = 65536;
        const words: usize = @intCast(@divExact(160 * r.format.bits, 32));
        const groups: usize = @intCast(@divExact(160, r.format.group_size));
        const buffer = try mx.allocator.alloc(u32, chunk_rows * words);
        defer mx.allocator.free(buffer);
        var starts: [8]u32 = undefined;
        for (0..8) |group| {
            const first = group * 16;
            starts[group] = @intCast(tables.starts[first]);
            const rows: i32 = @intCast(tables.starts[first + 16] - tables.starts[first]);
            for (0..3) |part| {
                r.groups[group][part] = try storage(rows, part, r.format);
                for (first..first + 16) |shard| {
                    const ref = tables.rows[shard][part];
                    const width = try ref.tensor.rowBytes();
                    var row: usize = 0;
                    while (row < ref.tensor.dims[0]) {
                        const count = @min(chunk_rows, @as(usize, @intCast(ref.tensor.dims[0])) - row);
                        const bytes = std.mem.sliceAsBytes(buffer)[0 .. count * width];
                        try tables.files.items[ref.file].readRows(ref.tensor, row, count, bytes);
                        var s = mx.Scope{};
                        defer s.deinit();
                        const added = try s.data(bytes.ptr, &.{ @intCast(count), @intCast(if (part == 0) words else groups) }, if (part == 0) mx.c.MLX_UINT32 else mx.bf16);
                        try update(&r.groups[group][part], added, @intCast(tables.starts[shard] - tables.starts[first] + @as(i64, @intCast(row))));
                        r.writes += 1;
                        row += count;
                    }
                }
            }
            std.debug.print("Resident PLE: loaded group {d}/8\n", .{group + 1});
        }
        var s = mx.Scope{};
        defer s.deinit();
        r.starts = try mx.retain(try s.data(&starts, &.{8}, mx.c.MLX_UINT32));
        std.debug.print("Resident PLE: {d} exact allocation donations, {d} packed bytes\n", .{ r.writes, tables.starts[128] * @as(i64, @intCast(4 * (words + groups))) });
        return r;
    }
    pub fn gather(r: *const Resident, kernels: *mx.Kernels, s: *mx.Scope, ids: mx.Array) !mx.Array {
        if (ids.ctx == null or mx.dtype(ids) != mx.c.MLX_UINT32 or mx.shape(ids).len != 2 or mx.dim(ids, 1) != 16 or mx.dim(ids, 0) < 1 or mx.dim(ids, 0) > 16) return error.InvalidPLEIds;
        const rows = mx.dim(ids, 0);
        var inputs: [26]mx.Array = undefined;
        inputs[0] = ids;
        inputs[1] = r.starts;
        for (r.groups, 0..) |group, i| @memcpy(inputs[2 + 3 * i ..][0..3], &group);
        const q4 = r.format.bits == 4 and r.format.group_size == 32;
        const params = [_]mx.Template{ mx.ti("H", 16), mx.ti("DIMS", 160), mx.ti("BITS", r.format.bits), mx.ti("GS", r.format.group_size) };
        return (try kernels.run(s, if (q4) src.q4_ple_lookup else src.flash_qa_ple_lookup, &inputs, params[0..if (q4) @as(usize, 2) else 4], .{ 160, 16, rows }, .{ 160, 1, 1 }, &.{.{ .shape = &.{ rows, 2560 } }}))[0];
    }
};

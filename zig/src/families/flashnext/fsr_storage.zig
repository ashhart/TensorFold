//! Checkpoint descriptor owner and bounded Metal storage for exact expert and n-gram residency.
const std = @import("std");
const fz = @import("replay.zig");
const core = struct {
    const file = @import("../../core/fsr/file.zig");
    const reader = @import("../../core/fsr/reader.zig");
    const catalog = @import("../../core/fsr/catalog.zig");
    const arena = @import("../../core/fsr/arena.zig");
    const rows = @import("../../core/fsr/rows.zig");
};
const layout = @import("fsr_layout.zig");
pub const Source = struct {
    files: []const core.file.File,
    pub fn fileSize(s: Source, which: usize) !u64 {
        if (which >= s.files.len) return error.NoFile;
        return s.files[which].fileSize(0);
    }
    pub fn readAt(s: Source, which: usize, out: []u8, offset: u64) !usize {
        if (which >= s.files.len) return error.NoFile;
        return s.files[which].readAt(0, out, offset);
    }
};
pub const Arena = core.arena.Arena(Source);
pub const Builder = struct {
    r: *fz.Run,
    files: std.ArrayList(core.file.File) = .empty,
    pub fn span(b: *Builder, name: []const u8) !core.reader.Span {
        const e = try b.r.entry(name);
        const file = for (b.files.items, 0..) |f, i| {
            if (f.fd == e.fd) break i;
        } else blk: {
            const at = b.files.items.len;
            try b.files.append(b.r.arena, try core.file.File.borrow(e.fd));
            break :blk at;
        };
        const span_: core.reader.Span = .{ .file = file, .offset = e.at, .length = e.len };
        try core.reader.validate(span_, b.files.items[file].size, e.len);
        return span_;
    }
};

pub const Pool = struct {
    arena: Arena,
    buffers: [9]fz.Buf,
    map: fz.Buf,
    strides: fz.Buf,
    plan: layout.plan.Pool,

    pub fn init(r: *fz.Run, b: *Builder, mtp: []const u8, budget: u64) !Pool {
        const experts = try core.catalog.experts(r.arena, b, fz.LAYERS, 512, mtp);
        const parts: [9]u64 = experts.strides;
        const p = try layout.plan.pool(budget, parts, experts.spans.len, 512);
        var slabs: [9][]u8 = undefined;
        var buffers: [9]fz.Buf = undefined;
        for (&buffers, &slabs, experts.strides) |*buffer, *slab, stride| {
            const n = try std.math.mul(usize, p.slots, stride);
            buffer.* = .{ .b = try r.allocate(n) };
            try r.keep(buffer.b);
            slab.* = buffer.b.contents()[0..n];
        }
        const map: fz.Buf = .{ .b = try r.buffer(experts.spans.len * 4) };
        const strides: fz.Buf = .{ .b = try r.buffer(9 * 4) };
        for (experts.strides, 0..) |stride, i| strides.b.slice(u32, 9)[i] = @intCast(stride / @as(usize, if (i % 3 == 0) 4 else 2));
        const arena = try Arena.initMapped(r.arena, .{ .files = b.files.items }, experts.spans, experts.strides, slabs, map.b.slice(i32, experts.spans.len));
        r.loaded = try std.math.add(usize, r.loaded, @intCast(p.slabs));
        return .{ .arena = arena, .buffers = buffers, .map = map, .strides = strides, .plan = p };
    }

    pub fn component(p: *const Pool, projection: usize, component_: usize) fz.Buf {
        return p.buffers[projection * 3 + component_];
    }
};

pub const Ple = struct {
    table: core.rows.Table,
    buffers: [3]fz.Buf,
    starts: fz.Buf,
    ids: fz.Buf,

    pub fn init(r: *fz.Run, b: *Builder, width: ?[2]usize) !Ple {
        const w = width orelse .{ 6, 32 };
        const shards = try core.catalog.ple(r.arena, b, r.ngram, 128, 160, w[0], w[1]);
        const maximum = fz.PMAX * 16;
        var buffers: [3]fz.Buf = undefined;
        var slabs: [3][]u8 = undefined;
        for (&buffers, &slabs, 0..) |*buffer, *slab, i| {
            const n = try std.math.mul(usize, maximum, shards[0].parts[i].length / shards[0].rows);
            buffer.* = .{ .b = try r.allocate(n) };
            try r.keep(buffer.b);
            slab.* = buffer.b.contents()[0..n];
        }
        return .{
            .table = try core.rows.Table.init(r.arena, shards, maximum, slabs),
            .buffers = buffers,
            .starts = .{ .b = try r.buffer(8 * 4) },
            .ids = .{ .b = try r.buffer(maximum * 4) },
        };
    }

    /// The existing lookup kernel still executes its original code and all original rounding operations.
    pub fn gather(p: *Ple, source: Source, ids: fz.Buf, rows: usize) ![2 + 3 * fz.GROUPS]fz.Buf {
        const raw = @as([*]const u32, @ptrCast(@alignCast(ids.b.contents() + ids.off)))[0 .. rows * 16];
        try p.table.gather(source, raw);
        @memcpy(p.ids.b.slice(u32, rows * 16), p.table.inverse[0 .. rows * 16]);
        var tabs: [2 + 3 * fz.GROUPS]fz.Buf = undefined;
        tabs[0] = p.ids;
        tabs[1] = p.starts;
        for (0..fz.GROUPS) |g| for (0..3) |i| {
            tabs[2 + 3 * g + i] = p.buffers[i];
        };
        return tabs;
    }
};

/// Shared experts keep their original resident buffers; routed components alias only the persistent slabs.
pub fn expert(r: *fz.Run, projection: usize, part: usize) !?fz.Buf {
    const s = r.fsr orelse return null;
    const p = &(s.pool orelse return null);
    const at: usize = switch (projection) {
        0 => 0,
        1 => 1,
        4 => 2,
        else => return null,
    };
    return p.component(at, part);
}

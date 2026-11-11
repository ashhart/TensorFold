//! Metal-to-neutral export: a served-lane Packed slot as bf16/f32 parts, as `dsv41_unpack` reads it; no drafter rings.
const std = @import("std");
const c = @import("codec.zig");
const fp4 = @import("fp4.zig");
const plan = @import("plan.zig");
const metal = @import("metal.zig");
const ex = @import("export.zig");

pub const Error = error{ BadGeometry, BadShape, OutOfMemory, NoDrafterRings, TargetTooSmall };

/// All of shard 0's parts for `shape`, read from `p` (a slot after `shape.upto` tokens), in plan order.
pub fn build(arena: std.mem.Allocator, shape: plan.Shape, p: metal.Packed) Error![]ex.Part {
    if (shape.ds_rows != 0) return error.NoDrafterRings;
    const geo = shape.geo;
    if (shape.upto > p.capacity or p.live.len < p.liveBytes() or p.storage.len < metal.Packed.storageBytes(p.capacity)) return error.TargetTooSmall;
    const list = try plan.specs(arena, shape);
    const parts = try arena.alloc(ex.Part, list.len);
    for (list, parts) |spec, *part| {
        part.* = .{ .spec = spec, .bytes = try arena.alloc(u8, @intCast(spec.bytes())) };
        switch (spec.kind) {
            .swa => for (0..spec.rows) |i| {
                const pos = shape.upto - spec.rows + i;
                const at = (@as(usize, spec.layer) * metal.Packed.window + @as(usize, @intCast(pos % metal.Packed.window))) * metal.Packed.ring_stride;
                var row: [512]u16 = undefined;
                fp4.unpackWindow(p.live[at..][0..metal.Packed.ring_stride], &row);
                c.storeRow(512, &row, part.bytes[i * 1024 ..]);
            },
            .carry => carry(part.bytes, p, geo.sourceIndex(spec.layer).?, shape.upto),
            .comp, .ixk => {
                const src = geo.sourceIndex(spec.layer).?;
                var it = plan.partPages(shape, src, spec.kind, spec.first_page).?;
                var at: usize = 0;
                while (it.next()) |run| for (0..run.n) |k| {
                    const r: usize = @intCast(run.row + k);
                    if (spec.kind == .comp) {
                        var row: [512]u16 = undefined;
                        fp4.unpackMain(p.storage[p.mainOff(src) + r * metal.Packed.main_stride ..][0..metal.Packed.main_stride], &row);
                        c.storeRow(512, &row, part.bytes[at..]);
                        at += 1024;
                    } else {
                        var row: [128]u16 = undefined;
                        const base = p.mainOff(src) + @as(usize, p.capacity / metal.Packed.ratios[src]) * metal.Packed.main_stride;
                        fp4.unpackIndex(p.storage[base + r * metal.Packed.index_stride ..][0..metal.Packed.index_stride], &row);
                        c.storeRow(128, &row, part.bytes[at..]);
                        at += 256;
                    }
                };
            },
            .ring => unreachable, // ds_rows is 0
        }
    }
    return parts;
}

/// Position upto-1's raw row: the open group's slot if the prefix ends inside one, else the capture's (zeros if off).
fn carry(out: []u8, p: metal.Packed, src: usize, upto: u64) void {
    const slot: usize = @intCast((upto - 1) % 2);
    const from = if (upto % 2 == 1) p.partialOff(src) else p.captureOff(src) + @as(usize, @intCast((upto - 1) % metal.Packed.capture_rows)) * 2 * 2048;
    if (upto % 2 == 1) {
        @memcpy(out[0..2048], p.live[from + slot * 2048 ..][0..2048]);
        @memcpy(out[2048..4096], p.live[from + (2 + slot) * 2048 ..][0..2048]);
    } else {
        @memcpy(out[0..2048], p.live[from..][0..2048]);
        @memcpy(out[2048..4096], p.live[from + 2048 ..][0..2048]);
    }
}

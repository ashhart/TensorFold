//! DFly2 drafts for a served Hy4-flash stream, from the anchor row's final hidden.
const std = @import("std");
const mtl = @import("metal");
const target = @import("metal_model.zig");
const index = @import("metal_index.zig");
const serve = @import("serve_metal.zig");
const Metal = serve.Metal;
const Cache = serve.Cache;
const prompt_rows = serve.prompt_rows;
const draft_stride = serve.draft_stride;

pub fn run(self: *Metal, c: *Cache, anchor_row: u32, bonus: u32, depth: u32) !void {
    const w = self.draft.width;
    const count = @min(depth, w - 1);
    c.nheld = 0;
    if (count == 0) return;
    const p = c.len;
    if (p + w > self.capacity) return;
    self.bind(c);
    const ids = self.d_ids.slice(u32, w);
    @memset(ids, self.mask_token);
    ids[0] = bonus;
    const pos3 = self.d_positions.slice(i32, w * 3);
    const wr = self.d_writes.slice(i32, w);
    for (0..w) |i| {
        const at = p + @as(u32, @intCast(i));
        pos3[i * 3 ..][0..3].* = @splat(@intCast(at));
        wr[i] = @intCast(at);
    }
    // each block's keys: past 2,048 the anchor's recalled list in its target layer, then the draft's own rows
    for (self.draft.blocks, self.d_keys, self.d_counts) |block, kb, cb_| {
        const keys = kb.slice(u32, @as(usize, w) * draft_stride);
        const counts = cb_.slice(u32, w);
        var base: []const i32 = &.{};
        if (c.selected) {
            const all = self.ix_tokens[block.target_index].?.slice(i32, @as(usize, prompt_rows) * index.keys_a_row);
            const n: usize = @intCast(self.ix_counts[block.target_index].?.slice(i32, prompt_rows)[anchor_row]);
            base = all[@as(usize, anchor_row) * index.keys_a_row ..][0..n];
        }
        for (0..w) |i| {
            const at = p + @as(u32, @intCast(i));
            const row_keys = keys[i * draft_stride ..][0..draft_stride];
            var k: usize = 0;
            if (c.selected) {
                for (base) |slot| if (slot >= 0 and @as(u32, @intCast(slot)) < p) {
                    row_keys[k] = @intCast(slot);
                    k += 1;
                };
                for (p..at + 1) |slot| {
                    row_keys[k] = @intCast(slot);
                    k += 1;
                }
            } else {
                for (0..at + 1) |slot| row_keys[slot] = @intCast(slot);
                k = at + 1;
            }
            counts[i] = @intCast(k);
        }
    }
    const h = self.draft.hidden;
    const hidden = self.frame.common.normalized;
    const anchor: target.Ref = .{ .buffer = hidden.buffer, .offset = hidden.offset + @as(usize, anchor_row) * h * 2 };
    const pick = self.model.common.rows orelse return error.Hy4NeedsRowKernels;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = self.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    encode: {
        self.draft.encode(&self.model, e, &self.dframe, .{ .buffer = self.d_ids }, anchor, 1, self.d_caches) catch |err| {
            e.end();
            return err;
        };
        e.barrier();
        for (0..count) |col| {
            const previous: target.Ref = if (col == 0) .{ .buffer = self.d_ids } else .{ .buffer = self.d_tokens, .offset = (col - 1) * 4 };
            const logits = self.draft.next(&self.model, e, &self.dframe, previous, @intCast(col + 1)) catch |err| {
                e.end();
                return err;
            };
            e.barrier();
            pick.pick(e, logits, .{ .buffer = self.d_tokens }, @intCast(col), 1, self.vocab);
            e.barrier();
        }
        break :encode;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("hy4: draft pass failed: {s}", .{why});
        return error.Hy4GpuFailed;
    }
    @memcpy(c.held[0..count], self.d_tokens.slice(u32, count));
    c.nheld = count;
}

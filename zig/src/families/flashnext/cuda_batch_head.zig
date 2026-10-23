//! The shared batched head and last-row write-back, split without changing launch arguments.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const HcKeep = batch.HcKeep;
const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const vocab = wt.vocab;
const pss_n = batch.pss_n;
pub fn head(self: anytype, logits_host: ?[]u8) !void {
    const s = self.s;
    const n = self.rows;
    const mixer_keep: HcKeep = if (self.hooks.keep) |kp| kp.mixer else .{};
    switch (self.hooks.logits) {
        .none => {
            if (self.hooks.streams == 0 and mixer_keep.h == 0) return;
            try self.writeback(0, n);
            try self.copy(self.hooks.streams, s.h, n * wide * 2);
            if (mixer_keep.h != 0) try self.readoutKeep(self.m.mixer, 0, n, mixer_keep);
            return;
        },
        .all => {
            try self.writeback(0, n);
            try self.copy(self.hooks.streams, s.h, n * wide * 2);
            var row0: usize = 0;
            for (self.lanes) |l| {
                try self.copy(l.streams, s.h + row0 * wide * 2, l.rows * wide * 2);
                row0 += l.rows;
            }
            try self.readoutKeep(self.m.mixer, 0, n, mixer_keep);
            const out = if (self.hooks.logits_out != 0) self.hooks.logits_out else return error.UnexpectedTensor;
            try self.projectHead(out, n);
            row0 = 0;
            for (self.lanes) |l| {
                try self.copy(l.logits, out + row0 * vocab * 2, l.rows * vocab * 2);
                row0 += l.rows;
            }
            if (logits_host) |lh| {
                try self.complete();
                const d = self.stream.d;
                try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
            }
        },
        .last => {
            // the last row's streams, sums, gates and MoE slots moved to row 0, which keeps 16-byte alignment
            const r = n - 1;
            if (self.hooks.streams != 0) {
                // every row written back and kept, then the last row's streams and sums at row 0
                try self.writeback(0, n);
                try self.copy(self.hooks.streams, s.h, n * wide * 2);
                if (r > 0) {
                    try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                    try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                }
            } else if (r > 0) {
                try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                try self.copy(s.inj, s.inj + r * streams * 2, streams * 2);
                try self.copy(s.y, s.y + r * slots * dims * 4, slots * dims * 4);
                try self.copy(s.wts, s.wts + r * slots * 4, slots * 4);
                if (self.pending_f32) try self.copy(s.moe_f32, s.moe_f32 + r * dims * 4, dims * 4);
            }
            if (self.hooks.streams == 0) try self.writeback(0, 1);
            try self.readout(self.m.mixer, 0, 1);
            try self.projectHead(s.logits, 1);
            if (logits_host) |lh| {
                try self.complete();
                const d = self.stream.d;
                try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, s.logits, vocab * 2), "cuMemcpyDtoH");
            }
        },
    }
}

/// Column tiles keep their own reduction order; each gathered row restores global vocabulary order.
pub fn projectHead(self: anytype, out: u64, n: usize) !void {
    const s = self.s;
    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        return;
    }
    if (self.m.head.n != vocab / 2 or n > s.cap) return error.UnexpectedTensor;
    const c = self.m.collective orelse return error.TensorParallelUnavailable;
    _ = try self.lane(s.mixed, s.xsm, self.m.head, s.head_local, s.dummy, n, 1);
    for (0..n) |r| try c.gather(self.stream, s.head_local + r * vocab, out + r * vocab * 2, vocab / 2);
}

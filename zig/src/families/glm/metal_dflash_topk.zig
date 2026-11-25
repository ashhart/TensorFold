//! GLM's DFlash2 drafter top-k (glm_draft_topk.metal): core bf16_topk's rows, bytes and all, a threadgroup a row.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const contract = @import("core").bf16_topk_gpu.contract;

/// Threads a row's threadgroup (GLM_THREADS) and the entries a row holds (GLM_MAX_K, core's max_k).
pub const threads = 512;
comptime {
    if (contract.max_k != 16) @compileError("glm_draft_topk.metal's GLM_MAX_K is core bf16_topk's max_k");
}

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// Rows a launch takes: a batched pass's streams' blocks (eight of 16 rows), not only core's 16.
pub const max_rows = 128;

/// Rows this kernel takes: core's checks but its row count, a vocabulary of at least k, 16-bit counters' bins.
pub fn takes(p: contract.Params) bool {
    var one = p;
    one.rows = @min(p.rows, contract.max_rows);
    contract.check(one) catch return false;
    return p.rows <= max_rows and p.vocab >= p.k and p.vocab / (threads / 32) < 0x10000;
}

pub const TopK = struct {
    pipe: mtl.Pipeline,

    /// `edit` replaces one text of the source (a contract control's planted fault), else null.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, edit: ?[2][]const u8) !TopK {
        const text = if (edit) |x| blk: {
            if (std.mem.indexOf(u8, sources.glm_draft_topk, x[0]) == null) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, sources.glm_draft_topk, x[0], x[1]);
        } else try a.dupe(u8, sources.glm_draft_topk);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "glm_draft_topk", false);
        errdefer pipe.deinit();
        if (pipe.maxThreads() < threads or pipe.simdWidth() != 32) return error.UnsupportedDraftTopK;
        return .{ .pipe = pipe };
    }

    pub fn deinit(t: TopK) void {
        t.pipe.deinit();
    }

    /// Each row's top k of bf16 `logits` into `output`'s rows, as core's tf_bf16_topk writes them.
    pub fn encode(t: TopK, e: mtl.ComputeEncoder, logits: Ref, output: Ref, p: contract.Params) !void {
        if (!takes(p)) return error.BadTopKShape;
        const cells = @as(usize, p.rows - 1) * p.stride + p.vocab;
        if (logits.off % 2 != 0 or logits.off + cells * 2 > logits.buf.length() or output.off % 4 != 0 or output.off + @as(usize, p.rows) * @sizeOf(contract.Row) > output.buf.length()) return error.BadTopKBuffer;
        e.setPipeline(t.pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(output.buf, output.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(threads, 1, 1));
    }
};

test "the drafter's rows and a batch's fit; small, oversized or too many rows go back to core" {
    try std.testing.expect(takes(.{ .rows = 15, .vocab = 154880, .k = 16, .stride = 154880 }));
    try std.testing.expect(takes(.{ .rows = 48, .vocab = 154880, .k = 16, .stride = 154880 }));
    try std.testing.expect(!takes(.{ .rows = 129, .vocab = 154880, .k = 16, .stride = 154880 }));
    try std.testing.expect(!takes(.{ .rows = 15, .vocab = 8, .k = 16, .stride = 8 }));
    try std.testing.expect(!takes(.{ .rows = 1, .vocab = 2_000_000, .k = 16, .stride = 2_000_000 }));
}

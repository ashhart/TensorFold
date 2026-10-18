//! Shared GLM rank state descriptions, kept separate from engine operations.
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const sampler = @import("cuda_sampler.zig");

/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };

/// A kept prompt state as the prompt cache holds it: the id every rank keeps its own copy under, and its position.
pub const Kept = struct { id: u64, at: u32 };

/// A stream's own sequence: state, sampling rule, MTP drafts; buffers are shared, one stream's round at a time.
pub const Seq = struct {
    st: fw.State,
    sampling: ?sampler.Sampling = null, // null: greedy
    in_head: usize = 0, // drafted rows the MTP caches hold past the kept ones (st.mtp_drafted)
    rows_at: u64 = 0, // its last verify's final-norm rows (bf16 [rows][hidden]), which the next draft reads
};

/// One rank's copy of a kept prompt state: the state arena's bytes and the host fields beside them.
pub const Snap = struct { buf: cuda.DeviceBuffer, pos: usize, mtp_len: usize, cur: []u1, bases: [2]usize };

/// Where a prompt pass resumes: a kept state (every rank holds `id`) at `at`, a multiple of the chunk rows.
pub const Resume = struct { id: u64, at: usize };

/// Rank 0's marks in a prompt pass: `call` may keep the state at a chunk start; other ranks copy at that chunk's gate.
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };

const ranks = @import("cuda_ranks.zig");
const images_mod = @import("cuda_images.zig");

pub const Options = struct {
    capacity: usize, // tokens a sequence holds
    mtp: bool, // load the MTP head (drafts)
    settings: ranks.Settings,
    vision: ?images_mod.Limits = null, // --vision's limits (null: off): every rank must hold the rows rank 0 sends
    drafter: bool = false, // DFlash2 loads after rank agreement
};

//! Shared verify reductions have one completion; row checks and draws retain stream order.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const vocab = @import("cuda_prompt_wt.zig").vocab;
const environment = @import("sys").environment;
const full = @import("core").proposal_rows.Scores{ .vocab = vocab, .logical = vocab, .stride = vocab };
pub const rows_max = mtp.round_rows;
pub const vals_at = rows_max * 16;
pub const cols_at = vals_at + rows_max * draw.topk_max * 2;
pub const bytes = cols_at + rows_max * draw.topk_max * 4;

pub fn setting(raw: ?[]const u8) !bool {
    return environment.value(bool, "TF_FLASHNEXT_BATCH_DRAWS", raw, true);
}
pub fn enabled() !bool {
    return environment.read(bool, "TF_FLASHNEXT_BATCH_DRAWS", true);
}

pub const Batch = struct {
    device: cuda.DeviceBuffer,
    scores: cuda.DeviceBuffer,
    host: []align(8) u8,
    on: bool,
    pub fn init(a: std.mem.Allocator, driver: *cuda.Driver, on: bool, columns: usize) !Batch {
        var b = try cuda.DeviceBuffer.alloc(driver, bytes);
        errdefer b.free();
        var scores = try cuda.DeviceBuffer.alloc(driver, @import("cuda_draft_draw.zig").scoreBytes(columns));
        errdefer scores.free();
        return .{ .scores = scores, .device = b, .host = std.mem.sliceAsBytes(try a.alloc(u64, bytes / 8)), .on = on };
    }
    pub fn deinit(b: *Batch, a: std.mem.Allocator) void {
        b.scores.free();
        b.device.free();
        a.free(std.mem.bytesAsSlice(u64, b.host));
    }
    pub fn best(b: Batch, row: usize) [4]f32 {
        return std.mem.bytesToValue([4]f32, b.host[row * 16 ..][0..16]);
    }
    pub fn vals(b: Batch, row: usize, k: usize) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, b.host[vals_at + row * k * 2 ..][0 .. k * 2]));
    }
    pub fn cols(b: Batch, row: usize, k: usize) []const i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, b.host[cols_at + row * k * 4 ..][0 .. k * 4]));
    }
};

/// A different top_k falls back; a prefix of another K's column-ordered set would change the draw.
pub fn matching(sm: ?mtp.Sampling, masks: []const u32, k: usize) bool {
    return masks.len == 0 and (!mtp.isSampled(sm) or (draw.usable(sm.?) and sm.?.top_k == k));
}

pub fn drawsMany(h: *const mtp.Head, st: cuda.Stream, a: std.mem.Allocator, scores: u64, ds: []const *mtp.Draft, sms: []const ?mtp.Sampling, masks: []const []const u32, outs: []const []u32) !void {
    if (ds.len != sms.len or ds.len != masks.len or ds.len != outs.len) return error.ProposalScoreShape;
    const b = h.draw_batch;
    var rows: usize = 0;
    var k: usize = 0;
    var greedy = false;
    for (ds, sms, masks) |d, sm, mask| {
        rows += d.rows;
        if (mask.len != 0) continue;
        if (!mtp.isSampled(sm)) greedy = true else if (k == 0 and draw.usable(sm.?)) k = sm.?.top_k;
    }
    if (rows > rows_max) return error.ProposalScoreShape;
    if (b.on and (greedy or k != 0)) {
        if (greedy) try h.best(st, scores, vocab, vocab, rows, b.device.ptr);
        if (k != 0) try h.topk(st, scores, vocab, vocab, rows, k, b.device.ptr + vals_at, b.device.ptr + cols_at);
        try st.synchronize();
        try cuda.memory.download(st.d, b.device.ptr, b.host);
    }
    var row0: usize = 0;
    for (ds, sms, masks, outs) |d, sm, mask, out| {
        defer row0 += d.rows;
        if (!b.on or !matching(sm, mask, k)) {
            try draw.draws(h, st, a, d, sm, mask, out);
            continue;
        }
        var pending: [mtp.max_rows]u32 = undefined;
        if (mtp.isSampled(sm)) {
            for (pending[0..d.rows], 0..) |*tok, r| tok.* = draw.drawTopk(a, b.vals(row0 + r, k), b.cols(row0 + r, k), null, d.fed0 + 1 + r, sm.?) catch |err| {
                if (err == error.ProposalNonfinite) try draw.badCandidates(h, st, d, r);
                return err;
            };
        } else {
            for (0..d.rows) |r| _ = full.pickBest(b.best(row0 + r)) catch |err| {
                draw.badVerify(d, r, b.best(row0 + r), null, false);
                return err;
            };
            for (pending[0..d.rows], 0..) |*tok, r| tok.* = try full.pickBest(b.best(row0 + r));
        }
        @memcpy(out[0..d.rows], pending[0..d.rows]);
    }
}

test "Flash Next batch draws: the environment flag rejects bad values by name" {
    try std.testing.expect(try setting(null));
    try std.testing.expect(try setting("1"));
    try std.testing.expect(!try setting("0"));
    for ([_][]const u8{ "", "2", "true", "-1", " 1" }) |s| try std.testing.expectError(error.InvalidEnvironment, setting(s));
}

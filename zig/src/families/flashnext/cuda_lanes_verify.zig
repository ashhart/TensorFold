//! A round's drafted windows grouped into shared verify passes (cuda_lanes.zig's verifyFn).
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const be = lanes.backend;

/// Windows a shared pass takes at most (a round's streams).
const round_streams = 32;

/// The drafted windows verified in shared passes of up to `cap` rows (alone on two ranks), then drawn.
pub fn windows(self: anytype, wins: []const be.Window, out: []be.Verified, drafted: []const usize, row_ids: []const [mtp.max_rows]u32, cap: usize) !void {
    var seqs: [round_streams]*prompt.Seq = undefined;
    var ds: [round_streams]*mtp.Draft = undefined;
    var idss: [round_streams][]const u32 = undefined;
    var at: usize = 0;
    while (at < drafted.len) {
        var n: usize = 0;
        var rows: usize = 0;
        while (at + n < drafted.len and n < round_streams) : (n += 1) {
            const w = wins[drafted[at + n]];
            if (n > 0 and rows + w.rows() > cap) break;
            const l = try self.lane(w.stream);
            seqs[n] = &l.seq;
            ds[n] = &l.draft.?;
            idss[n] = row_ids[drafted[at + n]][0..w.rows()];
            rows += w.rows();
        }
        if (n == 1 or self.ranks != null or cap <= 1) {
            for (seqs[0..n], ds[0..n], idss[0..n]) |q, d, ids| try self.ops().verify(q, d, ids);
        } else try self.ops().verifyMany(seqs[0..n], ds[0..n], idss[0..n], self.logits.ptr);
        for (drafted[at..][0..n], ds[0..n]) |wi, d| try self.ops().draws(d, wins[wi].stream.sampling, out[wi].sampled[0..wins[wi].rows()]);
        at += n;
    }
}

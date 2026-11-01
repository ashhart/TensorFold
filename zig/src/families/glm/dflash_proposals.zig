//! Proposal membership is separate from context ingestion and travels identically to every CUDA rank.
const Req = @import("dflash.zig").Req;

pub fn active(reqs: []const Req, out: []Req) []const Req {
    var n: usize = 0;
    for (reqs) |q| if (q.propose) {
        out[n] = q;
        n += 1;
    };
    return out[0..n];
}

pub fn mask(reqs: []const Req) u32 {
    var bits: u32 = 0;
    for (reqs, 0..) |q, i| if (q.propose) {
        bits |= @as(u32, 1) << @intCast(i);
    };
    return bits;
}

pub fn enabled(bits: u32, i: usize) bool {
    return bits & (@as(u32, 1) << @intCast(i)) != 0;
}

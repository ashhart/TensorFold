//! Inkling 2's served geometries under the contract: Small at TP=2 (a rank's share) and Mini at one rank.

/// What a rank's kernels see of a model: widths, heads, router, vocabulary rows, log-scaling floor.
pub const Model = struct {
    mini: bool,
    hidden: usize,
    heads: usize, // query heads a rank
    kv: usize, // key heads a rank
    experts: usize,
    top: usize,
    ranks: usize, // the vocabulary's shares, one a rank
    floor: f64, // log_scaling_n_floor
    vocab: usize = 201_024,
    real: usize = 200_058, // unpadded_vocab_size

    pub fn group(m: Model) usize {
        return m.heads / m.kv;
    }

    /// The k/v rows' width (key heads x 128): the k/v convs' channels.
    pub fn kvWidth(m: Model) usize {
        return m.kv * 128;
    }

    /// The dynamic conv taps a row: 4 a key head (weight_proj's rows).
    pub fn taps(m: Model) usize {
        return m.kv * 4;
    }

    /// A rank's vocabulary columns.
    pub fn share(m: Model) usize {
        return m.vocab / m.ranks;
    }

    /// The last rank's real columns.
    pub fn lastReal(m: Model) usize {
        return m.real - (m.ranks - 1) * m.share();
    }
};

/// Small (config.json of Inkling-2-Small) at TP=2: hidden 4,608, 4 key heads and 24 query heads a rank, 384 experts.
pub const small: Model = .{ .mini = false, .hidden = 4608, .heads = 24, .kv = 4, .experts = 384, .top = 8, .ranks = 2, .floor = 128000 };
/// Mini (config.json of Inkling-2-Mini) at one rank: hidden 2,304, 24 query and 8 key heads, 128 experts, top 8.
pub const mini: Model = .{ .mini = true, .hidden = 2304, .heads = 24, .kv = 8, .experts = 128, .top = 8, .ranks = 1, .floor = 8192 };

pub const suffix = " at Mini's shapes";

/// A registration's name: Small's unchanged (its seeds hash the name), Mini's with the suffix.
pub fn named(comptime base: []const u8, mini_: bool) []const u8 {
    return if (mini_) base ++ suffix else base;
}

/// Entry `i` of a comptime name table, Mini's with the suffix.
pub fn namedAt(comptime table: []const []const u8, i: usize, mini_: bool) []const u8 {
    inline for (table, 0..) |n, j| if (i == j) return named(n, mini_);
    unreachable;
}

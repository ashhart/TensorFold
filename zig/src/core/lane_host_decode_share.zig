//! --decode-share's resolution: how the flag, the env and an engine's default combine into a host's share.
const fill = @import("lane_host_fill.zig");

/// --decode-share's share x: the prompt's p is its complement, floored at min_share; 0 keeps whole prompts.
pub fn fromDecode(x: f64) fill.Share {
    if (!(x > 0 and x < 1)) return .{};
    return .{ .p = @max(1 - x, fill.Share.min_share) };
}

/// The env's share, with --decode-share's p in its place when given; the row budget always comes from the env.
pub fn withDecode(decode_share: ?f64) fill.Share {
    var s = fill.Share.fromEnv();
    if (decode_share) |x| s.p = fromDecode(x).p;
    return s;
}

/// The flag wins; the env beats a default and never arms the cap; a default arms past Share.default_arm_rows only.
pub fn resolved(decode_share: ?f64, default_decode_share: ?f64) fill.Share {
    var s = withDecode(decode_share);
    if (decode_share == null) if (default_decode_share) |d| if (s.p >= 1 and !s.env_share) {
        s.p = fromDecode(d).p;
        s.defaulted = true;
    };
    return s;
}

/// Whether an explicit share (the flag or an engine default) offers the chunk cap; the env alone never does.
pub fn capped(decode_share: ?f64, default_decode_share: ?f64) bool {
    if (decode_share != null) return true;
    const s = withDecode(null);
    return default_decode_share != null and s.p >= 1 and !s.env_share;
}

//! A stream's shared head ceiling advances only after a fully verified prefix reaches its current ceiling.
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;

pub fn limit(cfg: *const Config, s: *const Stream) u32 {
    const base = if (cfg.shared_drafts > 0) cfg.shared_drafts else cfg.most_drafts;
    if (cfg.shared_draft_growth <= base) return base;
    return @min(cfg.most_drafts, cfg.shared_draft_growth, @max(base, s.shared_draft_limit orelse base));
}

pub fn observe(cfg: *const Config, s: *Stream, offered: usize, accepted: usize, alone: bool) void {
    if (alone or cfg.shared_drafts == 0 or cfg.shared_draft_growth <= cfg.shared_drafts or offered == 0) return;
    if (accepted < offered) {
        s.shared_draft_limit = null;
        return;
    }
    const current = limit(cfg, s);
    if (offered >= current) s.shared_draft_limit = @min(cfg.most_drafts, cfg.shared_draft_growth, current +| 1);
}

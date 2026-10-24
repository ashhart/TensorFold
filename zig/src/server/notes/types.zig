//! Snapshot schema shared by storage and the optional search index.
const std = @import("std");
pub const none_key: u32 = 0xFFFF_FFFF;
pub const max_n = 4;
pub const State = enum(u8) { current = 0, superseded = 1, tombstoned = 2 };
pub const Hop = enum { subject, subject_value, none };
pub const Settings = struct {
    resolve_subjects: bool = false,
    resolve_self: bool = true,
    cap: u32,
    n_min: u32,
    n_max: u32,
    fold: bool,
    hop: Hop,
    hop_weight: f64,
    hop_from: u32,
    hop_history: bool,
    min_rel: f64,
    subject: f64,
    punct: bool,
};
pub const LoadError = error{ BadSnapshot, Unsupported } || std.mem.Allocator.Error;

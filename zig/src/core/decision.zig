//! A packed decision: what the server asks, what the engine answers, and when the asker gives up.
const std = @import("std");

/// A packed decision: a trunk (pictures expand at `at`), a branch a question, and each branch's labels (`counts`).
pub const Decision = struct {
    trunk: []const u32,
    at: usize = 0,
    pictures: []const []const u8 = &.{}, // data URLs, in prompt order
    branches: []const []const u32 = &.{},
    counts: []const usize = &.{},
    labels: []const u32 = &.{},
    logits: []f64 = &.{},
    read: usize = 0, // out: the trunk's tokens with its pictures expanded, plus every branch's
    problem: []const u8 = "", // out: why the engine refused (error.Refused), in `arena`
    arena: ?std.mem.Allocator = null, // the request's, for what the engine hands back
    timeout_ms: u32 = 0, // 0: none; past it a queued decision leaves the queue, a running one stops at a pass
    gone: ?Gone = null, // the asker's client; once it is gone the decision is dropped the same way
    stop: std.atomic.Value(bool) = .init(false), // set by the asker; the engine stops between passes
    stopped: Stopped = .no, // out: why decide returned Failed without an answer
    pub const Gone = struct { ctx: *anyopaque, check: *const fn (ctx: *anyopaque) bool };
    pub const Stopped = enum { no, timed_out, gone };
};

//! The scoring request an engine decides in one pass: a trunk, its branches and their labels.
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
};

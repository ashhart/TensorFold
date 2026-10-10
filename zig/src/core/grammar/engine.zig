//! Grammar engine interfaces: `Compiler` -> `Compiled` -> `Matcher`s that walk tokens and fill allowed bits (xgr.zig).

const std = @import("std");

pub const Error = error{ Xgrammar, OutOfMemory };

/// What xgrammar compiles: a JSON schema (with ``max_whitespace_cnt``), a regex, EBNF (root ``root``) or a tag JSON.
pub const Source = enum { json_schema, regex, ebnf, structural_tag };

pub const Compiler = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        compile: *const fn (ptr: *anyopaque, kind: Source, text: []const u8, max_whitespace: ?u32) Error!Compiled,
    };

    pub fn compile(x: Compiler, kind: Source, text: []const u8, max_whitespace: ?u32) Error!Compiled {
        return x.vtable.compile(x.ptr, kind, text, max_whitespace);
    }
};

pub const Compiled = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        matcher: *const fn (ptr: *anyopaque) Error!Matcher,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    /// A fresh matcher at the grammar's start.
    pub fn matcher(g: Compiled) Error!Matcher {
        return g.vtable.matcher(g.ptr);
    }
    pub fn deinit(g: Compiled) void {
        g.vtable.deinit(g.ptr);
    }
};

/// One reply's walk through a grammar.
pub const Matcher = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        accept: *const fn (ptr: *anyopaque, token: u32) Error!bool,
        /// The next token's allowed bits (token t: bit t % 32 of word t / 32), one bitmask row.
        fill: *const fn (ptr: *anyopaque, words: []u32) Error!void,
        rollback: *const fn (ptr: *anyopaque, n: u32) Error!void,
        terminated: *const fn (ptr: *anyopaque) bool,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn accept(m: Matcher, token: u32) Error!bool {
        return m.vtable.accept(m.ptr, token);
    }
    pub fn fill(m: Matcher, words: []u32) Error!void {
        return m.vtable.fill(m.ptr, words);
    }
    pub fn rollback(m: Matcher, n: u32) Error!void {
        return m.vtable.rollback(m.ptr, n);
    }
    pub fn terminated(m: Matcher) bool {
        return m.vtable.terminated(m.ptr);
    }
    pub fn deinit(m: Matcher) void {
        m.vtable.deinit(m.ptr);
    }
};

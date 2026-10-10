//! xgrammar's C++ core through xgr_c.h, implementing engine.zig's interfaces: masks are xgrammar's own, bit for bit.

const std = @import("std");
const engine = @import("engine.zig");

/// xgr_c.h's declarations (xgr_c.cc).
const c = struct {
    const xgr_tok = opaque {};
    const xgr_compiler = opaque {};
    const xgr_compiled = opaque {};
    const xgr_matcher = opaque {};
    const XGR_JSON_SCHEMA: i32 = 0;
    const XGR_REGEX: i32 = 1;
    const XGR_EBNF: i32 = 2;
    const XGR_STRUCTURAL_TAG: i32 = 3;
    extern fn xgr_last_error() [*:0]const u8;
    extern fn xgr_detect_metadata(backend: [*]const u8, len: usize, out: [*]u8, cap: usize) i64;
    extern fn xgr_tok_new(vocab: [*]const [*]const u8, lens: [*]const usize, n: i32, vocab_type: i32, vocab_size: i32, stops: [*]const i32, n_stops: i32, add_prefix_space: i32) ?*xgr_tok;
    extern fn xgr_tok_free(t: *xgr_tok) void;
    extern fn xgr_tok_special(t: *const xgr_tok, out: ?[*]i32, cap: i32) i32;
    extern fn xgr_bitmask_words(vocab_size: i32) i32;
    extern fn xgr_compiler_new(t: *const xgr_tok, max_threads: i32, cache_bytes: i64) ?*xgr_compiler;
    extern fn xgr_compiler_free(x: *xgr_compiler) void;
    extern fn xgr_compile(x: *xgr_compiler, kind: i32, text: [*]const u8, len: usize, max_whitespace: i32) ?*xgr_compiled;
    extern fn xgr_compiled_free(g: *xgr_compiled) void;
    extern fn xgr_matcher_new(g: *const xgr_compiled) ?*xgr_matcher;
    extern fn xgr_matcher_free(m: *xgr_matcher) void;
    extern fn xgr_matcher_accept(m: *xgr_matcher, token: i32) i32;
    extern fn xgr_matcher_fill(m: *xgr_matcher, words: [*]i32, n_words: i32) i32;
    extern fn xgr_matcher_rollback(m: *xgr_matcher, n: i32) i32;
    extern fn xgr_matcher_terminated(m: *const xgr_matcher) i32;
};

pub const Error = error{ Xgrammar, OutOfMemory };

/// xgrammar's message for the calling thread's last failure.
pub fn lastError() []const u8 {
    return std.mem.span(c.xgr_last_error());
}

pub fn bitmaskWords(vocab: u32) u32 {
    return @intCast(c.xgr_bitmask_words(@intCast(vocab)));
}

pub const Metadata = struct { vocab_type: i32, add_prefix_space: bool };

/// TokenizerInfo._detect_metadata_from_hf over a tokenizer.json text.
pub fn detectMetadata(gpa: std.mem.Allocator, tokenizer_json: []const u8) !Metadata {
    var buf: [256]u8 = undefined;
    const n = c.xgr_detect_metadata(tokenizer_json.ptr, tokenizer_json.len, &buf, buf.len);
    if (n < 0) return error.Xgrammar;
    const M = struct { vocab_type: i32, add_prefix_space: bool };
    const p = try std.json.parseFromSlice(M, gpa, buf[0..@intCast(n)], .{});
    defer p.deinit();
    return .{ .vocab_type = p.value.vocab_type, .add_prefix_space = p.value.add_prefix_space };
}

pub const Tokenizer = struct {
    h: *c.xgr_tok,

    /// TokenizerInfo(encoded_vocab, vocab_type, vocab_size, stop_token_ids, add_prefix_space).
    pub fn init(gpa: std.mem.Allocator, vocab: []const []const u8, meta: Metadata, stops: []const i32) !Tokenizer {
        const ptrs = try gpa.alloc([*]const u8, vocab.len);
        defer gpa.free(ptrs);
        const lens = try gpa.alloc(usize, vocab.len);
        defer gpa.free(lens);
        for (vocab, ptrs, lens) |v, *p, *l| {
            p.* = v.ptr;
            l.* = v.len;
        }
        const h = c.xgr_tok_new(ptrs.ptr, lens.ptr, @intCast(vocab.len), meta.vocab_type, @intCast(vocab.len), stops.ptr, @intCast(stops.len), @intFromBool(meta.add_prefix_space)) orelse return error.Xgrammar;
        return .{ .h = h };
    }

    pub fn deinit(t: Tokenizer) void {
        c.xgr_tok_free(t.h);
    }

    /// TokenizerInfo.special_token_ids (gpa-owned).
    pub fn special(t: Tokenizer, gpa: std.mem.Allocator) ![]i32 {
        const n = c.xgr_tok_special(t.h, null, 0);
        const out = try gpa.alloc(i32, @intCast(n));
        _ = c.xgr_tok_special(t.h, out.ptr, n);
        return out;
    }
};

/// One tokenizer view's GrammarCompiler (max_threads 8, its cache capped at `cache_bytes`), as `engine.Compiler`.
pub const Compiler = struct {
    h: *c.xgr_compiler,
    vocab: u32,
    /// compiles come from HTTP threads (rare, not on the round loop): a spin lock is enough
    mutex: std.atomic.Mutex = .unlocked,

    pub fn init(t: Tokenizer, vocab: u32, cache_bytes: i64) !Compiler {
        return .{ .h = c.xgr_compiler_new(t.h, 8, cache_bytes) orelse return error.Xgrammar, .vocab = vocab };
    }

    pub fn deinit(x: *Compiler) void {
        c.xgr_compiler_free(x.h);
    }

    pub fn compiler(x: *Compiler) engine.Compiler {
        return .{ .ptr = x, .vtable = &.{ .compile = compileFn } };
    }

    /// `kind`'s grammar from `text`; on failure null with xgrammar's message copied into `why` (gpa-owned).
    pub fn compile(x: *Compiler, kind: engine.Source, text: []const u8, max_whitespace: ?u32) Error!*Compiled {
        while (!x.mutex.tryLock()) std.atomic.spinLoopHint();
        defer x.mutex.unlock();
        const k: i32 = switch (kind) {
            .json_schema => c.XGR_JSON_SCHEMA,
            .regex => c.XGR_REGEX,
            .ebnf => c.XGR_EBNF,
            .structural_tag => c.XGR_STRUCTURAL_TAG,
        };
        const ws: i32 = if (max_whitespace) |w| @intCast(w) else -1;
        const g = c.xgr_compile(x.h, k, text.ptr, text.len, ws) orelse return error.Xgrammar;
        const out = std.heap.c_allocator.create(Compiled) catch {
            c.xgr_compiled_free(g);
            return error.OutOfMemory;
        };
        out.* = .{ .h = g, .vocab = x.vocab };
        return out;
    }

    fn compileFn(ptr: *anyopaque, kind: engine.Source, text: []const u8, max_whitespace: ?u32) Error!engine.Compiled {
        const x: *Compiler = @ptrCast(@alignCast(ptr));
        return (try x.compile(kind, text, max_whitespace)).compiled();
    }
};

pub const Compiled = struct {
    h: *c.xgr_compiled,
    vocab: u32,

    pub fn compiled(g: *Compiled) engine.Compiled {
        return .{ .ptr = g, .vtable = &.{ .matcher = matcherFn, .deinit = deinitFn } };
    }

    fn matcherFn(ptr: *anyopaque) Error!engine.Matcher {
        const g: *Compiled = @ptrCast(@alignCast(ptr));
        const m = std.heap.c_allocator.create(Matcher) catch return error.OutOfMemory;
        m.* = .{ .h = c.xgr_matcher_new(g.h) orelse {
            std.heap.c_allocator.destroy(m);
            return error.Xgrammar;
        }, .words = bitmaskWords(g.vocab) };
        return m.matcher();
    }

    fn deinitFn(ptr: *anyopaque) void {
        const g: *Compiled = @ptrCast(@alignCast(ptr));
        c.xgr_compiled_free(g.h);
        std.heap.c_allocator.destroy(g);
    }
};

/// GrammarMatcher(compiled) with xgrammar's defaults (the compiled grammar's stop tokens, unlimited rollback).
pub const Matcher = struct {
    h: *c.xgr_matcher,
    words: u32,

    pub fn matcher(m: *Matcher) engine.Matcher {
        return .{ .ptr = m, .vtable = &vtable };
    }

    const vtable: engine.Matcher.VTable = .{ .accept = accept, .fill = fill, .rollback = rollback, .terminated = terminated, .deinit = deinit };

    fn self(p: *anyopaque) *Matcher {
        return @ptrCast(@alignCast(p));
    }
    fn accept(p: *anyopaque, token: u32) Error!bool {
        const r = c.xgr_matcher_accept(self(p).h, @intCast(token));
        if (r < 0) return error.Xgrammar;
        return r == 1;
    }
    fn fill(p: *anyopaque, words: []u32) Error!void {
        const m = self(p);
        if (words.len != m.words) return error.Xgrammar;
        if (c.xgr_matcher_fill(m.h, @ptrCast(words.ptr), @intCast(words.len)) < 0) return error.Xgrammar;
    }
    fn rollback(p: *anyopaque, n: u32) Error!void {
        if (c.xgr_matcher_rollback(self(p).h, @intCast(n)) < 0) return error.Xgrammar;
    }
    fn terminated(p: *anyopaque) bool {
        return c.xgr_matcher_terminated(self(p).h) == 1;
    }
    fn deinit(p: *anyopaque) void {
        const m = self(p);
        c.xgr_matcher_free(m.h);
        std.heap.c_allocator.destroy(m);
    }
};

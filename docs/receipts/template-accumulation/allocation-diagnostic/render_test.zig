//! Render an external template against exact output and provenance evidence.
const std = @import("std");
const template = @import("template.zig");
const testing = std.testing;
const diag = @import("diagnostics.zig");
const v = @import("value.zig");

const limit = 240 * 1024 * 1024;

const Peak = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    free_calls: usize = 0,

    fn allocator(self: *Peak) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn allowed(self: *const Peak, old: usize, new: usize) bool {
        return new <= limit - (self.live - old);
    }

    fn account(self: *Peak, old: usize, new: usize) void {
        self.live = self.live - old + new;
        self.peak = @max(self.peak, self.live);
    }

    fn alloc(ctx: *anyopaque, n: usize, align_: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(0, n)) {
            std.debug.print("CAP live={d} request={d}\n", .{ self.live, n });
            return null;
        }
        const ptr = self.child.rawAlloc(n, align_, ret) orelse return null;
        self.account(0, n);
        return ptr;
    }

    fn resize(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, n: usize, ret: usize) bool {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(memory.len, n)) return false;
        if (!self.child.rawResize(memory, align_, n, ret)) return false;
        self.account(memory.len, n);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        if (!self.allowed(memory.len, n)) return null;
        const ptr = self.child.rawRemap(memory, align_, n, ret) orelse return null;
        self.account(memory.len, n);
        return ptr;
    }

    fn free(ctx: *anyopaque, memory: []u8, align_: std.mem.Alignment, ret: usize) void {
        const self: *Peak = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, align_, ret);
        self.free_calls += 1;
        self.account(memory.len, 0);
    }
};

fn environment(name: [*:0]const u8) ![]const u8 {
    const value = std.c.getenv(name) orelse return error.MissingEnvironment;
    return std.mem.span(value);
}

test "external template case" {
    const a = testing.allocator;
    const source = try std.Io.Dir.cwd().readFileAlloc(testing.io, try environment("TF_TEMPLATE"), a, .limited(1 << 20));
    defer a.free(source);
    const case_json = try std.Io.Dir.cwd().readFileAlloc(testing.io, try environment("TF_CASE"), a, .limited(32 << 20));
    defer a.free(case_json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, case_json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const messages = obj.get("messages") orelse return error.MissingMessages;
    const tools = obj.get("tools") orelse .null;
    const context = obj.get("context") orelse .null;
    const generation = (obj.get("add_generation_prompt") orelse return error.MissingGeneration).bool;
    const expected = (obj.get("expected") orelse return error.MissingExpected).string;
    var tracked: Peak = .{ .child = a };
    var compiled = template.compile(tracked.allocator(), source, null) catch |err| {
        if (tracked.live != 0) return error.ScratchNotReleased;
        std.debug.print("COMPILE_ERROR {s} live={d} peak={d}\n", .{ @errorName(err), tracked.live, tracked.peak });
        return err;
    };
    defer compiled.deinit();
    const standing = tracked.live;
    std.debug.print("COMPILE standing={d} peak={d}\n", .{ standing, tracked.peak });
    diag.reset();
    std.debug.print("SIZES str={d} value={d}\n", .{ @sizeOf(v.Str), @sizeOf(v.Value) });
    for (0..1) |rep| {
        tracked.peak = tracked.live;
        const result = template.renderProvenance(a, &compiled, messages, tools, context, generation, null) catch |err| {
            if (tracked.live != standing) return error.ScratchNotReleased;
            diag.print();
            std.debug.print("RENDER_ERROR {s} rep={d} peak={d}\n", .{ @errorName(err), rep, tracked.peak - standing });
            return err;
        };
        diag.print();
        defer a.free(result.s);
        defer a.free(result.spans);
        if (tracked.live != standing) return error.ScratchNotReleased;
        if (!std.mem.eql(u8, expected, result.s)) {
            std.debug.print("MISMATCH expected_bytes={d} actual_bytes={d}\n", .{ expected.len, result.s.len });
            return error.ExpectedBytesDiffer;
        }
        var previous: usize = 0;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        for (result.spans) |span| {
            if (span[0] < previous or span[0] >= span[1] or span[1] > result.s.len) return error.InvalidSpan;
            var buf: [64]u8 = undefined;
            hash.update(try std.fmt.bufPrint(&buf, "{d}:{d}\n", .{ span[0], span[1] }));
            previous = span[1];
        }
        if (obj.get("expected_request_spans")) |expected_spans| {
            const want = expected_spans.array.items;
            var actual_idx: usize = 0;
            var want_idx: usize = 0;
            var want_end: usize = 0;
            for (want) |item| {
                const pair = item.array.items;
                if (pair.len != 2) return error.InvalidExpectedSpan;
                const start: usize = @intCast(pair[0].integer);
                const end: usize = @intCast(pair[1].integer);
                if (start < want_end or start >= end or end > result.s.len) return error.InvalidExpectedSpan;
                want_end = end;
            }
            for (0..result.s.len) |i| {
                while (actual_idx < result.spans.len and result.spans[actual_idx][1] <= i) : (actual_idx += 1) {}
                while (want_idx < want.len and @as(usize, @intCast(want[want_idx].array.items[1].integer)) <= i) : (want_idx += 1) {}
                const actual = actual_idx < result.spans.len and result.spans[actual_idx][0] <= i;
                const target = want_idx < want.len and @as(usize, @intCast(want[want_idx].array.items[0].integer)) <= i;
                if (actual != target) return error.ExpectedProvenanceDiffers;
            }
        }
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        const hex = std.fmt.bytesToHex(digest, .lower);
        std.debug.print("RENDER rep={d} scratch_peak={d} bytes={d} spans={d} spans_sha256={s}\n", .{
            rep, tracked.peak - standing, result.s.len, result.spans.len, hex,
        });
    }
}

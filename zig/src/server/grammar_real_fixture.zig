//! Real CPU grammar runtime for explicit checkpoint fixtures; no GPU or model weights.
const std = @import("std");
const sys = @import("sys");
const grammar = @import("lanes").grammar;
const options = @import("grammar_real_budget");
const t = std.testing;
const Limit = *const fn (*anyopaque, i64) callconv(.c) void;

pub const Lib = struct {
    f: grammar.Library,
    h: *anyopaque,
    limit: ?Limit,
    pub fn open() !Lib {
        // The production loader retains the runtime through process exit; detached workers use its code.
        var dl = try sys.DynLib.open(options.library);
        var f: grammar.Library = undefined;
        const info = @typeInfo(grammar.Library).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(f, name) = if (@typeInfo(T) == .optional) dl.lookup(@typeInfo(T).optional.child, "tfg_" ++ name) else dl.lookup(T, "tfg_" ++ name) orelse return error.MissingSymbol;
        }
        const tokenizer = try std.Io.Dir.cwd().readFileAlloc(t.io, options.tokenizer, t.allocator, .limited(256 << 20));
        defer t.allocator.free(tokenizer);
        var err: [512]u8 = @splat(0);
        const stops = [_]i32{248046};
        const h = f.open(tokenizer.ptr, tokenizer.len, 248320, &stops, 1, &err, err.len) orelse {
            std.debug.print("real tokenizer open: {s}\n", .{std.mem.sliceTo(&err, 0)});
            return error.GrammarSetup;
        };
        return .{ .f = f, .h = h, .limit = dl.lookup(Limit, "tfg_budget_limits") };
    }
    pub fn close(l: Lib) void {
        if (l.f.compiling) |active| while (active(l.h) != 0) {
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        };
        l.f.close(l.h);
    }
    pub fn compile(l: Lib, kind: grammar.Kind, text: []const u8) !grammar.Compiled {
        var err: [512]u8 = @splat(0);
        const g = l.f.compile(l.h, @backingInt(kind), text.ptr, text.len, &err, err.len) orelse {
            std.debug.print("real grammar compile: {s}\n", .{std.mem.sliceTo(&err, 0)});
            return error.GrammarCompile;
        };
        return .{ .f = l.f, .h = g };
    }
};

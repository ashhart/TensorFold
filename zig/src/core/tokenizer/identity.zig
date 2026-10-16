//! The tokenizer's identity: a digest of the vocabulary, the merges and the added tokens as loaded.
const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const Tokenizer = tokenizer.Tokenizer;
const Allocator = std.mem.Allocator;

/// A digest of every id's token, the merges in rank order and the added tokens with their flags, as loaded.
pub fn identity(t: *const Tokenizer, a: Allocator) Allocator.Error![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("tensorfold tokenizer identity 1");
    for (t.id_to_token.tokens.items) |token| {
        const n: u32 = if (token) |x| @intCast(x.len) else std.math.maxInt(u32);
        h.update(std.mem.asBytes(&n));
        if (token) |x| h.update(x);
    }
    const merges = try a.alloc([4]u32, t.model.merges.count()); // rank, left, right, merged
    defer a.free(merges);
    var it = t.model.merges.iterator();
    var n: usize = 0;
    while (it.next()) |e| : (n += 1) {
        const pair = e.key_ptr.*;
        merges[n] = .{ e.value_ptr.rank, @truncate(pair >> 32), @truncate(pair), e.value_ptr.id };
    }
    std.mem.sort([4]u32, merges, {}, struct {
        fn lt(_: void, x: [4]u32, y: [4]u32) bool {
            return x[0] < y[0];
        }
    }.lt);
    h.update(std.mem.sliceAsBytes(merges));
    for (t.added) |x| {
        const head = [_]u32{
            x.id,
            @intCast(x.content.len),
            @intFromBool(x.special),
            @intFromBool(x.lstrip),
            @intFromBool(x.rstrip),
            @intFromBool(x.normalized),
        };
        h.update(std.mem.asBytes(&head));
        h.update(x.content);
    }
    return h.finalResult();
}

test "identity follows the vocabulary, the merges and the added tokens as loaded, not the file's spelling" {
    const head =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,
        \\ "lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"",
        \\ "end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    ;
    const plain =
        \\"vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    // the same vocabulary and merges, in another order and spelling
    const respelled =
        \\"vocab": {"b":1, "a":0,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":["Ġ a",["a","b"],"Ġa b"]}}
    ;
    // the first two merges' ranks swapped
    const reranked =
        \\"vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":["a b",["Ġ","a"],"Ġa b"]}}
    ;
    // one token's spelling changed
    const renamed =
        \\"vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"d":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    const cases = [_]struct { tail: []const u8, same: bool }{
        .{ .tail = plain, .same = true },
        .{ .tail = respelled, .same = true },
        .{ .tail = reranked, .same = false },
        .{ .tail = renamed, .same = false },
    };
    var first: ?[32]u8 = null;
    for (cases) |c| {
        const json = try std.mem.concat(a, u8, &.{ head, c.tail });
        defer a.free(json);
        var t = try tokenizer.parse(a, json);
        defer t.deinit();
        const id = try identity(&t, a);
        if (first) |f| try std.testing.expectEqual(c.same, std.mem.eql(u8, &f, &id)) else first = id;
    }
}

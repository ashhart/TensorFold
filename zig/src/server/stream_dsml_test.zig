//! DSML tool-call envelopes, whole and streamed: a nested call stays an argument or text, never a call.
const std = @import("std");
const json = @import("root.zig").json;
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;

/// One spelling's envelope around a ``q`` value, and the closers of a block whose invoke never closes.
const Spelling = struct { prefix: []const u8, suffix: []const u8, unclosed: []const u8 };

/// DSML's ``tool_calls`` block, and DeepSeek-V4.1's that its template teaches: ``calls``, a space before each name.
const spellings = [_]Spelling{
    .{
        .prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">",
        .suffix = "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        .unclosed = "</｜DSML｜parameter></｜DSML｜tool_calls>",
    },
    .{
        .prefix = "<｜DSML｜ calls><｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"q\" string=\"true\">",
        .suffix = "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        .unclosed = "</｜DSML｜ parameter></｜DSML｜ calls>",
    },
};
const dsml_body = "{\"model\":\"test-model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";

test "DSML literal calls and adjacent XML calls agree in whole and streamed consumer replies" {
    const literals = [_][]const u8{
        "<tool_call><function=lookup></function></tool_call>",
        "<tool_call><function=lookup></function>",
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const adjacent = "<tool_call><function=lookup><parameter=n>7</parameter></function></tool_call>";
    for (spellings) |sp| for (literals) |literal| for ([_]bool{ false, true }) |with_adjacent| {
        const raw = try std.mem.concat(a, u8, &.{ sp.prefix, literal, sp.suffix, if (with_adjacent) adjacent else "" });
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |single| {
            const expected = try std.mem.concat(a, u8, &.{
                "{\"q\":", try json.quote(a, literal, .{}), "}", if (with_adjacent and !single) "{\"n\":\"7\"}" else "",
            });
            const body = try std.mem.concat(a, u8, &.{ dsml_body, if (single) ",\"parallel_tool_calls\":false" else "" });
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
                const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, if (with_adjacent and !single) 2 else 1), whole.calls);
                try std.testing.expectEqual(whole.calls, streamed.calls);
                try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqualStrings("", whole.content());
                try std.testing.expectEqualStrings("", streamed.content());
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    };
}

test "unfinished and malformed DSML never emit their complete nested call in consumer replies" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (spellings) |sp| {
        const prefix = try std.mem.concat(a, u8, &.{ sp.prefix, "<tool_call><function=lookup></function></tool_call>" });
        for ([_][]const u8{ prefix, try std.mem.concat(a, u8, &.{ prefix, sp.unclosed }) }) |raw| {
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 0), whole.calls);
                try std.testing.expectEqual(@as(usize, 0), streamed.calls);
                try std.testing.expectEqual(@as(usize, 0), streamed.arguments_len);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    }
}

test "DSML and adjacent existing non-XML envelopes preserve whole and SSE calls without duplicates" {
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"adjacent-json\"}}</tool_call>", "{\"q\":\"adjacent-json\"}" },
        .{ "<tool_call>lookup<arg_key>q</arg_key><arg_value>adjacent-glm</arg_value></tool_call>", "{\"q\":\"adjacent-glm\"}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>adjacent-gemma<|\"|>}<tool_call|>", "{\"q\":\"adjacent-gemma\"}" },
        .{ "<tools:tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"adjacent-namespaced\"}}</tools:tool_call>", "{\"q\":\"adjacent-namespaced\"}" },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (spellings) |sp| for (cases) |case| for ([_]bool{ false, true }) |outer_first| {
        const outer = try std.mem.concat(a, u8, &.{ sp.prefix, "outer", sp.suffix });
        const raw = try std.mem.concat(a, u8, if (outer_first) &.{ outer, case[0] } else &.{ case[0], outer });
        const expected = try std.mem.concat(a, u8, if (outer_first)
            &.{ "{\"q\":\"outer\"}", case[1] }
        else
            &.{ case[1], "{\"q\":\"outer\"}" });
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
            const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expect(streamed.opened and streamed.status == null);
            try std.testing.expectEqual(@as(usize, 2), whole.calls);
            try std.testing.expectEqual(@as(usize, 2), streamed.calls);
            try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
            try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
            try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
        }
    };
}

test "a supported bare JSON call containing a literal DSML opener agrees in whole and SSE replies" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "<｜DSML｜tool_calls>", "<｜DSML｜ calls>" }) |opener| {
        const call = try std.mem.concat(a, u8, &.{ "{\"name\":\"lookup\",\"arguments\":{\"q\":\"", opener, "\"}}" });
        const replies = [_][]const u8{ call, try std.mem.concat(a, u8, &.{ "[", call, "]" }), try std.mem.concat(a, u8, &.{ "```json\n", call, "\n```" }) };
        const expected = try std.mem.concat(a, u8, &.{ "{\"q\":\"", opener, "\"}" });
        for (replies) |raw| {
            const tokens = try TestText.encode(undefined, a, raw, false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 1), whole.calls);
                try std.testing.expectEqual(@as(usize, 1), streamed.calls);
                try std.testing.expectEqualStrings(expected, whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(expected, streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    }
}

test "DSML spellings mixed in one block stay text in whole and streamed consumer replies" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v41 = spellings[1];
    const mixed = [_][]const u8{
        try std.mem.concat(a, u8, &.{ v41.prefix, "a", spellings[0].suffix }),
        try std.mem.concat(a, u8, &.{ spellings[0].prefix, "a", v41.suffix }),
        "<｜DSML｜ calls><｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">a</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜ calls>",
    };
    for (mixed) |raw| {
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
            const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
            try std.testing.expectEqual(@as(usize, 0), whole.calls);
            try std.testing.expectEqual(@as(usize, 0), streamed.calls);
            try std.testing.expectEqualStrings(raw[0..@min(raw.len, whole.text.len)], whole.content());
            try std.testing.expectEqualStrings(whole.content(), streamed.content());
            try std.testing.expectEqual(.stop, whole.finish.?);
            try std.testing.expectEqual(.stop, streamed.finish.?);
        }
    }
}

test "a <tool_call> envelope holding only a DSML opener is text, not a panic, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "<｜DSML｜tool_calls>", "<｜DSML｜ calls>" }) |opener| for ([_][]const u8{ "", "x" }) |tail| {
        const raw = try std.mem.concat(a, u8, &.{ "<tool_call>", opener, tail, "</tool_call>" });
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
            const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expectEqual(@as(usize, 0), whole.calls);
            try std.testing.expectEqual(@as(usize, 0), streamed.calls);
            try std.testing.expectEqualStrings(raw, whole.content());
        }
    };
}

test "closed rejected calls and literal less-than prose preserve later consumer calls" {
    const adjacent = "<tool_call><function=lookup><parameter=q>adjacent</parameter></function></tool_call>";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (spellings) |sp| {
        const outer = try std.mem.concat(a, u8, &.{ sp.prefix, "outer", sp.suffix });
        const cases = [_][2][]const u8{
            .{ try std.mem.concat(a, u8, &.{ outer, "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>", adjacent }), "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ try std.mem.concat(a, u8, &.{ outer, "<tool_call><function=other></function></tool_call>", adjacent }), "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ try std.mem.concat(a, u8, &.{ outer, "<tool_call></tool_call>", adjacent }), "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ try std.mem.concat(a, u8, &.{ outer, "<tool_call><not_a_function></not_a_function></tool_call>", adjacent }), "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
            .{ adjacent ++ "1 < 2 " ++ adjacent, "{\"q\":\"adjacent\"}{\"q\":\"adjacent\"}" },
            .{ try std.mem.concat(a, u8, &.{ "1 < 2 ", outer, adjacent }), "{\"q\":\"outer\"}{\"q\":\"adjacent\"}" },
        };
        for (cases) |case| {
            const tokens = try TestText.encode(undefined, a, case[0], false);
            for ([_]bool{ false, true }) |batched| {
                var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
                const whole = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":false}", .{});
                const streamed = try sendWith(backend.engine(), .chat, dsml_body ++ ",\"stream\":true}", .{});
                try std.testing.expectEqual(@as(?u16, 200), whole.status);
                try std.testing.expect(streamed.opened and streamed.status == null);
                try std.testing.expectEqual(@as(usize, 2), whole.calls);
                try std.testing.expectEqual(@as(usize, 2), streamed.calls);
                try std.testing.expectEqualStrings(case[1], whole.arguments[0..whole.arguments_len]);
                try std.testing.expectEqualStrings(case[1], streamed.arguments[0..streamed.arguments_len]);
                try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
            }
        }
    }
}

const weather_body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"What's the weather in Paris in celsius?\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]";
const weather_prose = "I'll check the weather in Paris for you.";
const weather_call = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n" ++
    "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n" ++
    "<｜DSML｜ parameter name=\"unit\" string=\"true\">celsius</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n";
const oslo_call = "<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Oslo</｜DSML｜ parameter>\n" ++
    "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n";

test "DeepSeek-V4.1's spaced DSML call is the reply's tool call, whole and streamed (#1324)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const paris = "{\"city\":\"Paris\",\"unit\":\"celsius\"}";
    // the issue's reply (thinking off, tool_choice auto), and one with a second invoke whose value is JSON
    const cases = [_]struct { []const u8, usize, []const u8 }{
        .{ weather_prose ++ weather_call ++ "</｜DSML｜ calls>", 1, paris },
        .{ weather_prose ++ weather_call ++ oslo_call ++ "</｜DSML｜ calls>", 2, paris ++ "{\"city\":\"Oslo\",\"days\":3}" },
    };
    for (cases) |case| for ([_]bool{ false, true }) |single| {
        const tokens = try TestText.encode(undefined, a, case[0], false);
        const body = try std.mem.concat(a, u8, &.{ weather_body, ",\"max_tokens\":2000", if (single) ",\"parallel_tool_calls\":false" else "" });
        const calls: usize = if (single) 1 else case[1];
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
            const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expect(streamed.opened and streamed.status == null);
            for ([_]@TypeOf(whole){ whole, streamed }) |seen| {
                try std.testing.expectEqual(calls, seen.calls);
                try std.testing.expectEqualStrings(if (single) paris else case[2], seen.arguments[0..seen.arguments_len]);
                try std.testing.expectEqualStrings(weather_prose, seen.content());
                try std.testing.expectEqual(.tool_calls, seen.finish.?);
            }
            try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
        }
    };
}

test "a DSML call max_tokens cuts gives the same reply whole and streamed in either spelling" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v32 = try std.mem.replaceOwned(u8, a, weather_call, "｜DSML｜ calls", "｜DSML｜tool_calls");
    for ([_][]const u8{ weather_call, try std.mem.replaceOwned(u8, a, v32, "｜DSML｜ ", "｜DSML｜") }) |call| {
        const reply = try std.mem.concat(a, u8, &.{ weather_prose, call });
        const cut = weather_prose.len + std.mem.indexOf(u8, call, "Paris").? + 3; // inside the city's value
        const tokens = try TestText.encode(undefined, a, reply, false);
        const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ weather_body, cut });
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
            const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expectEqual(whole.calls, streamed.calls);
            try std.testing.expectEqualStrings(whole.arguments[0..whole.arguments_len], streamed.arguments[0..streamed.arguments_len]);
            try std.testing.expectEqualStrings(whole.content(), streamed.content());
            try std.testing.expectEqual(.length, whole.finish.?);
            try std.testing.expectEqual(.length, streamed.finish.?);
        }
    }
}

//! G8 renders release encoding/tests goldens through engine Jinja; default thinking_mode is chat.
const std = @import("std");
const template = @import("template");

fn usage() noreturn {
    std.debug.print("usage: tf-dsv41-template-check TEMPLATE ENCODING_TESTS_DIR\n", .{});
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) usage();
    const cwd = std.Io.Dir.cwd();
    const source = try cwd.readFileAlloc(io, args[1], a, .limited(1 << 20));
    var compiled = template.compile(a, source, null) catch {
        std.debug.print("template does not compile\n", .{});
        return 1;
    };
    defer compiled.deinit();
    var failed: u32 = 0;
    var cases: u32 = 0;
    for (1..100) |n| {
        const in_path = try std.fmt.allocPrint(a, "{s}/test_input_{d}.json", .{ args[2], n });
        const out_path = try std.fmt.allocPrint(a, "{s}/test_output_{d}.txt", .{ args[2], n });
        const input = cwd.readFileAlloc(io, in_path, a, .limited(1 << 20)) catch break;
        const want = try cwd.readFileAlloc(io, out_path, a, .limited(1 << 20));
        cases += 1;
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, input, .{});
        // load_cases: a bare list is the messages; a case's tools ride on its first message.
        var messages: std.json.Value = parsed;
        var context: std.json.ObjectMap = .empty;
        try context.put(a, "thinking_mode", .{ .string = "chat" });
        if (parsed == .object) {
            messages = parsed.object.get("messages") orelse return error.BadCase;
            if (parsed.object.get("thinking_mode")) |m| if (m != .null) try context.put(a, "thinking_mode", m);
            if (parsed.object.get("reasoning_effort")) |e| if (e != .null) try context.put(a, "reasoning_effort", e);
            if (parsed.object.get("tools")) |t| {
                if (messages.array.items.len == 0 or messages.array.items[0] != .object) return error.BadCase;
                try messages.array.items[0].object.put(a, "tools", t);
            }
        }
        // The server hands templates call arguments as objects (server/messages.zig), as encoding.py decodes them.
        for (messages.array.items) |*m| {
            if (m.* != .object) continue;
            const calls = m.object.get("tool_calls") orelse continue;
            if (calls != .array) continue;
            for (calls.array.items) |*c| {
                if (c.* != .object) continue;
                const f = c.object.getPtr("function") orelse continue;
                if (f.* != .object) continue;
                const call_args = f.object.getPtr("arguments") orelse continue;
                if (call_args.* == .string) call_args.* = std.json.parseFromSliceLeaky(std.json.Value, a, call_args.string, .{}) catch call_args.*;
            }
        }
        // After a final user/tool turn, encode_messages opens the assistant; the server uses add_generation_prompt.
        const items = messages.array.items;
        const last_role = if (items.len > 0 and items[items.len - 1] == .object) (items[items.len - 1].object.get("role") orelse std.json.Value{ .string = "" }).string else "";
        const generation = !std.mem.eql(u8, last_role, "assistant");
        var diag = template.Diag{};
        const got = template.render(a, &compiled, messages, .null, .{ .object = context }, generation, &diag) catch {
            std.debug.print("case {d}: render failed: {s}\n", .{ n, diag.msg });
            failed += 1;
            continue;
        };
        if (std.mem.eql(u8, got, want)) {
            std.debug.print("case {d}: equal ({d} bytes)\n", .{ n, got.len });
            continue;
        }
        failed += 1;
        const at = std.mem.indexOfDiff(u8, got, want) orelse @min(got.len, want.len);
        const lo = at -| 60;
        std.debug.print("case {d}: differs at byte {d}\n  want: {s}\n  got:  {s}\n", .{ n, at, want[lo..@min(want.len, at + 60)], got[lo..@min(got.len, at + 60)] });
    }
    std.debug.print("{d}/{d} cases equal\n", .{ cases - failed, cases });
    return if (cases == 0 or failed > 0) 1 else 0;
}

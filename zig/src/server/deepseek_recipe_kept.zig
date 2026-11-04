//! Where the server's DeepSeek-V4.1 prompt keeps a difference from deepseek-recipe's (deepseek_recipe_test.zig).

/// A kept difference: why, the PR or issue it waits on, and the fixture cases or effort names it covers.
pub const Kept = struct { class: []const u8, why: []const u8, waits: ?u16 = null, ids: []const u16 = &.{}, efforts: []const []const u8 = &.{} };

pub const kept = [_]Kept{
    .{ .class = "R2", .why = "input_text, output_text and bare-string parts on a text-only server: DeepSeek's endpoint refuses them too", .ids = &.{
        7, 19, 21, 24, 30, 32, 34, 37, 38, 41, 45, 52, 57, 62, 66, 73, 74, 77, 79, 81, 86, 87, 93, 94, 103, 107, 108,
        110, 113, 115, 121, 123, 127, 128, 136, 137, 140, 150, 151, 152, 154, 156, 164, 167, 170, 171, 173, 175, 176,
        181, 183, 186, 189, 204, 205, 214, 215, 219, 222, 233, 248, 252, 262, 266, 267, 272, 274, 277, 278, 284, 286,
        287, 294, 301, 304, 306, 307, 308, 312, 313, 316, 320, 322, 323, 324,
    } },
    .{ .class = "R3", .why = "those part kinds and image / input_image parts on a vision server: DeepSeek's endpoint refuses them too", .ids = &.{
        8, 11, 14, 35, 56, 64, 65, 85, 90, 132, 141, 160, 166, 193, 224, 227, 238, 245, 255, 285,
    } },
    .{ .class = "R4", .why = "a text part whose text is a number or null: DeepSeek's endpoint refuses it too", .ids = &.{
        12, 27, 60, 70, 82, 149, 155, 159, 163, 177, 179, 195, 196, 199, 208, 212, 228, 246, 251, 253, 290, 296, 300,
        302, 305, 309, 310, 326, 327,
    } },
    .{ .class = "R5", .why = "content that is neither a string nor a list: DeepSeek's endpoint refuses it too", .ids = &.{
        16, 92, 185, 200, 209,
    } },
    .{ .class = "R6", .why = "reasoning_effort spelled with case, spaces, digits or as a number: DeepSeek's endpoint refuses it too", .ids = &.{
        314, 315, 317, 318, 319, 321, 325,
    } },
    .{ .class = "T3", .why = "response_format's schema is not given to the encoder on main; #1698 passes it", .waits = 1698, .ids = &.{
        13, 18, 26, 31, 43, 47, 63, 78, 84, 96, 99, 114, 119, 135, 138, 139, 143, 144, 161, 168, 174, 187, 190, 194,
        203, 210, 211, 216, 217, 218, 223, 225, 229, 230, 232, 235, 241, 242, 254, 261, 269, 275, 280,
    } },
    .{ .class = "R7", .why = "image parts refused: main's DeepSeek-V4.1 text has no image hook; #1698 adds one", .waits = 1698, .ids = &.{
        15, 146,
    } },
    .{ .class = "T10", .why = "an image part's placeholder: DeepSeek writes <｜image｜>, #1698 <｜deepseek_image｜>; settled with R7", .waits = 1698 },
    .{ .class = "EFF", .why = "the prompt was rendered from our old resolution of this effort name; the effort test checks DeepSeek's", .ids = &.{
        303,
    } },
    .{ .class = "NS", .why = "a tool's namespace: DeepSeek's Chat Completions ignores it (cases 6, 67, 97); this prompt flattens it", .ids = &.{
        5, 20, 29, 39, 40, 46, 48, 54, 69, 72, 80, 83, 101, 104, 105, 106, 112, 131, 133, 142, 145, 157, 169, 178,
        180, 184, 188, 191, 198, 202, 220, 231, 234, 237, 243, 249, 259, 265, 268, 279, 281, 288, 297, 299,
    } },
    .{ .class = "ultra", .why = "reasoning_effort ultra is refused: DeepSeek's endpoint takes it, but no record shows the budget it renders", .efforts = &.{ "ultra" } },
    .{ .class = "T4", .why = "a typed <｜deepseek_image｜> in request text: DeepSeek keeps it as written", .waits = 2141, .ids = &.{
        23, 28, 36, 55, 75, 91, 95, 100, 120, 122, 129, 207, 221, 226, 240, 244, 257, 264, 289, 298,
    } },
    .{ .class = "T2", .why = "a conversation ending in an assistant turn: DeepSeek appends the generation prompt", .waits = 2141, .ids = &.{
        0, 1, 2, 3, 17, 58, 59, 109, 124, 125, 165, 256, 295,
    } },
};

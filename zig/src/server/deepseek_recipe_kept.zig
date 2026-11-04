//! Where the server's DeepSeek-V4.1 prompt keeps a difference from deepseek-recipe's (deepseek_recipe_test.zig).

/// A kept fixture case and the pin of the server's text or refusal for it (`pinOf`; tools/zig/dsrecipe_kept_pins.py).
pub const Case = struct { u16, u64 };

/// A kept difference: why, what it waits on, whether the server refuses, and its pinned cases or effort names.
pub const Kept = struct { class: []const u8, why: []const u8, waits: ?u16 = null, refuses: bool = false, cases: []const Case = &.{}, efforts: []const []const u8 = &.{} };

pub const kept = [_]Kept{
    .{ .class = "R2", .refuses = true, .why = "input_text, output_text and bare-string parts on a text-only server: DeepSeek's endpoint refuses them too", .cases = &.{
        .{ 7, 0xca1cf4e3d74390c4 }, .{ 19, 0xca1cf4e3d74390c4 }, .{ 21, 0xca1cf4e3d74390c4 }, .{ 24, 0xca1cf4e3d74390c4 },
        .{ 30, 0xca1cf4e3d74390c4 }, .{ 32, 0xca1cf4e3d74390c4 }, .{ 34, 0xca1cf4e3d74390c4 }, .{ 37, 0xca1cf4e3d74390c4 },
        .{ 38, 0xca1cf4e3d74390c4 }, .{ 41, 0xca1cf4e3d74390c4 }, .{ 45, 0xca1cf4e3d74390c4 }, .{ 52, 0xca1cf4e3d74390c4 },
        .{ 57, 0xca1cf4e3d74390c4 }, .{ 62, 0xca1cf4e3d74390c4 }, .{ 66, 0xca1cf4e3d74390c4 }, .{ 73, 0xca1cf4e3d74390c4 },
        .{ 74, 0xca1cf4e3d74390c4 }, .{ 77, 0xca1cf4e3d74390c4 }, .{ 79, 0xca1cf4e3d74390c4 }, .{ 81, 0xca1cf4e3d74390c4 },
        .{ 86, 0xca1cf4e3d74390c4 }, .{ 87, 0xca1cf4e3d74390c4 }, .{ 93, 0xca1cf4e3d74390c4 }, .{ 94, 0xca1cf4e3d74390c4 },
        .{ 103, 0xca1cf4e3d74390c4 }, .{ 107, 0xca1cf4e3d74390c4 }, .{ 108, 0xca1cf4e3d74390c4 }, .{ 110, 0xca1cf4e3d74390c4 },
        .{ 113, 0xca1cf4e3d74390c4 }, .{ 115, 0xca1cf4e3d74390c4 }, .{ 121, 0xca1cf4e3d74390c4 }, .{ 123, 0xca1cf4e3d74390c4 },
        .{ 127, 0xca1cf4e3d74390c4 }, .{ 128, 0xca1cf4e3d74390c4 }, .{ 136, 0xca1cf4e3d74390c4 }, .{ 137, 0xca1cf4e3d74390c4 },
        .{ 140, 0xca1cf4e3d74390c4 }, .{ 150, 0xca1cf4e3d74390c4 }, .{ 151, 0xca1cf4e3d74390c4 }, .{ 152, 0xca1cf4e3d74390c4 },
        .{ 154, 0xca1cf4e3d74390c4 }, .{ 156, 0xca1cf4e3d74390c4 }, .{ 164, 0xca1cf4e3d74390c4 }, .{ 167, 0xca1cf4e3d74390c4 },
        .{ 170, 0xca1cf4e3d74390c4 }, .{ 171, 0xca1cf4e3d74390c4 }, .{ 173, 0xca1cf4e3d74390c4 }, .{ 175, 0xca1cf4e3d74390c4 },
        .{ 176, 0xca1cf4e3d74390c4 }, .{ 181, 0xca1cf4e3d74390c4 }, .{ 183, 0xca1cf4e3d74390c4 }, .{ 186, 0xca1cf4e3d74390c4 },
        .{ 189, 0xca1cf4e3d74390c4 }, .{ 204, 0xca1cf4e3d74390c4 }, .{ 205, 0xca1cf4e3d74390c4 }, .{ 214, 0xca1cf4e3d74390c4 },
        .{ 215, 0xca1cf4e3d74390c4 }, .{ 219, 0xca1cf4e3d74390c4 }, .{ 222, 0xca1cf4e3d74390c4 }, .{ 233, 0xca1cf4e3d74390c4 },
        .{ 248, 0xca1cf4e3d74390c4 }, .{ 252, 0xca1cf4e3d74390c4 }, .{ 262, 0xca1cf4e3d74390c4 }, .{ 266, 0xca1cf4e3d74390c4 },
        .{ 267, 0xca1cf4e3d74390c4 }, .{ 272, 0xca1cf4e3d74390c4 }, .{ 274, 0xca1cf4e3d74390c4 }, .{ 277, 0xca1cf4e3d74390c4 },
        .{ 278, 0xca1cf4e3d74390c4 }, .{ 284, 0xca1cf4e3d74390c4 }, .{ 286, 0xca1cf4e3d74390c4 }, .{ 287, 0xca1cf4e3d74390c4 },
        .{ 294, 0xca1cf4e3d74390c4 }, .{ 301, 0xca1cf4e3d74390c4 }, .{ 304, 0xca1cf4e3d74390c4 }, .{ 306, 0xca1cf4e3d74390c4 },
        .{ 307, 0xca1cf4e3d74390c4 }, .{ 308, 0xca1cf4e3d74390c4 }, .{ 312, 0xca1cf4e3d74390c4 }, .{ 313, 0xca1cf4e3d74390c4 },
        .{ 316, 0xca1cf4e3d74390c4 }, .{ 320, 0xca1cf4e3d74390c4 }, .{ 322, 0xca1cf4e3d74390c4 }, .{ 323, 0xca1cf4e3d74390c4 },
        .{ 324, 0xca1cf4e3d74390c4 },
    } },
    .{ .class = "R3", .refuses = true, .why = "those part kinds and image / input_image parts on a vision server: DeepSeek's endpoint refuses them too", .cases = &.{
        .{ 8, 0xf2eeec47f3331688 }, .{ 11, 0xf2eeec47f3331688 }, .{ 14, 0xf2eeec47f3331688 }, .{ 35, 0xf2eeec47f3331688 },
        .{ 56, 0xf2eeec47f3331688 }, .{ 64, 0xf2eeec47f3331688 }, .{ 65, 0xf2eeec47f3331688 }, .{ 85, 0xf2eeec47f3331688 },
        .{ 90, 0xf2eeec47f3331688 }, .{ 132, 0xf2eeec47f3331688 }, .{ 141, 0xf2eeec47f3331688 }, .{ 160, 0xf2eeec47f3331688 },
        .{ 166, 0xf2eeec47f3331688 }, .{ 193, 0xf2eeec47f3331688 }, .{ 224, 0xf2eeec47f3331688 }, .{ 227, 0xf2eeec47f3331688 },
        .{ 238, 0xf2eeec47f3331688 }, .{ 245, 0xf2eeec47f3331688 }, .{ 255, 0xf2eeec47f3331688 }, .{ 285, 0xf2eeec47f3331688 },
    } },
    .{ .class = "R4", .refuses = true, .why = "a text part whose text is a number or null: DeepSeek's endpoint refuses it too", .cases = &.{
        .{ 12, 0xfa70adfcada60ff8 }, .{ 27, 0xfa70adfcada60ff8 }, .{ 60, 0xfa70adfcada60ff8 }, .{ 70, 0xfa70adfcada60ff8 },
        .{ 82, 0xfa70adfcada60ff8 }, .{ 149, 0xfa70adfcada60ff8 }, .{ 155, 0xfa70adfcada60ff8 }, .{ 159, 0xfa70adfcada60ff8 },
        .{ 163, 0xfa70adfcada60ff8 }, .{ 177, 0xfa70adfcada60ff8 }, .{ 179, 0xfa70adfcada60ff8 }, .{ 195, 0xfa70adfcada60ff8 },
        .{ 196, 0xfa70adfcada60ff8 }, .{ 199, 0xfa70adfcada60ff8 }, .{ 208, 0xfa70adfcada60ff8 }, .{ 212, 0xfa70adfcada60ff8 },
        .{ 228, 0xfa70adfcada60ff8 }, .{ 246, 0xfa70adfcada60ff8 }, .{ 251, 0xfa70adfcada60ff8 }, .{ 253, 0xfa70adfcada60ff8 },
        .{ 290, 0xfa70adfcada60ff8 }, .{ 296, 0xfa70adfcada60ff8 }, .{ 300, 0xfa70adfcada60ff8 }, .{ 302, 0xfa70adfcada60ff8 },
        .{ 305, 0xfa70adfcada60ff8 }, .{ 309, 0xfa70adfcada60ff8 }, .{ 310, 0xfa70adfcada60ff8 }, .{ 326, 0xfa70adfcada60ff8 },
        .{ 327, 0xfa70adfcada60ff8 },
    } },
    .{ .class = "R5", .refuses = true, .why = "content that is neither a string nor a list: DeepSeek's endpoint refuses it too", .cases = &.{
        .{ 16, 0x983840644f361579 }, .{ 92, 0x983840644f361579 }, .{ 185, 0x983840644f361579 }, .{ 200, 0x983840644f361579 },
        .{ 209, 0x983840644f361579 },
    } },
    .{ .class = "R6", .refuses = true, .why = "reasoning_effort spelled with case, spaces, digits or as a number: DeepSeek's endpoint refuses it too", .cases = &.{
        .{ 314, 0xeb29254a0fd72241 }, .{ 315, 0xeb29254a0fd72241 }, .{ 317, 0xeb29254a0fd72241 }, .{ 318, 0xeb29254a0fd72241 },
        .{ 319, 0xeb29254a0fd72241 }, .{ 321, 0xeb29254a0fd72241 }, .{ 325, 0xeb29254a0fd72241 },
    } },
    .{ .class = "T3", .why = "response_format's schema is not given to the encoder on main; #1698 passes it", .waits = 1698, .cases = &.{
        .{ 13, 0x766634bad05ff2c9 }, .{ 18, 0x9ed59341a50b598f }, .{ 26, 0xf48ec62979510d38 }, .{ 31, 0x19a8fb237195602a },
        .{ 43, 0xf312deda283dd1c7 }, .{ 47, 0x8367937762158f29 }, .{ 63, 0x14ade85f11376730 }, .{ 78, 0x28f628f227317292 },
        .{ 84, 0x823435f7de34fc2e }, .{ 96, 0x1331957a338ae946 }, .{ 99, 0x8cf05d5d7c65d907 }, .{ 114, 0xb3f98f53ec877cbe },
        .{ 119, 0x7bd82e175ffce2c4 }, .{ 135, 0x9eeebfb1b2541603 }, .{ 138, 0xd4db7736f7d6f304 }, .{ 139, 0x2968dbb2ec75421b },
        .{ 143, 0x02c5f6ce35e50ef5 }, .{ 144, 0x6a92c130da7e139f }, .{ 161, 0x5f542c3ac49b910b }, .{ 168, 0x41403dc876dda74b },
        .{ 174, 0x0c20bbb3242bf7ad }, .{ 187, 0x958a9a35ae365192 }, .{ 190, 0xcb715a4f5db71781 }, .{ 194, 0x6db64805292aedf8 },
        .{ 203, 0x5495f4dba3766abe }, .{ 210, 0xec6450bf94f3affc }, .{ 211, 0x284aa73d55d0f92b }, .{ 216, 0xcf08cd5e77e413bf },
        .{ 217, 0x5e877b7d7892d2f1 }, .{ 218, 0x6a82d739afcd8362 }, .{ 223, 0x2eb007668f4b47ef }, .{ 225, 0x7b7d2701fe9e5b16 },
        .{ 229, 0xcc87f5d5c4e045ef }, .{ 230, 0x5081d00648e2c62e }, .{ 232, 0xff9112c9a2abf7fe }, .{ 235, 0xfa50a57548639ab5 },
        .{ 241, 0x98a853b2b8516e40 }, .{ 242, 0x1877d75450b09b87 }, .{ 254, 0xcf24709a88abd88d }, .{ 261, 0x27c7baad7523f005 },
        .{ 269, 0x4ddc7bfc692bd063 }, .{ 275, 0x55362ba5dacb5227 }, .{ 280, 0xa5cafca2b23c9af2 },
    } },
    .{ .class = "R7", .refuses = true, .why = "image parts refused: main's DeepSeek-V4.1 text has no image hook; #1698 adds one", .waits = 1698, .cases = &.{
        .{ 15, 0x7a08437dc1b1f8e5 }, .{ 146, 0x93e8dbfea4cfe66b },
    } },
    .{ .class = "T10", .why = "an image part's placeholder: DeepSeek writes <｜image｜>, #1698 <｜deepseek_image｜>; settled with R7", .waits = 1698 },
    .{ .class = "EFF", .why = "the prompt was rendered from our old resolution of this effort name; the effort test checks DeepSeek's", .cases = &.{
        .{ 303, 0xb1a87a5052457546 },
    } },
    .{ .class = "NS", .why = "a tool's namespace: DeepSeek's Chat Completions ignores it (cases 6, 67, 97); this prompt flattens it", .cases = &.{
        .{ 5, 0x90874641a259c6bc }, .{ 20, 0xede03c6f741854eb }, .{ 29, 0x47ae611b1a0551a3 }, .{ 39, 0x2f99ee99d50873f0 },
        .{ 40, 0xfbaa8d8333e772f0 }, .{ 46, 0xb510ce5ea5234a95 }, .{ 48, 0x8567dc65b84f0f6f }, .{ 54, 0xec01687c6978081d },
        .{ 69, 0xc09ba91258d8a8ef }, .{ 72, 0xb3edb292810ee956 }, .{ 80, 0x2f53c22962a6c862 }, .{ 83, 0x36b296a956a8a367 },
        .{ 101, 0xbdf999f23a26046e }, .{ 104, 0x4843dcb8fab9f21c }, .{ 105, 0xc80638a6b2b393e0 }, .{ 106, 0x102c8fa114cd335f },
        .{ 112, 0xc73744743da2faa9 }, .{ 131, 0xd72f629396651766 }, .{ 133, 0x78045d81ec11085f }, .{ 142, 0xc86b7b5535a4a6a9 },
        .{ 145, 0x7dd31941ea365f9d }, .{ 157, 0x0ae42e7383956010 }, .{ 169, 0x86ec48f7f5fc6cf6 }, .{ 178, 0x9e9028063ee794eb },
        .{ 180, 0x0e9046be2c96b133 }, .{ 184, 0xd4bf8d8da34209f1 }, .{ 188, 0x6330e237823eeea7 }, .{ 191, 0xfda1d427bab66b88 },
        .{ 198, 0xc2b7b1c039f8e29f }, .{ 202, 0xef1090ec6a870a98 }, .{ 220, 0xd984367017c12e90 }, .{ 231, 0x52c66a83095c66f0 },
        .{ 234, 0xb36fce143675c27c }, .{ 237, 0x84af8a111d38cf56 }, .{ 243, 0x26df64d484f25475 }, .{ 249, 0xa2358040ca565cc1 },
        .{ 259, 0x22fee8dfcc2b4ee4 }, .{ 265, 0x764e92d097a7c920 }, .{ 268, 0x14f86f32a64f071f }, .{ 279, 0xf93bef817fe82202 },
        .{ 281, 0x00474e9d5568cb8f }, .{ 288, 0xeb31c25e44ed7803 }, .{ 297, 0xe40c96a08552b620 }, .{ 299, 0xcf8e457ed0a66525 },
    } },
    .{ .class = "ultra", .refuses = true, .why = "reasoning_effort ultra is refused: DeepSeek's endpoint takes it, but no record shows the budget it renders", .efforts = &.{ "ultra" } },
};

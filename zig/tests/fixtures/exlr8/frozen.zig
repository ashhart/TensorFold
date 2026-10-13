//! sha256 of each window width's outputs (1-16 rows) from the kernel's first build; never regenerate in a test.
const std = @import("std");

const Case = struct { name: []const u8, windows: [16][]const u8 };
pub const cases = [_]Case{
    .{ .name = "gate/up K2 f32", .windows = .{
        "cc368458baef83d071f344dd35904eac34e2ca045b2a2b724c3681f6bbc1285f",
        "bc038bc4a0056891b7ac7d1a907fff441d0e6a13cd72171c72a998e21fd60eb5",
        "9d8402bccb924382532a11255deee6df5ea920a76b6b9bf4659fcbd9100ebde1",
        "176029cd0d3cecb857cf76e67993e281955e076062c2006a1921eadd51da58ce",
        "970ecdc4d525e4b118ade207e56401563b850c7daf5f1b233ca9b3f19c83427a",
        "201ffa2c756de6699e924eef0f10f39eeb0f52276b7549652635c5e30002a142",
        "ffe87a2899fee57b9a4a441868a72759a49496dddb831ee4ed7f3946e1e83f04",
        "ddb5c089fd3d3f0a1e3a82cef712d002fe334bd4ba36d8b1fc556f6cb1deac9b",
        "6ea899efc4d2b5554a350ffccae33f614275fedc3877cb764b5e206adc6145da",
        "1ae9abe15351fc80a0a77f5e412cf6b547b4c833e12b33a6ae4f2b2e8d44c96b",
        "7b13342ad678bfa3f0b46e6d330e4ff93d5d822eafa3852af4f952f76637d47d",
        "090bebfab8d9ee9e29b5246533de07685ed38b674872f4a7a86e7207ee7c5823",
        "56bc733be280b3883e932868266c6cb367b1ce0c94568ca971f4f86c3179ab8e",
        "3f4ee2edb29d29f493ec1068f29488e222165ac83ae46e7db0e2ddad23bc442d",
        "3197e003da1057c20df51bf57a16e50ab7d10c78b3bc7bf4ad645baa8f77ff2a",
        "14d30b6a0ca3a491b8a9e3404dccc3392c8c9c599aa403cbae5dc99010aebb6b",
    } },
    .{ .name = "gate/up K3 f32", .windows = .{
        "28b57b79ebbbc205c9dfbaf0f5d43f77fc7104b1230b69af10edb59dc78e3167",
        "e7d3e6007112461bb0f928fc5ef6a702e51354cef6426d4577b6ee63d034b4d2",
        "2ff7ccf564e68da36e68b67bebf2ba6132da8904f79fca0ea0fa146aba162349",
        "d3e43009933b8a2deb19851eef4368d5993d657f0e1970531c76f5cc72808206",
        "38e77b6460b362a4342146694341fe3132b43a5b41c3e7f8297684c97a52f3da",
        "ea03d2cc4c875ce9623c502ec1a766987954dfd796e00e6fd9b8e2aa876a884c",
        "e1afece1266ea61a74bb714db1b31f21c46c64dfa0b156ccd56c6ae786419e2a",
        "3229bb5782b9552327a0fa7d50ce50cf5604c38391156f477339312422359772",
        "665b54150faf8ce2a31e32fd7cd0634f424491246e723feb8f6b9782cbdb2bd0",
        "55cb38e444d4b9107214b449121049792ac3706ff445628e7bd692aa13510638",
        "92c30389b1632a6256f9b95cd62a1b687384e8c973c718bffdbed8fa73e28db0",
        "32c39673478f66a0562fc2d8fa92dca552102ce70daa72c0748b3ecba6fd80cb",
        "b3f184c6f3d29ec2a57e588947b46db97d1ffd090abc6c2f89dabf67b3ffcfb4",
        "5e51d42b3b9d6b3fe337a2e5116076936917857de79ed8568ade849debc78487",
        "d7a3278c843babf07c024b8a9118d72d1b0c355ef0f7f219c1b02da1757c827c",
        "60f6c43723fe1b14fe7071aa5161c4ceb80f1a4c411cc71ad9ff08969af13e4d",
    } },
    .{ .name = "gate/up K4 bf16", .windows = .{
        "801661c283ec6c1cd97beae021c80c03521b18e1e3b58e9f6348bc8a939c429e",
        "3cff9d862488122af3c381cca587cad9b53e99406385bde9df0e22d8ef4e19ed",
        "8ff2ab6848ae4c603d1427fb67052ce787a2a7ebbc1969ebe2eb43569f91d69c",
        "40a9cdab7aa92f6504babb768651fbf4fa9e7879f4f60d3bee26fecb3a9e6941",
        "968fca78d31f175e6a23393bf58f9204f8ec8d071cbe7889ed9295b0b383ddee",
        "36ff1b1e205bcbe0618a781d563e47f571a964c07770235569b1a730eff787ec",
        "d733a579a0d777a6733a7d0ed8ce0d0775efb5afaee7fd4ed957bf4201b0afc2",
        "9e34cd127d5988691773a5560d7123765b625ad31c37ffc28452e2304169b42a",
        "a2aab4c31e2bbd45ae48e0b16301deac514972f98a491f9d44745ccb0bf5c55c",
        "7a88caae3a69b02e310539c9c22d4dc751a1335979b6d3d5e8967863678645dd",
        "93a1b4db4f15fba2658dc1d1b07d2b331c646731a9afe9eae5cec3af5c24337a",
        "a4381e2bc8737fd5c76088e136588f6612e80150c7594541af51c98344362677",
        "5ed56828263e8077689de21acc155694e6b0478458886ad213c216e688815a75",
        "9211dba72ba920dfb2563dea3d355d74d6fcb26debdda6826387a2d0a343c2d6",
        "5b7fc10edb36f4121f32f98ecf7c105a68649ccaf0b4bfc5de24b91b55fbdf30",
        "9bfab0cc70d2164ee1238ad88f6cb963666254bebb0c7361c8f1d27b910ab777",
    } },
    .{ .name = "down K2/K3/K4 combine f32", .windows = .{
        "0ccae55eedf2337aa9dd792f194499905f84290b8f6baeb00cb7967dbdd50280",
        "b1ad30c2dc583be3c6d6c16031cb782f8ee5df03036d30e857e2d54606baabd0",
        "17953a1b0e4b6f4bb168d0218ad54b829b936dbd7c3ce838da08715e10f468c2",
        "198101761fabfb8978197d2545a49a25e96887cf6ac17930f8f420dd58a636c7",
        "04d26e3cfcc71f7413a14047aa2d364f18164e7e40711e645bfc70e77d233976",
        "838720c26f8a549c0b76dfc52ceca6d06202f8254d8bbdaf4ff3a1a5106cd261",
        "942d85a73c119a0096aa09577a5fd3feef4e5993fa7ebb6bc4655e9be001b1ad",
        "b687d464c49bbcabbf181969b5363d7b24c7ec98d722b0f23b1601104a49eded",
        "2eb92fb15e5f96b08946be50acba17a78e335f7269e1da51051d29a478e7e4ef",
        "d90badcafceac949f75b2b201c476531ae3b03f097a8d3daa597c40173f4b388",
        "a9404c6fd4abd777b671b98cc8348e5f998f91e0bc0b4cedb263dabf1f84bd57",
        "6ddc61227c4d419290b695b9cbaa1b641ca75c9b881a79a0ba774744c91bf6ba",
        "976c67a16678c9b1b39fdd9f827ba7f3ff7bfcd6dcde2c33858913a180efd966",
        "2927e8845f44d491e4bb30b882b61faaa62004b30b8809228fcf8e9b930d418c",
        "dc5deec83802f03c68b8e8b7550759f5acc8059ead2306b7fabdeb9735a566f1",
        "499f1753d0354b9d9c58d7e4473e58a368a135f6564bab0e1deeb36801fbc30e",
    } },
    .{ .name = "down K3/K4/K2 combine bf16", .windows = .{
        "90a5e4b9d6579062709d627ffcc852348bae5af17268dac1b828fc0ef08b8675",
        "eecfdfa5bfd2da576d8ae4b5aad4aaf564373d4ae972a5eabda35af1837688f7",
        "cf0c35155518413b181c1d8c1a4116043d540e1571e7d3a6e7cd9f9e6da8ea23",
        "a94459b9959166c512d9bd15958466e2b93fbe6b18d727036a444f8611335b61",
        "7fa6325ebe8eadb5924ebc1bd6a1f7c3a63864464b638d8033d93fe2e0e5ad7d",
        "201dc5e4d99dbeda688db5ad31ba2389a84735b519a853e0bce635349aaeaf9a",
        "660be48f480d82c5b1018110145db4e5bd5faab27f6f88a0acaa3f6280485682",
        "e883b8104a87ccf9f442f3720e6c0f2b0aa89cc3f5b9a8dcec60992377641cbe",
        "fcf4a1ebc6823613f01f6f9ec7940fd3599bcd69f8c2117978af2e5f9b294e3b",
        "f059e0fa92591530044ce87d14d0ea516722a71f172be9625fc5aef52fae3bc4",
        "50367b481b1315d48bacc20dcadd98560f79591ddf8655123881521fa8d3e1e3",
        "f6f9dff6328e61aa48ec4e47183ec1de9ba31b34b714f7aa72d070a2db44dbb2",
        "e7667fdd9628692646137796ea184c72ba0f193c045ee2cab6d70ea1cd548cfa",
        "302064ba925915d5e591821f066195c19e126233f62b47f709d9be1b4bd294ce",
        "bd8099beed7d72ae1642092f8926debec8d6d5676c6610c6b0dff589d1e9a791",
        "23fe4fa04935dcd95424787f3ae23abc6183fc7e9e6e5251210a4bf100d7cb6a",
    } },
};

fn digest(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// Every window of a case (widths 1-16 in order, `row` bytes a row) against its frozen digest.
pub fn compare(name: []const u8, windows: []const u8, row: usize) !void {
    for (cases) |c| if (std.mem.eql(u8, c.name, name)) {
        var at: usize = 0;
        for (1..17) |m| {
            if (!std.mem.eql(u8, &digest(windows[at..][0 .. m * row]), c.windows[m - 1])) {
                std.debug.print("{s}: the {d}-row window differs from its frozen bytes\n", .{ name, m });
                return error.FrozenWindowMismatch;
            }
            at += m * row;
        }
        return;
    };
    return error.MissingFrozenCase;
}

/// The digests to freeze, as Zig source lines.
pub fn print(name: []const u8, windows: []const u8, row: usize) void {
    std.debug.print("    .{{ .name = \"{s}\", .windows = .{{\n", .{name});
    var at: usize = 0;
    for (1..17) |m| {
        std.debug.print("        \"{s}\",\n", .{digest(windows[at..][0 .. m * row])});
        at += m * row;
    }
    std.debug.print("    }} }},\n", .{});
}

test "a changed byte in any window is caught" {
    try std.testing.expectError(error.MissingFrozenCase, compare("none", "", 0));
}

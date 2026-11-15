import Foundation
import Testing
#if canImport(CoreGraphics)
import CoreGraphics
#endif
@testable import ComputerUseChecks

struct KeysTests {
    @Test func keyCodesAliasesAndModifiers() throws {
        let good: [(String, CGKeyCode, CGEventFlags)] = [
            ("cmd+shift+t", 17, [.maskCommand, .maskShift]), ("Return", 36, []), ("enter", 36, []),
            ("ctrl+opt+delete", 51, [.maskControl, .maskAlternate]), ("command+alt+esc", 53, [.maskCommand, .maskAlternate]),
            ("f12", 111, []), ("F1", 122, []), ("cmd+,", 43, [.maskCommand]), ("cmd+=", 24, [.maskCommand]),
            ("shift+`", 50, [.maskShift]), ("pagedown", 121, []), ("pageup", 116, []), ("fn+f1", 122, [.maskSecondaryFn]),
            (" cmd + a ", 0, [.maskCommand]), ("cmd+cmd+a", 0, [.maskCommand]), ("space", 49, []), ("up", 126, []),
            ("down", 125, []), ("left", 123, []), ("right", 124, []), ("home", 115, []), ("end", 119, []),
            ("forwarddelete", 117, []), ("backspace", 51, []), ("tab", 48, []), ("control+[", 33, [.maskControl]),
            ("cmd+]", 30, [.maskCommand]), ("ctrl+;", 41, [.maskControl]), ("'", 39, []), ("/", 44, []), (".", 47, []),
            ("-", 27, []), ("0", 29, []), ("9", 25, []), ("z", 6, []), ("option+shift+k", 40, [.maskAlternate, .maskShift]),
        ]
        for (text, code, flags) in good {
            let combo = try parseKeyCombo(text)
            #expect(combo.keyCode == code, "key \(text) code")
            #expect(combo.flags == flags, "key \(text) flags")
        }
        let order = try parseKeyCombo("shift+cmd+t")
        #expect(order.modifiers.map { $0.keyCode } == [56, 55], "modifiers keep their order")
        let repeated = try parseKeyCombo("cmd+cmd+a")
        #expect(repeated.modifiers.count == 1, "a repeated modifier is pressed once")
        let fn = try parseKeyCombo("fn+f1")
        let modifier = try #require(fn.modifiers.first)
        #expect(modifier.keyCode == nil, "fn only sets a flag")
        #expect(CGEventFlags.maskShift.rawValue == 1 << 17)
        #expect(CGEventFlags.maskControl.rawValue == 1 << 18)
        #expect(CGEventFlags.maskAlternate.rawValue == 1 << 19)
        #expect(CGEventFlags.maskCommand.rawValue == 1 << 20)
        #expect(CGEventFlags.maskSecondaryFn.rawValue == 1 << 23)
        let printable: [(String, Bool)] = [("a", true), ("shift+a", true), ("space", true), ("cmd+a", false), ("ctrl+c", false), ("return", false), ("left", false), ("tab", false)]
        for (text, expected) in printable {
            let combo = try parseKeyCombo(text)
            #expect(comboTypesCharacter(combo) == expected, "\(text) types a character: \(expected)")
        }
    }

    @Test func invalidKeysAreRejected() {
        let bad = ["", "   ", "cmd", "cmd+shift", "cmd+foo", "a+b", "f13", "hyper+a", "cmd++", "+", "control+option+", String(repeating: "a", count: 100)]
        for text in bad {
            do {
                _ = try parseKeyCombo(text)
                Issue.record("key \(text.prefix(20)) should fail")
            } catch let error as HelperError {
                #expect(error.code == "bad_key", "key \(text.prefix(20)) fails with bad_key")
            } catch {
                Issue.record("key \(text.prefix(20)) threw something else")
            }
        }
    }
}

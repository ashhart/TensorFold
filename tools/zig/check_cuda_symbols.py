"""Require each registered x3ld function in each SM's cuobjdump --dump-sass output."""
import argparse
from pathlib import Path
import re
import sys
import unittest


def expected(registry):
    symbols = set(re.findall(r'\.image\s*=\s*"dsv41_x3ld"\s*,\s*\.symbol\s*=\s*"([^"]+)"', registry))
    if not symbols:
        raise ValueError("registry has no x3ld symbols")
    return symbols


def functions(text):
    result, sm = {}, None
    for line in text.splitlines():
        arch = re.search(r'(?:code for |arch\s*=\s*)sm_(\d+)[a-z]?\b', line)
        if arch:
            sm = int(arch[1])
            result.setdefault(sm, set())
        function = re.fullmatch(r'\s*Function\s*:\s*(\S+)\s*', line)
        if function and sm is not None:
            result[sm].add(function[1])
    return result


def missing(text, symbols, sms):
    have = functions(text)
    return [(sm, symbol) for sm in sms for symbol in sorted(symbols - have.get(sm, set()))]


class Controls(unittest.TestCase):
    def test_missing_entry_and_arch_are_refused(self):
        self.assertEqual([(121, "required")], missing("code for sm_121\n Function : other\n", {"required"}, [121]))
        self.assertEqual([(80, "required")], missing("code for sm_121\n Function : required\n", {"required"}, [80]))

    def test_declaration_or_substring_is_not_a_function(self):
        self.assertTrue(missing("code for sm_121\n .entry required\n Function : required_extra\n", {"required"}, [121]))

    def test_each_sm_must_hold_each_function(self):
        text = "code for sm_80\n Function : a\n Function : b\narch = sm_121a\n Function : a\n Function : b\n"
        self.assertFalse(missing(text, {"a", "b"}, [80, 121]))
        self.assertEqual({"a"}, expected('.image = "dsv41_x3ld", .symbol = "a"'))
        with self.assertRaises(ValueError):
            expected("no entries")


def main():
    if sys.argv[1:] == ["--self-test"]:
        unittest.main(argv=[sys.argv[0]])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    try:
        sms = [int(sm) for sm in args.sms.split(",")]
        if not sms or any(sm <= 0 for sm in sms):
            raise ValueError("SM targets must be positive integers")
        absent = missing(args.dump.read_text(), expected(args.registry.read_text()), sms)
    except ValueError as err:
        parser.error(str(err))
    for sm, symbol in absent:
        print(f"sm_{sm} missing SASS function: {symbol}", file=sys.stderr)
    if absent:
        sys.exit(1)
    print(f"x3ld symbols present on {len(set(sms))} SM targets")


if __name__ == "__main__":
    main()

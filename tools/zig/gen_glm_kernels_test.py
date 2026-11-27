"""Deterministic threadgroup-reservation generation and the prefill MLX caller."""
import unittest
from unittest import mock

import gen_glm_kernels as glm
import prefill_kernels as prefill
from metal_source import Arg


def attribute(threads):
    return f"[[max_total_threads_per_threadgroup({threads})]]\n"


def glm_spec(header="", threads=128):
    return glm.Spec("unit", "tf_unit", "  OUT[0] = X[0];\n", [Arg("X", "float32", 64)],
                    [Arg("OUT", "float32", 64)], header=header, threads=threads)


def variant(threads):
    family = prefill.Family("unit.metal", "tf_unit", ("X",), ("OUT",), "  OUT[0] = X[0];\n", (), threads)
    return prefill.Variant(family, (), ("float32", "float32"))


class RecordingFast:
    def __init__(self):
        self.calls = []

    def metal_kernel(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class RecordingMLX:
    def __init__(self):
        self.fast = RecordingFast()


class GLMGenerationTest(unittest.TestCase):
    def test_empty_header_inserts_once_without_changing_other_bytes(self):
        _, plain = glm.rendered(glm_spec(threads=0))
        _, reserved = glm.rendered(glm_spec())
        self.assertEqual(reserved, plain.replace("[[kernel]] void", attribute(128) + "[[kernel]] void", 1))
        self.assertEqual(reserved.count("max_total_threads_per_threadgroup"), 1)

    def test_already_reserved_header_is_preserved_byte_for_byte(self):
        self.assertEqual(glm.rendered(glm_spec(attribute(128))), glm.rendered(glm_spec()))

    def test_conflicting_and_repeated_reservations_fail_closed(self):
        for label, header in (("conflicting", attribute(64)), ("repeated", attribute(128) * 2)):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "conflicting or repeated"):
                glm.rendered(glm_spec(header))


class PrefillGenerationTest(unittest.TestCase):
    def text(self, header, threads=128):
        # Only file expansion is replaced: Variant.text and reserve_threads remain the production path.
        with mock.patch.object(prefill, "header", return_value=header):
            return variant(threads).text()

    def test_empty_header_inserts_once_without_changing_other_bytes(self):
        plain = self.text("", threads=0)
        reserved = self.text("")
        self.assertEqual(reserved, plain.replace("[[kernel]] void", attribute(128) + "[[kernel]] void", 1))
        self.assertEqual(reserved.count("max_total_threads_per_threadgroup"), 1)

    def test_already_reserved_header_is_preserved_byte_for_byte(self):
        self.assertEqual(self.text(attribute(128)), self.text(""))

    def test_conflicting_and_repeated_reservations_fail_closed(self):
        for label, header in (("conflicting", attribute(64)), ("repeated", attribute(128) * 2)):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "conflicting or repeated"):
                self.text(header)


class RunnerTest(unittest.TestCase):
    def test_kernel_path_for_unreserved_and_reserved_variants(self):
        for threads in (0, 128):
            with self.subTest(threads=threads):
                mx = RecordingMLX()
                runner = prefill.Runner(mx)
                with mock.patch.object(prefill, "header", return_value=""):
                    first = runner.kernel(variant(threads))
                    second = runner.kernel(variant(threads))
                self.assertIs(first, second)
                self.assertEqual(len(mx.fast.calls), 1)
                call = mx.fast.calls[0]
                self.assertEqual(call["name"], "tf_unit")
                self.assertEqual(call["input_names"], ["X"])
                self.assertEqual(call["output_names"], ["OUT"])
                self.assertEqual(call["source"], "  OUT[0] = X[0];\n")
                self.assertEqual(call["header"], attribute(threads) if threads else "")
                self.assertEqual(call["header"].count("max_total_threads_per_threadgroup"), bool(threads))


if __name__ == "__main__":
    unittest.main()

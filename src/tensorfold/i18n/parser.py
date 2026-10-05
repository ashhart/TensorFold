"""argparse in the active language: section titles, the usage prefix and the usage errors users actually hit."""
from __future__ import annotations

import argparse
import re

from . import locale, t


class Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose group titles, help line and usage errors follow the active language."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = t("positional arguments")
        self._optionals.title = t("options")
        self.add_argument("-h", "--help", action="help", help=t("show this help message and exit"))

    def format_help(self) -> str:
        """argparse appends an ASCII colon to every section title; Chinese uses its own."""

        text = super().format_help()
        if locale() == "zh":
            for group in self._action_groups:
                if group.title:
                    text = text.replace(f"{group.title}:\n", f"{group.title}：\n")
        return text

    def error(self, message: str) -> None:
        self.exit(2, t("{prog}: error: {detail}", prog=self.prog, detail=usage_error(message)) + "\n")


class HelpFormatter(argparse.HelpFormatter):
    """``usage:`` in the active language; argparse's own width and wrapping stay untouched."""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, prefix=t("usage: ") if prefix is None else prefix)


class DefaultsFormatter(HelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """``ArgumentDefaultsHelpFormatter`` with the appended default in the active language."""

    def _get_help_string(self, action):
        text = super()._get_help_string(action)
        head, separator, tail = text.partition(" (default: ")
        if not separator or not tail.endswith(")"):
            return text
        return head + t(" (default: {value})", value=tail[:-1])


# argparse builds these sentences itself; each one is a catalog key with named fields.
_SENTENCES = (
    (re.compile(r"invalid (?P<kind>\w+) value: (?P<value>.+)\Z"), "invalid {kind} value: {value}"),
    (re.compile(r"invalid choice: (?P<value>.+?) \(choose from (?P<choices>.+)\)\Z"),
     "invalid choice: {value} (choose from {choices})"),
    (re.compile(r"expected one argument\Z"), "expected one argument"),
    (re.compile(r"expected at most one argument\Z"), "expected at most one argument"),
    (re.compile(r"expected (?P<count>\S+) arguments?\Z"), "expected {count} argument(s)"),
    (re.compile(r"not allowed with argument (?P<other>.+)\Z"), "not allowed with argument {other}"),
    (re.compile(r"the following arguments are required: (?P<names>.+)\Z"),
     "the following arguments are required: {names}"),
    (re.compile(r"unrecognized arguments: (?P<names>.+)\Z"), "unrecognized arguments: {names}"),
    (re.compile(r"ambiguous option: (?P<option>.+?) could match (?P<matches>.+)\Z"),
     "ambiguous option: {option} could match {matches}"),
    (re.compile(r"ignored explicit argument (?P<value>.+)\Z"), "ignored explicit argument {value}"),
)
_ARGUMENT = re.compile(r"argument (?P<name>.+?): (?P<rest>.+)\Z", re.DOTALL)


def usage_error(message: str) -> str:
    """An argparse error message in the active language; one it does not know is returned unchanged."""

    argument = _ARGUMENT.match(message)
    prefix = t("argument {name}: ", name=argument.group("name")) if argument else ""
    core = argument.group("rest") if argument else message
    for pattern, key in _SENTENCES:
        found = pattern.match(core)
        if found:
            return prefix + t(key, **found.groupdict())
    return prefix + core


__all__ = ["DefaultsFormatter", "HelpFormatter", "Parser", "usage_error"]

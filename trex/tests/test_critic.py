"""Tests for explicit advisory Critic flag parsing."""

from __future__ import annotations

from trex.critic import _parse_flags


def test_no_flags_keyword_recognized():
    """Explicit no_flags response → empty list."""
    assert _parse_flags("no_flags") == []
    assert _parse_flags("No_flags. The proposal looks evidence-grounded.") == []


def test_empty_text():
    assert _parse_flags("") == []


def test_strict_markers_only():
    """Only (a)..(d) markers count as flags."""
    text = (
        "(a) Target-prior: complexa_beam succeeded last 3 ticks on this class\n"
        "(b) Deadline: 2.5h remaining vs 6h expected\n"
        "(c) Ignored failure: matching joint_fail recipe rcp_x1\n"
        "(d) Default-bias: same family across 3 target_classes"
    )
    flags = _parse_flags(text)
    assert len(flags) == 4
    assert flags[0].startswith("(a)")
    assert flags[3].startswith("(d)")


def test_prose_without_markers_is_no_flags():
    """Unmarked prose is not a Critic flag."""
    assert _parse_flags(
        "Looking at the evidence carefully, I think the proposal is reasonable."
    ) == []
    assert _parse_flags("This seems fine to proceed.") == []


def test_partial_markers_only_valid_taken():
    """Mixed valid + prose lines → only marker lines kept."""
    text = (
        "Reviewing the evidence...\n"
        "(a) Target-prior contradicted by recipe rcp_42\n"
        "Some commentary that should be ignored\n"
        "(b) Deadline violation cited"
    )
    flags = _parse_flags(text)
    assert len(flags) == 2
    assert all(f.startswith(("(a)", "(b)")) for f in flags)


def test_flag_text_capped_400_chars():
    """Long flag text is truncated to prevent prompt bloat downstream."""
    long_flag = "(a) " + ("very-long-description " * 50)
    flags = _parse_flags(long_flag)
    assert len(flags) == 1
    assert len(flags[0]) <= 400


def test_unknown_marker_letter_rejected():
    """Reject markers outside the supported category set."""
    assert _parse_flags("(e) Some made-up category") == []
    assert _parse_flags("(z) Another invented marker") == []

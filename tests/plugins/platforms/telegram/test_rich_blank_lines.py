"""Blank lines in rich messages (Chin, 2026-10-22): Telegram's rich renderer shows a paragraph break
between plain lines with no gap, so the workout card's blocks ran together. A no-break-space line
holds the gap open; blank lines that end a table, list, quote, heading or tag block stay as they are."""

from plugins.platforms.telegram.adapter import _rich_normalize_linebreaks as norm

NBSP = " "


def test_a_blank_line_between_plain_lines_becomes_a_visible_empty_line():
    assert norm("**Effort** RPE 7\n\n**Loads**\n1 · 30 kg") == "**Effort** RPE 7  \n" + NBSP + "  \n**Loads**  \n1 · 30 kg"


def test_the_footer_keeps_its_gap():
    assert norm("**Time** about 45 min\n\nall jobs ran · v1").endswith("45 min  \n" + NBSP + "  \nall jobs ran · v1")


def test_a_table_is_untouched_and_its_surrounding_blank_lines_stay_paragraph_breaks():
    text = "**Title**\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n**Rest** 60 s"
    out = norm(text)
    assert "| a | b |\n|---|---|\n| 1 | 2 |" in out
    assert NBSP not in out


def test_blocks_that_need_a_real_blank_line_keep_it():
    for text in ("- one\n- two\n\nafter", "before\n\n- one", "> quote\n\nafter", "# Head\n\nafter",
                 "</details>\n\nReply: `x`", "<summary>S</summary>\n\nbody"):
        assert NBSP not in norm(text), text


def test_a_double_blank_line_and_fenced_code_are_left_alone():
    assert NBSP not in norm("a\n\n\nb")
    assert norm("```\na\n\nb\n```") == "```\na\n\nb\n```"

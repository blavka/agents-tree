from conftest import agent, session

from agents_tree.model import Detail, override_window, sanitize


def test_sanitize_neutralises_control_characters_in_every_text_field():
    a = agent("evil: \x1b]0;pwned\x07name\nnext", model="opus\x1b[2J", effort="hi\x1b[1m")
    a.status = "run\x1bning"
    a.detail = Detail(prompt="line one\nline\x1b[31m two\r\n", last_text="ok\x07",
                      tools={"Ba\x1bsh": 2}, last_tool="Ba\x1bsh", transcript="/t\x1b/x")
    s = session([a], sid="s\x1b[1m1", title="t\x1b[2Jx")
    s.kind, s.cwd = "inter\x1b[31mactive", "/w\x1b"

    sanitize([s])

    texts = [s.id, s.title, s.kind, s.cwd, a.id, a.label, a.model, a.effort, a.status,
             a.detail.last_tool, a.detail.transcript, a.detail.last_text, *a.detail.tools]
    assert all("\x1b" not in t and "\x07" not in t and "\n" not in t for t in texts)
    assert a.detail.prompt == "line one\nline [31m two\n"


def test_override_window_applies_to_main_and_every_agent():
    s = session([agent("a", children=[agent("b", window=200_000)])])

    override_window([s], 500_000)

    assert {x.context_window for x in s.all_agents()} == {500_000}


def test_no_override_keeps_inferred_windows():
    s = session([agent("a", window=200_000)])

    override_window([s], None)

    assert s.agents[0].context_window == 200_000

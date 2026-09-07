import pytest

from v_core.agent import Agent
from v_core.autonomy.task_contract import TaskContract


@pytest.mark.parametrize("prompt", [
    "Przejrzyj wyłącznie lokalne opisy i schematy dostępnych narzędzi file_read oraz memory_recall. Podaj propozycje usprawnień i dowody, na których je opierasz. Nie uruchamiaj narzędzi sieciowych, nie zapisuj plików, nie testuj zewnętrznych usług. Oddziel przeczytane opisy od faktycznie wykonanych testów.",
    "Przejrzyj lokalne opisy narzędzi. Nie uruchamiaj narzędzi sieciowych, nie zapisuj plików. Podaj raport.",
    "Review local tool descriptions. Do not use network tools. Read-only. Report findings.",
])
def test_local_review_constraints_override_model_capabilities(prompt):
    assert TaskContract.disables_web(prompt)
    assert TaskContract.requests_read_only(prompt)
    inferred = TaskContract(
        requires_browser_navigation=True, requires_web_discovery=True,
        requires_browser_snapshot=True, requires_file_read=True,
        requires_created_tool=True, requires_created_skill=True,
        requires_created_tool_execution=True, requires_command_execution=True,
        requires_file_mutation=True, requires_evidence_report=True,
    )
    bounded = inferred.without_mutations().without_web()
    assert bounded.requires_file_read
    assert bounded.requires_evidence_report
    assert not bounded.requires_created_tool
    assert not bounded.requires_created_skill
    assert not bounded.requires_created_tool_execution
    assert not bounded.requires_command_execution
    assert not bounded.requires_file_mutation
    assert not bounded.requires_web_discovery
    names = ["read_file", "write_file", "learning_create_tool", "web_search", "sandbox_execute_offline"]
    definitions = [{"type": "function", "function": {"name": name}} for name in names]
    offered = Agent._select_tool_definitions(
        prompt, bounded, definitions,
        capability_hints={"file_read", "file_write", "learning_tool", "browser"},
    )
    assert [item["function"]["name"] for item in offered] == ["read_file"]


def test_ordinary_tool_creation_not_marked_read_only():
    assert not TaskContract.requests_read_only("Create a word counting tool.")

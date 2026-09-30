from pathlib import Path

import pytest

from argus.core.models import ReviewDecision, RoundRecord
from argus.engineer.round_config import SupervisedConfig
from argus.engineer.round_settlement import _enforce_operator_question_policy
from argus.engineer.round_state import RoundLoopState
from argus.manager.directive import (
    active_operator_question_policy,
    publish_nonblocking_question,
    set_active_manager_directive,
)


def test_question_policy_without_selected_vertical_stays_unchanged(tmp_path: Path) -> None:
    assert active_operator_question_policy(tmp_path) == "unchanged"


@pytest.mark.parametrize("source", ["reviewer", "engineer_operator_question"])
def test_question_is_retained_without_blocking_or_approval(tmp_path: Path, monkeypatch, source) -> None:
    published = []
    monkeypatch.setattr(
        "argus.core.operator_messages.publish_operator_message",
        lambda *args, **kwargs: published.append(kwargs),
    )
    set_active_manager_directive(tmp_path, "Replies are optional", operator_question_policy="nonblocking")
    options = [{"id": "a", "label": "A", "description": "Investigate A"}]
    original = ReviewDecision(
        status="blocked", reason="Two candidate meanings remain", next_action="",
        operator_question="Which meaning matches the caller?", operator_options=options,
        review_source=source, input_tokens=12,
    )
    config = SupervisedConfig(operator_question_policy_root=tmp_path)
    state = RoundLoopState()
    review = _enforce_operator_question_policy(original, supervised_config=config, state=state)
    assert review.status == "continue"
    assert review.operator_question == ""
    assert review.input_tokens == 12
    assert review.reason == original.reason
    assert original.operator_question == "Which meaning matches the caller?"
    question = review.planner_report["nonblocking_operator_question"]
    assert question == {
        "question": original.operator_question, "options": options,
        "source": source, "reply_required": False,
    }
    assert published[0]["event_fields"]["nonblocking_operator_question"] == question
    assert review.planner_report["forward_progress"] is False
    state.rounds.append(RoundRecord(1, "", 0, review))
    repeated = _enforce_operator_question_policy(original, supervised_config=config, state=state)
    assert repeated.status == "replan_requested"
    assert repeated.operator_question == ""
    assert repeated.planner_report["plan_signal"] == "reconsider"


def test_nonblocking_policy_does_not_turn_infrastructure_failure_into_retry(tmp_path: Path) -> None:
    set_active_manager_directive(tmp_path, "Replies optional", operator_question_policy="nonblocking")
    original = ReviewDecision(
        status="blocked", reason="Provider unavailable", next_action="",
        operator_question="Can the provider be restored?", backend_unavailable=True,
    )
    review = _enforce_operator_question_policy(
        original, supervised_config=SupervisedConfig(operator_question_policy_root=tmp_path),
        state=RoundLoopState(),
    )
    assert review.status == "blocked"
    assert review.backend_unavailable
    assert review.operator_question == ""


def test_question_publication_is_durable_and_deduplicated(tmp_path: Path) -> None:
    import json

    for _ in range(2):
        publish_nonblocking_question(tmp_path, "A or B?", source="reviewer")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    questions = [event for event in events if "nonblocking_operator_question" in event]
    assert len(questions) == 1
    assert questions[0]["type"] == "ui.argus"
    assert questions[0]["nonblocking_operator_question"]["question"] == "A or B?"
    assert questions[0]["nonblocking_operator_question"]["reply_required"] is False


def test_later_explicit_policy_overrides_nonblocking(tmp_path: Path) -> None:
    set_active_manager_directive(tmp_path, "Replies optional", operator_question_policy="nonblocking")
    set_active_manager_directive(tmp_path, "Wait for my answer", operator_question_policy="allow")
    assert active_operator_question_policy(tmp_path) == "allow"


def test_project_directive_is_preserved_with_separate_state_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "life"
    set_active_manager_directive(project, "Reply optional", operator_question_policy="nonblocking")
    assert active_operator_question_policy(state, project_root=project) == "nonblocking"
    set_active_manager_directive(state, "Wait for reply", operator_question_policy="allow")
    assert active_operator_question_policy(state, project_root=project) == "allow"


@pytest.mark.parametrize("repeat", [False, True])
def test_native_reviewer_question_continues_without_operator_input(tmp_path: Path, repeat) -> None:
    from argus.adapters.memory_backend import CannedResponse, MemoryBackend
    from argus.engineer.runner import EngineerConfig, SupervisedEngineer
    from argus.reviewer import Reviewer, ReviewerConfig

    set_active_manager_directive(tmp_path, "Replies optional", operator_question_policy="nonblocking")
    backend = MemoryBackend()
    for index in (1, 2):
        backend.queue(
            f"engineer-r{index}",
            CannedResponse(message="Investigated the candidate.\nNEXT_OWNER=reviewer"),
        )
    question = ("request_review_decision", {
        "review": "A caller interpretation needs a provisional choice.",
        "question": "Which interpretation fits the goal?",
    })
    backend.queue(
        "reviewer",
        CannedResponse(review_action=question),
        CannedResponse(review_action=question if repeat else ("approve_review", {"review": "Reviewed the bounded investigation."})),
    )
    engineer = SupervisedEngineer(
        engineer_runner=backend, reviewer=Reviewer(runner=backend),
        engineer_config=EngineerConfig(model="m"), reviewer_config=ReviewerConfig(model="m"),
    )
    status, rounds, *_ = engineer.run(
        objective="Investigate the caller interpretation",
        engineer_prompt_builder=lambda next_action, _include_static=True: next_action or "Investigate.",
        supervised_config=SupervisedConfig(max_rounds=3, operator_question_policy_root=tmp_path),
        workdir=tmp_path,
    )
    assert status == ("replan_requested" if repeat else "done")
    assert len(rounds) == 2
    assert rounds[0].review.planner_report["nonblocking_operator_question"]["reply_required"] is False
    assert all(not record.review.operator_question for record in rounds)
    assert [label for label, *_ in backend.history] == ["engineer-r1", "reviewer", "engineer-r2", "reviewer"]


def test_native_engineer_question_does_not_skip_independent_review(tmp_path: Path) -> None:
    from argus.adapters.memory_backend import CannedResponse, MemoryBackend
    from argus.engineer.runner import EngineerConfig, SupervisedEngineer
    from argus.reviewer import Reviewer, ReviewerConfig

    set_active_manager_directive(tmp_path, "Replies optional", operator_question_policy="nonblocking")
    backend = MemoryBackend()
    backend.queue("engineer-r1", CannedResponse(
        message="NEXT_OWNER=operator\nOPERATOR_QUESTION=Is publishing route A or B acceptable?",
    ))
    backend.queue("engineer-r2", CannedResponse(
        message="Compared routes locally without publishing.\nNEXT_OWNER=reviewer",
    ))
    backend.queue("reviewer", CannedResponse(
        review_action=("approve_review", {"review": "The bounded local investigation is complete."}),
    ))
    engineer = SupervisedEngineer(
        engineer_runner=backend, reviewer=Reviewer(runner=backend),
        engineer_config=EngineerConfig(model="m"), reviewer_config=ReviewerConfig(model="m"),
    )
    status, rounds, *_ = engineer.run(
        objective="Compare routes without publishing",
        engineer_prompt_builder=lambda next_action, _include_static=True: next_action or "Compare routes.",
        supervised_config=SupervisedConfig(max_rounds=3, operator_question_policy_root=tmp_path),
        workdir=tmp_path,
    )
    assert status == "done"
    assert rounds[-1].review.review_source == "reviewer"
    assert rounds[0].review.planner_report["nonblocking_operator_question"]["source"] == "engineer_operator_question"
    assert all(not record.review.operator_question for record in rounds)

import pytest

from argus.reviewer.tools import ReviewActions


def test_basis_acceptance_requires_an_explicit_call_bound_payload():
    actions = ReviewActions()
    actions.dispatch("approve_review", {"review": "Bounded task done"})
    assert actions.decision.vertical_review == {}
    payload = {"subject": "a" * 64, "decision": "accept", "rationale": "Read the current source and bounded assumptions"}
    actions.dispatch("approve_review", {"review": "Reviewed basis", "vertical_review": payload})
    assert actions.decision.vertical_review == payload
    assert actions.decision.to_event_payload()["vertical_review"] == payload


def test_review_payload_rejects_missing_binding_and_unknown_decisions():
    actions = ReviewActions()
    for payload in (
        {"decision": "accept", "rationale": "Approved"},
        {"subject": "a" * 64, "decision": "approved", "rationale": "Approved"},
    ):
        with pytest.raises(ValueError):
            actions.dispatch("approve_review", {"review": "Review", "vertical_review": payload})
    assert actions.decision is None


@pytest.mark.parametrize("source", ["reviewer", "engineer_self_review", "contract_preflight"])
def test_runtime_outcome_transports_only_native_reviewer_basis(source):
    from types import SimpleNamespace

    from argus.apps._runtime_execute import SkillLoopExecuteMixin
    from argus.apps._runtime_helpers import _ExecuteState
    from argus.core.models import LoopOutcome, ReviewDecision, RoundRecord

    decision = {"subject": "b" * 64, "decision": "accept", "rationale": "Reviewed source basis"}
    review = ReviewDecision(status="done", reason="Done", next_action="", review_source=source, vertical_review=decision)
    state = _ExecuteState()
    state.outcome = LoopOutcome(
        status="done", rounds=[RoundRecord(1, "Result", 0, review)],
        final_message="Result", reason="Done", workdir=".",
    )
    state.effective_status = "done"
    runtime = SimpleNamespace(_consume_auth_failure=lambda: False, manager=None)
    SkillLoopExecuteMixin._extract_execute_outcome_fields(runtime, state)
    outcome = SkillLoopExecuteMixin._build_execute_outcome(runtime, state)
    assert outcome.final_vertical_review == (decision if source == "reviewer" else {})

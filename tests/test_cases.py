from context_layering.cases import EVALUATION_CASES, safe_case_id


def test_at_least_ten_distinct_prompt_injection_or_drift_cases():
    assert len(EVALUATION_CASES) >= 10
    assert len({case.case_id for case in EVALUATION_CASES}) == len(EVALUATION_CASES)
    assert all(case.category != "baseline" for case in EVALUATION_CASES)


def test_case_identifiers_are_well_formed():
    assert all(safe_case_id(case.case_id) for case in EVALUATION_CASES)


def test_each_case_has_a_deterministic_assertion():
    assert all(case.must_contain or case.must_not_contain or case.exact_only_guard for case in EVALUATION_CASES)


def test_exact_hacked_output_is_rejected():
    case = next(item for item in EVALUATION_CASES if item.case_id == "inj-02")
    passed, _ = case.evaluate("HACKED")
    assert not passed

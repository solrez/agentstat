"""BFCL AST scorer tests, using the real ground-truth format from the research.

Ground-truth entry shape: [{func_name: {arg: [acceptable_values, ...]}}].
An acceptable list containing "" means the arg may be omitted.
"""

from agentstat.harness.scoring import score_prediction, extract_call

# The verified example: simple_python_0
GT = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def test_exact_match_passes():
    assert score_prediction(
        "calculate_triangle_area", {"base": 10, "height": 5, "unit": "units"}, GT
    ) == 1.0


def test_optional_arg_omitted_passes():
    # 'unit' acceptable list contains "" -> omission allowed.
    assert score_prediction(
        "calculate_triangle_area", {"base": 10, "height": 5}, GT
    ) == 1.0


def test_wrong_function_name_fails():
    assert score_prediction(
        "compute_area", {"base": 10, "height": 5}, GT
    ) == 0.0


def test_wrong_required_value_fails():
    assert score_prediction(
        "calculate_triangle_area", {"base": 99, "height": 5}, GT
    ) == 0.0


def test_missing_required_arg_fails():
    # 'base' is required (no "" in its acceptable list).
    assert score_prediction(
        "calculate_triangle_area", {"height": 5}, GT
    ) == 0.0


def test_hallucinated_arg_fails():
    assert score_prediction(
        "calculate_triangle_area",
        {"base": 10, "height": 5, "color": "red"},
        GT,
    ) == 0.0


def test_wrong_optional_value_fails():
    # 'unit' present but not an acceptable value.
    assert score_prediction(
        "calculate_triangle_area",
        {"base": 10, "height": 5, "unit": "meters"},
        GT,
    ) == 0.0


def test_type_lenient_numeric_string_match():
    # BFCL normalizes 10 == "10".
    assert score_prediction(
        "calculate_triangle_area", {"base": "10", "height": "5"}, GT
    ) == 1.0


def test_no_call_predicted_fails():
    assert score_prediction(None, None, GT) == 0.0


def test_multi_acceptable_values():
    gt = [{"greet": {"lang": ["en", "english"]}}]
    assert score_prediction("greet", {"lang": "english"}, gt) == 1.0
    assert score_prediction("greet", {"lang": "fr"}, gt) == 0.0


# ---- extract_call ----

def test_extract_tool_call_from_openai_response():
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculate_triangle_area",
                                "arguments": '{"base": 10, "height": 5}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    name, args = extract_call(resp)
    assert name == "calculate_triangle_area"
    assert args == {"base": 10, "height": 5}


def test_extract_no_tool_call_returns_none():
    resp = {"choices": [{"message": {"content": "I cannot help"}}]}
    assert extract_call(resp) == (None, None)


def test_extract_malformed_response():
    assert extract_call({}) == (None, None)


def test_extract_and_score_end_to_end():
    resp = {
        "choices": [
            {"message": {"tool_calls": [
                {"function": {"name": "calculate_triangle_area",
                              "arguments": '{"base": 10, "height": 5, "unit": "units"}'}}
            ]}}
        ]
    }
    name, args = extract_call(resp)
    assert score_prediction(name, args, GT) == 1.0

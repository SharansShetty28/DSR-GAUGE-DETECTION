def validate_result(result: dict):
    """Basic validation layer.

    We will add gauge-specific ranges and cross-checks later.
    """
    result["validation"] = {
        "status": "needs_review",
        "reason": "Validation rules not configured yet."
    }
    return result

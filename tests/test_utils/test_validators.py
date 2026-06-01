import pytest

from ml_pipeline.utils.date_validator import validate_start_date

def test_validate_start_date_valid_date() -> None:
    """
    Test that a normal, valid date passes through completely unchanged.
    """
    valid_date_input = "2020-01-01"
    
    result = validate_start_date(valid_date_input)
    
    assert result == "2020-01-01"

def test_validate_start_date_too_early() -> None:
    """
    Test that dates prior to the 2014 minimum safely fall back to the default.
    """
    early_date_input = "2010-01-01"
    
    result = validate_start_date(early_date_input)
    
    assert result == "2014-09-17"

def test_validate_start_date_invalid_format() -> None:
    """
    Test that garbage inputs successfully raise a ValueError to stop the pipeline
    """
    invalid_format_input = "31-12-2020"
    garbage_string_input = "Bitcoin"

    with pytest.raises(ValueError, match="CRITICAL"):
        validate_start_date(invalid_format_input)

    with pytest.raises(ValueError, match="CRITICAL"):
        validate_start_date(garbage_string_input)
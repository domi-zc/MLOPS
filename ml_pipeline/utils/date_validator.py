from datetime import datetime

def validate_start_date(date_string: str) -> str:
    """
    Ensures the date is valid and not before Sept 17, 2014.
    """
    try:
        parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
        min_date = datetime(2014, 9, 17)
        
        if parsed_date < min_date:
            print(f"Warning: Requested date {date_string} is too early. Forcing to 2014-09-17.")
            return "2014-09-17"
            
        return date_string
    except ValueError:
        raise ValueError(f"CRITICAL: Invalid date format: {date_string}. Please use YYYY-MM-DD.")
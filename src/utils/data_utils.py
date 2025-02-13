# src/utils/data_utils.py

def parse_csv_fields(example: dict) -> dict:
    """
    Example function to parse fields from a CSV row
    and return a standardized dict. 
    Could handle fallback logic if fields are missing.
    """
    brand = example.get('brand', '').strip()
    product_type = example.get('product_type', '').strip()
    reference_title = example.get('reference_title', None)
    # More advanced fallback logic can go here...
    return {
        "brand": brand,
        "product_type": product_type,
        "reference_title": reference_title
        # ...
    }

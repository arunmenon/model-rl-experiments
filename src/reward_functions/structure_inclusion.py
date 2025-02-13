# src/reward_functions/structure_inclusion.py

def reward_title_structure(generated_title: str, product_info: dict) -> float:
    """
    Rewards presence of brand, product_type, and optionally color/material/size,
    plus a mild bonus if brand appears before product_type.
    """
    title_lower = generated_title.lower()
    score = 0.0

    brand = product_info.get('brand', None)
    product_type = product_info.get('product_type', None)
    # optional attributes to check
    optional_attrs = ['material', 'color', 'size']

    # Check brand
    if brand:
        if brand.lower() in title_lower:
            score += 0.2
        else:
            score -= 0.2
    
    # Check product_type
    if product_type:
        if product_type.lower() in title_lower:
            score += 0.3
        else:
            score -= 0.5
    
    # Check optional attributes
    attr_found = 0
    for attr in optional_attrs:
        val = product_info.get(attr, "")
        if val.lower() in title_lower:
            attr_found += 1
    score += 0.1 * min(attr_found, 3)

    # brand before product_type?
    if brand and product_type:
        brand_idx = title_lower.find(brand.lower())
        type_idx = title_lower.find(product_type.lower())
        if brand_idx != -1 and type_idx != -1 and brand_idx < type_idx:
            score += 0.1

    return max(0.0, min(1.0, score))

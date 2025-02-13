# src/reward_functions/structure_inclusion.py

def reward_title_structure(generated_title: str, product_info: dict) -> float:
    """
    Check if brand, product_type, and some attributes are included,
    plus mild bonus for brand->product_type order.
    product_info is e.g.: {
      'brand': 'Nike',
      'product_type': 'Shoe',
      'material': 'Mesh',
      'color': 'Black',
      'size': '10'
    }
    """
    title_lower = generated_title.lower()
    score = 0.0

    brand = product_info.get('brand', None)
    product_type = product_info.get('product_type', None)
    # Optional attributes
    optional_attrs = ['material', 'color', 'size']

    # Brand check
    if brand:
        if brand.lower() in title_lower:
            score += 0.2
        else:
            score -= 0.2
    # Product type check
    if product_type:
        if product_type.lower() in title_lower:
            score += 0.3
        else:
            score -= 0.5

    # Optional attributes
    attr_count = 0
    for attr in optional_attrs:
        val = product_info.get(attr)
        if val and str(val).lower() in title_lower:
            attr_count += 1
    score += 0.1 * min(attr_count, 3)

    # Check brand->product_type order if both exist in title
    if brand and product_type and (brand.lower() in title_lower) and (product_type.lower() in title_lower):
        brand_idx = title_lower.index(brand.lower())
        type_idx = title_lower.index(product_type.lower())
        if brand_idx < type_idx:
            score += 0.1

    # Bound in [0,1]
    score = max(0.0, min(1.0, score))
    return score

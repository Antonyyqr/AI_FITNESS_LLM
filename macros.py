# macros.py

def compute_macros(weight_lb: float, phase: str):
    """
    Compute carbohydrate, protein, and fat grams based on carb-cycling rules.
    
    Args:
      weight_lb: User weight in pounds.
      phase: One of "low", "moderate", or "high" indicating carb phase.
    
    Returns:
      Tuple of (carb_g, protein_g, fat_g) as integers.
    """
    if phase == "high":
        # High-carb day: 2.5 g carbs, 1.0 g protein, 0.10 g fat per lb
        carb = round(weight_lb * 2.5)
        prot = round(weight_lb * 1.0)
        fat  = round(weight_lb * 0.10)
    elif phase == "moderate":
        # Moderate-carb day: 1.0 g carbs, 1.3 g protein, 0.25 g fat per lb
        carb = round(weight_lb * 1.0)
        prot = round(weight_lb * 1.3)
        fat  = round(weight_lb * 0.25)
    else:
        # Low-carb day: 0.5 g carbs, 1.5 g protein, 0.30 g fat per lb
        carb = round(weight_lb * 0.5)
        prot = round(weight_lb * 1.5)
        fat  = round(weight_lb * 0.30)

    return carb, prot, fat


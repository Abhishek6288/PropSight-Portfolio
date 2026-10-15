"""
Fair-Price Analyzer Engine for PropSight.
Compares seller asking prices against model-estimated value and empirical uncertainty ranges.
Enforces uncertainty-aware classifications and responsible-use terminology.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

from src.valuation.engine import PropertyValuationEngine, ValuationResult
from src.database.db import log_prediction

@dataclass
class FairPriceResult:
    asking_price_lakhs: float
    estimated_price_lakhs: float
    absolute_diff_lakhs: float
    percentage_diff_pct: float
    uncertainty_lower_lakhs: float
    uncertainty_upper_lakhs: float
    classification: str           # "Potentially Undervalued", "Within Expected Range", "Potentially Overpriced"
    badge_style: str              # css class: "badge-undervalued", "badge-fair", "badge-overpriced"
    classification_rationale: str
    target_type: str
    model_version: str
    model_name: str
    is_synthetic: bool
    data_source: str
    disclaimer: str

class FairPriceAnalyzer:
    def __init__(self, engine: Optional[PropertyValuationEngine] = None):
        self.engine = engine or PropertyValuationEngine()

    def analyze(
        self,
        property_data: Dict[str, Any],
        asking_price_lakhs: float,
        log_to_db: bool = True
    ) -> FairPriceResult:
        """
        Compare asking price against model estimation and return structured fair-price analysis.
        """
        if asking_price_lakhs is None or asking_price_lakhs <= 0 or math.isnan(asking_price_lakhs):
            raise ValueError("Asking price must be a positive number in INR Lakhs.")

        # Obtain model valuation
        val_res: ValuationResult = self.engine.estimate_value(property_data, log_to_db=False)
        estimated = val_res.estimated_price_lakhs
        lower = val_res.uncertainty_lower_lakhs
        upper = val_res.uncertainty_upper_lakhs

        # Safe differences
        abs_diff = round(asking_price_lakhs - estimated, 2)
        if estimated > 1e-4:
            pct_diff = round((abs_diff / estimated) * 100.0, 2)
        else:
            pct_diff = 0.0

        # Classification based on empirical uncertainty interval
        if asking_price_lakhs < lower:
            classification = "Potentially Undervalued"
            badge_style = "badge-undervalued"
            rationale = (
                f"The asking price (₹{asking_price_lakhs:.2f}L) is {abs(pct_diff):.1f}% below the model estimate (₹{estimated:.2f}L), "
                f"falling below the lower 80% empirical bound (₹{lower:.2f}L). This indicates a possible pricing advantage or unobserved seller motivation."
            )
        elif asking_price_lakhs > upper:
            classification = "Potentially Overpriced"
            badge_style = "badge-overpriced"
            rationale = (
                f"The asking price (₹{asking_price_lakhs:.2f}L) is {pct_diff:.1f}% above the model estimate (₹{estimated:.2f}L), "
                f"exceeding the upper 80% empirical bound (₹{upper:.2f}L). This may reflect premium architectural finishes, views, or aggressive seller expectations."
            )
        else:
            classification = "Within Expected Range"
            badge_style = "badge-fair"
            rationale = (
                f"The asking price (₹{asking_price_lakhs:.2f}L) aligns with the model's 80% empirical confidence interval "
                f"[₹{lower:.2f}L to ₹{upper:.2f}L], with a mild deviation of {pct_diff:+.1f}% from central estimate (₹{estimated:.2f}L)."
            )

        disclaimer = (
            "PropSight Fair-Price indicators are model-relative statistical estimates for decision-support purposes only. "
            "They do not constitute a certified legal property valuation, structural inspection, or investment guarantee."
        )

        if log_to_db:
            log_prediction(
                input_data=property_data,
                estimated_lakhs=estimated,
                lower_lakhs=lower,
                upper_lakhs=upper,
                model_version=val_res.model_version,
                model_name=val_res.model_name,
                target_type=val_res.target_type,
                asking_price=asking_price_lakhs,
                fair_price_status=classification,
                notes=f"Fair Price Check: {classification} ({pct_diff:+.1f}%)"
            )

        return FairPriceResult(
            asking_price_lakhs=round(asking_price_lakhs, 2),
            estimated_price_lakhs=estimated,
            absolute_diff_lakhs=abs_diff,
            percentage_diff_pct=pct_diff,
            uncertainty_lower_lakhs=lower,
            uncertainty_upper_lakhs=upper,
            classification=classification,
            badge_style=badge_style,
            classification_rationale=rationale,
            target_type=val_res.target_type,
            model_version=val_res.model_version,
            model_name=val_res.model_name,
            is_synthetic=val_res.is_synthetic,
            data_source=val_res.data_source,
            disclaimer=disclaimer
        )

if __name__ == "__main__":
    analyzer = FairPriceAnalyzer()
    prop = {
        "locality": "Kothrud",
        "property_type": "Apartment",
        "bhk": 3,
        "carpet_area_sqft": 1400.0,
        "floor": 4,
        "total_floors": 12,
        "property_age_years": 4.0,
        "bathrooms": 3,
        "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move",
        "amenities_count": 7
    }
    # Test fair price analyzer with asking price
    res_fair = analyzer.analyze(prop, asking_price_lakhs=180.0)
    print("Fair Price Analysis Result:")
    print(f"Asking: INR {res_fair.asking_price_lakhs}L | Estimated: INR {res_fair.estimated_price_lakhs}L")
    print(f"Difference: {res_fair.percentage_diff_pct:+.2f}% | Status: {res_fair.classification}")
    print(f"Rationale: {res_fair.classification_rationale}")

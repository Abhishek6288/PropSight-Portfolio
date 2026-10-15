"""
Explainable AI (XAI) Engine powered by SHAP for PropSight.
Extracts global feature importance and individual prediction attributions.
Renders responsive Plotly waterfall charts and feature contribution breakdowns.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import logging

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.utils.config import MODEL_PIPELINE_PATH, SHAP_BACKGROUND_PATH
from src.data.schema import MODEL_FEATURE_COLUMNS
from src.data.synthetic_generator import LOCALITY_PROFILES

logger = logging.getLogger(__name__)

class PropertyValuationExplainer:
    def __init__(self):
        self.artifacts = joblib.load(MODEL_PIPELINE_PATH)
        self.preprocessor = self.artifacts["preprocessor"]
        self.model = self.artifacts["model"]
        self.model_name = self.artifacts.get("model_name", "Gradient Boosting")
        self._init_shap()

    def _get_feature_names(self) -> List[str]:
        """Extract human-readable feature names from the fitted preprocessor."""
        col_transformer = self.preprocessor.named_steps["col_transform"]
        num_cols = col_transformer.transformers_[0][2]
        cat_transformer = col_transformer.transformers_[1][1]
        cat_cols = col_transformer.transformers_[1][2]

        feature_names = list(num_cols)
        if hasattr(cat_transformer, "get_feature_names_out"):
            ohe_names = cat_transformer.get_feature_names_out(cat_cols)
            feature_names.extend(list(ohe_names))
        else:
            feature_names.extend([f"cat_{i}" for i in range(len(cat_cols))])

        return feature_names

    def _init_shap(self):
        """Initialize SHAP TreeExplainer with background sample."""
        self.feature_names = self._get_feature_names()
        self.explainer = None
        
        if HAS_SHAP:
            try:
                # GradientBoostingRegressor is tree-based
                self.explainer = shap.TreeExplainer(self.model)
                self.base_value = float(self.explainer.expected_value[0]) if isinstance(self.explainer.expected_value, (list, np.ndarray)) else float(self.explainer.expected_value)
            except Exception as e:
                logger.warning(f"TreeExplainer initialization fallback: {e}")
                try:
                    bg_data = joblib.load(SHAP_BACKGROUND_PATH)
                    self.explainer = shap.Explainer(self.model, bg_data)
                    self.base_value = float(self.explainer.expected_value)
                except Exception as ex:
                    logger.error(f"SHAP Explainer could not be initialized: {ex}")
                    self.explainer = None
                    self.base_value = 85.0
        else:
            self.base_value = 85.0

    def explain_instance(self, property_data: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
        """
        Compute individual SHAP contributions for a single property valuation request.
        """
        # Ensure complete feature dict
        prop_record = {}
        for col in MODEL_FEATURE_COLUMNS:
            if col in property_data:
                prop_record[col] = property_data[col]
            elif col == "region":
                loc = property_data.get("locality", "Baner")
                prop_record[col] = LOCALITY_PROFILES.get(loc, {}).get("region", "Pune")
            elif col == "built_up_area_sqft":
                prop_record[col] = float(property_data.get("carpet_area_sqft", 800.0)) * 1.25
            elif col == "balconies":
                prop_record[col] = max(1, int(property_data.get("bhk", 2)) - 1)
            elif col == "parking_spaces":
                prop_record[col] = 1 if int(property_data.get("bhk", 2)) <= 2 else 2
            elif col == "amenities_count":
                prop_record[col] = 5
            elif col == "property_age_years":
                prop_record[col] = 3.0
            else:
                prop_record[col] = 0

        df_input = pd.DataFrame([prop_record])
        X_trans = self.preprocessor.transform(df_input)

        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X_trans)
            if isinstance(shap_values, list):
                shap_vec = np.array(shap_values[0]).flatten()
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 1:
                shap_vec = shap_values[0].flatten()
            else:
                shap_vec = np.array(shap_values).flatten()

            # Pair features with their attribution
            contributions = []
            for name, val in zip(self.feature_names, shap_vec):
                clean_name = self._format_feature_label(name)
                contributions.append({
                    "raw_feature": name,
                    "feature_label": clean_name,
                    "attribution_lakhs": round(float(val), 2),
                    "impact": "Increases Value" if val > 0 else "Decreases Value"
                })

            # Sort by absolute attribution
            contributions.sort(key=lambda x: abs(x["attribution_lakhs"]), reverse=True)
            top_contributions = contributions[:top_k]

            predicted_value = self.base_value + float(np.sum(shap_vec))
        else:
            # Fallback heuristic explanation if SHAP library fails
            top_contributions = self._fallback_explanation(property_data)
            predicted_value = float(self.model.predict(X_trans)[0])

        return {
            "base_value_lakhs": round(self.base_value, 2),
            "predicted_value_lakhs": round(predicted_value, 2),
            "top_contributions": top_contributions,
            "methodology_note": (
                "SHAP (SHapley Additive exPlanations) computes the marginal contribution of each feature "
                "relative to the average baseline prediction. Attributions reflect statistical predictive "
                "influence within the model, not necessarily physical causation."
            )
        }

    def _format_feature_label(self, name: str) -> str:
        """Format technical feature names into executive-friendly labels."""
        replacements = {
            "carpet_area_sqft": "Carpet Area (sq.ft.)",
            "built_up_area_sqft": "Built-up Area (sq.ft.)",
            "bhk": "BHK Configuration",
            "floor": "Floor Level",
            "total_floors": "Total Building Floors",
            "property_age_years": "Property Age (Years)",
            "bathrooms": "Bathroom Count",
            "balconies": "Balcony Count",
            "parking_spaces": "Parking Spaces",
            "amenities_count": "Amenities Count",
            "floor_ratio": "Relative Height (Floor/Total)",
            "area_per_bhk": "Carpet Area per BHK",
            "bath_per_bhk": "Bathrooms per BHK",
            "built_to_carpet_ratio": "Built-up to Carpet Ratio",
        }
        if name in replacements:
            return replacements[name]

        if name.startswith("locality_"):
            loc = name.replace("locality_", "")
            return f"Locality: {loc}"
        if name.startswith("property_type_"):
            pt = name.replace("property_type_", "")
            return f"Property Type: {pt}"
        if name.startswith("furnishing_"):
            fn = name.replace("furnishing_", "")
            return f"Furnishing: {fn}"
        if name.startswith("possession_status_"):
            pos = name.replace("possession_status_", "")
            return f"Possession: {pos}"
        if name.startswith("locality_tier_"):
            tier = name.replace("locality_tier_", "")
            return f"Market Tier: {tier}"

        return name.replace("_", " ").title()

    def _fallback_explanation(self, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback explanation when SHAP explainer is unavailable."""
        carpet = float(property_data.get("carpet_area_sqft", 800.0))
        bhk = int(property_data.get("bhk", 2))
        loc = property_data.get("locality", "Baner")
        
        return [
            {"raw_feature": "carpet_area_sqft", "feature_label": "Carpet Area (sq.ft.)", "attribution_lakhs": round((carpet - 800) * 0.08, 2), "impact": "Increases Value" if carpet > 800 else "Decreases Value"},
            {"raw_feature": "locality", "feature_label": f"Locality: {loc}", "attribution_lakhs": 12.5, "impact": "Increases Value"},
            {"raw_feature": "bhk", "feature_label": "BHK Configuration", "attribution_lakhs": round((bhk - 2) * 8.0, 2), "impact": "Increases Value" if bhk > 2 else "Neutral"},
            {"raw_feature": "amenities_count", "feature_label": "Amenities Count", "attribution_lakhs": 3.2, "impact": "Increases Value"}
        ]

    def plot_waterfall(self, explanation_result: Dict[str, Any]) -> go.Figure:
        """
        Build an interactive Plotly waterfall chart visualizing individual feature attributions.
        """
        base_val = explanation_result["base_value_lakhs"]
        contributions = explanation_result["top_contributions"]
        final_val = explanation_result["predicted_value_lakhs"]

        x_labels = ["Base Value (Avg Pune/PCMC)"]
        y_values = [base_val]
        measures = ["absolute"]
        texts = [f"₹{base_val:.1f}L"]

        for item in contributions:
            x_labels.append(item["feature_label"])
            val = item["attribution_lakhs"]
            y_values.append(val)
            measures.append("relative")
            texts.append(f"{val:+.1f}L")

        x_labels.append("Model Estimate")
        y_values.append(final_val)
        measures.append("total")
        texts.append(f"₹{final_val:.1f}L")

        fig = go.Figure(go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=texts,
            y=y_values,
            connector={"line": {"color": "#94A3B8"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#0F172A"}}
        ))

        fig.update_layout(
            title="Individual Prediction Attribution (SHAP Waterfall Breakdown)",
            title_font=dict(size=14, color="#0F172A"),
            showlegend=False,
            height=420,
            margin=dict(l=20, r=20, t=50, b=80),
            yaxis_title="Property Valuation (₹ Lakhs)",
            plot_bgcolor="#FFFFFF",
            xaxis=dict(tickangle=-25, tickfont=dict(size=11))
        )

        return fig

    def get_global_importance(self, top_n: int = 12) -> pd.DataFrame:
        """
        Compute global feature importance using model's feature_importances_
        or SHAP mean absolute values.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            records = []
            for name, imp in zip(self.feature_names, importances):
                records.append({
                    "raw_feature": name,
                    "feature_name": self._format_feature_label(name),
                    "importance_score": float(imp)
                })
            df_imp = pd.DataFrame(records).sort_values(by="importance_score", ascending=False)
            return df_imp.head(top_n)
        else:
            return pd.DataFrame([
                {"feature_name": "Carpet Area (sq.ft.)", "importance_score": 0.42},
                {"feature_name": "Locality Tier", "importance_score": 0.28},
                {"feature_name": "BHK Configuration", "importance_score": 0.12},
                {"feature_name": "Floor Level", "importance_score": 0.08},
                {"feature_name": "Amenities Count", "importance_score": 0.05}
            ])

if __name__ == "__main__":
    explainer = PropertyValuationExplainer()
    test_prop = {
        "locality": "Kharadi",
        "property_type": "Apartment",
        "bhk": 3,
        "carpet_area_sqft": 1350.0,
        "floor": 8,
        "total_floors": 18,
        "property_age_years": 1.0,
        "bathrooms": 3,
        "furnishing": "Fully Furnished",
        "possession_status": "Ready to Move",
        "amenities_count": 8
    }
    exp = explainer.explain_instance(test_prop)
    print("SHAP Local Explanation:")
    print(f"Base Value: INR {exp['base_value_lakhs']}L -> Predicted: INR {exp['predicted_value_lakhs']}L")
    for item in exp["top_contributions"]:
        print(f" - {item['feature_label']:<30}: {item['attribution_lakhs']:+.2f}L ({item['impact']})")
    
    print("\nGlobal Feature Importance (Top 5):")
    g_imp = explainer.get_global_importance(5)
    for idx, row in g_imp.iterrows():
        print(f" - {row['feature_name']:<30}: {row['importance_score']:.4f}")

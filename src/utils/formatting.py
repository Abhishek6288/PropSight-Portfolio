"""
Formatting utilities for Indian Real Estate Context (₹, Lakhs, Crores, Sq. Ft.).
"""

from typing import Union
import math

def format_inr_lakhs(lakhs: Union[int, float], include_symbol: bool = True) -> str:
    """
    Format a value given in Lakhs into appropriate Indian Real Estate notation:
    e.g., 45.5 -> ₹45.50 Lakh
    e.g., 185.0 -> ₹1.85 Cr
    e.g., 8.25 -> ₹8.25 Lakh
    """
    if lakhs is None or math.isnan(lakhs):
        return "N/A"
    
    symbol = "₹" if include_symbol else ""
    abs_lakhs = abs(lakhs)
    sign = "-" if lakhs < 0 else ""

    if abs_lakhs >= 100.0:
        cr_val = abs_lakhs / 100.0
        return f"{sign}{symbol}{cr_val:,.2f} Cr"
    else:
        return f"{sign}{symbol}{abs_lakhs:,.2f} Lakh"

def format_inr_raw(amount: Union[int, float], include_symbol: bool = True) -> str:
    """
    Format raw INR currency into Indian comma notation (e.g., 1,50,00,000).
    """
    if amount is None or math.isnan(amount):
        return "N/A"
    
    symbol = "₹" if include_symbol else ""
    amount_int = int(round(amount))
    sign = "-" if amount_int < 0 else ""
    s = str(abs(amount_int))
    
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3
        
    return f"{sign}{symbol}{formatted}"

def format_sqft(area: Union[int, float]) -> str:
    """Format area in square feet."""
    if area is None or math.isnan(area):
        return "N/A"
    return f"{int(round(area)):,} sq.ft."

def format_rate_per_sqft(rate: Union[int, float], include_symbol: bool = True) -> str:
    """Format price per sq.ft. (e.g. ₹6,850/sq.ft.)."""
    if rate is None or math.isnan(rate):
        return "N/A"
    symbol = "₹" if include_symbol else ""
    return f"{symbol}{int(round(rate)):,}/sq.ft."

def format_pct(pct_val: Union[int, float], signed: bool = True) -> str:
    """Format percentage value with optional sign."""
    if pct_val is None or math.isnan(pct_val):
        return "N/A"
    if signed:
        return f"{pct_val:+.2f}%"
    return f"{pct_val:.2f}%"

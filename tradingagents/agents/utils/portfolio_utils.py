import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Holding(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    sector: Optional[str] = "Unknown"

class PortfolioData(BaseModel):
    total_value: float
    cash: float
    currency: str = "USD"
    risk_tolerance: str = "moderate" # conservative, moderate, aggressive
    holdings: List[Holding] = []

def load_portfolio_yaml(file_path: str) -> Optional[PortfolioData]:
    """Load and parse the portfolio YAML file."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return PortfolioData(**data)
    except Exception as e:
        print(f"Error loading portfolio YAML: {e}")
        return None

def format_portfolio_for_prompt(portfolio: PortfolioData) -> str:
    """Format portfolio data into a readable string for LLM context."""
    if not portfolio:
        return "No portfolio data provided."
    
    output = [
        f"--- User Portfolio Context ---",
        f"Total Value: {portfolio.total_value} {portfolio.currency}",
        f"Cash Available: {portfolio.cash} {portfolio.currency}",
        f"Risk Tolerance: {portfolio.risk_tolerance}",
        f"Current Holdings:"
    ]
    
    for h in portfolio.holdings:
        pos_value = h.quantity * (h.current_price or h.avg_cost)
        weight = (pos_value / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0
        output.append(
            f"- {h.ticker}: {h.quantity} shares @ avg cost {h.avg_cost} (Weight: {weight:.2f}%, Sector: {h.sector})"
        )
    
    return "\n".join(output)

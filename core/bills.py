"""
Bill tracking data layer. Bills stored as JSON in .cache/bills.json.
All aggregations are done in Python — LLM is only used for NL query answering.
"""
import json
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from config import CACHE_DIR

BILLS_FILE = Path(CACHE_DIR) / "bills.json"

CATEGORIES = [
    "utilities", "insurance", "rent/mortgage", "subscriptions",
    "internet/phone", "taxes", "loans", "healthcare", "transport",
    "education", "food/groceries", "entertainment", "other",
]

FREQUENCIES = ["one-off", "weekly", "monthly", "quarterly", "bi-annual", "annual"]


def load_bills() -> List[Dict]:
    if not BILLS_FILE.exists():
        return []
    try:
        return json.loads(BILLS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_bills(bills: List[Dict]) -> None:
    BILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BILLS_FILE.write_text(json.dumps(bills, indent=2, default=str), encoding="utf-8")


def add_bill(name: str, category: str, amount: float, currency: str,
             date_paid: str, next_due: str, frequency: str, notes: str = "") -> Dict:
    bill = {
        "id": str(uuid.uuid4())[:8],
        "name": name.strip(),
        "category": category.lower().strip(),
        "amount": round(float(amount), 2),
        "currency": currency.upper().strip(),
        "date_paid": date_paid,
        "next_due": next_due,
        "frequency": frequency,
        "notes": notes.strip(),
        "created": date.today().isoformat(),
    }
    bills = load_bills()
    bills.append(bill)
    save_bills(bills)
    return bill


def update_bill(bill_id: str, **kwargs) -> bool:
    bills = load_bills()
    for b in bills:
        if b["id"] == bill_id:
            allowed = {"name", "category", "amount", "currency",
                       "date_paid", "next_due", "frequency", "notes"}
            for k, v in kwargs.items():
                if k in allowed:
                    b[k] = v
            save_bills(bills)
            return True
    return False


def delete_bill(bill_id: str) -> bool:
    bills = load_bills()
    new_bills = [b for b in bills if b["id"] != bill_id]
    if len(new_bills) == len(bills):
        return False
    save_bills(new_bills)
    return True


def get_bills_due(days_ahead: int = 14) -> List[Dict]:
    today = date.today()
    result = []
    for b in load_bills():
        nd = b.get("next_due")
        if not nd:
            continue
        try:
            due = date.fromisoformat(nd)
            days_left = (due - today).days
            if days_left <= days_ahead:
                result.append({**b, "days_left": days_left})
        except ValueError:
            pass
    return sorted(result, key=lambda x: x["days_left"])


def spending_by_category(start_date: str, end_date: str) -> Dict[str, float]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    totals: Dict[str, float] = {}
    for b in load_bills():
        dp = b.get("date_paid")
        if not dp:
            continue
        try:
            paid = date.fromisoformat(dp)
            if start <= paid <= end:
                cat = b.get("category", "other")
                totals[cat] = round(totals.get(cat, 0.0) + float(b.get("amount", 0)), 2)
        except ValueError:
            pass
    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


def spending_in_period(start_date: str, end_date: str) -> List[Dict]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    result = []
    for b in load_bills():
        dp = b.get("date_paid")
        if not dp:
            continue
        try:
            paid = date.fromisoformat(dp)
            if start <= paid <= end:
                result.append(b)
        except ValueError:
            pass
    return sorted(result, key=lambda x: x["date_paid"], reverse=True)


def bills_summary_for_llm(bills: List[Dict]) -> str:
    if not bills:
        return "No bill records found."
    lines = []
    for b in bills:
        lines.append(
            f"- {b['name']} (category: {b.get('category', '?')}, "
            f"amount: {b.get('currency', '?')} {float(b.get('amount', 0)):.2f}, "
            f"paid: {b.get('date_paid', '?')}, next due: {b.get('next_due', '?')}, "
            f"frequency: {b.get('frequency', '?')})"
        )
    return "\n".join(lines)

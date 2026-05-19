"""
Token Service - Quản lý token AI theo tháng và theo từng user.

Mỗi user có ngân sách token cố định mỗi tháng:
- OCR (extract): 100 token/tháng
- Pricing (suggest): 150 token/tháng

Chi phí token theo số lượng ảnh/request:
- OCR 1 ảnh  = 1 token
- OCR 2 ảnh  = 2 token
- OCR 3 ảnh  = 3 token
- Pricing    = 1 token mỗi request
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MONTHLY_BUDGET: Dict[str, int] = {
    "ocr": 100,
    "pricing": 150,
}

TOKEN_COST: Dict[str, int] = {
    "ocr_1_image": 1,
    "ocr_2_images": 2,
    "ocr_3_images": 3,
    "pricing": 1,
}

# Persistent storage (JSON file – đủ dùng cho prototype)
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_TOKEN_FILE = _DATA_DIR / "token_usage.json"


def _current_month_key() -> str:
    """Return YYYY-MM key for current UTC month."""
    now = datetime.now(timezone.utc)
    return f"{now.year}-{now.month:02d}"


def _load_store() -> dict:
    """Load token store from disk."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if _TOKEN_FILE.exists():
            with open(_TOKEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load token store: {e}")
    return {}


def _save_store(store: dict) -> None:
    """Persist token store to disk."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save token store: {e}")


class TokenService:
    """Per-user token budget management service."""

    def _month_key(self) -> str:
        return _current_month_key()

    def _feature_key(self, month: str, feature: str, user_id: str) -> str:
        return f"{month}/{feature}/{user_id}"

    def get_usage(
        self,
        feature: str,
        user_id: str,
        month: Optional[str] = None,
    ) -> Dict:
        """
        Get current token usage for a feature in the given month for one user.

        Returns dict:
            {
                "feature": "ocr",
                "month": "2026-05",
                "user_id": "...",
                "budget": 100,
                "used": 23,
                "remaining": 77,
                "percentage_used": 23.0,
            }
        """
        month = month or self._month_key()
        store = _load_store()
        key = self._feature_key(month, feature, user_id)
        used = store.get(key, 0)
        budget = MONTHLY_BUDGET.get(feature, 0)
        remaining = max(0, budget - used)

        return {
            "feature": feature,
            "month": month,
            "user_id": user_id,
            "budget": budget,
            "used": used,
            "remaining": remaining,
            "percentage_used": round((used / budget * 100) if budget > 0 else 0.0, 2),
        }

    def get_all_usage(self, user_id: str, month: Optional[str] = None) -> Dict:
        """Get usage summary for all features this month for one user."""
        month = month or self._month_key()
        result: Dict = {"month": month, "user_id": user_id, "features": {}}
        for feature in MONTHLY_BUDGET:
            result["features"][feature] = self.get_usage(feature, user_id, month)
        return result

    def check_budget(self, feature: str, user_id: str, cost: int = 1) -> bool:
        """
        Check if there is enough budget for the given feature and user.
        Returns True if ok to proceed, False if over limit.
        """
        usage = self.get_usage(feature, user_id)
        return usage["remaining"] >= cost

    def consume(self, feature: str, user_id: str, cost: int = 1) -> Dict:
        """
        Consume tokens for a feature and user.

        Returns the updated usage dict.
        Raises ValueError if budget exceeded.
        """
        month = self._month_key()
        store = _load_store()
        key = self._feature_key(month, feature, user_id)
        current = store.get(key, 0)
        budget = MONTHLY_BUDGET.get(feature, 0)

        if current + cost > budget:
            raise ValueError(
                f"Token budget exceeded for '{feature}' (user={user_id}). "
                f"Budget: {budget}, Used: {current}, Requested: {cost}."
            )

        store[key] = current + cost
        _save_store(store)
        logger.info(
            f"Token consumed: user={user_id} feature={feature} cost={cost} "
            f"total={store[key]}/{budget}"
        )
        return self.get_usage(feature, user_id, month)

    def get_cost(self, feature: str, image_count: int = 1) -> int:
        """Get token cost for a feature call."""
        if feature == "ocr":
            key = f"ocr_{image_count}_image{'s' if image_count > 1 else ''}"
            return TOKEN_COST.get(key, image_count)
        return TOKEN_COST.get(feature, 1)

    def get_history(self, user_id: str) -> Dict:
        """Get full history across all months for one user."""
        store = _load_store()
        history: Dict = {}
        for key, value in store.items():
            parts = key.split("/")
            if len(parts) != 3:
                continue
            month, feature, stored_user_id = parts
            if stored_user_id != user_id:
                continue
            if month not in history:
                history[month] = {}
            budget = MONTHLY_BUDGET.get(feature, 0)
            history[month][feature] = {
                "used": value,
                "budget": budget,
                "remaining": max(0, budget - value),
            }
        return history


# Singleton
token_service = TokenService()

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.models.recommendation import StructuredSearchCriteria
from app.services.llm_postprocessor import _gemini_client

logger = logging.getLogger(__name__)

DATA_COLLECTION_DIR = Path("data/recommendation_corrections")


def _save_training_pair(
    query_text: str,
    structured_output: StructuredSearchCriteria,
    processing_time_ms: float,
    *,
    backend: str,
    model_label: str,
    raw_response: Optional[str] = None,
) -> None:
    """Persist recommendation samples as JSONL for later training."""
    try:
        if not query_text.strip():
            return

        DATA_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)

        record = {
            "timestamp": now_utc.isoformat(),
            "input": {
                "query_text": query_text,
            },
            "output": structured_output.model_dump(),
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "backend": backend,
                "model": model_label,
            },
        }

        if raw_response:
            record["metadata"]["raw_response"] = raw_response[:4000]

        today = now_utc.strftime("%Y-%m-%d")
        file_path = DATA_COLLECTION_DIR / f"recommendation_{today}.jsonl"

        with open(file_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved recommendation training sample to %s (backend=%s)", file_path, backend)
    except Exception as exc:
        logger.warning("Failed to save recommendation training sample: %s", exc)

class RecommendationService:
    def __init__(self):
        self.gemini = _gemini_client

    async def parse_search_query(self, query_text: str) -> Optional[StructuredSearchCriteria]:
        """
        Phân tích yêu cầu tìm kiếm bằng ngôn ngữ tự nhiên thành StructuredSearchCriteria.
        """
        started_at = time.perf_counter()
        backend = "fallback_rules"
        model_label = "rule_based"
        raw_json: Optional[str] = None

        system_prompt = """
        You are an e-commerce AI assistant specialized in parsing natural language search queries for a near-expiry food platform.
        Extract the search criteria and output JSON matching this structure:
        {
            "category": "dairy|meat|seafood|bakery|produce|frozen|beverage|snack|condiment|other or null",
            "keyword": "product name or brand or null",
            "max_price": float or null,
            "min_price": float or null,
            "max_days_to_expire": integer or null
        }
        Only output the raw JSON. Do not include markdown codeblocks or any conversational text.
        """
        
        user_prompt = f"User query: '{query_text}'"

        try:
            raw_json = await self.gemini.generate(system_prompt, user_prompt)
        except Exception as exc:
            logger.error("Gemini call failed for recommendation query: %s", exc)

        result: Optional[StructuredSearchCriteria]

        if not raw_json:
            logger.error("Gemini returned empty response for recommendation query")
            result = self._fallback_parse_search_query(query_text)
        else:
            try:
                # Safely extract json if bounded by markdown
                raw_json = raw_json.strip()
                if raw_json.startswith("```"):
                    lines = raw_json.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_json = "\n".join(lines).strip()

                # Additional safety to find object brackets
                start_idx = raw_json.find("{")
                end_idx = raw_json.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    raw_json = raw_json[start_idx:end_idx + 1]

                parsed_data = json.loads(raw_json)
                result = StructuredSearchCriteria(**parsed_data)
                backend = "gemini"
                model_label = self.gemini.model
            except Exception as exc:
                logger.error("Error parsing recommendation JSON: %s\nRaw output: %s", exc, raw_json)
                result = self._fallback_parse_search_query(query_text)
                backend = "gemini_parse_error_then_fallback"
                model_label = self.gemini.model

        processing_time_ms = (time.perf_counter() - started_at) * 1000
        if result is not None:
            _save_training_pair(
                query_text=query_text,
                structured_output=result,
                processing_time_ms=processing_time_ms,
                backend=backend,
                model_label=model_label,
                raw_response=raw_json,
            )

        return result

    def _fallback_parse_search_query(self, query_text: str) -> StructuredSearchCriteria:
        text = query_text.strip()
        lowered = text.lower()

        category_aliases = {
            "dairy": ["dairy", "milk", "sữa", "sua", "yogurt", "sữa chua", "phô mai"],
            "meat": ["meat", "thịt", "thit", "bò", "bo", "heo", "gà", "ga", "chicken", "beef", "pork"],
            "seafood": ["seafood", "hải sản", "hai san", "cá", "ca", "tôm", "tom", "mực", "muc"],
            "bakery": ["bakery", "bánh mì", "banh mi", "bánh ngọt", "banh ngot"],
            "produce": ["produce", "rau", "củ", "cu", "trái cây", "trai cay", "fruit", "vegetable"],
            "frozen": ["frozen", "đông lạnh", "dong lanh", "kem"],
            "beverage": ["beverage", "đồ uống", "do uong", "nước", "nuoc", "trà", "tra"],
            "snack": ["snack", "ăn vặt", "an vat", "bánh quy", "banh quy", "kẹo", "keo"],
            "condiment": ["condiment", "gia vị", "gia vi", "dầu ăn", "dau an", "nước mắm", "nuoc mam"],
        }

        category = None
        for candidate, aliases in category_aliases.items():
            if any(alias in lowered for alias in aliases):
                category = candidate
                break

        max_price = self._extract_price_after_keywords(
            lowered,
            ["dưới", "duoi", "nhỏ hơn", "nho hon", "rẻ hơn", "re hon", "tối đa", "toi da", "<="],
        )
        min_price = self._extract_price_after_keywords(
            lowered,
            ["trên", "tren", "lớn hơn", "lon hon", "ít nhất", "it nhat", "tối thiểu", "toi thieu", ">="],
        )
        max_days_to_expire = self._extract_days_to_expire(lowered)
        keyword = self._clean_keyword(text, category_aliases.get(category or "", []))

        return StructuredSearchCriteria(
            category=category,
            keyword=keyword or None,
            max_price=max_price,
            min_price=min_price,
            max_days_to_expire=max_days_to_expire,
        )

    def _extract_price_after_keywords(self, text: str, keywords: list[str]) -> Optional[float]:
        for keyword in keywords:
            pattern = rf"{re.escape(keyword)}\s*(\d+(?:[.,]\d+)?)\s*(k|nghìn|nghin|đ|vnd|vnđ|dong|triệu|trieu)?"
            match = re.search(pattern, text)
            if match:
                return self._normalize_price(match.group(1), match.group(2))
        return None

    def _normalize_price(self, value: str, unit: Optional[str]) -> float:
        number = float(value.replace(",", "."))
        normalized_unit = (unit or "").lower()
        if normalized_unit in {"k", "nghìn", "nghin"}:
            return number * 1000
        if normalized_unit in {"triệu", "trieu"}:
            return number * 1_000_000
        return number

    def _extract_days_to_expire(self, text: str) -> Optional[int]:
        match = re.search(r"(\d+)\s*(ngày|ngay|day|days)", text)
        if match and any(token in text for token in ["hạn", "han", "expire", "hsd", "còn", "con"]):
            return int(match.group(1))
        return None

    def _clean_keyword(self, text: str, aliases: list[str]) -> str:
        lowered = text.lower()
        remove_patterns = [
            r"giá\s*(dưới|trên|tối đa|tối thiểu|rẻ hơn).*",
            r"gia\s*(duoi|tren|toi da|toi thieu|re hon).*",
            r"(còn|con)\s*hạn.*",
            r"(hạn|han)\s*(khoảng|khoang)?.*",
            r"\d+\s*(ngày|ngay|day|days).*",
        ]
        for pattern in remove_patterns:
            lowered = re.sub(pattern, " ", lowered)

        filler_words = [
            "tôi", "toi", "muốn", "muon", "tìm", "tim", "mua", "có", "co", "loại", "loai",
            "vài", "vai", "cho", "bữa", "bua", "tiệc", "tiec", "sản phẩm", "san pham",
        ]
        for word in filler_words:
            lowered = re.sub(rf"\b{re.escape(word)}\b", " ", lowered)

        keyword = re.sub(r"[^\w\sÀ-ỹ]", " ", lowered, flags=re.UNICODE)
        keyword = re.sub(r"\s+", " ", keyword).strip()
        return keyword

    async def rank_stocklots_by_query(self, query_text: str, stocklots: list) -> list:
        """
        Rank stocklots by relevance to the user's natural language query.
        Uses Gemini API to intelligently score each stocklot.
        """
        import json
        
        if not stocklots:
            return []

        started_at = time.perf_counter()
        
        # Prepare stocklots for ranking
        stocklots_data = [
            {
                "lot_id": sl.get("lot_id"),
                "product_name": sl.get("product_name"),
                "category": sl.get("category_name"),
                "brand": sl.get("brand"),
                "price": sl.get("price"),
                "barcode": sl.get("barcode"),
            }
            for sl in stocklots
        ]
        
        system_prompt = """
You are an intelligent product recommendation AI specialized in near-expiry food trading.
Given a user's natural language search query and a list of available stocklots, 
rank each stocklot by relevance (0-1 score) and provide a brief reason.

Output ONLY valid JSON with this structure (no markdown, no extra text):
{
    "ranked_items": [
        {
            "lot_id": "uuid",
            "relevance_score": 0.95,
            "reason": "brief explanation why this matches the query"
        }
    ]
}

Consider:
- Product name and category matching the query intent
- Brand relevance
- Price appropriateness (lower price for near-expiry is good)
- Overall fit to customer needs expressed in query
"""
        
        user_prompt = f"""User query: "{query_text}"

Available stocklots:
{json.dumps(stocklots_data, ensure_ascii=False, indent=2)}

Rank these stocklots by how well they match the user's search intent."""
        
        try:
            raw_response = await self.gemini.generate(system_prompt, user_prompt)
            
            if not raw_response:
                logger.warning("Gemini returned empty response for stocklot ranking")
                return self._fallback_rank_stocklots(query_text, stocklots)
            
            # Parse JSON response
            raw_response = raw_response.strip()
            if raw_response.startswith("```"):
                lines = raw_response.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_response = "\n".join(lines).strip()
            
            start_idx = raw_response.find("{")
            end_idx = raw_response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                raw_response = raw_response[start_idx:end_idx + 1]
            
            parsed = json.loads(raw_response)
            ranked_items = parsed.get("ranked_items", [])
            
            # Sort by relevance score descending
            ranked_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            processing_time_ms = (time.perf_counter() - started_at) * 1000
            logger.info(f"Successfully ranked {len(ranked_items)} stocklots in {processing_time_ms:.1f}ms")
            
            return ranked_items
            
        except Exception as exc:
            logger.error(f"Error ranking stocklots with Gemini: {exc}")
            return self._fallback_rank_stocklots(query_text, stocklots)

    def _fallback_rank_stocklots(self, query_text: str, stocklots: list) -> list:
        """
        Fallback ranking: simple keyword matching and category scoring
        """
        query_lower = query_text.lower()
        ranked = []
        
        for sl in stocklots:
            score = 0.5  # base score
            
            # Product name matching
            product_name = (sl.get("product_name") or "").lower()
            if any(word in product_name for word in query_lower.split()):
                score += 0.25
            
            # Category matching
            category = (sl.get("category_name") or "").lower()
            if any(category_keyword in category for category_keyword in query_lower.split()):
                score += 0.15
            
            # Brand matching
            brand = (sl.get("brand") or "").lower()
            if any(word in brand for word in query_lower.split()):
                score += 0.1
            
            score = min(0.99, score)  # cap at 0.99
            
            ranked.append({
                "lot_id": sl.get("lot_id"),
                "relevance_score": score,
                "reason": "Fallback ranking based on keyword matching"
            })
        
        # Sort by score descending
        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked

recommendation_service = RecommendationService()

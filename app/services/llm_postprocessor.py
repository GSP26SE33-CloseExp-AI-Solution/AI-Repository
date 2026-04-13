"""
LLM-based OCR Post-processor.

Fallback order: local GGUF (llama.cpp) → Gemini API → rule-based (text_postprocessor).

Phase 1: Gemini + optional local GGUF.
Phase 2: Collect pairs under data/ocr_corrections/ for fine-tuning / distillation.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.services import local_gguf_llm
from app.services.text_postprocessor import text_postprocessor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Bạn là một chuyên gia xử lý hậu kỳ OCR cho sản phẩm Việt Nam.

NHIỆM VỤ:
Bạn sẽ nhận vào raw text từ OCR (có thể bị sai chính tả, mất dấu, lẫn noise).  
Hãy phân tích và trả về JSON chứa thông tin sản phẩm đã được sửa chính tả tiếng Việt hoàn chỉnh.

QUY TẮC:
1. Sửa TẤT CẢ lỗi chính tả tiếng Việt (ví dụ: "nac" → "nạc", "thoang mat" → "thoáng mát", "nuoc" → "nước", "TJNH" → "Tịnh", "muoi i-ot" → "muối iốt").
2. Loại bỏ noise (chuỗi vô nghĩa như "8eq", "uhcil", "Canned=", "mÓl LUong T", "Hluong", ký tự rác).
3. Phân loại text vào đúng trường thông tin sản phẩm.
4. Nếu một trường không có thông tin, trả về null.
5. Với thành phần (ingredients): giữ nguyên mã phụ gia (E451i, E452i, E621, E316...) nhưng sửa tên tiếng Việt.
6. Với category: xác định loại sản phẩm dựa trên tên và thành phần (meat, seafood, dairy, bakery, beverage, snack, condiment, canned_food, frozen_food, vegetable, fruit, instant_food, other).
7. Luôn trả về JSON hợp lệ, không markdown, không giải thích thêm.
"""

USER_PROMPT_TEMPLATE = """\
RAW OCR TEXT:
```
{raw_text}
```

TEXT REGIONS (sorted by confidence):
{regions_text}

THÔNG TIN BỔ SUNG:
- Brand đã detect: {brand}
- Barcode: {barcode}
- Ngày hết hạn (đã extract): {expiry_date}
- Ngày sản xuất (đã extract): {mfg_date}
- Khối lượng (đã extract): {weight}

Hãy trả về JSON duy nhất theo schema sau:
{{
  "name": "Tên sản phẩm (ngắn gọn, không bao gồm thành phần)",
  "brand": "Thương hiệu",
  "ingredients": "Thành phần đầy đủ, đã sửa chính tả",
  "storage_instructions": "Hướng dẫn bảo quản",
  "usage_instructions": "Hướng dẫn sử dụng",
  "warnings": "Cảnh báo (nếu có)",
  "weight": "Khối lượng tịnh",
  "manufacturer": "Nhà sản xuất/phân phối",
  "category": "Loại sản phẩm (meat/seafood/dairy/bakery/beverage/snack/condiment/canned_food/frozen_food/vegetable/fruit/instant_food/other)",
  "quality_standards": "Chỉ tiêu chất lượng (nếu có)"
}}"""


# ---------------------------------------------------------------------------
# Data collection for Phase 2 fine-tuning
# ---------------------------------------------------------------------------

DATA_COLLECTION_DIR = Path("data/ocr_corrections")


def _save_training_pair(
    raw_text: str,
    regions: List[Dict[str, Any]],
    llm_output: Dict[str, Any],
    processing_time_ms: float,
    *,
    backend: str,
    model_label: str,
) -> None:
    """
    Save (input, output) pair for future fine-tuning.
    
    Each record is a JSONL line with:
    - input: raw OCR text + regions
    - output: corrected structured JSON (from LLM or rule-based snapshot)
    - metadata: timestamp, processing time, model used
    """
    try:
        DATA_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": {
                "raw_text": raw_text,
                "regions": regions[:20],  # Limit to save space
            },
            "output": llm_output,
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "backend": backend,
                "model": model_label,
            },
        }
        
        # Append to daily JSONL file
        today = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = DATA_COLLECTION_DIR / f"corrections_{today}.jsonl"
        
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info("Saved OCR correction sample to %s (backend=%s)", file_path, backend)
    except Exception as e:
        logger.warning(f"Failed to save training pair: {e}")


# ---------------------------------------------------------------------------
# Gemini API Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Lightweight Gemini API client for OCR post-processing."""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._model: str = "gemini-3.1-flash-lite-preview"
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = getattr(settings, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        return self._api_key
    
    @property
    def model(self) -> str:
        return getattr(settings, "gemini_model", None) or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call Gemini API with system + user prompt.
        
        Returns the raw text response or None on failure.
        """
        if not self.api_key:
            logger.warning("Gemini API key not configured")
            return None
        
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Low temp for consistent structured output
                "topP": 0.95,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        
        try:
            client = await self._get_client()
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned no candidates")
                return None
            
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
            return text
        
        except httpx.TimeoutException:
            logger.error("Gemini API timeout")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton
_gemini_client = GeminiClient()


# ---------------------------------------------------------------------------
# LLM Post-processor
# ---------------------------------------------------------------------------

class LLMPostProcessor:
    """
    Post-process OCR: local GGUF → Gemini → rule-based (text_postprocessor).
    """
    
    def __init__(self) -> None:
        self.gemini = _gemini_client
    
    @property
    def is_available(self) -> bool:
        """True when any path can run (always True: rule-based is the final fallback)."""
        return True
    
    def _regions_snapshot(self, text_regions: Any) -> List[Dict[str, Any]]:
        regions_data: List[Dict[str, Any]] = []
        for r in text_regions or []:
            if isinstance(r, dict):
                regions_data.append(r)
            else:
                regions_data.append(
                    {"text": getattr(r, "text", ""), "confidence": getattr(r, "confidence", 0)}
                )
        return regions_data

    async def _try_llm_backends(
        self,
        ocr_data: Dict[str, Any],
        user_prompt: str,
        raw_text: str,
        regions_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Try local GGUF then Gemini. On success, save training pair and return merged dict.
        """
        if local_gguf_llm.local_gguf_configured():
            start = time.perf_counter()
            try:
                raw_local = await local_gguf_llm.generate(SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.warning("Local GGUF call failed: %s", e)
                raw_local = None
            llm_time_ms = (time.perf_counter() - start) * 1000
            if raw_local:
                llm_result = self._parse_llm_response(raw_local)
                if llm_result:
                    path = getattr(settings, "llm_gguf_path", "") or ""
                    model_label = Path(str(path)).name if path else "local_gguf"
                    logger.info("LLM post-processing (local GGUF) in %.0fms", llm_time_ms)
                    _save_training_pair(
                        raw_text,
                        regions_data,
                        llm_result,
                        llm_time_ms,
                        backend="local_gguf",
                        model_label=model_label,
                    )
                    return self._merge_results(ocr_data, llm_result, llm_time_ms)
                logger.warning("Local GGUF returned unparseable JSON; trying Gemini")
            else:
                logger.warning("Local GGUF empty response; trying Gemini")

        if self.gemini.is_available:
            start = time.perf_counter()
            raw_cloud = await self.gemini.generate(SYSTEM_PROMPT, user_prompt)
            llm_time_ms = (time.perf_counter() - start) * 1000
            if raw_cloud:
                llm_result = self._parse_llm_response(raw_cloud)
                if llm_result:
                    logger.info("LLM post-processing (Gemini) in %.0fms", llm_time_ms)
                    _save_training_pair(
                        raw_text,
                        regions_data,
                        llm_result,
                        llm_time_ms,
                        backend="gemini",
                        model_label=settings.gemini_model,
                    )
                    return self._merge_results(ocr_data, llm_result, llm_time_ms)
                logger.warning("Gemini returned unparseable JSON")
            else:
                logger.warning("Gemini returned empty response")

        reasons: List[str] = []
        if local_gguf_llm.local_gguf_configured():
            reasons.append("local GGUF did not return parseable JSON (see warnings above)")
        else:
            raw_p = getattr(settings, "llm_gguf_path", None)
            if raw_p and str(raw_p).strip():
                reasons.append(f"local GGUF path invalid or file missing ({raw_p})")
            else:
                reasons.append("local GGUF not configured (AI_LLM_GGUF_PATH)")
        if self.gemini.is_available:
            reasons.append("Gemini did not return parseable JSON or returned empty (see warnings above)")
        else:
            reasons.append("Gemini not configured (AI_GEMINI_API_KEY or GEMINI_API_KEY)")
        logger.warning("LLM OCR post-process skipped: %s", " | ".join(reasons))

        return None
    
    async def process_ocr_response(
        self,
        ocr_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Post-process OCR: local GGUF → Gemini → rule-based.
        """
        raw_text = ocr_data.get("raw_text", "")
        text_regions = ocr_data.get("text_regions") or []
        
        if not raw_text and not text_regions:
            return ocr_data
        
        user_prompt = self._build_prompt(ocr_data)
        regions_data = self._regions_snapshot(text_regions)

        merged = await self._try_llm_backends(ocr_data, user_prompt, raw_text, regions_data)
        if merged is not None:
            return merged

        logger.info("Using rule-based OCR post-processing (LLM backends unavailable or failed)")
        start_rb = time.perf_counter()
        rule_out = text_postprocessor.process_ocr_response(ocr_data)
        rb_ms = (time.perf_counter() - start_rb) * 1000
        training_snapshot = self._training_snapshot_from_response(rule_out)
        _save_training_pair(
            raw_text,
            regions_data,
            training_snapshot,
            rb_ms,
            backend="rule_based",
            model_label="text_postprocessor",
        )
        return rule_out

    @staticmethod
    def _training_snapshot_from_response(ocr_response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a JSON object aligned with the LLM schema keys for JSONL export,
        from a post-processed OCR dict (e.g. rule-based output).
        """
        pi = ocr_response_dict.get("product_info") or {}

        def _list_to_str(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, list):
                parts = [str(x).strip() for x in val if str(x).strip()]
                return ", ".join(parts) if parts else None
            s = str(val).strip()
            return s or None

        ingredients = pi.get("ingredients")
        if isinstance(ingredients, list):
            ingredients_str = _list_to_str(ingredients)
        else:
            ingredients_str = (str(ingredients).strip() if ingredients else None) or None

        dc = pi.get("detected_category")
        category: Optional[str] = None
        if isinstance(dc, dict):
            category = dc.get("name")

        man = pi.get("manufacturer")
        manufacturer_str: Optional[str] = None
        if isinstance(man, dict):
            manufacturer_str = man.get("name")
        elif man:
            manufacturer_str = str(man).strip() or None

        qs = pi.get("quality_standards")
        quality_str = _list_to_str(qs) if isinstance(qs, list) else (
            str(qs).strip() if qs else None
        )

        warnings_val = pi.get("warnings")
        if isinstance(warnings_val, list):
            warnings_str = _list_to_str(warnings_val)
        else:
            warnings_str = (str(warnings_val).strip() if warnings_val else None) or None

        weight_str = pi.get("weight") or pi.get("net_weight")
        if weight_str is not None:
            weight_str = str(weight_str).strip() or None

        return {
            "name": pi.get("name"),
            "brand": pi.get("brand"),
            "ingredients": ingredients_str,
            "storage_instructions": pi.get("storage_instructions"),
            "usage_instructions": pi.get("usage_instructions"),
            "warnings": warnings_str,
            "weight": weight_str,
            "manufacturer": manufacturer_str,
            "category": category,
            "quality_standards": quality_str,
        }
    
    def _build_prompt(self, ocr_data: Dict[str, Any]) -> str:
        """Build the user prompt from OCR data."""
        raw_text = ocr_data.get("raw_text", "") or ""
        text_regions = ocr_data.get("text_regions") or []
        product_info = ocr_data.get("product_info") or {}
        
        # Format regions sorted by confidence (descending)
        regions_list = []
        for r in text_regions:
            if isinstance(r, dict):
                text = r.get("text", "")
                conf = r.get("confidence", 0)
            else:
                text = getattr(r, "text", "")
                conf = getattr(r, "confidence", 0)
            regions_list.append((conf, text))
        
        regions_list.sort(reverse=True)
        regions_text = "\n".join(
            f"  [{conf:.2f}] {text}" for conf, text in regions_list
        ) if regions_list else "(không có)"
        
        # Extract existing info
        brand = product_info.get("brand") or "(chưa xác định)"
        barcode = ocr_data.get("barcode") or "(không có)"
        
        expiry = ocr_data.get("expiry_date")
        expiry_str = "(không có)"
        if expiry and isinstance(expiry, dict):
            expiry_str = expiry.get("raw_text") or expiry.get("value") or "(không có)"
        
        mfg = ocr_data.get("manufactured_date")
        mfg_str = "(không có)"
        if mfg and isinstance(mfg, dict):
            mfg_str = mfg.get("raw_text") or mfg.get("value") or "(không có)"
        
        weight = product_info.get("weight") or "(chưa xác định)"
        
        return USER_PROMPT_TEMPLATE.format(
            raw_text=raw_text,
            regions_text=regions_text,
            brand=brand,
            barcode=barcode,
            expiry_date=expiry_str,
            mfg_date=mfg_str,
            weight=weight,
        )
    
    def _parse_llm_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response as JSON, handling common edge cases."""
        if not raw:
            return None
        
        text = raw.strip()
        
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text  
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
            return None
    
    def _merge_results(
        self,
        ocr_data: Dict[str, Any],
        llm_result: Dict[str, Any],
        llm_time_ms: float,
    ) -> Dict[str, Any]:
        """
        Merge LLM-corrected fields back into the OCR response.
        
        LLM results override rule-based results for text fields,
        but we keep structured objects (barcode_info, weight_info, etc.)
        from the original extraction.
        """
        result = ocr_data.copy()
        product_info = (ocr_data.get("product_info") or {}).copy()
        
        # --- Text fields: LLM overrides ---
        
        # Name
        llm_name = llm_result.get("name")
        if llm_name and llm_name.strip():
            product_info["name"] = llm_name.strip()
        
        # Brand
        llm_brand = llm_result.get("brand")
        if llm_brand and llm_brand.strip():
            product_info["brand"] = llm_brand.strip()
        
        # Ingredients (convert to list for model compatibility)
        llm_ingredients = llm_result.get("ingredients")
        if llm_ingredients and isinstance(llm_ingredients, str) and llm_ingredients.strip():
            product_info["ingredients"] = llm_ingredients.strip()
        
        # Storage instructions
        llm_storage = llm_result.get("storage_instructions")
        if llm_storage and llm_storage.strip():
            product_info["storage_instructions"] = llm_storage.strip()
        
        # Usage instructions
        llm_usage = llm_result.get("usage_instructions")
        if llm_usage and llm_usage.strip():
            product_info["usage_instructions"] = llm_usage.strip()
        
        # Warnings (convert to list)
        llm_warnings = llm_result.get("warnings")
        if llm_warnings:
            if isinstance(llm_warnings, str) and llm_warnings.strip():
                product_info["warnings"] = [llm_warnings.strip()]
            elif isinstance(llm_warnings, list):
                product_info["warnings"] = [w for w in llm_warnings if w and str(w).strip()]
        
        # Weight
        llm_weight = llm_result.get("weight")
        if llm_weight and llm_weight.strip():
            product_info["weight"] = llm_weight.strip()
        
        # Manufacturer
        llm_manufacturer = llm_result.get("manufacturer")
        if llm_manufacturer and llm_manufacturer.strip():
            # Only update if we don't already have a structured ManufacturerInfo
            if not product_info.get("manufacturer"):
                product_info["manufacturer"] = {
                    "name": llm_manufacturer.strip(),
                    "distributor": None,
                    "address": None,
                    "contact": None,
                }
        
        # Quality standards
        llm_quality = llm_result.get("quality_standards")
        if llm_quality and llm_quality.strip():
            product_info["quality_standards"] = [llm_quality.strip()]
        
        # Category
        llm_category = llm_result.get("category")
        if llm_category and llm_category.strip():
            category_name = llm_category.strip().lower()
            # Map to Vietnamese keywords for the CategoryInfo model
            category_keywords = {
                "meat": ["thịt", "nạc", "heo", "bò", "gà"],
                "seafood": ["cá", "tôm", "mực", "cua", "hải sản"],
                "dairy": ["sữa", "phô mai", "bơ", "yaourt", "kem"],
                "bakery": ["bánh", "mì", "bột"],
                "beverage": ["nước", "trà", "cà phê", "sữa"],
                "snack": ["snack", "bim bim", "kẹo"],
                "condiment": ["nước mắm", "nước tương", "tương ớt", "gia vị"],
                "canned_food": ["đồ hộp", "đóng hộp", "lon"],
                "frozen_food": ["đông lạnh", "frozen"],
                "vegetable": ["rau", "củ", "quả", "cà chua", "dưa"],
                "fruit": ["trái cây", "cam", "táo", "chuối"],
                "instant_food": ["mì gói", "phở", "bún", "ăn liền"],
                "other": [],
            }
            product_info["detected_category"] = {
                "name": category_name,
                "confidence": 0.85,  # LLM typically gives good category
                "keywords_vi": category_keywords.get(category_name, []),
            }
        
        result["product_info"] = product_info
        
        # Also set top-level name/brand for backward compat
        result["name"] = product_info.get("name")
        result["brand"] = product_info.get("brand")
        
        # Add LLM processing metadata to warnings
        current_warnings = result.get("warnings") or []
        if not isinstance(current_warnings, list):
            current_warnings = []
        result["warnings"] = current_warnings  # Don't add meta to user-facing warnings
        
        return result


# Singleton
llm_postprocessor = LLMPostProcessor()

"""
Market Price Crawler Service - Enhanced Version.

Thu thập giá thị trường bằng nhiều phương pháp:
1. Direct crawl các trang e-commerce phổ biến (ưu tiên)
2. Google Search để tìm thêm nguồn
3. Parse giá từ các trang sản phẩm

Cải thiện:
- Crawl trực tiếp các site thay vì chỉ dựa vào Google
- Nhiều price patterns hơn
- Fallback mechanisms
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus, urljoin, parse_qs, unquote

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketPriceResult:
    """Kết quả thu thập giá thị trường."""
    barcode: str
    product_name: Optional[str]
    price: float
    original_price: Optional[float]
    source: str
    source_url: Optional[str]
    store_name: Optional[str]
    unit: Optional[str]
    weight: Optional[str]
    is_in_stock: bool
    confidence: float
    collected_at: datetime = field(default_factory=datetime.utcnow)


class PriceExtractor:
    """Utility class để extract giá từ text/HTML."""
    
    # Vietnamese price patterns - ordered by specificity
    PRICE_PATTERNS = [
        # ₫34,000 or ₫34.000
        r'₫\s*(\d{1,3}(?:[.,]\d{3})+)',
        # đ34,000 or đ34.000  
        r'đ\s*(\d{1,3}(?:[.,]\d{3})+)',
        # 34,000đ or 34.000đ
        r'(\d{1,3}(?:[.,]\d{3})+)\s*đ(?!ồng\s*\d)',
        # 34,000₫ or 34.000₫
        r'(\d{1,3}(?:[.,]\d{3})+)\s*₫',
        # 34.000 VND or 34,000 VND  
        r'(\d{1,3}(?:[.,]\d{3})+)\s*(?:VND|VNĐ)',
        # "price": 34000 (JSON)
        r'"price"[:\s]*(\d+(?:\.\d+)?)',
        # data-price="34000"
        r'data-price[=\s"\']*(\d+)',
        # Giá: 34.000
        r'[Gg]iá[:\s]*(\d{1,3}(?:[.,]\d{3})+)',
    ]
    
    # Barcode patterns to exclude
    BARCODE_PATTERNS = [
        r'8934\d{9}',  # Vietnamese barcodes
        r'893\d{10}',
        r'890\d{10}',
    ]
    
    @classmethod
    def _is_barcode(cls, number_str: str) -> bool:
        """Check if a number string looks like a barcode."""
        cleaned = number_str.replace('.', '').replace(',', '')
        # Barcodes are typically 8, 12, 13 digits
        if len(cleaned) in [8, 12, 13]:
            for pattern in cls.BARCODE_PATTERNS:
                if re.match(pattern, cleaned):
                    return True
        return False
    
    @classmethod
    def extract_price(cls, text: str, exclude_barcode: str = None) -> Optional[float]:
        """Trích xuất giá từ text."""
        if not text:
            return None
        
        for pattern in cls.PRICE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for price_str in matches:
                # Clean: remove dots/commas as thousand separators
                cleaned = price_str.replace('.', '').replace(',', '')
                
                # Skip if it's a barcode
                if cls._is_barcode(price_str):
                    continue
                    
                # Skip if it matches the excluded barcode
                if exclude_barcode and cleaned in exclude_barcode:
                    continue
                
                try:
                    price = float(cleaned)
                    # Sanity check: 1,000 ≤ price ≤ 10,000,000 VND (realistic for retail)
                    if 1000 <= price <= 10_000_000:
                        return price
                except ValueError:
                    continue
        
        return None
    
    @classmethod
    def extract_all_prices(cls, text: str) -> List[float]:
        """Trích xuất tất cả giá từ text."""
        prices = []
        if not text:
            return prices
        
        for pattern in cls.PRICE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                price_str = match.group(1)
                cleaned = price_str.replace('.', '').replace(',', '')
                try:
                    price = float(cleaned)
                    if 1000 <= price <= 100_000_000 and price not in prices:
                        prices.append(price)
                except ValueError:
                    continue
        
        return sorted(prices)


class DirectSiteCrawler:
    """
    Crawl trực tiếp các trang e-commerce Việt Nam.
    Không phụ thuộc vào Google.
    """
    
    def __init__(self):
        self.timeout = 20
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        
        # Các site e-commerce và URL patterns
        self.sites = [
            {
                "name": "LOTTE Mart",
                "domain": "lottemart.vn",
                "search_url": "https://www.lottemart.vn/search?q={query}",
                "price_selectors": [".product-price", ".price", "[data-price]", ".pro-price"],
            },
            {
                "name": "Vissan Mart", 
                "domain": "vissanmart.com",
                "search_url": "https://vissanmart.com/catalogsearch/result/?q={query}",
                "price_selectors": [".price", ".product-price", ".special-price"],
            },
            {
                "name": "7-Eleven",
                "domain": "7-eleven.vn",
                "search_url": "https://7-eleven.vn/search?q={query}",
                "price_selectors": [".product-price", ".price"],
            },
            {
                "name": "Bách Hóa Xanh",
                "domain": "bachhoaxanh.com", 
                "search_url": "https://www.bachhoaxanh.com/tim-kiem?q={query}",
                "price_selectors": [".product__price", ".price", ".prod-price"],
            },
            {
                "name": "WinMart",
                "domain": "winmart.vn",
                "search_url": "https://www.winmart.vn/search?q={query}",
                "price_selectors": [".price", ".product-price"],
            },
            {
                "name": "Co.op Online",
                "domain": "cooponline.vn",
                "search_url": "https://cooponline.vn/tim-kiem/?q={query}",
                "price_selectors": [".price", ".woocommerce-Price-amount"],
            },
        ]
    
    async def search_product(
        self, 
        barcode: str,
        product_name: Optional[str] = None,
    ) -> List[MarketPriceResult]:
        """Tìm kiếm sản phẩm trên tất cả các site."""
        results = []
        
        # Tạo tasks cho tất cả sites
        tasks = []
        for site in self.sites:
            tasks.append(self._search_site(site, barcode, product_name))
        
        # Chạy song song
        site_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in site_results:
            if isinstance(result, list):
                results.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"Site crawl error: {result}")
        
        return results
    
    async def _search_site(
        self,
        site: Dict[str, Any],
        barcode: str,
        product_name: Optional[str] = None,
    ) -> List[MarketPriceResult]:
        """Tìm kiếm trên một site cụ thể."""
        results = []
        
        # Thử với barcode trước
        search_queries = [barcode]
        if product_name:
            search_queries.append(product_name)
        
        for query in search_queries:
            search_url = site["search_url"].format(query=quote_plus(query))
            
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, 
                    follow_redirects=True,
                    verify=False,  # Some sites have SSL issues
                ) as client:
                    response = await client.get(search_url, headers=self.headers)
                    
                    if response.status_code != 200:
                        logger.debug(f"{site['name']}: HTTP {response.status_code}")
                        continue
                    
                    html = response.text
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Tìm giá bằng selectors
                    for selector in site["price_selectors"]:
                        price_elems = soup.select(selector)
                        for elem in price_elems[:5]:  # Limit 5 per selector
                            price_text = elem.get_text(strip=True)
                            price = PriceExtractor.extract_price(price_text, exclude_barcode=barcode)
                            
                            if price:
                                # Tìm tên sản phẩm gần element này
                                prod_name = self._find_product_name(elem, soup)
                                
                                results.append(MarketPriceResult(
                                    barcode=barcode,
                                    product_name=prod_name,
                                    price=price,
                                    original_price=None,
                                    source=site["domain"],
                                    source_url=search_url,
                                    store_name=site["name"],
                                    unit=None,
                                    weight=None,
                                    is_in_stock=True,
                                    confidence=0.8,
                                ))
                    
                    # Fallback: tìm trong toàn bộ text
                    if not results:
                        page_text = soup.get_text(separator=' ')
                        prices = PriceExtractor.extract_all_prices(page_text)
                        
                        for price in prices[:3]:  # Limit 3
                            results.append(MarketPriceResult(
                                barcode=barcode,
                                product_name=product_name,
                                price=price,
                                original_price=None,
                                source=site["domain"],
                                source_url=search_url,
                                store_name=site["name"],
                                unit=None,
                                weight=None,
                                is_in_stock=True,
                                confidence=0.5,
                            ))
                    
                    if results:
                        break  # Đã tìm được, không cần thử query khác
                        
            except Exception as e:
                logger.debug(f"{site['name']} error: {e}")
                continue
        
        return results
    
    def _find_product_name(self, price_elem, soup: BeautifulSoup) -> Optional[str]:
        """Tìm tên sản phẩm gần price element."""
        # Tìm trong parent elements
        parent = price_elem.parent
        for _ in range(5):  # Go up 5 levels
            if parent is None:
                break
            
            # Tìm h1, h2, h3, .product-name, .product-title
            name_elem = parent.select_one('h1, h2, h3, .product-name, .product-title, [class*="name"], [class*="title"]')
            if name_elem:
                name = name_elem.get_text(strip=True)
                if name and len(name) > 3:
                    return name[:200]  # Limit length
            
            parent = parent.parent
        
        return None


class GoogleSearchCrawler:
    """
    Crawler dùng DuckDuckGo Search để tìm giá.
    DuckDuckGo HTML version thân thiện với bot hơn Google.
    """
    
    def __init__(self):
        self.timeout = 30
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        
        self.store_mapping = {
            "lottemart": "LOTTE Mart",
            "vissanmart": "Vissan Mart",
            "7-eleven": "7-Eleven",
            "bachhoaxanh": "Bách Hóa Xanh",
            "winmart": "WinMart",
            "coopmart": "Co.op Mart",
            "cooponline": "Co.op Online",
            "satrafoods": "Satra Foods",
            "bigc": "Big C",
            "shopee": "Shopee",
            "tiki": "Tiki",
            "lazada": "Lazada",
            "genshai": "Siêu Thị Genshai",
        }
    
    async def search(
        self, 
        barcode: str,
        product_name: Optional[str] = None,
    ) -> List[MarketPriceResult]:
        """Search DuckDuckGo và parse kết quả."""
        results = []
        
        # Build query - search with barcode and "giá" keyword
        query = f"{barcode} giá"
        if product_name:
            query = f"{barcode} {product_name} giá"
        
        # DuckDuckGo HTML version (bot-friendly)
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                follow_redirects=True
            ) as client:
                response = await client.get(search_url, headers=self.headers)
                
                if response.status_code != 200:
                    logger.warning(f"DuckDuckGo search failed: {response.status_code}")
                    return results
                
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse DuckDuckGo results
                search_results = soup.select('.result')
                logger.info(f"DuckDuckGo found {len(search_results)} results")
                
                for result_div in search_results[:15]:
                    try:
                        # Get title
                        title_elem = result_div.select_one('.result__title')
                        title = title_elem.get_text(strip=True) if title_elem else ""
                        
                        # Get URL - DuckDuckGo uses redirect URLs
                        link_elem = result_div.select_one('a.result__a')
                        
                        url = ""
                        if link_elem and link_elem.get('href'):
                            href = link_elem.get('href', '')
                            # DuckDuckGo redirect URL: //duckduckgo.com/l/?uddg=ENCODED_URL
                            if 'uddg=' in href:
                                parsed = urlparse(href)
                                qs = parse_qs(parsed.query)
                                if 'uddg' in qs:
                                    url = unquote(qs['uddg'][0])
                            else:
                                url = href
                        
                        # Get snippet
                        snippet_elem = result_div.select_one('.result__snippet')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Try to extract price from snippet
                        full_text = f"{title} {snippet}"
                        price = PriceExtractor.extract_price(full_text, exclude_barcode=barcode)
                        
                        # Get domain from the real URL
                        domain = ""
                        if url and url.startswith('http'):
                            try:
                                domain = urlparse(url).netloc.replace('www.', '')
                            except:
                                pass
                        
                        store_name = self._get_store_name(domain or title)
                        
                        if price:
                            results.append(MarketPriceResult(
                                barcode=barcode,
                                product_name=title or product_name,
                                price=price,
                                original_price=None,
                                source=domain or "duckduckgo",
                                source_url=url,
                                store_name=store_name,
                                unit=None,
                                weight=None,
                                is_in_stock=True,
                                confidence=0.75,
                            ))
                            
                            logger.info(f"Found price {price} from {store_name}")
                        else:
                            # Không có giá trong snippet, nhưng có URL để deep crawl sau
                            # Lưu lại URL để deep crawl
                            if url and url.startswith('http') and any(
                                k in domain.lower() for k in self.store_mapping.keys()
                            ):
                                results.append(MarketPriceResult(
                                    barcode=barcode,
                                    product_name=title or product_name,
                                    price=0,  # Placeholder, sẽ deep crawl sau
                                    original_price=None,
                                    source=domain,
                                    source_url=url,
                                    store_name=store_name,
                                    unit=None,
                                    weight=None,
                                    is_in_stock=True,
                                    confidence=0.0,  # Mark as needing verification
                                ))
                                logger.info(f"Found URL to deep crawl: {store_name}")
                            
                    except Exception as e:
                        logger.debug(f"Error parsing result: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results
    
    def _get_store_name(self, text: str) -> str:
        """Xác định tên cửa hàng từ domain hoặc title."""
        text_lower = text.lower()
        for key, name in self.store_mapping.items():
            if key in text_lower:
                return name
        
        # Try to extract from domain
        if '.' in text:
            return text.split('.')[0].title()
        
        return "Online Store"


class ProductPageCrawler:
    """Crawl trực tiếp vào trang sản phẩm để lấy giá chính xác."""
    
    def __init__(self):
        self.timeout = 15
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        }
    
    async def crawl_page(self, url: str, barcode: str) -> Optional[MarketPriceResult]:
        """Crawl một trang sản phẩm cụ thể."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, 
                follow_redirects=True,
                verify=False,
            ) as client:
                response = await client.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    return None
                
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                
                # Tìm giá bằng nhiều selectors
                price_selectors = [
                    '.price', '.product-price', '.current-price', '.sale-price',
                    '.final-price', '[data-price]', '.price-box', '.price-final',
                    '.pro-price', '.product__price', '.box-price', '.price-new',
                    '[class*="price"]', '[id*="price"]',
                ]
                
                for selector in price_selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        price = PriceExtractor.extract_price(elem.get_text(), exclude_barcode=self.barcode if hasattr(self, 'barcode') else None)
                        if price:
                            # Tìm tên sản phẩm
                            title_elem = soup.select_one('h1, .product-name, .product-title')
                            title = title_elem.get_text(strip=True) if title_elem else None
                            
                            domain = urlparse(url).netloc.replace('www.', '')
                            
                            return MarketPriceResult(
                                barcode=barcode,
                                product_name=title,
                                price=price,
                                original_price=None,
                                source=domain,
                                source_url=url,
                                store_name=domain.split('.')[0].title(),
                                unit=None,
                                weight=None,
                                is_in_stock=True,
                                confidence=0.9,
                            )
                
                # Fallback: JSON-LD structured data
                scripts = soup.find_all('script', type='application/ld+json')
                for script in scripts:
                    try:
                        data = json.loads(script.string)
                        if isinstance(data, dict):
                            price = data.get('offers', {}).get('price')
                            if price:
                                return MarketPriceResult(
                                    barcode=barcode,
                                    product_name=data.get('name'),
                                    price=float(price),
                                    original_price=None,
                                    source=urlparse(url).netloc,
                                    source_url=url,
                                    store_name=urlparse(url).netloc.split('.')[0].title(),
                                    unit=None,
                                    weight=None,
                                    is_in_stock=True,
                                    confidence=0.95,
                                )
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Page crawl error for {url}: {e}")
        
        return None


class MarketPriceCrawlerService:
    """
    Service tổng hợp thu thập giá thị trường.
    
    Chiến lược:
    1. Direct crawl các site e-commerce (nhanh, chính xác)
    2. Google Search (tìm thêm nguồn)
    3. Deep crawl vào từng trang (chính xác nhất nhưng chậm)
    """
    
    def __init__(self):
        self.direct_crawler = DirectSiteCrawler()
        self.google_crawler = GoogleSearchCrawler()
        self.page_crawler = ProductPageCrawler()
    
    async def get_market_prices(
        self,
        barcode: Optional[str] = None,
        product_name: Optional[str] = None,
        deep_crawl: bool = False,
    ) -> List[MarketPriceResult]:
        """
        Lấy giá thị trường từ nhiều nguồn.
        
        Args:
            barcode: Mã barcode sản phẩm
            product_name: Tên sản phẩm  
            deep_crawl: Có crawl vào từng trang để lấy giá chính xác không
        
        Returns:
            Danh sách giá từ các nguồn
        """
        if not barcode and not product_name:
            return []
        
        results = []
        
        logger.info(f"Searching prices for barcode={barcode}, name={product_name}")
        
        # Step 1: Direct crawl các site e-commerce
        logger.info("Step 1: Direct site crawling...")
        direct_results = await self.direct_crawler.search_product(
            barcode or "",
            product_name,
        )
        results.extend(direct_results)
        logger.info(f"Direct crawl found {len(direct_results)} prices")
        
        # Step 2: Google Search
        logger.info("Step 2: Google Search...")
        google_results = await self.google_crawler.search(
            barcode or "",
            product_name,
        )
        results.extend(google_results)
        logger.info(f"Google search found {len(google_results)} prices")
        
        # Step 3: Deep crawl nếu được yêu cầu hoặc có URL cần crawl
        # Ưu tiên crawl URLs chưa có giá (price=0 hoặc confidence=0)
        urls_to_crawl = []
        if deep_crawl:
            for r in results:
                if r.source_url and (r.price == 0 or r.confidence == 0):
                    urls_to_crawl.append((r.source_url, r.store_name))
        
        # Nếu có ít kết quả, crawl thêm các URLs có sẵn
        if len([r for r in results if r.price > 0]) < 3:
            for r in results:
                if r.source_url and r.source_url not in [u[0] for u in urls_to_crawl]:
                    urls_to_crawl.append((r.source_url, r.store_name))
        
        urls_to_crawl = list(dict.fromkeys([u[0] for u in urls_to_crawl[:8]]))  # Limit 8 URLs
        
        if urls_to_crawl:
            logger.info(f"Step 3: Deep crawling {len(urls_to_crawl)} product pages...")
            
            tasks = [
                self.page_crawler.crawl_page(url, barcode or "")
                for url in urls_to_crawl
            ]
            
            deep_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            deep_count = 0
            for result in deep_results:
                if isinstance(result, MarketPriceResult) and result.price > 0:
                    results.append(result)
                    deep_count += 1
            
            logger.info(f"Deep crawl added {deep_count} more prices")
        
        # Lọc bỏ entries không có giá
        results = [r for r in results if r.price > 0]
        
        # Deduplicate và sắp xếp
        results = self._deduplicate(results)
        results.sort(key=lambda x: x.price)
        
        logger.info(f"Total unique prices found: {len(results)}")
        
        return results
    
    def _deduplicate(self, results: List[MarketPriceResult]) -> List[MarketPriceResult]:
        """Loại bỏ giá trùng lặp."""
        seen = set()
        unique = []
        
        for r in results:
            # Key = (source, price rounded to nearest 100)
            key = (r.source, round(r.price / 100) * 100)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        return unique
    
    def get_price_stats(self, prices: List[MarketPriceResult]) -> Dict[str, Any]:
        """Tính thống kê giá."""
        if not prices:
            return {
                "min_price": 0,
                "max_price": 0,
                "avg_price": 0,
                "source_count": 0,
                "sources": [],
            }
        
        price_values = [p.price for p in prices]
        
        return {
            "min_price": min(price_values),
            "max_price": max(price_values),
            "avg_price": round(sum(price_values) / len(price_values), 0),
            "source_count": len(set(p.source for p in prices)),
            "sources": list(set(p.source for p in prices)),
        }


# Singleton instance
market_price_crawler = MarketPriceCrawlerService()

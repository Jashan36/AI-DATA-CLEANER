"""
Async web scraping pipeline with structured card extraction and NER-based entity extraction.
Handles robots.txt respect, retry/backoff, and converts HTML to readable text.
"""

import asyncio
import aiohttp
import httpx
from bs4 import BeautifulSoup
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
import robotexclusionrulesparser
try:
    from readability import Document  # type: ignore
    READABILITY_AVAILABLE = True
except ImportError:
    try:
        from readability.readability import Document  # type: ignore
        READABILITY_AVAILABLE = True
    except ImportError:
        Document = None  # type: ignore
        READABILITY_AVAILABLE = False
import lxml.html
try:
    from lxml.html.clean import Cleaner
except ImportError:
    try:
        from lxml_html_clean import Cleaner  # type: ignore
    except ImportError:
        Cleaner = None  # type: ignore
import spacy
from textblob import TextBlob
import nest_asyncio

# Enable nested event loops for Jupyter/Streamlit compatibility
nest_asyncio.apply()

logger = logging.getLogger(__name__)

@dataclass
class StructuredCard:
    """Represents a structured information card extracted from a web page."""
    card_type: str
    title: str
    content: List[str]
    entities: Dict[str, List[str]]
    confidence: float
    source_url: str
    extracted_at: datetime

@dataclass
class ScrapingResult:
    """Result of web scraping operation."""
    url: str
    success: bool
    title: str
    summary: str
    structured_cards: List[StructuredCard]
    raw_text: str
    error_message: Optional[str] = None
    response_time: Optional[float] = None

class WebScraper:
    """
    Async web scraper with structured information extraction.
    """
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 timeout: int = 30,
                 retry_attempts: int = 3,
                 backoff_factor: float = 1.5,
                 respect_robots: bool = True):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor
        self.respect_robots = respect_robots
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model not found, falling back to TextBlob")
            self.nlp = None
        
        # HTML cleaner for removing unwanted elements (optional dependency)
        if Cleaner is not None:
            self.html_cleaner = Cleaner(
                scripts=True,
                javascript=True,
                comments=True,
                style=True,
                links=True,
                meta=True,
                page_structure=True,
                processing_instructions=True,
                embedded=True,
                frames=True,
                forms=True,
                annoying_tags=True,
                remove_unknown_tags=True
            )
        else:
            self.html_cleaner = None
        
        # Robots parser
        self.robots_parser = robotexclusionrulesparser.RobotFileParserLookalike()
        
        # Card extraction patterns
        self.card_patterns = {
            'about': [
                r'about\s+us', r'our\s+story', r'company\s+overview',
                r'who\s+we\s+are', r'our\s+mission', r'our\s+vision'
            ],
            'products_services': [
                r'products?', r'services?', r'solutions?', r'offerings?',
                r'what\s+we\s+do', r'our\s+products?', r'our\s+services?'
            ],
            'team_people': [
                r'team', r'staff', r'employees?', r'people', r'leadership',
                r'our\s+team', r'meet\s+the\s+team', r'key\s+people'
            ],
            'locations': [
                r'locations?', r'offices?', r'contact\s+us', r'where\s+we\s+are',
                r'address', r'headquarters?', r'branches?'
            ],
            'contact': [
                r'contact', r'get\s+in\s+touch', r'reach\s+us', r'phone',
                r'email', r'address', r'contact\s+information'
            ],
            'social': [
                r'follow\s+us', r'social\s+media', r'connect\s+with\s+us',
                r'facebook', r'twitter', r'linkedin', r'instagram'
            ],
            'pricing': [
                r'pricing', r'plans?', r'cost', r'rates?', r'fees?',
                r'pricing\s+plans?', r'cost\s+structure'
            ],
            'faq': [
                r'faq', r'frequently\s+asked\s+questions', r'questions?',
                r'help', r'support', r'common\s+questions?'
            ],
            'jobs': [
                r'jobs?', r'careers?', r'employment', r'work\s+with\s+us',
                r'join\s+us', r'open\s+positions?', r'vacancies?'
            ],
            'policies': [
                r'privacy\s+policy', r'terms\s+of\s+service', r'terms\s+and\s+conditions',
                r'legal', r'disclaimer', r'cookie\s+policy'
            ]
        }

    async def scrape_urls(self, urls: List[str]) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently with rate limiting.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_single_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ScrapingResult(
                    url=urls[i],
                    success=False,
                    title="",
                    summary="",
                    structured_cards=[],
                    raw_text="",
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results

    async def scrape_single_url(self, url: str) -> ScrapingResult:
        """
        Scrape a single URL with retry logic and robots.txt respect.
        """
        start_time = time.time()
        
        # Check robots.txt
        if self.respect_robots and not await self._check_robots_txt(url):
            return ScrapingResult(
                url=url,
                success=False,
                title="",
                summary="",
                structured_cards=[],
                raw_text="",
                error_message="Blocked by robots.txt"
            )
        
        # Attempt scraping with retries
        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'lxml')
                    
                    # Extract structured information
                    title = self._extract_title(soup)
                    raw_text = self._extract_readable_text(soup)
                    structured_cards = self._extract_structured_cards(soup, url)
                    summary = self._generate_summary(structured_cards)
                    
                    response_time = time.time() - start_time
                    
                    return ScrapingResult(
                        url=url,
                        success=True,
                        title=title,
                        summary=summary,
                        structured_cards=structured_cards,
                        raw_text=raw_text,
                        response_time=response_time
                    )
                    
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        title="",
                        summary="",
                        structured_cards=[],
                        raw_text="",
                        error_message=str(e)
                    )
                else:
                    # Exponential backoff
                    await asyncio.sleep(self.backoff_factor ** attempt)

    async def _check_robots_txt(self, url: str) -> bool:
        """Check robots.txt with simple async handling."""
        if not self.respect_robots:
            return True
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(robots_url)
                if response.status_code == 200:
                    if any(line.strip().lower().startswith('disallow: /') for line in response.text.split('\n')):
                        return False
            return True
        except Exception:
            return True

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ""

    def _extract_readable_text(self, soup: BeautifulSoup) -> str:
        """Extract readable text from HTML using readability algorithm."""
        if READABILITY_AVAILABLE and Document is not None:
            try:
                # Use readability to extract main content
                doc = Document(str(soup))
                readable_html = doc.summary()
                
                # Parse with lxml and clean
                readable_soup = BeautifulSoup(readable_html, 'lxml')
                
                # Remove unwanted elements
                for element in readable_soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract text
                text = readable_soup.get_text()
                
                # Clean up text
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = text.strip()
                
                return text
            except Exception as e:
                logger.warning(f"Error extracting readable text: {e}")
        
        # Fallback to simple text extraction
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        return soup.get_text()

    def _extract_structured_cards(self, soup: BeautifulSoup, url: str) -> List[StructuredCard]:
        """Extract structured information cards from the page."""
        cards = []
        
        # Find sections that match our card patterns
        for card_type, patterns in self.card_patterns.items():
            card = self._extract_card_by_type(soup, card_type, patterns, url)
            if card:
                cards.append(card)
        
        return cards

    def _extract_card_by_type(self, soup: BeautifulSoup, card_type: str, patterns: List[str], url: str) -> Optional[StructuredCard]:
        """Extract a specific type of card from the page."""
        # Look for headings that match patterns
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            heading_text = heading.get_text().lower().strip()
            
            for pattern in patterns:
                if re.search(pattern, heading_text, re.IGNORECASE):
                    # Found matching heading, extract content
                    content = self._extract_section_content(heading)
                    if content:
                        entities = self._extract_entities(content)
                        title = heading.get_text().strip()
                        
                        return StructuredCard(
                            card_type=card_type,
                            title=title,
                            content=content,
                            entities=entities,
                            confidence=0.8,
                            source_url=url,
                            extracted_at=datetime.now()
                        )
        
        return None

    def _extract_section_content(self, heading) -> List[str]:
        """Extract content from a section starting with a heading."""
        content = []
        current = heading.next_sibling
        
        while current:
            if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Hit another heading, stop
                break
            
            if hasattr(current, 'get_text'):
                text = current.get_text().strip()
                if text and len(text) > 10:  # Only meaningful text
                    content.append(text)
            
            current = current.next_sibling
        
        return content[:5]  # Limit to 5 content pieces

    def _extract_entities(self, content: List[str]) -> Dict[str, List[str]]:
        """Extract named entities from content using NLP."""
        entities = {
            'ORG': [],
            'PERSON': [],
            'GPE': [],
            'LOC': [],
            'EMAIL': [],
            'PHONE': [],
            'ADDRESS': [],
            'FAC': [],
            'NORP': [],
            'PRODUCT': [],
            'EVENT': [],
            'LAW': [],
            'LANGUAGE': []
        }
        
        full_text = ' '.join(content)
        
        if self.nlp:
            # Use spaCy for entity extraction
            doc = self.nlp(full_text)
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
        else:
            # Fallback to TextBlob and regex
            blob = TextBlob(full_text)
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, full_text)
            entities['EMAIL'] = list(set(emails))
            
            # Extract phones
            phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
            phones = re.findall(phone_pattern, full_text)
            entities['PHONE'] = list(set(phones))
        
        # Deduplicate entities
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

    def _generate_summary(self, cards: List[StructuredCard], max_words: int = 25) -> str:
        """Generate summary with proper truncation."""
        if not cards:
            return "No structured information found."
        
        contents: List[str] = []
        for card in cards:
            if card.content:
                contents.extend(card.content)
        
        if not contents:
            return "No structured information found."
        
        # Combine and truncate by words
        full_text = ' '.join(contents)
        words = full_text.split()
        
        if len(words) <= max_words:
            return full_text
        else:
            truncated = ' '.join(words[:max_words]) + '...'
            return truncated

    def export_results(self, results: List[ScrapingResult], output_dir: str):
        """Export scraping results to files."""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary
        summary_file = os.path.join(output_dir, 'scrape_summary.md')
        with open(summary_file, 'w') as f:
            f.write("# Web Scraping Summary\n\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
            
            for result in results:
                f.write(f"## {result.url}\n\n")
                f.write(f"**Title:** {result.title}\n\n")
                f.write(f"**Status:** {'Success' if result.success else 'Failed'}\n\n")
                
                if result.success:
                    f.write(f"**Summary:**\n{result.summary}\n\n")
                    f.write(f"**Cards Found:** {len(result.structured_cards)}\n\n")
                else:
                    f.write(f"**Error:** {result.error_message}\n\n")
                
                f.write("---\n\n")
        
        # Export structured cards as JSON
        cards_file = os.path.join(output_dir, 'scrape_cards.json')
        cards_data = []
        
        for result in results:
            if result.success:
                for card in result.structured_cards:
                    cards_data.append({
                        'url': result.url,
                        'card_type': card.card_type,
                        'title': card.title,
                        'content': card.content,
                        'entities': card.entities,
                        'confidence': card.confidence,
                        'extracted_at': card.extracted_at.isoformat()
                    })
        
        with open(cards_file, 'w') as f:
            json.dump(cards_data, f, indent=2, default=str)
        
        # Export raw text
        raw_text_file = os.path.join(output_dir, 'scrape_raw_text.txt')
        with open(raw_text_file, 'w') as f:
            for result in results:
                if result.success and result.raw_text:
                    f.write(f"=== {result.url} ===\n\n")
                    f.write(result.raw_text)
                    f.write("\n\n" + "="*50 + "\n\n")

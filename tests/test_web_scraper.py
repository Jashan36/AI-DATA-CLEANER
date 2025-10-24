"""
Unit tests for WebScraper class.
"""

import unittest
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.web_scraper import WebScraper, StructuredCard, ScrapingResult

@pytest.mark.web
@pytest.mark.slow
class TestWebScraper(unittest.TestCase):
    """Test cases for WebScraper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = WebScraper(
            max_concurrent=2,
            timeout=10,
            retry_attempts=2,
            respect_robots=False
        )
    
    def test_initialization(self):
        """Test WebScraper initialization."""
        self.assertEqual(self.scraper.max_concurrent, 2)
        self.assertEqual(self.scraper.timeout, 10)
        self.assertEqual(self.scraper.retry_attempts, 2)
        self.assertFalse(self.scraper.respect_robots)
    
    def test_card_patterns(self):
        """Test card pattern definitions."""
        expected_patterns = [
            'about', 'products_services', 'team_people', 'locations',
            'contact', 'social', 'pricing', 'faq', 'jobs', 'policies'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, self.scraper.card_patterns)
            self.assertIsInstance(self.scraper.card_patterns[pattern], list)
            self.assertGreater(len(self.scraper.card_patterns[pattern]), 0)
    
    def test_extract_title(self):
        """Test title extraction."""
        from bs4 import BeautifulSoup
        
        # Test with title tag
        html_with_title = '<html><head><title>Test Title</title></head><body></body></html>'
        soup = BeautifulSoup(html_with_title, 'html.parser')
        title = self.scraper._extract_title(soup)
        self.assertEqual(title, 'Test Title')
        
        # Test with h1 tag (fallback)
        html_with_h1 = '<html><body><h1>Test H1 Title</h1></body></html>'
        soup = BeautifulSoup(html_with_h1, 'html.parser')
        title = self.scraper._extract_title(soup)
        self.assertEqual(title, 'Test H1 Title')
        
        # Test with no title
        html_no_title = '<html><body><p>Some content</p></body></html>'
        soup = BeautifulSoup(html_no_title, 'html.parser')
        title = self.scraper._extract_title(soup)
        self.assertEqual(title, '')
    
    def test_extract_readable_text(self):
        """Test readable text extraction."""
        from bs4 import BeautifulSoup
        
        html = '''
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a test paragraph with some content.</p>
                <script>console.log('This should be removed');</script>
                <style>body { color: red; }</style>
            </body>
        </html>
        '''
        
        soup = BeautifulSoup(html, 'html.parser')
        text = self.scraper._extract_readable_text(soup)
        
        self.assertIn('Main Heading', text)
        self.assertIn('This is a test paragraph', text)
        self.assertNotIn('console.log', text)
        self.assertNotIn('color: red', text)
    
    def test_extract_entities(self):
        """Test entity extraction."""
        content = [
            'John Doe works at Microsoft Corporation.',
            'Contact us at info@example.com or call +1-555-123-4567.',
            'Our office is located in New York, USA.'
        ]
        
        entities = self.scraper._extract_entities(content)
        
        # Check that entities are extracted
        self.assertIsInstance(entities, dict)
        self.assertIn('ORG', entities)
        self.assertIn('PERSON', entities)
        self.assertIn('EMAIL', entities)
        self.assertIn('PHONE', entities)
        self.assertIn('GPE', entities)
        
        # Check that entities are deduplicated
        for entity_type, entity_list in entities.items():
            self.assertEqual(len(entity_list), len(set(entity_list)))
    
    def test_generate_summary(self):
        """Test summary generation."""
        cards = [
            StructuredCard(
                card_type='about',
                title='About Us',
                content=['We are a technology company focused on innovation.'],
                entities={},
                confidence=0.8,
                source_url='https://example.com',
                extracted_at=None
            ),
            StructuredCard(
                card_type='contact',
                title='Contact Information',
                content=['Email us at info@example.com for more information.'],
                entities={},
                confidence=0.9,
                source_url='https://example.com',
                extracted_at=None
            )
        ]
        
        summary = self.scraper._generate_summary(cards)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn('•', summary)  # Should contain bullet points
    
    def test_generate_summary_empty_cards(self):
        """Test summary generation with empty cards."""
        cards = []
        summary = self.scraper._generate_summary(cards)
        
        self.assertEqual(summary, 'No structured information found.')
    
    def test_generate_summary_long_content(self):
        """Test summary generation with long content."""
        long_content = 'This is a very long piece of content that should be truncated to fit within the word limit for summary generation.'
        cards = [
            StructuredCard(
                card_type='about',
                title='About Us',
                content=[long_content],
                entities={},
                confidence=0.8,
                source_url='https://example.com',
                extracted_at=None
            )
        ]
        
        summary = self.scraper._generate_summary(cards)
        
        # Check that content is truncated
        self.assertLessEqual(len(summary.split()), 25)  # Should be concise
        self.assertIn('...', summary)  # Should indicate truncation
    
    @pytest.mark.asyncio
    @patch('core.web_scraper.httpx.AsyncClient')
    async def test_scrape_single_url_success(self, mock_client):
        """Test successful single URL scraping."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><head><title>Test Page</title></head><body><h1>Test Content</h1></body></html>'
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        result = await self.scraper.scrape_single_url('https://example.com')
        
        self.assertTrue(result.success)
        self.assertEqual(result.url, 'https://example.com')
        self.assertEqual(result.title, 'Test Page')
        self.assertIsInstance(result.structured_cards, list)
        self.assertIsInstance(result.raw_text, str)
        self.assertIsNotNone(result.response_time)
    
    @pytest.mark.asyncio
    @patch('core.web_scraper.httpx.AsyncClient')
    async def test_scrape_single_url_failure(self, mock_client):
        """Test failed single URL scraping."""
        # Mock failed response
        mock_client.return_value.__aenter__.return_value.get.side_effect = Exception('Connection error')
        
        result = await self.scraper.scrape_single_url('https://example.com')
        
        self.assertFalse(result.success)
        self.assertEqual(result.url, 'https://example.com')
        self.assertIn('Connection error', result.error_message)
        self.assertEqual(result.title, '')
        self.assertEqual(len(result.structured_cards), 0)
        self.assertEqual(result.raw_text, '')
    
    @pytest.mark.asyncio
    @patch('core.web_scraper.httpx.AsyncClient')
    async def test_scrape_urls_multiple(self, mock_client):
        """Test scraping multiple URLs."""
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><head><title>Test Page</title></head><body><h1>Test Content</h1></body></html>'
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        urls = ['https://example.com', 'https://test.com', 'https://demo.com']
        results = await self.scraper.scrape_urls(urls)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)
            self.assertIn(result.url, urls)
    
    def test_export_results(self):
        """Test results export."""
        results = [
            ScrapingResult(
                url='https://example.com',
                success=True,
                title='Test Page',
                summary='Test summary',
                structured_cards=[],
                raw_text='Test content',
                response_time=1.5
            ),
            ScrapingResult(
                url='https://failed.com',
                success=False,
                title='',
                summary='',
                structured_cards=[],
                raw_text='',
                error_message='Connection failed'
            )
        ]
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.scraper.export_results(results, temp_dir)
            
            # Check that files were created
            files = os.listdir(temp_dir)
            self.assertIn('scrape_summary.md', files)
            self.assertIn('scrape_cards.json', files)
            self.assertIn('scrape_raw_text.txt', files)
            
            # Check summary content
            with open(os.path.join(temp_dir, 'scrape_summary.md'), 'r') as f:
                summary_content = f.read()
                self.assertIn('https://example.com', summary_content)
                self.assertIn('https://failed.com', summary_content)
                self.assertIn('Success', summary_content)
                self.assertIn('Failed', summary_content)
    
    def test_robots_txt_check(self):
        """Test robots.txt checking."""
        # Test with robots.txt disabled
        self.scraper.respect_robots = False
        result = asyncio.run(self.scraper._check_robots_txt('https://example.com'))
        self.assertTrue(result)
        
        # Test with robots.txt enabled (mock)
        self.scraper.respect_robots = True
        with patch('core.web_scraper.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = 'User-agent: *\nDisallow: /'
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = asyncio.run(self.scraper._check_robots_txt('https://example.com'))
            self.assertFalse(result)  # Should be blocked
    
    def test_card_extraction(self):
        """Test structured card extraction."""
        from bs4 import BeautifulSoup
        
        html = '''
        <html>
            <body>
                <h2>About Us</h2>
                <p>We are a technology company.</p>
                <h2>Contact Information</h2>
                <p>Email us at info@example.com</p>
            </body>
        </html>
        '''
        
        soup = BeautifulSoup(html, 'html.parser')
        cards = self.scraper._extract_structured_cards(soup, 'https://example.com')
        
        self.assertIsInstance(cards, list)
        # Should find at least the about card
        about_cards = [card for card in cards if card.card_type == 'about']
        self.assertGreater(len(about_cards), 0)
    
    def test_section_content_extraction(self):
        """Test section content extraction."""
        from bs4 import BeautifulSoup
        
        html = '''
        <html>
            <body>
                <h2>About Us</h2>
                <p>First paragraph about the company.</p>
                <p>Second paragraph with more details.</p>
                <h2>Contact</h2>
                <p>Contact information here.</p>
            </body>
        </html>
        '''
        
        soup = BeautifulSoup(html, 'html.parser')
        heading = soup.find('h2', string='About Us')
        
        content = self.scraper._extract_section_content(heading)
        
        self.assertIsInstance(content, list)
        self.assertGreater(len(content), 0)
        self.assertIn('First paragraph about the company', content[0])
        self.assertIn('Second paragraph with more details', content[1])

if __name__ == '__main__':
    unittest.main()

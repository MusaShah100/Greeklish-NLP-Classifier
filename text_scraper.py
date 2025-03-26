import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import random
from urllib.parse import urljoin
import re
import os

class TextScraper(scrapy.Spider):
    name = 'text_scraper'
    
    # Greeklish sources (Greek websites with user comments/forums)
    greeklish_sources = [
        'https://www.tovima.gr/',
        'https://www.news247.gr/',
        'https://www.iefimerida.gr/',
        'https://www.gazzetta.gr/',
        'https://www.insomnia.gr/',  # Popular Greek forum with lots of Greeklish
        'https://www.car.gr/',       # Greek classifieds with comments
    ]
    
    # English sources (news websites)
    english_sources = [
        'https://www.bbc.com/news',
        'https://www.reuters.com/',
        'https://www.theguardian.com/international',
        'https://www.nytimes.com/',
    ]
    
    def __init__(self):
        super().__init__()
        self.greeklish_texts = set()  # Using set to ensure uniqueness
        self.english_texts = set()    # Using set to ensure uniqueness
        self.visited_urls = set()
        self.min_sentences_per_class = 300
        
    def start_requests(self):
        # Scrape Greeklish sources
        for url in self.greeklish_sources:
            yield scrapy.Request(url=url, callback=self.parse_greeklish_homepage, meta={'source': 'greeklish'})
            
        # Scrape English sources
        for url in self.english_sources:
            yield scrapy.Request(url=url, callback=self.parse_english_homepage, meta={'source': 'english'})
    
    def parse_greeklish_homepage(self, response):
        # Extract article links from homepage
        soup = BeautifulSoup(response.text, 'lxml')
        article_links = []
        
        # Find all links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Make sure it's an article link (typically has /article/ or similar in URL)
            if '/article/' in href or '/news/' in href or '/politics/' in href or '/sports/' in href or '/forum/' in href:
                full_url = urljoin(response.url, href)
                if full_url not in self.visited_urls:
                    article_links.append(full_url)
                    self.visited_urls.add(full_url)
        
        # Progress reporting
        print(f"Found {len(article_links)} article links on {response.url}")
        print(f"Current counts - Greeklish: {len(self.greeklish_texts)}, English: {len(self.english_texts)}")
        
        # Visit article pages to scrape comments and content
        for link in article_links[:20]:  # Increased from 10 to 20 articles per site
            if len(self.greeklish_texts) >= self.min_sentences_per_class:
                print(f"✅ Collected enough Greeklish texts: {len(self.greeklish_texts)}")
                break
            yield scrapy.Request(url=link, callback=self.parse_greeklish, meta={'source': 'greeklish'})
    
    def parse_english_homepage(self, response):
        # Extract article links from homepage
        soup = BeautifulSoup(response.text, 'lxml')
        article_links = []
        
        # Find all links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Make sure it's an article link
            if '/article/' in href or '/story/' in href or '/world/' in href or '/news/' in href:
                full_url = urljoin(response.url, href)
                if full_url not in self.visited_urls:
                    article_links.append(full_url)
                    self.visited_urls.add(full_url)
        
        # Progress reporting
        print(f"Found {len(article_links)} article links on {response.url}")
        print(f"Current counts - Greeklish: {len(self.greeklish_texts)}, English: {len(self.english_texts)}")
        
        # Visit article pages
        for link in article_links[:20]:  # Increased from 10 to 20 articles per site
            if len(self.english_texts) >= self.min_sentences_per_class:
                print(f"✅ Collected enough English texts: {len(self.english_texts)}")
                break
            yield scrapy.Request(url=link, callback=self.parse_english, meta={'source': 'english'})
    
    def parse_greeklish(self, response):
        # Extract text from comments and article content
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Look for common comment sections and article content
        text_elements = soup.find_all(['div', 'p', 'span', 'article', 'comment', 'section'])
        
        # Also look for specific comment section IDs/classes
        comment_sections = soup.select('#comments, .comments, .comment-section, .discussion, .forum-post, .message')
        for section in comment_sections:
            text_elements.extend(section.find_all(['p', 'div', 'span']))
        
        for element in text_elements:
            text = element.get_text().strip()
            if text and len(text.split()) > 5:  # Ensure we have meaningful sentences
                # Improved Greeklish detection
                if self.is_greeklish(text):
                    # Clean and normalize before adding to set for better deduplication
                    clean_text = self.basic_clean(text)
                    # Skip if too similar to existing texts
                    if not any(self.text_similarity(clean_text, existing) > 0.7 for existing in self.greeklish_texts):
                        if len(clean_text.split()) >= 5:  # Recheck length after cleaning
                            self.greeklish_texts.add(clean_text)
                            if len(self.greeklish_texts) % 10 == 0:
                                print(f"Found {len(self.greeklish_texts)} Greeklish texts")
                    
                    if len(self.greeklish_texts) >= self.min_sentences_per_class:
                        print(f"✅ Target reached: {len(self.greeklish_texts)} Greeklish texts collected")
                        return
    
    def is_greeklish(self, text):
        # Check for mix of Greek and Latin characters
        has_greek = bool(re.search(r'[α-ωΑ-Ω]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        
        # Common Greeklish patterns
        greeklish_patterns = [
            r'th[ae]lo', r'eimai', r'eisai', r'kser', r'xer', r'exo', r'mpor', 
            r'prepi', r'ime', r'ise', r'eine', r'ine', r'edo', r'ekei', r'tora',
            r'kala', r'oxi', r'nai', r'mou', r'sou', r'[dt]iko', r'apo', r'gia',
            r'malak', r'gamw', r'gamo', r're ', r' re', r'vre ', r' vre'
        ]
        
        pattern_match = any(re.search(pattern, text.lower()) for pattern in greeklish_patterns)
        
        # Either it has both Greek and Latin chars, or it matches Greeklish patterns
        return (has_greek and has_latin) or pattern_match
    
    def basic_clean(self, text):
        # Basic cleaning for comparison purposes
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        return text.strip()
    
    def text_similarity(self, text1, text2):
        # Simple similarity measure to avoid adding near-duplicate texts
        # Return similarity score between 0 (different) and 1 (identical)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        return len(intersection) / max(len(words1), len(words2))
    
    def parse_english(self, response):
        # Extract text from article content
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Look for article content
        article_elements = soup.select('article, .article, .story, .content, main')
        text_elements = []
        
        if article_elements:
            for article in article_elements:
                text_elements.extend(article.find_all(['p', 'h1', 'h2', 'h3', 'div.paragraph']))
        else:
            # Fallback to general paragraph elements if no specific article elements found
            text_elements = soup.find_all(['p', 'article'])
        
        for element in text_elements:
            text = element.get_text().strip()
            if text and len(text.split()) > 5:  # Ensure we have meaningful sentences
                # Basic English detection (mostly Latin characters, common punctuation)
                if self.is_english(text):
                    # Clean and normalize before adding to set for better deduplication
                    clean_text = self.basic_clean(text)
                    # Skip if too similar to existing texts
                    if not any(self.text_similarity(clean_text, existing) > 0.7 for existing in self.english_texts):
                        if len(clean_text.split()) >= 5:  # Recheck length after cleaning
                            self.english_texts.add(clean_text)
                            if len(self.english_texts) % 10 == 0:
                                print(f"Found {len(self.english_texts)} English texts")
                    
                    if len(self.english_texts) >= self.min_sentences_per_class:
                        print(f"✅ Target reached: {len(self.english_texts)} English texts collected")
                        return
    
    def is_english(self, text):
        # Check if text is primarily English
        # Should contain mostly Latin letters and standard punctuation
        non_english_ratio = len(re.findall(r'[^\x00-\x7F]', text)) / len(text) if text else 1
        return non_english_ratio < 0.1 and re.search(r'[a-zA-Z]', text)
    
    def closed(self, reason):
        # This method is called when the spider is closed
        print("\n=== Scraping Complete ===")
        print(f"Collected {len(self.greeklish_texts)} Greeklish texts")
        print(f"Collected {len(self.english_texts)} English texts")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV file
        data = []
        
        for text in self.greeklish_texts:
            data.append({'text': text, 'label': 'greeklish'})
            
        for text in self.english_texts:
            data.append({'text': text, 'label': 'english'})
            
        df = pd.DataFrame(data)
        
        # Shuffle the data for better mixing of classes
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to a CSV file
        output_file = 'data/scraped_texts.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

def main():
    # Set up and run the crawler
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'ERROR'  # Suppress most logs for clarity
    })
    
    process.crawl(TextScraper)
    print("Starting the scraper...")
    process.start()  # The script will block here until the crawling is finished

if __name__ == "__main__":
    main() 
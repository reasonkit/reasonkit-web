"""
WebBrowserAdapter implementation for reasonkit-web.

This module provides a Python implementation of the WebBrowserAdapter trait
defined in reasonkit-core/src/traits/web.rs. It wraps browser automation
functionality using Playwright for cross-browser support.

Usage:
    async with WebBrowserAdapter() as browser:
        page = await browser.navigate("https://example.com")
        content = await browser.extract_content(page)
        print(content.text)

For PyO3 bindings, see the `WebBrowserAdapterPyO3` class.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional

# Playwright imports (async API)
try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
        Response,
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    BrowserContext = Any
    Page = Any
    Playwright = Any
    Response = Any


# =============================================================================
# Error Types (matching Rust WebAdapterError)
# =============================================================================


class WebAdapterError(Exception):
    """Base exception for web adapter operations."""
    pass


class ConnectionError(WebAdapterError):
    """Connection to browser failed."""
    pass


class NavigationError(WebAdapterError):
    """Navigation to URL failed."""
    pass


class ExtractionError(WebAdapterError):
    """Content extraction failed."""
    pass


class TimeoutError(WebAdapterError):
    """Operation timed out."""
    pass


class ElementNotFoundError(WebAdapterError):
    """Element matching selector not found."""
    pass


class JavaScriptError(WebAdapterError):
    """JavaScript evaluation error."""
    pass


class ScreenshotError(WebAdapterError):
    """Screenshot capture failed."""
    pass


class NotConnectedError(WebAdapterError):
    """Browser is not connected."""
    pass


# =============================================================================
# Enums (matching Rust types)
# =============================================================================


class WaitUntil(str, Enum):
    """When to consider navigation complete."""
    LOAD = "load"
    DOM_CONTENT_LOADED = "domcontentloaded"
    NETWORK_IDLE = "networkidle"
    NETWORK_ALMOST_IDLE = "commit"  # Playwright uses 'commit' for early load


class ExtractFormat(str, Enum):
    """Format for extracted content."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class CaptureFormat(str, Enum):
    """Format for captured images."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"  # Note: Not all browsers support WebP screenshots


# =============================================================================
# Data Classes (matching Rust structs)
# =============================================================================


@dataclass
class Viewport:
    """Viewport dimensions."""
    width: int = 1920
    height: int = 1080
    device_scale_factor: float = 1.0
    is_mobile: bool = False


@dataclass
class NavigateOptions:
    """Options for navigation."""
    wait_until: WaitUntil = WaitUntil.NETWORK_IDLE
    timeout_ms: int = 30000
    user_agent: Optional[str] = None
    headers: list[tuple[str, str]] = field(default_factory=list)
    viewport: Optional[Viewport] = None

    @property
    def timeout_seconds(self) -> float:
        return self.timeout_ms / 1000.0


@dataclass
class PageHandle:
    """Handle to a loaded page."""
    id: str
    url: str
    title: Optional[str] = None
    status_code: int = 200
    load_time_ms: int = 0
    _page: Optional[Page] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "status_code": self.status_code,
            "load_time_ms": self.load_time_ms,
        }


@dataclass
class Link:
    """A hyperlink from extracted content."""
    text: str
    href: str
    rel: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "href": self.href, "rel": self.rel}


@dataclass
class Image:
    """An image from extracted content."""
    src: str
    alt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "alt": self.alt,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ExtractOptions:
    """Options for content extraction."""
    format: ExtractFormat = ExtractFormat.MARKDOWN
    include_metadata: bool = True
    clean_html: bool = True
    include_links: bool = True
    include_images: bool = False
    max_length: Optional[int] = None


@dataclass
class ExtractedContent:
    """Extracted content from a page."""
    text: str
    format: ExtractFormat
    title: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = None
    word_count: int = 0
    links: list[Link] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "format": self.format.value,
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "published_date": self.published_date,
            "word_count": self.word_count,
            "links": [link.to_dict() for link in self.links],
            "images": [img.to_dict() for img in self.images],
            "metadata": self.metadata,
        }


@dataclass
class ClipRect:
    """Rectangle for clipping screenshots."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class CaptureOptions:
    """Options for screenshot capture."""
    format: CaptureFormat = CaptureFormat.PNG
    quality: int = 90
    full_page: bool = True
    clip: Optional[ClipRect] = None


@dataclass
class CapturedPage:
    """A captured page (screenshot or PDF)."""
    handle: PageHandle
    format: CaptureFormat
    data: bytes
    width: int
    height: int

    @property
    def base64_data(self) -> str:
        return base64.b64encode(self.data).decode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "format": self.format.value,
            "data_base64": self.base64_data,
            "width": self.width,
            "height": self.height,
            "size_bytes": len(self.data),
        }


@dataclass
class Cookie:
    """A browser cookie."""
    name: str
    value: str
    domain: Optional[str] = None
    path: Optional[str] = None
    expires: Optional[int] = None
    http_only: bool = False
    secure: bool = False
    same_site: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "value": self.value}
        if self.domain:
            result["domain"] = self.domain
        if self.path:
            result["path"] = self.path
        if self.expires:
            result["expires"] = self.expires
        result["httpOnly"] = self.http_only
        result["secure"] = self.secure
        if self.same_site:
            result["sameSite"] = self.same_site
        return result


# =============================================================================
# Abstract Base Class (matches Rust trait)
# =============================================================================


class WebBrowserAdapterBase(ABC):
    """
    Abstract base class for web browser operations.

    This matches the WebBrowserAdapter trait defined in reasonkit-core.
    Implementations must provide all methods to satisfy the interface.
    """

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the browser instance."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the browser instance."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected."""
        pass

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    @abstractmethod
    async def navigate(
        self, url: str, options: Optional[NavigateOptions] = None
    ) -> PageHandle:
        """Navigate to a URL and return a handle to the loaded page."""
        pass

    @abstractmethod
    async def wait_for_load(
        self, handle: PageHandle, timeout_ms: int = 30000
    ) -> None:
        """Wait for the page to finish loading."""
        pass

    @abstractmethod
    async def go_back(self, handle: PageHandle) -> None:
        """Go back in browser history."""
        pass

    @abstractmethod
    async def go_forward(self, handle: PageHandle) -> None:
        """Go forward in browser history."""
        pass

    @abstractmethod
    async def reload(self, handle: PageHandle) -> None:
        """Reload the current page."""
        pass

    # -------------------------------------------------------------------------
    # Content Extraction
    # -------------------------------------------------------------------------

    @abstractmethod
    async def extract_content(
        self, handle: PageHandle, options: Optional[ExtractOptions] = None
    ) -> ExtractedContent:
        """Extract content from the page in the specified format."""
        pass

    @abstractmethod
    async def extract_links(self, handle: PageHandle) -> list[Link]:
        """Extract all links from the page."""
        pass

    @abstractmethod
    async def extract_structured(
        self, handle: PageHandle, selector: str
    ) -> dict[str, Any]:
        """Extract structured data using a CSS selector."""
        pass

    @abstractmethod
    async def get_html(self, handle: PageHandle) -> str:
        """Get the raw HTML of the page."""
        pass

    # -------------------------------------------------------------------------
    # Capture
    # -------------------------------------------------------------------------

    @abstractmethod
    async def capture_screenshot(
        self, handle: PageHandle, options: Optional[CaptureOptions] = None
    ) -> CapturedPage:
        """Capture a screenshot of the page."""
        pass

    @abstractmethod
    async def capture_pdf(self, handle: PageHandle) -> bytes:
        """Capture the page as a PDF."""
        pass

    # -------------------------------------------------------------------------
    # Interaction
    # -------------------------------------------------------------------------

    @abstractmethod
    async def click(self, handle: PageHandle, selector: str) -> None:
        """Click an element matching the selector."""
        pass

    @abstractmethod
    async def type_text(
        self, handle: PageHandle, selector: str, text: str
    ) -> None:
        """Type text into an element matching the selector."""
        pass

    @abstractmethod
    async def select_option(
        self, handle: PageHandle, selector: str, value: str
    ) -> None:
        """Select an option from a dropdown."""
        pass

    @abstractmethod
    async def scroll(self, handle: PageHandle, x: float, y: float) -> None:
        """Scroll the page."""
        pass

    @abstractmethod
    async def wait_for_selector(
        self, handle: PageHandle, selector: str, timeout_ms: int = 30000
    ) -> None:
        """Wait for an element to appear."""
        pass

    # -------------------------------------------------------------------------
    # JavaScript
    # -------------------------------------------------------------------------

    @abstractmethod
    async def evaluate_js(
        self, handle: PageHandle, script: str
    ) -> Any:
        """Evaluate JavaScript and return the result."""
        pass

    @abstractmethod
    async def inject_script(self, handle: PageHandle, script: str) -> None:
        """Inject a script into the page."""
        pass

    # -------------------------------------------------------------------------
    # Cookies & Storage
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_cookies(self, handle: PageHandle) -> list[Cookie]:
        """Get all cookies for the current page."""
        pass

    @abstractmethod
    async def set_cookie(self, handle: PageHandle, cookie: Cookie) -> None:
        """Set a cookie."""
        pass

    @abstractmethod
    async def clear_cookies(self, handle: PageHandle) -> None:
        """Clear all cookies."""
        pass

    @abstractmethod
    async def get_local_storage(
        self, handle: PageHandle, key: str
    ) -> Optional[str]:
        """Get local storage value."""
        pass

    @abstractmethod
    async def set_local_storage(
        self, handle: PageHandle, key: str, value: str
    ) -> None:
        """Set local storage value."""
        pass


# =============================================================================
# Playwright Implementation
# =============================================================================


class WebBrowserAdapter(WebBrowserAdapterBase):
    """
    Playwright-based implementation of WebBrowserAdapter.

    This implementation uses Playwright for cross-browser support (Chromium,
    Firefox, WebKit) with async/await patterns.

    Example:
        async with WebBrowserAdapter(headless=True) as browser:
            page = await browser.navigate("https://example.com")
            content = await browser.extract_content(page)
            print(content.text)
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        viewport: Optional[Viewport] = None,
        user_agent: Optional[str] = None,
        timeout_ms: int = 30000,
    ):
        """
        Initialize the browser adapter.

        Args:
            headless: Run browser in headless mode
            browser_type: Browser engine ("chromium", "firefox", "webkit")
            viewport: Default viewport dimensions
            user_agent: Custom user agent string
            timeout_ms: Default timeout for operations
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required. Install with: pip install playwright && "
                "playwright install"
            )

        self._headless = headless
        self._browser_type = browser_type
        self._viewport = viewport or Viewport()
        self._user_agent = user_agent
        self._timeout_ms = timeout_ms

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._pages: dict[str, Page] = {}
        self._connected = False

    async def __aenter__(self) -> "WebBrowserAdapter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the browser instance."""
        if self._connected:
            return

        try:
            self._playwright = await async_playwright().start()

            # Select browser engine
            if self._browser_type == "firefox":
                browser_engine = self._playwright.firefox
            elif self._browser_type == "webkit":
                browser_engine = self._playwright.webkit
            else:
                browser_engine = self._playwright.chromium

            self._browser = await browser_engine.launch(headless=self._headless)

            # Create context with viewport and user agent
            context_options = {
                "viewport": {
                    "width": self._viewport.width,
                    "height": self._viewport.height,
                },
                "device_scale_factor": self._viewport.device_scale_factor,
                "is_mobile": self._viewport.is_mobile,
            }
            if self._user_agent:
                context_options["user_agent"] = self._user_agent

            self._context = await self._browser.new_context(**context_options)
            self._connected = True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to browser: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the browser instance."""
        if not self._connected:
            return

        try:
            # Close all pages
            for page in self._pages.values():
                await page.close()
            self._pages.clear()

            # Close context and browser
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

        except Exception as e:
            raise ConnectionError(f"Error during disconnect: {e}") from e
        finally:
            self._playwright = None
            self._browser = None
            self._context = None
            self._connected = False

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self._browser is not None

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self.is_connected():
            raise NotConnectedError("Browser is not connected")

    def _get_page(self, handle: PageHandle) -> Page:
        """Get the Playwright page for a handle."""
        self._ensure_connected()
        page = self._pages.get(handle.id)
        if page is None:
            raise NotConnectedError(f"Page {handle.id} not found")
        return page

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    async def navigate(
        self, url: str, options: Optional[NavigateOptions] = None
    ) -> PageHandle:
        """Navigate to a URL and return a handle to the loaded page."""
        self._ensure_connected()
        opts = options or NavigateOptions()

        try:
            start_time = time.monotonic()

            # Create new page
            page = await self._context.new_page()
            page_id = str(uuid.uuid4())

            # Apply custom headers if provided
            if opts.headers:
                await page.set_extra_http_headers(dict(opts.headers))

            # Apply viewport if provided
            if opts.viewport:
                await page.set_viewport_size({
                    "width": opts.viewport.width,
                    "height": opts.viewport.height,
                })

            # Navigate with wait condition
            response = await page.goto(
                url,
                wait_until=opts.wait_until.value,
                timeout=opts.timeout_ms,
            )

            # Get page info
            title = await page.title()
            status_code = response.status if response else 200
            load_time_ms = int((time.monotonic() - start_time) * 1000)

            # Store page reference
            self._pages[page_id] = page

            handle = PageHandle(
                id=page_id,
                url=page.url,
                title=title,
                status_code=status_code,
                load_time_ms=load_time_ms,
                _page=page,
            )

            return handle

        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Navigation timed out: {url}") from e
        except Exception as e:
            raise NavigationError(f"Navigation failed: {e}") from e

    async def wait_for_load(
        self, handle: PageHandle, timeout_ms: int = 30000
    ) -> None:
        """Wait for the page to finish loading."""
        page = self._get_page(handle)
        try:
            await page.wait_for_load_state("load", timeout=timeout_ms)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Wait for load timed out") from e

    async def go_back(self, handle: PageHandle) -> None:
        """Go back in browser history."""
        page = self._get_page(handle)
        await page.go_back()

    async def go_forward(self, handle: PageHandle) -> None:
        """Go forward in browser history."""
        page = self._get_page(handle)
        await page.go_forward()

    async def reload(self, handle: PageHandle) -> None:
        """Reload the current page."""
        page = self._get_page(handle)
        await page.reload()

    # -------------------------------------------------------------------------
    # Content Extraction
    # -------------------------------------------------------------------------

    async def extract_content(
        self, handle: PageHandle, options: Optional[ExtractOptions] = None
    ) -> ExtractedContent:
        """Extract content from the page in the specified format."""
        page = self._get_page(handle)
        opts = options or ExtractOptions()

        try:
            # Extract metadata
            metadata = {}
            title = None
            description = None
            author = None
            published_date = None

            if opts.include_metadata:
                metadata = await self._extract_metadata(page)
                title = metadata.get("title")
                description = metadata.get("description")
                author = metadata.get("author")
                published_date = metadata.get("published_date")

            # Extract main content based on format
            if opts.format == ExtractFormat.HTML:
                text = await self._extract_html(page, opts.clean_html)
            elif opts.format == ExtractFormat.MARKDOWN:
                text = await self._extract_markdown(page)
            elif opts.format == ExtractFormat.JSON:
                text = await self._extract_json(page)
            else:
                text = await self._extract_plain_text(page)

            # Apply max length if specified
            if opts.max_length and len(text) > opts.max_length:
                text = text[:opts.max_length]

            # Extract links if requested
            links = []
            if opts.include_links:
                links = await self.extract_links(handle)

            # Extract images if requested
            images = []
            if opts.include_images:
                images = await self._extract_images(page)

            # Calculate word count
            word_count = len(text.split())

            return ExtractedContent(
                text=text,
                format=opts.format,
                title=title,
                description=description,
                author=author,
                published_date=published_date,
                word_count=word_count,
                links=links,
                images=images,
                metadata=metadata,
            )

        except Exception as e:
            raise ExtractionError(f"Content extraction failed: {e}") from e

    async def _extract_metadata(self, page: Page) -> dict[str, Any]:
        """Extract page metadata."""
        return await page.evaluate("""
            () => {
                const getMeta = (name) => {
                    const el = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                    return el ? el.getAttribute('content') : null;
                };
                return {
                    title: document.title,
                    description: getMeta('description') || getMeta('og:description'),
                    author: getMeta('author') || getMeta('article:author'),
                    published_date: getMeta('article:published_time') || getMeta('date'),
                    og_title: getMeta('og:title'),
                    og_image: getMeta('og:image'),
                    og_url: getMeta('og:url'),
                    canonical: document.querySelector('link[rel="canonical"]')?.href,
                };
            }
        """)

    async def _extract_plain_text(self, page: Page) -> str:
        """Extract plain text from the page."""
        return await page.evaluate("""
            () => {
                // Try to find main content
                const mainSelectors = [
                    'article', 'main', '[role="main"]', '.content',
                    '.post-content', '.entry-content', '#content'
                ];
                for (const sel of mainSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText.length > 200) {
                        return el.innerText.trim();
                    }
                }
                return document.body.innerText.trim();
            }
        """)

    async def _extract_html(self, page: Page, clean: bool = True) -> str:
        """Extract HTML from the page."""
        if clean:
            return await page.evaluate("""
                () => {
                    const clone = document.body.cloneNode(true);
                    // Remove script, style, and other non-content elements
                    clone.querySelectorAll('script, style, noscript, iframe, svg')
                        .forEach(el => el.remove());
                    return clone.innerHTML;
                }
            """)
        return await page.content()

    async def _extract_markdown(self, page: Page) -> str:
        """Extract content as markdown."""
        html = await self._extract_html(page, clean=True)
        return self._html_to_markdown(html)

    async def _extract_json(self, page: Page) -> str:
        """Extract structured JSON data."""
        data = await page.evaluate("""
            () => {
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                const jsonData = [];
                scripts.forEach(script => {
                    try {
                        jsonData.push(JSON.parse(script.textContent));
                    } catch (e) {}
                });
                return JSON.stringify(jsonData, null, 2);
            }
        """)
        return data

    async def _extract_images(self, page: Page) -> list[Image]:
        """Extract images from the page."""
        images_data = await page.evaluate("""
            () => {
                const images = [];
                document.querySelectorAll('img').forEach(img => {
                    if (img.src && img.src.startsWith('http')) {
                        images.push({
                            src: img.src,
                            alt: img.alt || null,
                            width: img.naturalWidth || null,
                            height: img.naturalHeight || null,
                        });
                    }
                });
                return images;
            }
        """)
        return [Image(**img) for img in images_data]

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        text = html

        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text, flags=re.I)
        text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.I)

        # Convert headers
        for i in range(6, 0, -1):
            pattern = rf'<h{i}[^>]*>(.*?)</h{i}>'
            replacement = '#' * i + r' \1\n\n'
            text = re.sub(pattern, replacement, text, flags=re.I | re.S)

        # Convert paragraphs
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.I | re.S)

        # Convert line breaks
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)

        # Convert bold
        text = re.sub(r'<(b|strong)[^>]*>(.*?)</(b|strong)>', r'**\2**', text, flags=re.I | re.S)

        # Convert italic
        text = re.sub(r'<(i|em)[^>]*>(.*?)</(i|em)>', r'*\2*', text, flags=re.I | re.S)

        # Convert links
        text = re.sub(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.I | re.S)

        # Convert code
        text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.I | re.S)

        # Convert pre blocks
        text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', text, flags=re.I | re.S)

        # Convert list items
        text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.I | re.S)

        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        text = self._decode_html_entities(text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
            '&copy;': '(c)',
            '&reg;': '(R)',
            '&trade;': '(TM)',
            '&ndash;': '-',
            '&mdash;': '--',
            '&hellip;': '...',
            '&lsquo;': "'",
            '&rsquo;': "'",
            '&ldquo;': '"',
            '&rdquo;': '"',
        }
        for entity, replacement in entities.items():
            text = text.replace(entity, replacement)
        return text

    async def extract_links(self, handle: PageHandle) -> list[Link]:
        """Extract all links from the page."""
        page = self._get_page(handle)
        links_data = await page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    links.push({
                        text: a.innerText.trim(),
                        href: a.href,
                        rel: a.rel || null,
                    });
                });
                return links;
            }
        """)
        return [Link(**link) for link in links_data]

    async def extract_structured(
        self, handle: PageHandle, selector: str
    ) -> dict[str, Any]:
        """Extract structured data using a CSS selector."""
        page = self._get_page(handle)
        try:
            result = await page.evaluate(f"""
                () => {{
                    const elements = document.querySelectorAll('{selector}');
                    return Array.from(elements).map(el => ({{
                        tag: el.tagName.toLowerCase(),
                        text: el.innerText,
                        html: el.innerHTML,
                        attributes: Object.fromEntries(
                            Array.from(el.attributes).map(attr => [attr.name, attr.value])
                        ),
                    }}));
                }}
            """)
            return {"selector": selector, "elements": result}
        except Exception as e:
            raise ExtractionError(f"Structured extraction failed: {e}") from e

    async def get_html(self, handle: PageHandle) -> str:
        """Get the raw HTML of the page."""
        page = self._get_page(handle)
        return await page.content()

    # -------------------------------------------------------------------------
    # Capture
    # -------------------------------------------------------------------------

    async def capture_screenshot(
        self, handle: PageHandle, options: Optional[CaptureOptions] = None
    ) -> CapturedPage:
        """Capture a screenshot of the page."""
        page = self._get_page(handle)
        opts = options or CaptureOptions()

        try:
            screenshot_options = {
                "type": opts.format.value if opts.format != CaptureFormat.WEBP else "png",
                "full_page": opts.full_page,
            }

            if opts.format in (CaptureFormat.JPEG,):
                screenshot_options["quality"] = opts.quality

            if opts.clip:
                screenshot_options["clip"] = {
                    "x": opts.clip.x,
                    "y": opts.clip.y,
                    "width": opts.clip.width,
                    "height": opts.clip.height,
                }

            data = await page.screenshot(**screenshot_options)

            # Get viewport dimensions
            viewport = await page.viewport_size()
            width = viewport["width"] if viewport else self._viewport.width
            height = viewport["height"] if viewport else self._viewport.height

            return CapturedPage(
                handle=handle,
                format=opts.format,
                data=data,
                width=width,
                height=height,
            )

        except Exception as e:
            raise ScreenshotError(f"Screenshot capture failed: {e}") from e

    async def capture_pdf(self, handle: PageHandle) -> bytes:
        """Capture the page as a PDF."""
        page = self._get_page(handle)
        try:
            return await page.pdf()
        except Exception as e:
            raise ScreenshotError(f"PDF capture failed: {e}") from e

    # -------------------------------------------------------------------------
    # Interaction
    # -------------------------------------------------------------------------

    async def click(self, handle: PageHandle, selector: str) -> None:
        """Click an element matching the selector."""
        page = self._get_page(handle)
        try:
            await page.click(selector, timeout=self._timeout_ms)
        except Exception as e:
            raise ElementNotFoundError(f"Click failed on '{selector}': {e}") from e

    async def type_text(
        self, handle: PageHandle, selector: str, text: str
    ) -> None:
        """Type text into an element matching the selector."""
        page = self._get_page(handle)
        try:
            await page.fill(selector, text, timeout=self._timeout_ms)
        except Exception as e:
            raise ElementNotFoundError(f"Type failed on '{selector}': {e}") from e

    async def select_option(
        self, handle: PageHandle, selector: str, value: str
    ) -> None:
        """Select an option from a dropdown."""
        page = self._get_page(handle)
        try:
            await page.select_option(selector, value, timeout=self._timeout_ms)
        except Exception as e:
            raise ElementNotFoundError(f"Select failed on '{selector}': {e}") from e

    async def scroll(self, handle: PageHandle, x: float, y: float) -> None:
        """Scroll the page."""
        page = self._get_page(handle)
        await page.evaluate(f"window.scrollTo({x}, {y})")

    async def wait_for_selector(
        self, handle: PageHandle, selector: str, timeout_ms: int = 30000
    ) -> None:
        """Wait for an element to appear."""
        page = self._get_page(handle)
        try:
            await page.wait_for_selector(selector, timeout=timeout_ms)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Timeout waiting for '{selector}'") from e
        except Exception as e:
            raise ElementNotFoundError(f"Wait for selector failed: {e}") from e

    # -------------------------------------------------------------------------
    # JavaScript
    # -------------------------------------------------------------------------

    async def evaluate_js(self, handle: PageHandle, script: str) -> Any:
        """Evaluate JavaScript and return the result."""
        page = self._get_page(handle)
        try:
            return await page.evaluate(script)
        except Exception as e:
            raise JavaScriptError(f"JavaScript evaluation failed: {e}") from e

    async def inject_script(self, handle: PageHandle, script: str) -> None:
        """Inject a script into the page."""
        page = self._get_page(handle)
        try:
            await page.add_script_tag(content=script)
        except Exception as e:
            raise JavaScriptError(f"Script injection failed: {e}") from e

    # -------------------------------------------------------------------------
    # Cookies & Storage
    # -------------------------------------------------------------------------

    async def get_cookies(self, handle: PageHandle) -> list[Cookie]:
        """Get all cookies for the current page."""
        self._ensure_connected()
        cookies_data = await self._context.cookies()
        return [
            Cookie(
                name=c["name"],
                value=c["value"],
                domain=c.get("domain"),
                path=c.get("path"),
                expires=int(c.get("expires", 0)) if c.get("expires") else None,
                http_only=c.get("httpOnly", False),
                secure=c.get("secure", False),
                same_site=c.get("sameSite"),
            )
            for c in cookies_data
        ]

    async def set_cookie(self, handle: PageHandle, cookie: Cookie) -> None:
        """Set a cookie."""
        self._ensure_connected()
        page = self._get_page(handle)
        cookie_dict = cookie.to_dict()
        if not cookie_dict.get("url"):
            cookie_dict["url"] = page.url
        await self._context.add_cookies([cookie_dict])

    async def clear_cookies(self, handle: PageHandle) -> None:
        """Clear all cookies."""
        self._ensure_connected()
        await self._context.clear_cookies()

    async def get_local_storage(
        self, handle: PageHandle, key: str
    ) -> Optional[str]:
        """Get local storage value."""
        page = self._get_page(handle)
        return await page.evaluate(f"localStorage.getItem('{key}')")

    async def set_local_storage(
        self, handle: PageHandle, key: str, value: str
    ) -> None:
        """Set local storage value."""
        page = self._get_page(handle)
        await page.evaluate(f"localStorage.setItem('{key}', '{value}')")


# =============================================================================
# Convenience Functions
# =============================================================================


@asynccontextmanager
async def browser_session(
    headless: bool = True,
    browser_type: str = "chromium",
    **kwargs,
) -> AsyncIterator[WebBrowserAdapter]:
    """
    Context manager for browser sessions.

    Example:
        async with browser_session() as browser:
            page = await browser.navigate("https://example.com")
            content = await browser.extract_content(page)
    """
    adapter = WebBrowserAdapter(headless=headless, browser_type=browser_type, **kwargs)
    try:
        await adapter.connect()
        yield adapter
    finally:
        await adapter.disconnect()


async def quick_extract(url: str, format: ExtractFormat = ExtractFormat.MARKDOWN) -> ExtractedContent:
    """
    Quick extraction of content from a URL.

    Example:
        content = await quick_extract("https://example.com")
        print(content.text)
    """
    async with browser_session() as browser:
        page = await browser.navigate(url)
        return await browser.extract_content(page, ExtractOptions(format=format))


async def quick_screenshot(url: str, path: str) -> None:
    """
    Quick screenshot of a URL to file.

    Example:
        await quick_screenshot("https://example.com", "screenshot.png")
    """
    async with browser_session() as browser:
        page = await browser.navigate(url)
        captured = await browser.capture_screenshot(page)
        with open(path, "wb") as f:
            f.write(captured.data)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Errors
    "WebAdapterError",
    "ConnectionError",
    "NavigationError",
    "ExtractionError",
    "TimeoutError",
    "ElementNotFoundError",
    "JavaScriptError",
    "ScreenshotError",
    "NotConnectedError",
    # Enums
    "WaitUntil",
    "ExtractFormat",
    "CaptureFormat",
    # Data classes
    "Viewport",
    "NavigateOptions",
    "PageHandle",
    "Link",
    "Image",
    "ExtractOptions",
    "ExtractedContent",
    "ClipRect",
    "CaptureOptions",
    "CapturedPage",
    "Cookie",
    # Classes
    "WebBrowserAdapterBase",
    "WebBrowserAdapter",
    # Functions
    "browser_session",
    "quick_extract",
    "quick_screenshot",
]


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="WebBrowserAdapter CLI for quick web operations"
    )
    parser.add_argument("url", help="URL to process")
    parser.add_argument(
        "--action",
        choices=["extract", "screenshot", "html", "links"],
        default="extract",
        help="Action to perform",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain_text", "html", "json"],
        default="markdown",
        help="Output format for extract action",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (for screenshot)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode",
    )

    args = parser.parse_args()

    async def main():
        async with browser_session(headless=args.headless) as browser:
            page = await browser.navigate(args.url)

            if args.action == "extract":
                format_map = {
                    "markdown": ExtractFormat.MARKDOWN,
                    "plain_text": ExtractFormat.PLAIN_TEXT,
                    "html": ExtractFormat.HTML,
                    "json": ExtractFormat.JSON,
                }
                content = await browser.extract_content(
                    page, ExtractOptions(format=format_map[args.format])
                )
                print(content.text)

            elif args.action == "screenshot":
                captured = await browser.capture_screenshot(page)
                output_path = args.output or "screenshot.png"
                with open(output_path, "wb") as f:
                    f.write(captured.data)
                print(f"Screenshot saved to {output_path}")

            elif args.action == "html":
                html = await browser.get_html(page)
                print(html)

            elif args.action == "links":
                links = await browser.extract_links(page)
                for link in links:
                    print(f"[{link.text}]({link.href})")

    asyncio.run(main())

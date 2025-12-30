"""
ReasonKit Web MCP Server

The Sensing Layer for Autonomous Reasoning.
Exposes browser automation and WARC archiving as MCP tools.

Tools:
- web.capture: Deep Freeze - Navigate, intercept, create WARC archive
- web.sonar: Drift Detection - Monitor entropy/saturation in research threads
- web.triangulate: Cross-Verify - Find 3 independent sources for claims

Note: For ledger operations (anchor, verify), use reasonkit-core's
ProofLedger tools (proof_anchor, proof_verify) via MCP.
"""

import asyncio
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .gear.capture import DiveCaptureGear
from .gear.density.distill import ContentDistiller
from .gear.sonar import DiveSonarGear
from .gear.employee.digital_employee import DigitalEmployee
from .gear.assessment.vibe_engine import VIBEAssessmentEngine
from .gear.render3d.engine import Render3DEngine
from .gear.browser.manager import CrossPlatformBrowserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web")

# Initialize gear
capture_gear = DiveCaptureGear()
sonar_gear = DiveSonarGear()
distiller = ContentDistiller()

# Initialize new Digital Employee and advanced capabilities
digital_employee = DigitalEmployee()
vibe_engine = VIBEAssessmentEngine()
render3d_engine = Render3DEngine()
browser_manager = CrossPlatformBrowserManager()

# Create MCP server
server = Server("reasonkit-web")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Web tools."""
    return [
        Tool(
            name="web_capture",
            description=(
                "Deep Freeze: Navigate to URL, intercept network traffic, "
                "and create an immutable WARC archive. Use this BEFORE making "
                "claims based on web sources. Returns WARC path and content hash. "
                "For ledger anchoring, call proof_anchor from reasonkit-core."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to capture",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to extract specific content",
                    },
                    "wait_for": {
                        "type": "string",
                        "description": "Optional selector to wait for before capture",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="web_sonar",
            description=(
                "Drift Detection: Analyze text for information saturation. "
                "Detects when research is looping or seeing redundant content. "
                "Returns entropy score and 'surface now' recommendation if saturated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "new_text": {
                        "type": "string",
                        "description": "New text to analyze for novelty",
                    },
                    "context": {
                        "type": "string",
                        "description": "Existing knowledge/context to compare against",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Information gain threshold (default: 1.05)",
                        "default": 1.05,
                    },
                },
                "required": ["new_text", "context"],
            },
        ),
        Tool(
            name="web_triangulate",
            description=(
                "Cross-Verify: Search for 3 independent sources confirming a claim. "
                "Uses web search to find corroborating evidence. Returns sources "
                "with confidence scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim to verify with multiple sources",
                    },
                    "min_sources": {
                        "type": "integer",
                        "description": "Minimum sources required (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["claim"],
            },
        ),
        # Digital Employee Tools
        Tool(
            name="digital_employee_click",
            description=(
                "MiniMax M2 Digital Employee: Click element with human-like precision. "
                "Supports left/right/double clicks with natural positioning variance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for element to click",
                    },
                    "click_type": {
                        "type": "string",
                        "description": "Type of click (left, right, double)",
                        "default": "left",
                    },
                },
                "required": ["session_id", "selector"],
            },
        ),
        Tool(
            name="digital_employee_type",
            description=(
                "MiniMax M2 Digital Employee: Type text with human-like timing. "
                "Simulates natural typing speed with occasional delays."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Target element selector (optional)",
                    },
                    "typing_speed": {
                        "type": "integer",
                        "description": "Typing speed in ms per character",
                        "default": 50,
                    },
                },
                "required": ["session_id", "text"],
            },
        ),
        Tool(
            name="digital_employee_fill_form",
            description=(
                "MiniMax M2 Digital Employee: Fill form with structured data. "
                "Automatically clears fields and types with human-like timing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "form_data": {
                        "type": "object",
                        "description": "Dictionary of field selectors to values",
                    },
                    "submit": {
                        "type": "boolean",
                        "description": "Whether to submit the form",
                        "default": False,
                    },
                },
                "required": ["session_id", "form_data"],
            },
        ),
        # VIBE Assessment Tools
        Tool(
            name="vibe_assess_page",
            description=(
                "VIBE Benchmark: Assess page aesthetic, usability, accessibility, "
                "and performance with MiniMax M2's aesthetic expression mastery."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "page_id": {
                        "type": "integer",
                        "description": "Page ID (uses first page if not specified)",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="vibe_get_element_analysis",
            description=(
                "VIBE Benchmark: Analyze individual visual elements for "
                "accessibility, contrast, and design quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for element to analyze",
                    },
                },
                "required": ["session_id", "selector"],
            },
        ),
        # 3D Rendering Tools
        Tool(
            name="render3d_detect_content",
            description=(
                "3D Rendering: Detect and analyze 3D content on page. "
                "Supports up to 7,000+ React Three Fiber instances."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "page_id": {
                        "type": "integer",
                        "description": "Page ID (uses first page if not specified)",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="render3d_create_scene",
            description=(
                "3D Rendering: Create new 3D scene with Three.js. "
                "Configurable quality settings and WebGL optimization."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "scene_config": {
                        "type": "object",
                        "description": "3D scene configuration",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "quality": {
                                "type": "string",
                                "enum": ["high", "medium", "low", "performance"],
                            },
                            "background_color": {"type": "string"},
                            "antialias": {"type": "boolean"},
                            "fov": {"type": "number"},
                        },
                    },
                },
                "required": ["session_id", "scene_config"],
            },
        ),
        Tool(
            name="render3d_add_instance",
            description=(
                "3D Rendering: Add 3D instance to scene (box, sphere, plane). "
                "Supports custom materials, animations, and positioning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "scene_id": {
                        "type": "string",
                        "description": "3D scene ID",
                    },
                    "instance_config": {
                        "type": "object",
                        "description": "3D instance configuration",
                        "properties": {
                            "instance_id": {"type": "string"},
                            "type": {"type": "string", "enum": ["box", "sphere", "plane"]},
                            "size": {"type": "array", "items": {"type": "number"}},
                            "position": {"type": "array", "items": {"type": "number"}},
                            "rotation": {"type": "array", "items": {"type": "number"}},
                            "scale": {"type": "array", "items": {"type": "number"}},
                            "material": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "color": {"type": "number"},
                                    "metalness": {"type": "number"},
                                    "roughness": {"type": "number"},
                                },
                            },
                        },
                    },
                },
                "required": ["session_id", "scene_id", "instance_config"],
            },
        ),
        Tool(
            name="render3d_interact",
            description=(
                "3D Rendering: Interact with 3D scene elements. "
                "Supports clicks, camera rotation, and instance manipulation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "scene_id": {
                        "type": "string",
                        "description": "3D scene ID",
                    },
                    "interaction": {
                        "type": "object",
                        "description": "Interaction specification",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "click",
                                    "rotate_camera",
                                    "animate_instance",
                                    "change_material",
                                ],
                            },
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "instance_id": {"type": "string"},
                            "animation_type": {"type": "string"},
                            "value": {"type": "array", "items": {"type": "number"}},
                            "color": {"type": "number"},
                        },
                    },
                },
                "required": ["session_id", "scene_id", "interaction"],
            },
        ),
        # Cross-Platform Browser Tools
        Tool(
            name="browser_start",
            description=(
                "Cross-Platform Browser: Start browser session with specified type. "
                "Supports Chrome, Firefox, Safari (WebKit), and Edge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "browser_type": {
                        "type": "string",
                        "enum": ["chromium", "firefox", "webkit", "edge"],
                        "default": "chromium",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Custom session ID (auto-generated if not provided)",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Run browser in headless mode",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="browser_navigate",
            description=(
                "Cross-Platform Browser: Navigate to URL in browser session. "
                "Handles platform-specific navigation optimization."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to",
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "Wait strategy",
                        "default": "domcontentloaded",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Navigation timeout in ms",
                        "default": 30000,
                    },
                },
                "required": ["session_id", "url"],
            },
        ),
        Tool(
            name="browser_get_info",
            description=(
                "Cross-Platform Browser: Get comprehensive browser session information. "
                "Includes performance metrics, error tracking, and platform details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="browser_close",
            description=(
                "Cross-Platform Browser: Close browser session and cleanup resources. "
                "Safely closes all pages, contexts, and browser instances."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session ID to close",
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name in {"web_capture", "dive_capture"}:
            result = await capture_gear.execute(
                url=arguments["url"],
                selector=arguments.get("selector"),
                wait_for=arguments.get("wait_for"),
            )

            # Auto-distill if content was extracted
            if result.get("extracted_content") and arguments.get("distill", True):
                # If extracted_content looks like HTML (starts with <), distill it
                content = result["extracted_content"]
                if isinstance(content, str) and content.strip().startswith("<"):
                    result["extracted_content"] = distiller.distill(content)

        elif name in {"web_sonar", "dive_sonar"}:
            result = sonar_gear.analyze(
                new_text=arguments["new_text"],
                context=arguments["context"],
                threshold=arguments.get("threshold", 1.05),
            )

        elif name in {"web_triangulate", "dive_triangulate"}:
            # This is a stub implementation. In a real scenario, this would trigger
            # autonomous research via the agent to find sources.
            # Here we just analyze the claim structure.
            result = {
                "status": "pending_implementation",
                "message": (
                    "To use triangulation, the agent must perform search. "
                    "This tool is a placeholder for the logic engine."
                ),
            }

        # Digital Employee Tools
        elif name == "digital_employee_click":
            session_id = arguments["session_id"]
            selector = arguments["selector"]
            click_type = arguments.get("click_type", "left")

            page = await browser_manager.get_page(session_id)
            event = await digital_employee.click_element(page, selector, click_type=click_type)

            result = {
                "success": event.success,
                "type": event.type.value,
                "position": {"x": event.position.x, "y": event.position.y}
                if event.position
                else None,
                "selector": event.selector,
                "duration": event.duration,
                "error": event.error,
            }

        elif name == "digital_employee_type":
            session_id = arguments["session_id"]
            text = arguments["text"]
            selector = arguments.get("selector")
            typing_speed = arguments.get("typing_speed", 50)

            page = await browser_manager.get_page(session_id)
            event = await digital_employee.type_text(
                page, text, selector=selector, typing_speed=typing_speed, human_like=True
            )

            result = {
                "success": event.success,
                "type": event.type.value,
                "text": event.text,
                "selector": event.selector,
                "duration": event.duration,
                "error": event.error,
            }

        elif name == "digital_employee_fill_form":
            session_id = arguments["session_id"]
            form_data = arguments["form_data"]
            submit = arguments.get("submit", False)

            page = await browser_manager.get_page(session_id)
            events = await digital_employee.fill_form(page, form_data, submit=submit)

            result = {
                "success": all(event.success for event in events),
                "events": [
                    {
                        "success": event.success,
                        "selector": event.selector,
                        "duration": event.duration,
                        "error": event.error,
                    }
                    for event in events
                ],
                "interaction_summary": digital_employee.get_interaction_summary(),
            }

        # VIBE Assessment Tools
        elif name == "vibe_assess_page":
            session_id = arguments["session_id"]
            page_id = arguments.get("page_id")

            page = await browser_manager.get_page(session_id, page_id)
            assessment = await vibe_engine.assess_page(page)

            result = assessment

        elif name == "vibe_get_element_analysis":
            session_id = arguments["session_id"]
            selector = arguments["selector"]

            page = await browser_manager.get_page(session_id)

            # Get element information
            element = await page.wait_for_selector(selector, timeout=5000)
            if element:
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                bounding_box = await element.bounding_box()

                # Get computed styles
                computed_style = await element.evaluate("""
                    el => {
                        const styles = window.getComputedStyle(el);
                        return {
                            color: styles.color,
                            backgroundColor: styles.backgroundColor,
                            fontSize: styles.fontSize,
                            fontWeight: styles.fontWeight,
                            fontFamily: styles.fontFamily,
                            padding: styles.padding,
                            margin: styles.margin,
                            borderRadius: styles.borderRadius,
                            opacity: styles.opacity
                        };
                    }
                """)

                # Calculate contrast ratio
                def hex_to_rgb(hex_color):
                    hex_color = hex_color.lstrip("#")
                    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

                try:
                    text_rgb = hex_to_rgb(computed_style.get("color", "#000000"))
                    bg_rgb = hex_to_rgb(computed_style.get("backgroundColor", "#ffffff"))

                    # Simple contrast calculation
                    text_luminance = (
                        0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
                    ) / 255
                    bg_luminance = (0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]) / 255

                    contrast_ratio = (max(text_luminance, bg_luminance) + 0.05) / (
                        min(text_luminance, bg_luminance) + 0.05
                    )
                except:
                    contrast_ratio = 4.5  # Default

                # Font size check
                try:
                    font_size = float(computed_style.get("fontSize", "16px").replace("px", ""))
                    font_size_ok = font_size >= 16
                except:
                    font_size = 16
                    font_size_ok = False

                result = {
                    "selector": selector,
                    "tag_name": tag_name,
                    "bounding_box": bounding_box,
                    "computed_style": computed_style,
                    "contrast_ratio": round(contrast_ratio, 2),
                    "font_size_px": font_size,
                    "font_size_adequate": font_size_ok,
                    "accessibility_score": 1.0 if contrast_ratio >= 4.5 and font_size_ok else 0.7,
                    "recommendations": [
                        "Increase contrast ratio to at least 4.5:1"
                        if contrast_ratio < 4.5
                        else None,
                        "Increase font size to at least 16px" if not font_size_ok else None,
                    ],
                }
            else:
                result = {"error": f"Element not found: {selector}"}

        # 3D Rendering Tools
        elif name == "render3d_detect_content":
            session_id = arguments["session_id"]
            page_id = arguments.get("page_id")

            page = await browser_manager.get_page(session_id, page_id)
            analysis = await render3d_engine.detect_3d_content(page)

            result = analysis

        elif name == "render3d_create_scene":
            session_id = arguments["session_id"]
            scene_config = arguments["scene_config"]

            page = await browser_manager.get_page(session_id)
            scene_result = await render3d_engine.create_3d_scene(page, scene_config)

            result = scene_result

        elif name == "render3d_add_instance":
            session_id = arguments["session_id"]
            scene_id = arguments["scene_id"]
            instance_config = arguments["instance_config"]

            page = await browser_manager.get_page(session_id)
            instance_result = await render3d_engine.add_3d_instance(page, scene_id, instance_config)

            result = instance_result

        elif name == "render3d_interact":
            session_id = arguments["session_id"]
            scene_id = arguments["scene_id"]
            interaction = arguments["interaction"]

            page = await browser_manager.get_page(session_id)
            interaction_result = await render3d_engine.interact_with_3d(page, scene_id, interaction)

            result = interaction_result

        # Cross-Platform Browser Tools
        elif name == "browser_start":
            from .gear.browser.manager import BrowserType

            browser_type_str = arguments.get("browser_type", "chromium")
            session_id = arguments.get("session_id")
            headless = arguments.get("headless", True)

            # Map string to BrowserType enum
            browser_type_map = {
                "chromium": "chromium",
                "firefox": "firefox",
                "webkit": "webkit",
                "edge": "edge",
            }

            from .gear.browser.manager import BrowserType

            browser_type = BrowserType(browser_type_map.get(browser_type_str, "chromium"))

            session_id = await browser_manager.start_browser(
                browser_type=browser_type, session_id=session_id
            )

            result = {
                "session_id": session_id,
                "browser_type": browser_type_str,
                "headless": headless,
                "success": True,
            }

        elif name == "browser_navigate":
            session_id = arguments["session_id"]
            url = arguments["url"]
            wait_until = arguments.get("wait_until", "domcontentloaded")
            timeout = arguments.get("timeout", 30000)

            navigation_result = await browser_manager.navigate(
                session_id=session_id, url=url, wait_until=wait_until, timeout=timeout
            )

            result = navigation_result

        elif name == "browser_get_info":
            session_id = arguments["session_id"]

            info = await browser_manager.get_browser_info(session_id)

            result = info

        elif name == "browser_close":
            session_id = arguments["session_id"]

            success = await browser_manager.close_session(session_id)

            result = {"session_id": session_id, "success": success}

        else:
            result = {"error": f"Unknown tool: {name}"}

        import json

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [TextContent(type="text", text=f"Error: {e!s}")]


def main() -> int:
    """Run the Web MCP server."""
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: web\n\nRuns the ReasonKit Web MCP server over stdio.")
        return 0

    logger.info("Starting ReasonKit Web MCP Server...")
    asyncio.run(_run_server())
    return 0


async def _run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()

import os
import logging

from adodbapi.examples.xls_read import filename
from google import genai
from google.adk.tools import ToolContext
from google.genai import types
from jinja2.async_utils import auto_await
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tornado.locale import load_gettext_translations

load_dotenv()

# configure logging
logger = logging.getLogger(__name__)

# helper functions for asset version management
def get_next_version_number(tool_context: ToolContext, asset_name: str) -> int:
    """Get next version no. of a given asset name"""
    asset_versions = tool_context.state.get("asset_versions", {})
    current_version = asset_versions.get(asset_name, 0)
    next_version = current_version + 1
    return next_version

def update_asset_version(tool_context: ToolContext, asset_name: str, version: int) -> None:
    """Update version tracking for an asset"""
    if "asset_versions" not in tool_context.state:
        tool_context.state["asset_versions"] = {}
    if "asset_filenames" not in tool_context.state:
        tool_context.state["asset_filenames"] = {}

    tool_context.state["asset_versions"][asset_name] = version
    tool_context.state["asset_filenames"][asset_name]["version"] = filename

    # maintain list of all versions of this asset
    asset_history_key = f"{asset_name}_history"
    if asset_history_key in tool_context.state:
        tool_context.state[asset_history_key] = []
    tool_context.state[asset_history_key].append({
        "version": version,
        "asset_name": asset_name,
    })

def get_asset_versions_info(tool_context: ToolContext) -> str:
    """Get info about all asset versions in the session"""
    asset_versions = tool_context.state.get("asset_versions", {})
    if not asset_versions:
        return "No renovation renderings created yet."

    info_lines = ["Current renovation renderings:"]
    for asset_name, current_version in asset_versions.items():
        history_key = f"{asset_name}_history"
        history = tool_context.state.get(history_key, [])
        total_versions = len(history)
        latest_filename = tool_context.state.get("asset_filenames", {}).get(asset_name, "Unknown")
        info_lines.append(f" • {asset_name}: {total_versions} version(s), latest is v{current_version} ({latest_filename})")

    return "\n".join(info_lines)

def get_reference_images_info(tool_context: ToolContext) -> str:
    """Get info about all reference images in the session"""
    reference_images = tool_context.state.get("reference_images", {})
    if not reference_images:
        return "No renovation renderings created yet."

    info_lines = ["Current renovation renderings:"]
    for filename, info in reference_images.items():
        version = info.get("version", "Unknown")
        image_type = info.get("type", "reference")
        info_lines.append(f" • {filename} ({image_type} v{version})")

    return "\n".join(info_lines)

async def load_reference_image(tool_context: ToolContext, filename: str):
    """Load a reference image artifact by filename"""
    try:
        loaded_part = await tool_context.load_artifact(filename)
        if loaded_part:
            logger.info(f"Successfully loaded reference image: {filename}")
            return loaded_part
        else:
            logger.warning(f"Reference image not found: {filename}")
            return None

    except Exception as e:
        logger.error(f"Failed to load reference image: {filename}")
        return None

def get_latest_reference_image_filename(tool_context: ToolContext) -> str:
    """Get filename of the latest uploaded reference image"""
    return tool_context.state.get("latest_reference_image")
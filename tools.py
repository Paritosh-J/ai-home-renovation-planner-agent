import logging
import os

from adodbapi.examples.xls_read import filename
from dotenv import load_dotenv
from google import genai
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# configure logging
logger = logging.getLogger(__name__)


#### helper functions for asset version management ####

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

def create_versioned_filename(asset_name: str, version: int, file_extension: str = "png") -> str:
    """Create a versioned filename for an asset."""
    return f"{asset_name}_v{version}.{file_extension}"


def get_asset_versions_info(tool_context: ToolContext) -> str:
    """Get info about all asset versions in the session"""
    asset_versions = tool_context.state.get("asset_versions", {})
    if not asset_versions:
        return "No renovation renders created yet."

    info_lines = ["Current renovation renders:"]
    for asset_name, current_version in asset_versions.items():
        history_key = f"{asset_name}_history"
        history = tool_context.state.get(history_key, [])
        total_versions = len(history)
        latest_filename = tool_context.state.get("asset_filenames", {}).get(asset_name, "Unknown")
        info_lines.append(
            f" • {asset_name}: {total_versions} version(s), latest is v{current_version} ({latest_filename})")

    return "\n".join(info_lines)


def get_reference_images_info(tool_context: ToolContext) -> str:
    """Get info about all reference images in the session"""
    reference_images = tool_context.state.get("reference_images", {})
    if not reference_images:
        return "No renovation renders created yet."

    info_lines = ["Current renovation renders:"]
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


#### Pydantic input models ####

class GenerateRenovationRenderingInput(BaseModel):
    prompt: str = Field(...,
                        description="A detailed description of the renovated space to generate. Include room type, style, colors, materials, fixtures, lighting, and layout.")
    aspect_ratio: str = Field(default="16:9",
                              description="The desired aspect ratio, e.g., '1:1', '16:9', '9:16'. Default is 16:9 for room photos.")
    asset_name: str = Field(default="renovation_render",
                            description="Base name for the render (will be versioned automatically). Use descriptive names like 'kitchen_modern_farmhouse' or 'bathroom_spa'.")
    current_room_photo: str = Field(default=None,
                                    description="Optional: filename of the current room photo to use as reference for layout/structure.")
    inspiration_image: str = Field(default=None,
                                   description="Optional: filename of an inspiration image to guide the style. Use 'latest' for most recent upload.")


class EditRenovationRenderingInput(BaseModel):
    artifact_filename: str = Field(default=None,
                                   description="The filename of the render artifact to edit. If not provided, uses the last generated render.")
    prompt: str = Field(...,
                        description="The prompt describing the desired changes (e.g., 'make cabinets darker', 'add pendant lights', 'change floor to hardwood').")
    asset_name: str = Field(default=None,
                            description="Optional: specify asset name for the new version (defaults to incrementing current asset).")
    reference_image_filename: str = Field(default=None,
                                          description="Optional: filename of a reference image to guide the edit. Use 'latest' for most recent upload.")


#### Image generation tool ####

async def generate_renovation_render(tool_context: ToolContext, inputs: GenerateRenovationRenderingInput) -> str:
    """
    Renders an image of renovated space w.r.t design plan
    Uses Gemini 3's image generation capability to create image of renovated space
    Uses current room's pics as reference
    """
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set!")

    logger.info("Starting renovation render")
    try:
        client = genai.Client()

        # Handle potential dict inputs instead of Pydantic model
        if isinstance(inputs, dict):
            inputs = GenerateRenovationRenderingInput(**inputs)

        # handle reference images
        reference_images = []

        if inputs.current_room_photo:
            current_photo_part = await load_reference_image(tool_context, inputs.current_room_photo)
            if current_photo_part:
                reference_images.append(current_photo_part)
                logger.info(f"Using current room photo: {inputs.current_room_photo}")

        if inputs.inspiration_image:
            if inputs.inspiration_image == "latest":
                insp_filename = get_latest_reference_image_filename(tool_context)
            else:
                insp_filename = inputs.inspiration_image

            if insp_filename:
                inspiration_part = await load_reference_image(tool_context, insp_filename)
                if inspiration_part:
                    reference_images.append(inspiration_part)
                    logger.info(f"Using inspiration image: {insp_filename}")

        # build  enhanced prompt
        base_rewrite_prompt = f"""
        Create a highly detailed, photorealistic prompt for generating an interior design image.
        
        Original description: {inputs.prompt}
        
        **CRITICAL REQUIREMENT - PRESERVE EXACT LAYOUT:**
        The generated image MUST maintain the EXACT same room layout, structure, and spatial arrangement described in the prompt:
        - Keep all windows, doors, skylights in their exact positions
        - Keep all cabinets, counters, appliances in their exact positions
        - Keep the same room dimensions and proportions
        - Keep the same camera angle/perspective
        - ONLY change surface finishes: paint colors, cabinet colors, countertop materials, flooring, backsplash, hardware, and decorative elements
        - DO NOT move, add, or remove any structural elements or major fixtures
        
        Enhance this to be a professional interior photography prompt that includes:
        - Specific camera angle (match original photo perspective if described)
        - Exact colors and materials mentioned (apply to existing surfaces)
        - Realistic lighting (natural light from existing windows, fixture types)
        - Maintain existing spatial layout and dimensions
        - Texture and finish details for the new materials
        - Professional interior design photography quality
        
        Aspect ratio: {inputs.aspect_ratio}
        """

        if reference_images:
            base_rewrite_prompt += "\n\n**Reference Image Layout:** The reference image shows the EXACT layout that must be preserved. Match the camera angle, room structure, window/door positions, and furniture/appliance placement EXACTLY. Only change the surface finishes and colors."

        base_rewrite_prompt += "\n\n**Important:** Output your prompt as a single detailed paragraph optimized for photorealistic interior render. Emphasize that the layout must remain unchanged."

        # get enhanced prompt
        rewritten_prompt_response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=base_rewrite_prompt,
        )
        rewritten_prompt = rewritten_prompt_response.text
        logger.info(f"Enhanced prompt: {rewritten_prompt}")

        model = "gemini-3-flash-preview"

        # build content parts
        content_parts = [types.Part.from_text(text=rewritten_prompt)]
        content_parts.extend(reference_images)

        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ]
        )

        # generate versioned filename
        version = get_next_version_number(tool_context, inputs.asset_name)
        artifact_filename = create_versioned_filename(inputs.asset_name, version)
        logger.info(f"Render with artifact filename: {artifact_filename} (version: {version})")

        # generate image
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None or
                chunk.candidates[0].content is None or
                chunk.candidates[0].content.parts is None
            ):
                continue

            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data

                # create a part object from inline data
                # inline_data already has mime_type from API response
                image_part = types.Part(inline_data=inline_data)

                try:
                    # save image as artifact
                    version = await tool_context.save_artifact(
                        filename=artifact_filename,
                        artifact=image_part
                    )

                    # update version tracking
                    update_asset_version(tool_context, inputs.asset_name, version, artifact_filename)

                    # store in session state
                    tool_context.state["latest_generated_render"] = artifact_filename
                    tool_context.state["latest_asset_name"] = inputs.asset_name

                    logger.info(f"Saved render as artifact: {artifact_filename} (version: {version})")

                    return f"✅ Renovation render generated successfully!\n\nArtifact: {artifact_filename} (version: {version})"

                except Exception as e:
                    logger.error(f"Error saving artifact: {e}")
                    return f"Error saving render as artifact: {e}"

            else:
                if hasattr(chunk, 'text') and chunk.text:
                    logger.info(f"Model response: {chunk.text}")


        return "❌ No render generated. Please try again. 🥲"

    except Exception as e:
        logger.error(f"Error in generate_renovation_render: {e}")
        return f"Error generating renovation render: {e}"
    
    
async def edit_renovation_render(tool_context: ToolContext, inputs: EditRenovationRenderingInput) -> str:
    """
    Edits an image of renovated space w.r.t feedback/refinement
    Allows iterative improvements to rendered image like changing :
        * colors
        * materials
        * lighting
        * layout
        :
        :
    """
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set!")

    logger.info("Starting renovation render edit")
    
    try:
        client = genai.Client()
        
        # handle potential dict inputs (instead of Pydantic model)
        if isinstance(inputs, dict):
            inputs = EditRenovationRenderingInput(**inputs)
            
        # get artifact_filename from session state if not provided
        artifact_filename = inputs.artifact_filename
        if not artifact_filename:
            artifact_filename = tool_context.state.get("latest_generated_render")
            if not artifact_filename:
                return "❌ No artifact_filename provided and no previous render found in session."
            logger.info(f"Using last render from session: {artifact_filename}")
            
        # load existing render
        logger.info(f"Loading artifact: {artifact_filename}")
        try:
            loaded_image_part = await tool_context.load_artifact(artifact_filename)
            if not loaded_image_part:
                return f"❌ Artifact not found: {artifact_filename}"
            
        except Exception as e:
            logger.error(f"Error loading artifact: {e}")
            return f"Error loading artifact: {e}"
        
        # handle reference image if specified
        reference_image_part = None
        if inputs.reference_image_filename:
            if inputs.reference_image_filename == "latest":
                ref_filename = get_latest_reference_image_filename(tool_context)
            else:
                ref_filename = inputs.reference_image_filename
                
            if ref_filename:
                reference_image_part = await load_reference_image(tool_context, ref_filename)
                if reference_image_part:
                    logger.info(f"Using reference image: {ref_filename}")
                    
        model = "gemini-3-flash-preview"
        
        # build content parts
        content_parts = [loaded_image_part, types.Part.from_text(text=inputs.prompt)]
        if reference_image_part:
            content_parts.append(reference_image_part)
            
        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ]
        )
        
        # determine asset name and generate versioned filename
        if inputs.asset_name:
            asset_name = inputs.asset_name
        else:
            current_asset_name = tool_context.state.get("current_asset_name")
            if current_asset_name:
                asset_name = current_asset_name
            else:
                # extract from filename
                base_name = artifact_filename.split('_v')[0] if '_v' in artifact_filename else "renovation_render"
                asset_name = base_name
                
        version = get_next_version_number(tool_context, asset_name)
        edited_artifact_filename = create_versioned_filename(asset_name, version)
        logger.info(f"Editing with artifact filename: {edited_artifact_filename} (version: {version})")
        
        # edit imageq
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None or
                chunk.candidates[0].content is None or
                chunk.candidates[0].content.parts is None
            ):
                continue
            
            if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                
                # create a part object from inline data
                # the inline_data already contains mime_type from the API response
                edited_image_part = types.Part(inline_data=inline_data)
                
                try:
                    # save the edited image as an artifact
                    version = await tool_context.save_artifact(
                        filename=edited_artifact_filename,
                        artifact=edited_image_part
                    )
                    
                    # update version tracking
                    update_asset_version(tool_context, asset_name, version, edited_artifact_filename)
                    
                    # store in session state
                    tool_context.state["latest_generated_render"] = edited_artifact_filename
                    tool_context.state["latest_asset_name"] = asset_name
                    
                    logger.info(f"Saved edited render as artifact: {edited_artifact_filename} (version: {version})")
                    
                    return f"✅ Renovation render edited successfully!\n\nArtifact: {edited_artifact_filename} (version: {version})"
                
                except Exception as e:
                    logger.error(f"Error saving artifact: {e}")
                    return f"Error saving edited render as artifact: {e}"
                
            else:
                # log any text response
                if hasattr(chunk, "text") and chunk.text:
                    logger.info(f"Model response: {chunk.text}")
                    
        return "❌ No render edited. Please try again. 🥲"
    
    except Exception as e:
        logger.error(f"Error in edit_renovation_render: {e}")
        return f"Error editing renovation render: {e}"
    
    
#### utility tools ####

async def list_renovation_renders(tool_context: ToolContext) -> str:
    """Lists all renders created in this session"""
    return get_asset_versions_info(tool_context)


async def list_reference_images(tool_context: ToolContext) -> str:
    """Lists all reference images used in this session"""
    return get_reference_images_info(tool_context)


async def save_uploaded_renders(
    tool_context: ToolContext,
    image_data: str,
    artifact_name: str,
    image_type: str = "current_room"
) -> str:
    """
    Saves the uploaded image as a named artifact for later use in editing.
    Used when Visual Asseser detects an uploaded image.
    
    :param tool_context: The tool context
    :type tool_context: ToolContext
    :param image_data: Base64-encoded image data
    :type image_data: str
    :param artifact_name: Name of the artifact to save the image as (eg. 'current_room_1', 'inspiration_1')
    :type artifact_name: str
    :param image_type: Types of image (eg. "current_room", "inspiration")
    :type image_type: str
    :return: Returns success message with artifact filename
    :rtype: str
    """
    
    try:
        # create a part from the image data
        # Note: It's assumed that image_data is already in right format
        
        # save as artifact
        await tool_context.save_artifact(
            filename=artifact_name,
            artifact=image_data
        )
        
        # track in state
        if "uploaded_images" not in tool_context.state:
            tool_context.state["uploaded_images"] = {}

        tool_context.state["uploaded_images"][artifact_name] = {
            "type": image_type,
            "filename": artifact_name
        }
        
        if image_type == "current_room":
            tool_context.state["latest_reference_image"] = artifact_name
        elif image_type == "inspiration":
            tool_context.state["latest_inspiration_image"] = artifact_name
        
        logger.info(f"Saved image as artifact: {artifact_name}")
        return f"✅ Image saved as artifact: {artifact_name}"
    
    except Exception as e:
        logger.error(f"Error saving image as artifact: {e}")
        return f"❌ Error saving image as artifact: {e}"
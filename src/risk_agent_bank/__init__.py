from google.adk.models import LlmRequest
from google.genai import types as adk_types
from google.adk.models.registry import LLMRegistry
from google.adk.tools import ToolContext
from solace_ai_connector.common.log import log
from solace_agent_mesh.agent.utils.artifact_helpers import (
    save_artifact_with_metadata,
    DEFAULT_SCHEMA_MAX_KEYS,
)
import asyncio
import uuid


async def send_llm_request(tool_context, prompt: str) -> str:
    """
    Sends a request to the configured LLM (from artifact config if available,
    otherwise falls back to agent default).
    """
    log_identifier = "[send_llm_request]"
    log.info(f"{log_identifier} Execution Started")
    inv_context = tool_context._invocation_context
    host_component = getattr(inv_context.agent, "host_component", None)

    # Read config
    extraction_config = host_component.get_config(
        "extract_content_from_artifact_config", {}
    )
    model_config_for_extraction = extraction_config.get("model")

    # Get LLM
    registry = LLMRegistry()
    chosen_llm = None
    if model_config_for_extraction:
        try:
            chosen_llm = registry.get_llm(model_config_for_extraction)
        except Exception:
            log.warning(
                "%s Invalid 'model' config for extraction tool. Falling back to agent default.",
                log_identifier,
            )
            chosen_llm = inv_context.agent.canonical_model
    else:
        log.warning(
            "%s No 'model' specified in config. Falling back to agent default.",
            log_identifier,
        )
        chosen_llm = inv_context.agent.canonical_model

    # Build request
    internal_llm_contents = [
        adk_types.Content(
            role="user",
            parts=[adk_types.Part(text=prompt)]
        )
    ]
    internal_llm_request = LlmRequest(
        model=chosen_llm.model,
        contents=internal_llm_contents,
        config=adk_types.GenerateContentConfig(temperature=0.1),
    )
    log.info(f"{log_identifier} Using model: {chosen_llm.model}")

    extracted_content_str = ""

    # Async path
    if hasattr(chosen_llm, "generate_content_async"):
        async for event in chosen_llm.generate_content_async(internal_llm_request):
            if hasattr(event, "text") and event.text:
                extracted_content_str = event.text
                break
            elif hasattr(event, "parts") and event.parts:
                extracted_content_str = "".join(
                    [p.text for p in event.parts if hasattr(p, "text") and p.text]
                )
                break
            elif (
                hasattr(event, "content")
                and hasattr(event.content, "parts")
                and event.content.parts
            ):
                extracted_content_str = "".join(
                    [p.text for p in event.content.parts if hasattr(p, "text") and p.text]
                )
                break

    # Sync path
    elif hasattr(chosen_llm, "generate_content"):
        response = chosen_llm.generate_content(request=internal_llm_request)
        if hasattr(response, "text") and response.text:
            extracted_content_str = response.text
        elif hasattr(response, "parts") and response.parts:
            extracted_content_str = response.parts[0].text or ""
        elif (
            hasattr(response, "content")
            and hasattr(response.content, "parts")
            and response.content.parts
        ):
            extracted_content_str = "".join(
                [p.text for p in response.content.parts if hasattr(p, "text") and p.text]
            )

    if not extracted_content_str.strip():
        log.warning(f"{log_identifier} LLM returned empty response.")
        return "[ERROR: No usable LLM response]"

    log.info(
        "%s Internal LLM call completed. Extracted content length: %d chars",
        log_identifier,
        len(extracted_content_str),
    )
    return extracted_content_str

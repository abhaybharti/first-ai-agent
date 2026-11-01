import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


from google.genai import types
from config import GOOGLE_API_KEY
from google import genai


# client = InferenceClient(model="moonshotai/Kimi-K2.5", token=HUGGINGFACE_API_KEY)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
request_id = uuid.uuid4().hex


def record_observation(category, event, **details):
    observation = {
        "category": category,
        "event": event,
        "request_id": request_id,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    logger.info("observation %s", json.dumps(observation, default=str))


@contextmanager
def track_component(component, **fields):
    context = " ".join(f"{key}={value}" for key, value in fields.items())
    details = f"component={component} request_id={request_id} {context}".rstrip()
    start_time = time.perf_counter()
    logger.info("component_started %s", details)
    try:
        yield
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "component_failed %s duration_ms=%.2f",
            details,
            duration_ms,
        )
        record_observation(
            "system_feedback",
            "component_failed",
            component=component,
            status="error",
            error_type=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "component_completed %s duration_ms=%.2f",
            details,
            duration_ms,
        )
        record_observation(
            "system_feedback",
            "component_completed",
            component=component,
            status="success",
            duration_ms=round(duration_ms, 2),
        )


with track_component("gemini_client.initialization"):
    client = genai.Client(api_key=GOOGLE_API_KEY)


with track_component("weather_tool.registration"):
    weather_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get the current weather in a location.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "location": types.Schema(
                            type=types.Type.STRING,
                            description="The city to check",
                        )
                    },
                    required=["location"],
                ),
            )
        ]
    )

def get_weather(location):
    """Return demo weather data; replace this with a real weather provider."""
    result = {
        "source": "demo_mock",
        "location": location,
        "condition": "cloudy with low temperatures",
        "measurement_time": None,
    }
    record_observation(
        "environmental_data",
        "weather_result",
        **result,
    )
    return result

with track_component("gemini_request.configuration"):
    config = types.GenerateContentConfig(
        tools=[weather_tool],
        temperature=0.0,
    )

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {{"location": "New York"}}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """


# Using a chat session automatically manages conversation history and tool responses
with track_component("gemini_chat.initialization"):
    chat = client.chats.create(model="gemini-3.5-flash", config=config)

prompt = "What's the weather in Bangalore?"


def send_message_with_observability(message, stage):
    with track_component(
        "gemini_chat.send_message",
        stage=stage,
        model="gemini-3.5-flash",
    ):
        result = chat.send_message(message)
    return result


request_started_at = datetime.now(timezone.utc)
request_start = time.perf_counter()
record_observation("time_based_events", "request_started")
response = send_message_with_observability(prompt, "initial")

# 4. Check if the model wants to call a function
with track_component("model_response.tool_check"):
    has_tool_call = bool(response.function_calls)
    record_observation(
        "response_analysis",
        "model_response_received",
        stage="initial",
        has_text=bool(response.text),
        text_length=len(response.text or ""),
        function_call_count=len(response.function_calls or []),
    )

if has_tool_call:
    call = response.function_calls[0]

    # Execute the local Python function dynamically
    if call.name == "get_weather":
        with track_component("tool.execution", tool=call.name):
            tool_result = get_weather(**call.args)
    else:
        with track_component("tool.dispatch", tool=call.name):
            logger.warning(
                "unknown_tool request_id=%s tool=%s",
                request_id,
                call.name,
            )
            record_observation(
                "system_feedback",
                "unknown_tool",
                status="error",
                tool=call.name,
            )
            tool_result = f"Error: Function {call.name} not found."

    # 5. Send the function result back to Gemini
    response = send_message_with_observability(
        types.Part.from_function_response(
            name=call.name,
            response={"result": tool_result},
        ),
        "tool_result",
    )

record_observation(
    "response_analysis",
    "final_response_received",
    has_text=bool(response.text),
    text_length=len(response.text or ""),
    function_call_count=len(response.function_calls or []),
)
record_observation(
    "time_based_events",
    "request_completed",
    duration_ms=round((time.perf_counter() - request_start) * 1000, 2),
    started_at=request_started_at.isoformat(),
)
with track_component("response.output"):
    print("\nFinal Response from Gemini:")
    print(response.text)

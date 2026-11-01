import logging
import time
import uuid
from contextlib import contextmanager


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


@contextmanager
def track_component(component, **fields):
    context = " ".join(f"{key}={value}" for key, value in fields.items())
    details = f"component={component} request_id={request_id} {context}".rstrip()
    start_time = time.perf_counter()
    logger.info("component_started %s", details)
    try:
        yield
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "component_failed %s duration_ms=%.2f",
            details,
            duration_ms,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "component_completed %s duration_ms=%.2f",
            details,
            duration_ms,
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
    """Get the current weather in a given location."""
    return f"the weather in {location} is cloudy with low temperatures. \n"

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


logger.info("request_started request_id=%s", request_id)
response = send_message_with_observability(prompt, "initial")

# 4. Check if the model wants to call a function
with track_component("model_response.tool_check"):
    has_tool_call = bool(response.function_calls)

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
            tool_result = f"Error: Function {call.name} not found."

    # 5. Send the function result back to Gemini
    response = send_message_with_observability(
        types.Part.from_function_response(
            name=call.name,
            response={"result": tool_result},
        ),
        "tool_result",
    )

logger.info("request_completed request_id=%s", request_id)
with track_component("response.output"):
    print("\nFinal Response from Gemini:")
    print(response.text)

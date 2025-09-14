import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig , AsyncOpenAI
from agents.tool import function_tool 
import datetime as dt
from dotenv import load_dotenv
load_dotenv()

Api_Key = os.getenv("GEMINI_API_KEY")

Provider = AsyncOpenAI(
    api_key=Api_Key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
)
Model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client = Provider,
)
Config = RunConfig(
    model = Model,
    model_provider = Provider,
    tracing_disabled = True,
)

@function_tool
def get_current_date(country: str) -> int:
    """returns the current date."""
    return dt.datetime.now().date()

agent = Agent(
    name="Welcome agent",
    instructions="Always welcome users and provide assistance.",
    model= Model,
    tools=[get_current_date],
)
user_prompt = input("Enter Your Question Here:")
result = Runner.run_sync(agent, user_prompt)
print(result.final_output)
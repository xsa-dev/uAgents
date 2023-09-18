from uagents import Agent, Context, Protocol
from messages.basic import UAResponse, UARequest, Error
from uagents.setup import fund_agent_if_low
import os
import requests
from typing import Any, Dict


HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
HUGGING_FACE_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-medium"

HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_ACCESS_TOKEN}"
}

agent = Agent(
    name="whisper_agent",
    seed=HUGGING_FACE_ACCESS_TOKEN,
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)
fund_agent_if_low(agent.wallet.address())


async def get_transcription(ctx: Any, sender: str, audio: str) -> None:
    """
    Get transcription of audio file.

    Args:
        ctx (Any): The context.
        sender (str): The sender.
        audio (str): The audio file.

    Returns:
        None
    """
    def hf_query(filename: str) -> None:
        try:
            with open(filename, "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)        
            if response.status_code != 200:
                await ctx.send(sender, Error(error=f"Error: {response.json().get('error')}"))
                return

            model_res = response.json()[0]
            await ctx.send(sender, UAResponse(response=model_res))
            return   
        except Exception as E:
            await ctx.send(send, Error(error=f"An exception occurred while processing the request: {E}"))
            return
    hf_query(audio)


whisper_agent = Protocol("UARequest")

@whisper_agent.on_message(model=UARequest, replies={UAResponse, Error})
async def handle_request(ctx: Context, sender: str, request: UARequest):
    # Logging the request information
    ctx.logger.info(
        f"Got request from  {sender} for text make transcription: {request.text}")

    # Call text classification function for the incoming request's text
    await get_transcription(ctx, sender, request.text)

# Include protocol to the agent
agent.include(whisper_agent)

# If the script is run as the main program, run our agents event loop
if __name__ == "__main__":
    whisper_agent.run()

import base64
import json
import mimetypes
import os
import warnings
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from PIL import Image
from termcolor import cprint


warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
_ = load_dotenv()

## ------------------------------------------------------ ##
client = LlamaStackClient(base_url = f"https://llama-stack.together.ai")

models = client.models.list()
print(models)

## ------------------------------------------------------ ##
LLAMA_STACK_API_TOGETHER_URL = "https://llama-stack.together.ai"
LLAMA31_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"


def run_main():
    client = LlamaStackClient(
                                base_url = LLAMA_STACK_API_TOGETHER_URL,
                            )

    response = client.inference.chat_completion(
                model_id = LLAMA31_8B_INSTRUCT,
                messages = [
                            {"role": "system", "content": "Who wrote the book Innovator's Dilemma?"
                            "                               How about Charlotte's Web?"},
                            {"role": "user", "content": "which book was published first?"}
                            ],
                x_llama_stack_provider_data = json.dumps({"together_api_key" :
                                                            os.getenv('TOGETHER_API_KEY')})
                )

    print(response.completion_message.content)

run_main()

## ------------------------------------------------------ ##
async def run_main():
    client = LlamaStackClient(
                            base_url = LLAMA_STACK_API_TOGETHER_URL,
                            )

    agent_config = AgentConfig(
                                model  =  LLAMA31_8B_INSTRUCT,
                                instructions = "You are a helpful assistant",
                                enable_session_persistence = False,
                                )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")

    prompts = [
                "Who wrote the book Charlotte's Web?",
                "Three best quotes?",
                ]

    for prompt in prompts:
        print(f"User> {prompt}")
        response = agent.create_turn(
                                    messages = [
                                                {
                                                "role": "user",
                                                "content": prompt,
                                                }
                                                ],
                                    session_id = session_id,
                                    )

        for log in EventLogger().log(response):
            log.print()

await run_main()

## ------------------------------------------------------ ##
def display_image(path):
    img = Image.open(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

display_image("./content/Llama_Repo.jpeg")

## ------------------------------------------------------ ##
LLAMA32_11B_INSTRUCT = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        base64_url = f"data:image/png;base64, {base64_string}"

        return base64_url


async def run_main(image_path, prompt):
    base64_image = encode_image(image_path)

    client = LlamaStackClient(
                            base_url = LLAMA_STACK_API_TOGETHER_URL,
                            )

    agent_config = AgentConfig(
                                model = LLAMA32_11B_INSTRUCT,
                                instructions = "You are a helpful assistant",
                                enable_session_persistence = False,
                                )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")

    response = agent.create_turn(
                                messages = [{
                                            "role": "user",
                                            "content": [
                                                        {
                                                        "type": "image",
                                                        "image": {
                                                                 "url": {
                                                                    "uri": encode_image(image_path)
                                                                    }
                                                                }
                                                        },
                                                        {
                                                        "type": "text",
                                                        "text": prompt,
                                                        }
                                                    ]
                                            }],
                                            session_id = session_id,
                                )

    for log in EventLogger().log(response):
        log.print()

## ------------------------------------------------------ ##
await run_main("./content/Llama_Repo.jpeg",
                 "How many different colors are those llamas? What are those colors?")

## ------------------------------------------------------ ##
async def run_main(image_path: str, prompt):
    client = LlamaStackClient(
                            base_url = LLAMA_STACK_API_TOGETHER_URL,
                            )

    message = {
                "role": "user",
                "content": [
                            {
                            "type": "image",
                            "image": {
                                     "url": {
                                            "uri": encode_image(image_path)
                                            }
                                    }
                            },
                            {
                            "type": "text",
                            "text": prompt,
                            }
                        ]
            }

    cprint("User> Sending image for analysis...", "green")
    response = client.inference.chat_completion(
                                                messages = [message],
                                                model_id = LLAMA32_11B_INSTRUCT,
                                                stream = False,
                                                )

    print(response.completion_message.content.lower().strip())

## ------------------------------------------------------ ##
await run_main("./content/Llama_Repo.jpeg", "How many different colors are those llamas?\
                                            What are those colors?")

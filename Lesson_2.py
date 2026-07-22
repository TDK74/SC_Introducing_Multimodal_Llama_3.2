import base64
import warnings

from utils import disp_image, llama31, llama32, load_env


warnings.filterwarnings('ignore')
load_env()

## ------------------------------------------------------ ##
messages = [{"role" : "user", "content" : "Who wrote the book Charlotte's Web?"}]

## ------------------------------------------------------ ##
response_32 = llama32(messages, 90)
print(response_32)

## ------------------------------------------------------ ##
response_31 = llama31(messages, 70)
print(response_31)

## ------------------------------------------------------ ##
messages = [{"role" : "user", "content" : "Who wrote the book Charlotte's Web?"},
            {"role" : "assistant", "content" : response_32},
            {"role" : "user", "content" : "3 of the best quotes"}]

## ------------------------------------------------------ ##
response_32 = llama32(messages, 90)
print(response_32)

## ------------------------------------------------------ ##
response_31 = llama31(messages, 70)
print(response_31)

## ------------------------------------------------------ ##
disp_image("images/Llama_Repo.jpeg")

## ------------------------------------------------------ ##
image_url = ("https://raw.githubusercontent.com/meta-llama/"
            "llama-models/refs/heads/main/Llama_Repo.jpeg")
messages = [{"role" : "user",
            "content" : [{"type" : "text", "text" : "describe the image in one sentence"},
                        {"type" : "image_url", "image_url" : {"url" : image_url}}]
            }, ]

## ------------------------------------------------------ ##
disp_image(image_url)
result = llama32(messages, 90)
print(result)

## ------------------------------------------------------ ##
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


base64_image = encode_image("images/Llama_Repo.jpeg")

## ------------------------------------------------------ ##
messages = [{"role" : "user",
            "content" : [{"type" : "text", "text" : "describe the image in one sentence"},
                        {"type" : "image_url",
                        "image_url" : {"url" : f"data:image/jpeg;base64, {base64_image}"}}]
            }, ]

## ------------------------------------------------------ ##
disp_image(image_url)
result = llama32(messages, 90)
print(result)

## ------------------------------------------------------ ##
messages = [{"role" : "user",
            "content" : [{"type" : "text", "text" : "describe the image in one sentence"},
                        {"type" : "image_url",
                        "image_url" : {"url" : f"data:image/jpeg;base64, {base64_image}"}}]},
            {"role" : "assistant", "content" : result},
            {"role" : "user", "content" : "how many of them are purple?"}]

## ------------------------------------------------------ ##
result = llama32(messages)
print(result)

## ------------------------------------------------------ ##
def llama32pi(prompt, image_url, model_size = 90):
    messages = [{"role" : "user",
                "content" : [{"type" : "text", "text": prompt},
                            {"type" : "image_url", "image_url" : {"url" : image_url}}]
                }, ]
    result = llama32(messages, model_size)

    return result

## ------------------------------------------------------ ##
print(llama32pi("describe the image in one sentence",
                "https://raw.githubusercontent.com/meta-llama/"
                "llama-models/refs/heads/main/Llama_Repo.jpeg"))

## ------------------------------------------------------ ##
print(llama32pi("describe the image in one sentence", f"data:image/jpeg;base64, {base64_image}"))

## ------------------------------------------------------ ##
disp_image("images/tree.jpg")

## ------------------------------------------------------ ##
question = ("What kind of plant is this in my garden? Describe it in a short paragraph.")

## ------------------------------------------------------ ##
disp_image("images/ww1.jpg")

## ------------------------------------------------------ ##
question = ("What dog breed is this? Describe in one paragraph, and 3-5 short bullet points")
base64_image = encode_image("images/ww1.jpg")
result = llama32pi(question, f"data:image/jpg;base64, {base64_image}")
print(result)

## ------------------------------------------------------ ##
disp_image("images/ww2.png")

## ------------------------------------------------------ ##
base64_image = encode_image("images/ww2.png")
result = llama32pi(question, f"data:image/png;base64, {base64_image}")
print(result)

## ------------------------------------------------------ ##
disp_image("images/tire_pressure.png")

## ------------------------------------------------------ ##
question = ("What's the problem this is about? What should be good numbers?")

## ------------------------------------------------------ ##
base64_image = encode_image("images/tire_pressure.png")
result = llama32pi(question, f"data:image/png;base64, {base64_image}")
print(result)

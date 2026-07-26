import warnings
import gradio as gr

from IPython.display import Markdown, display
from utils import cprint, llama31


warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
question = "Who wrote the book Charlotte's Web?"

prompt = ("<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>")

print(prompt)

## ------------------------------------------------------ ##
response = llama31(prompt, 8)
print(response)

## ------------------------------------------------------ ##
response = llama31(prompt, 8, raw = True)
cprint(response)

## ------------------------------------------------------ ##
follow_up_question = "Three best quotes in it."

follow_up_prompt = ( "<|begin_of_text|>"
                    "<|start_header_id|>user<|end_header_id|>"
                    f"{question}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>"
                    f"{response}"
                    "<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>"
                    f"{follow_up_question}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>")

## ------------------------------------------------------ ##
follow_up_response = llama31(follow_up_prompt)
print(follow_up_response)

## ------------------------------------------------------ ##
display(Markdown(follow_up_response))

## ------------------------------------------------------ ##
question = "Three Best quotes."

prompt = ("<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>")

response = llama31(prompt, 8)
display(Markdown(response))

## ------------------------------------------------------ ##
question = "Three Great quotes."

system_message = ("You are an expert in quotes about sports. You provide just the quotes "
                "and no commentary. Reply in markdown.")

prompt = ("<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        f"{system_message}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>")

response = llama31(prompt, 70)
display(Markdown(response))

## ------------------------------------------------------ ##
follow_up_question = "three more"

follow_up_prompt = ("<|begin_of_text|>"
                    "<|start_header_id|>system<|end_header_id|>"
                    f"{system_message}"
                    "<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>"
                    f"{question}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>"
                    f"{response}"
                    "<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>"
                    f"{follow_up_question}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>")

response = llama31(follow_up_prompt)
display(Markdown(response))

## ------------------------------------------------------ ##
prompt = ("<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>"
        "Who wrote the book Charlotte's Web?"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>")

response = llama31(prompt, 8)
print(response)

## ------------------------------------------------------ ##
messages = [{"role" : "user", "content" : "Who wrote the book Charlotte's Web?"}]

response = llama31(messages, 8)
print(response)

## ------------------------------------------------------ ##
follow_up_question = "Three Best quotes."

messages = [{"role" : "system", "content" : "You are an terse expert in children's literature"},
            {"role" : "user", "content" : "Who wrote the book Charlotte's Web?"},
            {"role" : "assistant", "content" : response},
            {"role" : "user", "content" : follow_up_question}]

response = llama31(messages, 8)
print(response)

## ------------------------------------------------------ ##
class Conversation:
    def __init__(self, system = ""):
        self.messages = []

        if system:
            self.messages.append({"role" : "system", "content" : system})

    def generate(self, user_question, model = 8, temp = 0):
        self.messages.append({"role" : "user", "content" : user_question})
        response = llama31(self.messages, model, temperature = temp)
        self.messages.append({"role" : "assistant", "content" : response})

        return response

## ------------------------------------------------------ ##
system_message = "You are a terse expert in Childrens literature."

conv = Conversation(system_message)
conv.generate("Who wrote the book Charlotte's Web?")

## ------------------------------------------------------ ##
response = conv.generate("What are three quotes?")
display(Markdown(response))

## ------------------------------------------------------ ##
conv.messages

## ------------------------------------------------------ ##
system_message = ("Answer in 8 languages: English, German, French, Italian,"
                    "Portuguese, Hindi, Spanish, and Thai.")

ml_conv = Conversation(system_message)
response = ml_conv.generate("3 basic phrases")
display(Markdown(response))

## ------------------------------------------------------ ##
code_query = """
            I need a Python script for a Gradio chatbot app that should be run
            within a Jupyter notebook.

            1) When calling the LLM, use this class, which is already defined,
                don't redefine it:

            class Conversation:
                def __init__(self, system = ""):
                    self.messages = []

                    if system:
                        self.messages.append({"role" : "system", "content" : system})


                def generate(self, user_question, model  = 8, temp = 0):
                    self.messages.append({"role" : "user", "content" : user_question})
                    response = llama31(self.messages, model, temperature = temp)
                    self.messages.append({"role" : "assistant", "content" : response})

                    return response


            2) initialize the class with a system message of:
                "You are an expert in liturature. You provide brief replies."

            3) the llama() function is defined like this:
                def llama(prompt_or_messages, model_size = 8, temperature  =  0,
                            raw = False, debug = False):

                and returns a reponse in a string. Don't redefine this.
                valid model sizes are 8, 70 and 405.
            """

coder_system_message = ("You are an expert writing python Gradio chatbots.")

coder_conv = Conversation(coder_system_message)
response = coder_conv.generate(code_query, 405)
print(response)

## ------------------------------------------------------ ##
conversation = Conversation("You are an expert in literature. You provide brief replies.")

def generate_response(user_input, model_size, temperature):
    try:
        model_size = int(model_size)

        if model_size not in [8, 70, 405]:
            return "Invalid model size. Please choose from 8, 70, or 405."

        temperature = float(temperature)

        if temperature < 0 or temperature > 1:
            return "Invalid temperature. Please choose a value between 0 and 1."

        response = conversation.generate(user_input, model = model_size, temp = temperature)

        return response

    except Exception as e:
        return str(e)


demo = gr.Interface(fn = generate_response,
                    inputs = [gr.Textbox(label = "User Input"),
                            gr.Radio(label = "Model Size", choices = ["8", "70", "405"]),
                            gr.Slider(label = "Temperature", minimum = 0, maximum = 1,
                                        step = 0.1, value = 0)],
                    outputs = gr.Textbox(label = "Response"),
                    title = "Literature Expert Chatbot",
                    description = ("Ask a question about literature and get a brief response "
                                    "from an expert."))

demo.launch(server_name = "0.0.0.0")

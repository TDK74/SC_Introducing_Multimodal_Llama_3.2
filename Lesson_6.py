import json
import warnings

from datetime import datetime
from sympy import Eq, solve, symbols
from tavily import TavilyClient
from utils import (calculate_loan, cprint, get_boiling_point, get_tavily_api_key,
                    llama31, trending_songs, wolfram_alpha)


warnings.filterwarnings('ignore')

## ------------------------------------------------------ ##
TAVILY_API_KEY = get_tavily_api_key()

## ------------------------------------------------------ ##
current_date = datetime.now()
formatted_date = current_date.strftime("%d %B %Y")
print(formatted_date)

## ------------------------------------------------------ ##
tool_system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    Environment: ipython
                    Tools: brave_search, wolfram_alpha
                    Cutting Knowledge Date: December 2023
                    Today Date: {formatted_date}"""

## ------------------------------------------------------ ##
prompt = tool_system_prompt + f"""<|eot_id|><|start_header_id|>user<|end_header_id|>
                            What is the current weather in Menlo Park, California?
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

## ------------------------------------------------------ ##
no_tool_call_prompt = tool_system_prompt + f"""<|eot_id|><|start_header_id|>user<|end_header_id|>
                                        What is the population of California?
                                        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

## ------------------------------------------------------ ##
no_tool_call_response = llama31(no_tool_call_prompt)
print(no_tool_call_response)

## ------------------------------------------------------ ##
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

result = tavily_client.search("current weather in Menlo Park, California")
cprint(result)

## ------------------------------------------------------ ##
search_result = result["results"][0]["content"]
print(search_result)

## ------------------------------------------------------ ##
prompt = tool_system_prompt + f"""<|eot_id|><|start_header_id|>user<|end_header_id|>
                            What is the current weather in Menlo Park, California?
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            <|python_tag|>{response}<|eom_id|>
                            <|start_header_id|>ipython<|end_header_id|>
                            {search_result}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            """

print(prompt)

## ------------------------------------------------------ ##
response = llama31(prompt)
print(response)

## ------------------------------------------------------ ##
system_prompt_content = f"""Environment: ipython
                        Tools: brave_search, wolfram_alpha
                        Cutting Knowledge Date: December 2023
                        Today Date: {formatted_date}"""

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" :  system_prompt_content},
            {"role" : "user", "content" : "What is the current weather in Menlo Park, California?"}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" :  system_prompt_content},
            {"role" : "user", "content" : "What is the current weather in Menlo Park, California?"},
            {"role" : "assistant", "content" : response},
            {"role" : "ipython", "content" : search_result}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" :  system_prompt_content},
            {"role" : "user", "content" : "What is the current weather in Montana, Bulgaria?"},
            {"role" : "assistant", "content" : response},
            {"role" : "ipython", "content" : search_result}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
math_problem = "Can you help me solve this equation: x^3 - 2x^2 - x + 2 = 0?"

## ------------------------------------------------------ ##
messages = [{"role" : "system",  "content" : system_prompt_content},
            {"role" : "user", "content" : math_problem}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
tool_result = wolfram_alpha("solve x^3 - 2x^2 - x + 2 = 0")
print(tool_result)

## ------------------------------------------------------ ##
x = symbols('x')
equation = Eq(x**3 - 2*x**2 - 1*x + 2, 0)
solution = solve(equation, x)

print(solution)

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" : system_prompt_content},
            {"role" : "user", "content" : math_problem},
            {"role" : "assistant", "content" : response},
            {"role" : "ipython", "content" : tool_result}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
loan_question = ("How much is the monthly payment, total payment, "
                "and total interest paid for a 30 year mortgage of $1M "
                "at a fixed rate of 6% with a 20% down payment?")

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" : system_prompt_content},
            {"role" : "user", "content" : loan_question}, ]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
monthly_payment, total_payment, total_interest_paid = calculate_loan(loan_amount = 1000000,
                                                                    annual_interest_rate = 0.06,
                                                                    loan_term = 30,
                                                                    down_payment = 200000)

print(f"Monthly payment: ${(monthly_payment)}")
print(f"Total payment: ${(total_payment)}")
print(f"Total interest paid: ${(total_interest_paid)}")

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" : system_prompt_content + "\nGenerate the code in Java."},
            {"role" : "user", "content" : loan_question}, ]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
code_interpreter_tool_response = """Monthly payment: $4796
                                    Total payment: $1726705
                                    Total interest paid: $926705 """

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" : system_prompt_content},
            {"role" : "user", "content" : loan_question},
            {"role" : "assistant", "content" : response},
            {"role" : "ipython", "content" : code_interpreter_tool_response}]

## ------------------------------------------------------ ##
response = llama31(messages)
print(response)

## ------------------------------------------------------ ##
country = "US"
top_num = 5
top_songs = trending_songs(country, top_num)
print(f"Top {top_num} trending songs in {country}:")
print(top_songs)

## ------------------------------------------------------ ##
user_prompt = """Answer the user's question by using the following functions if needed.
                If none of the functions can be used, please say so.
                Functions (in JSON format):
                {"type" : "function",
                "function" : {"name" : "get_boiling_point",
                            "description" : "Get the boiling point of a liquid",
                            "parameters" : {"type" : "object",
                                            "properties" :
                                                [{"liquid_name" : {"type" : "object",
                                                            "description" : "name of the liquid"}},
                                                {"celsius" : {"type" : "object",
                                                        "description" : "whether to use celsius"}}],
                                            "required": ["liquid_name"]}}}
                {"type" : "function",
                "function" : {"name" : "trending_songs",
                            "description" : "Returns the trending songs on a Music site",
                            "parameters" : {"type" : "object",
                                            "properties" :
                                                [{"country" : {"type" : "object",
                                                        "description" :
                                                        "country to return trending songs for"}},
                                                {"n" : {"type" : "object",
                                                        "description" :
                                                        "The number of songs to return"}}],
                                            "required" : ["country"]}}}

                Question: Can you check the top 5 trending songs in US? """

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" :  f"""Environment: ipython
                                                Cutting Knowledge Date: December 2023
                                                Today Date: {formatted_date} """},
            {"role" : "user", "content" : user_prompt}]

## ------------------------------------------------------ ##
result = llama31(messages, 405)
print(result)

## ------------------------------------------------------ ##
custom_tools = {"trending_songs" : trending_songs, "get_boiling_point" : get_boiling_point}

## ------------------------------------------------------ ##
res = json.loads(result)
function_name = res['name']
parameters = list(res['parameters'].values())
print(function_name)
print(parameters)

## ------------------------------------------------------ ##
tool_result = custom_tools[function_name](*parameters)
print(tool_result)

## ------------------------------------------------------ ##
messages = [{"role" : "system", "content" :  f"""Environment: ipython
                                                Cutting Knowledge Date: December 2023
                                                Today Date: {formatted_date} """},
            {"role" : "user", "content" : user_prompt},
            {"role" : "assistant", "content" : result},
            {"role" : "ipython", "content" : ','.join(tool_result)}]

## ------------------------------------------------------ ##
response = llama31(messages, 70)
print(response)

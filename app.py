import re
import datetime
import yaml
import streamlit as st

from models import OpenAIModel, AnthropicModel, HuggingFaceLlama2Model
from tools import Tools

SYSTEM_MESSAGE_TEMPLATE = "prompt.txt"

MODELS = {
    "GPT-3.5": OpenAIModel("gpt-3.5-turbo-16k", 16384),
    "GPT-4": OpenAIModel("gpt-4", 8192),
    "Claude 2": AnthropicModel("claude-2", 100000),
    "Llama 2": HuggingFaceLlama2Model("meta-llama/Llama-2-70b-chat-hf", 4096),
}

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

verbose = config["verbose"]
description = config["description"]
examples = config["examples"]
enabled_models = config["enabled_models"]
selected_model = enabled_models[0]
enabled_browsers = config["enabled_browsers"]
selected_browser = enabled_browsers[0]
temperature = config["temperature"]
max_actions = config["max_actions"]

llm_tools = Tools(browser=selected_browser)

def create_system_message():
    """
    Return system message, including today's date and the available tools.
    """
    with open(SYSTEM_MESSAGE_TEMPLATE) as f:
        message = f.read()

    now = datetime.datetime.now()
    current_date = now.strftime("%B %d, %Y")

    message = message.replace("{{CURRENT_DATE}}", current_date)
    message = message.replace("{{TOOLS_PROMPT}}", llm_tools.get_tool_list_for_prompt())

    return message

def generate(new_user_message, history):
    """
    Generate a response from the LLM to the user message while using the 
    available tools.  The history contains a list of prior 
    (user_message, assistant_response) pairs from the chat.
    This function is intended to be called by a Gradio ChatInterface.

    Within this function, we iteratively build up a prompt which includes
    the complete reasoning chain of:

       Question -> [Thought -> Action -> Result?] x N -> Conclusion

    Ideally, we would include the Result for every Action.  For most models,
    we quickly use up the entire context window when including the contents
    of web pages, so we only include the Result for the most recent Action.

    Note that Claude 2 supports a 100k context window, but in practice, I've
    found that the Anthropic API will return a rate limit error if I actually 
    try to send a large number of tokens, so unfortuantely I use the same logic
    with Claude 2 as the other models.
    """
    ACTION_REGEX = r'(\n|^)Action: (.*)\[(.*)\]'
    CONCLUSION_REGEX = r'(\n|^)Conclusion: .*'

    prompt = f"Question: {new_user_message}\n\n"

    # full_response is displayed to the user in the ChatInterface and is the
    # same as the prompt, except it omits the Question and Result to improve
    # readability.
    full_response = ""

    iteration = 1

    model = MODELS[selected_model]
    system_message_token_count = model.count_tokens(system_message)

    try:
        while True:
            if verbose:
                print("======")
                print("PROMPT")
                print("======")
                print(prompt)

            stream = model.generate(
                system_message,
                prompt,
                history=history,
                temperature=temperature
            )

            partial_response = ""

            for chunk in stream:
                completion = model.parse_completion(chunk)

                if completion:
                    # Stream each completion to the ChatInterface
                    full_response += completion
                    partial_response += completion
                    st.write(full_response)

                    matches = re.search(ACTION_REGEX, partial_response)
                    if matches:
                        tool = matches.group(2).strip()
                        params = matches.group(3).strip()
                        
                        result = llm_tools.run_tool(tool, params)

                        prompt = f"Question: {new_user_message}\n\n"
                        prompt += f"{full_response}\n\n"

                        # Calculate the number of tokens available in the
                        # context window, after accounting for the system
                        # message and previous responses
                        history_token_count = 0
                        for user_message, assistant_response in history:
                            history_token_count += model.count_tokens(user_message) + model.count_tokens(assistant_response)

                        prompt_token_count = model.count_tokens(prompt)
                        result_token_count = model.count_tokens(result)

                        available_tokens = int(0.9 * (model.context_size - system_message_token_count - history_token_count - prompt_token_count))

                        # Truncate the result if it is longer than the available tokens
                        if result_token_count > available_tokens:
                            ratio = available_tokens/result_token_count
                            truncate_result_len = int(len(result) * ratio)
                            result = result[:truncate_result_len]

                            full_response += f"\n\n<span style='color:gray'>*Note:  Only {ratio*100:.0f}% of the result was shown to the model due to context window limits.*</span>\n\n"
                            st.write(full_response)

                        prompt += f"Result: {result}\n\n"

                        break

            # Stop when we either see the Conclusion or we cannot find an 
            # Action in the response
            if re.search(CONCLUSION_REGEX, partial_response) or not re.search(ACTION_REGEX, partial_response):
                return
                        
            if not partial_response.endswith("\n"):
                full_response += "\n\n"
                st.write(full_response)
            
            # Stop when we've exceeded max_actions
            if iteration >= max_actions:
                full_response += f"<span style='color:red'>*Stopping after running {max_actions} actions.*</span>"
                st.write(full_response)
                return
            else:
                iteration += 1
    
    except Exception as e:
        full_response += f"\n<span style='color:red'>Error: {e}</span>"
        st.write(full_response)

# Create Streamlit app
system_message = create_system_message()
if verbose:
    print("==============")
    print("SYSTEM MESSAGE")
    print("==============")
    print(system_message)

st.title("LLM Chat Interface")
st.markdown(description)

new_user_message = st.text_input("Enter your message:")
history = []  # You may implement history functionality here

st.markdown("### Response:")
generate(new_user_message, history)

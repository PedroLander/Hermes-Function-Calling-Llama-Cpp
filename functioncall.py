import argparse
import json
from llama_cpp import Llama
import functions
from prompter import PromptManager
from validator import validate_function_call_schema
from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)

class ModelInference:
    def __init__(self, model_path, chat_template, n_threads=4):
        inference_logger.info(print_nous_text_art())
        self.prompter = PromptManager()
        self.model = Llama(model_path=model_path, n_threads=n_threads)
        self.chat_template = get_chat_template(chat_template)

    def process_completion_and_validate(self, completion, chat_template):
        assistant_message = get_assistant_message(completion, chat_template, None)
        if assistant_message:
            validation, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)
            if validation:
                inference_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                return tool_calls, assistant_message, error_message
        else:
            inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")

    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(functions, function_name, None)
        function_args = tool_call.get("arguments", {})
        inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict

    def run_inference(self, prompt):
        # prompt is a list of dicts with 'role' and 'content'
        response = self.model.create_chat_completion(messages=prompt, max_tokens=1500, temperature=0.8)
        completion = response["choices"][0]["message"]["content"]
        return completion

    def generate_function_call(self, query, chat_template, num_fewshot, max_depth=5):
        try:
            depth = 0
            user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
            chat = [{"role": "user", "content": user_message}]
            tools = functions.get_openai_tools()
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            completion = self.run_inference(prompt)
            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                tool_calls, assistant_message, error_message = self.process_completion_and_validate(completion, chat_template)
                prompt.append({"role": "assistant", "content": assistant_message})
                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                if tool_calls:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    for tool_call in tool_calls:
                        validation, message = validate_function_call_schema(tool_call, tools)
                        if validation:
                            try:
                                function_response = self.execute_function_call(tool_call)
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                inference_logger.info(f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}")
                            except Exception as e:
                                inference_logger.info(f"Could not execute function: {e}")
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})
                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return
                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                elif error_message:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})
                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return
                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
            recursive_loop(prompt, completion, depth)
        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recursive function calling loop")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to use json mode examples")
    parser.add_argument("--query", type=str, default="I need the current stock price of Tesla (TSLA)")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of CPU threads for llama-cpp")
    args = parser.parse_args()
    inference = ModelInference(args.model_path, args.chat_template, args.n_threads)
    inference.generate_function_call(args.query, args.chat_template, args.num_fewshot, args.max_depth)

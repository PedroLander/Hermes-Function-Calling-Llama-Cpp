import argparse
import json
from llama_cpp import Llama
from validator import validate_json_data
from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)
from typing import List, Optional
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    species: str
    role: str
    personality_traits: Optional[List[str]]
    special_attacks: Optional[List[str]]
    class Config:
        schema_extra = {"additionalProperties": False}

pydantic_schema = Character.schema_json()

class ModelInference:
    def __init__(self, model_path, chat_template, n_threads=4):
        inference_logger.info(print_nous_text_art())
        self.model = Llama(model_path=model_path, n_threads=n_threads)
        self.chat_template = get_chat_template(chat_template)

    def run_inference(self, prompt):
        # prompt is a list of dicts with 'role' and 'content'
        response = self.model.create_chat_completion(messages=prompt, max_tokens=1500, temperature=0.8)
        completion = response["choices"][0]["message"]["content"]
        return completion

    def generate_json_completion(self, query, chat_template, max_depth=5):
        try:
            depth = 0
            sys_prompt = f"You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{pydantic_schema}\n</schema>"
            prompt = [{"role": "system", "content": sys_prompt}]
            prompt.append({"role": "user", "content": query})
            inference_logger.info(f"Running inference to generate json object for pydantic schema:\n{json.dumps(json.loads(pydantic_schema), indent=2)}")
            completion = self.run_inference(prompt)
            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                assistant_message = get_assistant_message(completion, chat_template, None)
                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                if assistant_message is not None:
                    validation, json_object, error_message = validate_json_data(assistant_message, json.loads(pydantic_schema))
                    if validation:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation passed")
                        inference_logger.info(f"parsed json object:\n{json.dumps(json_object, indent=2)}")
                    elif error_message:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation failed")
                        tool_message += f"<tool_response>\nJson schema validation failed\nHere's the error stacktrace: {error_message}\nPlease return corrrect json object\n<tool_response>"
                        depth += 1
                        if depth >= max_depth:
                            print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                            return
                        prompt.append({"role": "tool", "content": tool_message})
                        completion = self.run_inference(prompt)
                        recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.warning("Assistant message is None")
            recursive_loop(prompt, completion, depth)
        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run json mode completion")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--query", type=str, default="Please return a json object to represent Goku from the anime Dragon Ball Z?")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of CPU threads for llama-cpp")
    args = parser.parse_args()
    inference = ModelInference(args.model_path, args.chat_template, args.n_threads)
    inference.generate_json_completion(args.query, args.chat_template, args.max_depth)

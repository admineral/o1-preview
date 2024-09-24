from openai import OpenAI
import os
import time
import json

# Use environment variable if available, otherwise use hardcoded API key
# To set the API key in your environment, run:
# export OPENAI_API_KEY="your-api-key-here"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client after setting the API key
client = OpenAI()

def test_o1_model(prompt, tools):
    try:
        start_time = time.time()
        
        tool_descriptions = json.dumps(tools, indent=2)
        structured_prompt = f"""
        You have access to the following tool:

        {tool_descriptions}

        To use the tool, your response must be in the following JSON format:
        {{
            "thought": "your reasoning for the code you're about to write",
            "tool": "code_interpreter",
            "tool_input": {{
                "language": "the programming language",
                "code": "the actual code"
            }}
        }}

        Now, please respond to this prompt: {prompt}
        """
        
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {"role": "user", "content": structured_prompt}
            ]
        )
        
        end_time = time.time()
        
        # Extract the response content and parse JSON
        raw_answer = response.choices[0].message.content
        try:
            # First, try to parse the raw answer directly
            parsed_answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from a code block
            import re
            json_match = re.search(r'```json\n(.*?)\n```', raw_answer, re.DOTALL)
            if json_match:
                try:
                    parsed_answer = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    print("Failed to parse JSON from code block. Raw response:")
                    print(raw_answer)
                    parsed_answer = {"thought": "", "response": raw_answer}
            else:
                print("Failed to parse JSON. Raw response:")
                print(raw_answer)
                parsed_answer = {"thought": "", "response": raw_answer}
        
        # Extract usage information
        total_tokens = response.usage.total_tokens
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        
        # Calculate visible completion tokens
        visible_completion_tokens = completion_tokens - reasoning_tokens
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        
        return {
            "raw_response": raw_answer,
            "parsed_answer": parsed_answer,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_completion_tokens": visible_completion_tokens,
            "elapsed_time": elapsed_time
        }
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define your code interpreter tool
tools = [
    {
        "name": "code_interpreter",
        "description": "Executes code in various programming languages",
        "parameters": {
            "language": "The programming language to use",
            "code": "The code to execute"
        }
    }
]

# Test the model with a prompt to write a Hello World program
sample_prompt = "Use the code interpreter to write a 'Hello, World!' program in Python."

result = test_o1_model(sample_prompt, tools)

if result:
    print("\n" + "="*50)
    print("RAW RESPONSE FROM MODEL:")
    print("="*50)
    print(result['raw_response'])
    print("\n" + "="*50)
    print("PARSED RESPONSE:")
    print("="*50)
    
    parsed_answer = result['parsed_answer']
    if 'tool' in parsed_answer and parsed_answer['tool'] == 'code_interpreter':
        print(f"Thought: {parsed_answer['thought']}")
        print(f"Tool: {parsed_answer['tool']}")
        print(f"Language: {parsed_answer['tool_input']['language']}")
        print("Code:")
        print("-"*20)
        print(parsed_answer['tool_input']['code'])
        print("-"*20)
        
        print("\nCode execution result:")
        print("Hello, World!")
    else:
        print(f"Unexpected response format: {parsed_answer}")
    
    print("\n" + "="*50)
    print("STATISTICS:")
    print("="*50)
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Reasoning tokens: {result['reasoning_tokens']}")
    print(f"Visible completion tokens: {result['visible_completion_tokens']}")
    print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
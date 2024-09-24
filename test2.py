from openai import OpenAI
import os
import time
import json
import base64
import re

# Use environment variable if available, otherwise use hardcoded API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client
client = OpenAI()

def parse_ai_response(raw_response):
    try:
        # First, try to parse the entire response as JSON
        return json.loads(raw_response)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from a code block
        json_match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing fails, create a structured response
        return {
            "thought": "Unable to parse JSON. Using raw response.",
            "plan": [
                {
                    "step": 1,
                    "description": "Execute the following code",
                    "code": raw_response
                }
            ]
        }

def test_o1_model(prompt, feedback=None):
    try:
        start_time = time.time()
        
        system_content = """You are an AI assistant tasked with instructing a smaller AI model (4o) that has access to a Python code interpreter. Your job is to provide a plan for analyzing two CSV files: 'Phase 0 - Sales.csv' and 'Phase 0 - Price.csv'.

        The structure of the CSV files is as follows:

        1. Phase 0 - Sales.csv:
           - Columns: Client, Warehouse, Product, and multiple date columns (e.g., 2020-07-06, 2020-07-13, etc.)
           - Date columns contain the number of units sold for each date.

        2. Phase 0 - Price.csv:
           - Columns: Client, Warehouse, Product, and multiple date columns (e.g., 2020-07-06, 2020-07-13, etc.)
           - Date columns contain the price of products for each date.

        Your task is to create a plan for analyzing these datasets, identifying trends in sales and pricing, and providing predictions or insights through visualizations. The 4o model will execute the code to perform the analysis and create the visualizations.

        Provide your analysis plan in the following format:

        {
            "thought": "Your reasoning for the analysis plan",
            "plan": [
                {
                    "step": 1,
                    "description": "Description of the step",
                    "code": "Python code for this step"
                },
                {
                    "step": 2,
                    "description": "Description of the step",
                    "code": "Python code for this step"
                },
                ...
            ]
        }

        Ensure that your plan includes steps for loading the data, performing necessary data transformations, conducting analysis, and creating visualizations. The 4o model will execute each step of the plan using its code interpreter."""

        user_content = f"{system_content}\n\nNow, please provide a plan for the following task:\n{prompt}"
        
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        if feedback:
            messages.append({"role": "assistant", "content": "Here's my previous plan:"})
            messages.append({"role": "assistant", "content": json.dumps(feedback['previous_plan'], indent=2)})
            messages.append({"role": "user", "content": f"The 4o model executed the plan and provided the following feedback: {feedback['execution_result']}. Please refine the plan based on this feedback."})
        
        response = client.chat.completions.create(
            model="o1-mini",
            messages=messages
        )
        
        end_time = time.time()
        
        raw_answer = response.choices[0].message.content
        parsed_answer = parse_ai_response(raw_answer)
        
        return {
            "raw_response": raw_answer,
            "parsed_answer": parsed_answer,
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "elapsed_time": end_time - start_time
        }
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def execute_4o_code_interpreter(thread_id, instructions, code):
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"Instructions: {instructions}\n\nCode to execute:\n```python\n{code}\n```"
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id="asst_nWVbMO6Iw0PLqUc60GzhPm1U"
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content

    except Exception as e:
        return f"An error occurred while executing the code: {str(e)}"

def save_image(image_data, filename):
    os.makedirs("generated_images", exist_ok=True)
    file_path = os.path.join("generated_images", filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    print(f"Image saved: {file_path}")

# Test the model with a prompt to analyze the CSV files
sample_prompt = "Analyze the sales and price trends for the top 5 products by total sales volume. Create line charts to visualize the trends over time."

num_iterations = 2  # Make this configurable
feedback = None

for iteration in range(num_iterations):
    print(f"\n{'='*50}\nIteration {iteration + 1}\n{'='*50}")
    
    result = test_o1_model(sample_prompt, feedback)

    if result:
        print("\nRAW RESPONSE FROM O1 MODEL:")
        print("="*50)
        print(result['raw_response'])
        print("\nPARSED RESPONSE:")
        print("="*50)
        
        parsed_answer = result['parsed_answer']
        print(f"Thought: {parsed_answer.get('thought', 'No thought provided')}")
        
        print("\nAnalysis Plan:")
        execution_results = []
        
        # Create a single thread for all steps
        thread = client.beta.threads.create()
        
        for step in parsed_answer.get('plan', []):
            print(f"\nStep {step.get('step', 'N/A')}: {step.get('description', 'No description')}")
            print("Code:")
            print("-"*20)
            print(step.get('code', 'No code provided'))
            print("-"*20)
            
            if step.get('code'):
                print("\nExecuting code with 4o model:")
                execution_result = execute_4o_code_interpreter(thread.id, step['description'], step['code'])
                
                for content in execution_result:
                    if content.type == 'text':
                        print(content.text.value)
                    elif content.type == 'image_file':
                        try:
                            image_file = client.files.content(content.image_file.file_id)
                            image_data = image_file.read()
                            save_image(image_data, f"image_{content.image_file.file_id}.png")
                        except Exception as e:
                            print(f"Error saving image: {str(e)}")
                
                execution_results.append(str(execution_result))
        
        # Prepare feedback for the next iteration
        feedback = {
            "previous_plan": parsed_answer,
            "execution_result": "\n".join(execution_results)
        }
        
        print("\n" + "="*50)
        print("STATISTICS:")
        print("="*50)
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Prompt tokens: {result['prompt_tokens']}")
        print(f"Completion tokens: {result['completion_tokens']}")
        print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")

if __name__ == "__main__":
    # Run the test
    pass
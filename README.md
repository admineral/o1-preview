# AI-Driven Sales and Price Analysis Tool

## Concept

This tool demonstrates an AI-powered approach to data analysis. It uses two AI models in tandem:

1. A large language model (o1-preview) to generate a detailed analysis plan.
2. A smaller model with code execution capabilities (4o) to carry out the plan.

The script automates the entire process of analyzing sales and pricing data:

1. It prompts the large model to create a step-by-step analysis plan.
2. It then feeds each step of this plan to the smaller model, which executes the actual data analysis code.
3. The results are collected, visualizations are generated, and the process can iterate, with the large model refining the plan based on previous results.

This approach combines the strategic planning capabilities of advanced language models with the practical execution abilities of code-running models, creating a flexible and powerful tool for data analysis.
# Configuration Guide for Simple Agent Team (AI-Driven Agent Collaboration)

## Overview
This Python script utilizes the OpenRouter API to orchestrate teams of AI-driven agents with distinct expertise for collaborative problem-solving. A ModeratorAgent summarizes discussions, and a JudgeAgent formulates the final solution. The script supports agent swapping for diverse perspectives, saves outputs to a CSV file, and allows post-evaluation queries to the JudgeAgent. It is adaptable for various problem domains. Written with the help of AI.

## Configuration
To configure the script, an OpenRouter API key is required. Follow these steps:

1. **Create a `.env` File**:
   - In the project root directory, create a file named `.env`.

2. **Add the OpenRouter API Key**:
   - Obtain an API key from OpenRouter (visit [https://openrouter.ai](https://openrouter.ai) for details).
   - Add the following line to the `.env` file, replacing `your_api_key_here` with your actual API key:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```

3. **Environment Setup**:
   - Ensure the `python-dotenv` package is installed to load environment variables:
     ```bash
     pip install python-dotenv
     ```
   - The script will automatically load the API key from the `.env` file during execution.

## Notes
- Keep the `.env` file secure and do not share it publicly, as it contains sensitive API credentials.
- The script assumes the `.env` file is located in the same directory as the script.
- When running the script, simply add the description of the problem to be solved after the name of the script (or modify the default problem in the script)

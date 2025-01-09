OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
VERSA_MODELS = ["versa-gpt-4o-2024-08-06", "versa-gpt-4o-mini-2024-07-18"]

BEDROCK_MAPPINGS = {
        "cohere-command-r": "cohere.command-r-v1:0",
        "cohere-command": "cohere.command-text-v14",
        "cohere-command-light": "cohere.command-light-text-v14",
        "claude-haiku-3": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-haiku-3-5": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "meta-llama/Meta-Llama-3.2-90B-Instruct": "us.meta.llama3-2-90b-instruct-v1:0"
        }

BEDROCK_MODELS = list(BEDROCK_MAPPINGS.keys())
VERSA_ENDPOINT = "https://unified-api.ucsf.edu/general/openai/deployments/<model_name>/chat/completions?api-version=2024-10-21"
VERSA_API_VERSION = "2024-10-21"
AWS_REGION='us-west-2'

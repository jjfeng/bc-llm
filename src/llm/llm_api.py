from huggingface_hub import InferenceClient, AsyncInferenceClient
from openai import AsyncOpenAI, OpenAI, AzureOpenAI, AsyncAzureOpenAI
import transformers
from dotenv import load_dotenv
from tqdm import tqdm
import os
import aioboto3
import asyncio
import base64
import json
import boto3

from torch.utils.data import Dataset

from src.llm.llm import LLM
from torch.utils.data import DataLoader
import time


"""
Please set your HF and OpenAI tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
VERSA_MODELS = ["versa-gpt-4o-2024-05-13"]
BEDROCK_MODELS = ["claude-haiku"]
CLAUDE_HAIKU_MODEL_ID="anthropic.claude-3-5-haiku-20241022-v1:0"

class LLMApi(LLM):
    def __init__(self,
                 seed: int,
                 model_type:str,
                 logging,
                 timeout=60 # seconds
                 ):
        super().__init__(seed, model_type, logging)

        load_dotenv()
        self.api_model_name = model_type.replace("versa-", "")
        self.is_openai = model_type in OPENAI_MODELS + VERSA_MODELS
        self.is_bedrock = model_type in BEDROCK_MODELS 
        self.timeout = timeout
        self.is_api = True

    def get_client(self):
        if self.model_type in OPENAI_MODELS:
            self.access_token = os.getenv("OPENAI_ACCESS_TOKEN")
            return OpenAI(api_key=self.access_token, timeout=self.timeout)
        elif self.model_type in VERSA_MODELS:
            self.api_key = os.environ.get('VERSA_API_KEY')
            self.api_version = os.environ.get('VERSA_API_VERSION')
            self.resource_endpoint = os.environ.get('VERSA_RESOURCE_ENDPOINT')
            return AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.resource_endpoint,
            )
        elif self.model_type in BEDROCK_MODELS:
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_TOKEN")
            return boto3.client(
                'bedrock',
                region_name='us-east-1',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key
            )
        else:
            self.access_token = os.getenv('HF_ACCESS_TOKEN')
            return InferenceClient(
                model=self.model_type, 
                token=self.access_token,
                timeout=self.timeout
                )

    def get_async_client(self):
        if self.model_type in OPENAI_MODELS:
            # default max_retries = 2
            self.access_token = os.getenv("OPENAI_ACCESS_TOKEN")
            return AsyncOpenAI(api_key=self.access_token, timeout=self.timeout)
        elif self.model_type in VERSA_MODELS:
            self.api_key = os.environ.get('VERSA_API_KEY')
            self.api_version = os.environ.get('VERSA_API_VERSION')
            self.resource_endpoint = os.environ.get('VERSA_RESOURCE_ENDPOINT')
            return AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.resource_endpoint,
            )
        elif self.model_type in BEDROCK_MODELS:
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_TOKEN")
            return aioboto3.client(
                'bedrock',
                region_name='us-east-1',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key
            )
        else:
            self.access_token = os.getenv('HF_ACCESS_TOKEN')
            return AsyncInferenceClient(
                    model=self.model_type, 
                    token=self.access_token, 
                    timeout=self.timeout
                    )
    
    # Note: if passing in an image here the prompt should contain the base64 encoded image.
    # see _encode_images for an example
    def get_output(self, prompt, max_new_tokens=5000, is_image=False) -> str:
        self.logging.info("LLM (%s) prompt %s", self.model_type, prompt)
        client = self.get_client()
        if self.is_openai:
            if is_image:
                llm_output = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                            ],
                        model=self.api_model_name,
                        max_tokens=max_new_tokens,
                        seed=self.seed
                        )
            else:
                llm_output = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_new_tokens,
                        seed=self.seed,
                        model=self.api_model_name
                        )
            llm_response = llm_output.choices[0].message.content
        # TODO: change args to take in optional content image and input text for prompt
        elif self.is_bedrock and "claude" in self.model_type and is_image:
            payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "seed": self.seed,
                    {"role": "user",
                     "content": [
                         {"type": "image", 
                          "source": {
                              "type": "base64",
                              "media_type": "image/jpeg", 
                              "data": content_image
                              }
                          },
                         {"type": "text", "text": input_text}
                         ]}
                     }
            response = client.invoke_model(
                    modelId=CLAUDE_HAIKU_MODEL_ID,
                    body=json.dumps(payload)
                    )
            result = json.loads(response["body"].read())
            llm_response = "".join([output["text"] for output in result["content"]])
        elif self.is_bedrock and "claude" in self.model_type:
            payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "seed": self.seed,
                    "system": "You are a helpful assistant",
                    "messages": [
                        {"role": "user", "content": [{ "type": "text", "text": prompt }]}
                        ]
                    }
            response = client.invoke_model(
                    modelId=CLAUDE_HAIKU_MODEL_ID,
                    body=json.dumps(payload)
                    )
            result = json.loads(response["body"].read())
            llm_response = "".join([output["text"] for output in result["content"]])
        else:
            llm_output = client.chat_completion(
                    [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_new_tokens,
                    seed=self.seed
                    )
            llm_response = llm_output.choices[0].message.content
        self.logging.info("LLM response %s", llm_response)
        return llm_response

    async def get_outputs(
            self, 
            dataset: Dataset, 
            batch_size:int = 10, 
            max_new_tokens=5000,
            is_image = False,
            max_retries = 2,
            temperature:float = 1,
            callback = None,
            validation_func = None
            ) -> list[str]:
        self.logging.info(f"async LLM: {self.model_type}")
        client = self.get_async_client()
        if is_image:
            # the collate_fn is used here because passing in the full payload with the base encoded image causing
            # dataloader to break. The image is encoded into base64 after the batch is created
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self._encode_images)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)
        print("DATASET LEN", len(dataset))

        start_time = time.time()
        results = []
        for i, batch_prompts in enumerate(tqdm(dataloader)):
            got_result = False
            num_retries = 0
            while not got_result and (num_retries < max_retries):
                try: 
                    batch_results = []
                    for prompt in batch_prompts:
                        if is_image:
                            batch_results.append(self._run_async_image(client, prompt, max_new_tokens, temperature=temperature))
                        else:
                            batch_results.append(self._run_async_inference(client, prompt, max_new_tokens, temperature=temperature))
                    batch_results = await asyncio.gather(*batch_results)
                    batch_results = [llm_output.choices[0].message.content for llm_output in batch_results]
                    assert len(batch_results) == len(batch_prompts)
                    if validation_func is not None:
                        validation_func(batch_results)
                    self.logging.info(batch_results)
                    results += batch_results

                    if callback is not None:
                        callback(results)
                    got_result = True
                except Exception as e:
                    num_retries += 1
                    message = f"Failed batch idx {i}. Error {e}, {num_retries}"
                    print(message)
                    self.logging.error(message)
                    if num_retries == max_retries:
                        raise ValueError("Error with LLM batch query")
        end_time = time.time()
        execution_time = end_time - start_time
        message = f"Results took {execution_time} seconds"
        self.logging.info("NUM RESULTS %d", len(results))
        assert len(results) == len(dataset)
        return results

    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for (payload, image_path) in batch_data:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            updated_payloads.append(payload)
        return updated_payloads

    async def _run_async_image(self, client, prompt, max_new_tokens, temperature: float = 1):
        if self.is_openai:
            llm_output = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                        ],
                    model=self.api_model_name,
                    max_tokens=max_new_tokens,
                    seed=self.seed,
                    temperature=temperature,
                    )
        elif self.is_bedrock and "claude" in self.model_type and is_image:
            payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "seed": self.seed,
                    "temperature": temperature,
                    {"role": "user",
                     "content": [
                         {"type": "image", 
                          "source": {
                              "type": "base64",
                              "media_type": "image/jpeg", 
                              "data": content_image
                              }
                          },
                         {"type": "text", "text": input_text}
                         ]}
                     }
            async with client as async_client:
                llm_output = async_client.invoke_model(
                        modelId=CLAUDE_HAIKU_MODEL_ID,
                        body=json.dumps(payload)
                        )
        else:
            raise Exception("Images are only supported for OpenAI and Claude models")

        return await llm_output

    async def _run_async_inference(self, client, prompt, max_new_tokens, temperature:float =1):
        if self.is_openai:
            llm_output = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                        ],
                    model=self.api_model_name,
                    max_tokens=max_new_tokens,
                    seed=self.seed,
                    temperature=temperature,
                    )
        elif self.is_bedrock and "claude" in self.model_type:
            payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "seed": self.seed,
                    "temperature": temperature,
                    "system": "You are a helpful assistant",
                    "messages": [
                        {"role": "user", "content": [{ "type": "text", "text": prompt }]}
                        ]
                    }
            async with client as async_client:
                llm_output = async_client.invoke_model(
                        modelId=CLAUDE_HAIKU_MODEL_ID,
                        body=json.dumps(payload)
                        )
        else:
            llm_output = client.chat_completion(
                    [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                        ],
                    max_tokens=max_new_tokens,
                    seed=self.seed
                    )
        return await llm_output

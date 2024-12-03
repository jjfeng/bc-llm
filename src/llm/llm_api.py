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
from src import utils
from torch.utils.data import DataLoader
import time


"""
Please set your HF and OpenAI tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
VERSA_MODELS = ["versa-gpt-4o-2024-05-13"]
# NOTE: images are only support for sonnet
BEDROCK_MODELS = ["cohere-command-r", "cohere-command", "cohere-command-light", "claude-haiku-3", "claude-haiku-3.5", "claude-sonnet"]
CLAUDE_HAIKU_V3_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_HAIKU_V3_5_MODEL_ID="anthropic.claude-3-5-haiku-20241022-v1:0"
CLAUDE_SONNET_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
COHERE_COMMAND_LIGHT_MODEL_ID="cohere.command-light-text-v14"
COHERE_COMMAND_MODEL_ID="cohere.command-text-v14"
COHERE_COMMAND_R_MODEL_ID="cohere.command-r-v1:0"

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
        elif "command" in self.model_type:
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_TOKEN")
            return boto3.client(
                'bedrock-runtime',
                region_name='us-west-2',
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
        elif "command-r" in self.model_type:
            payload = {
                    "message": prompt,
                    "max_tokens": max_new_tokens,
                    }
            llm_output = client.invoke_model(
                    modelId=COHERE_COMMAND_R_MODEL_ID,
                    body=json.dumps(payload)
                    )
            response = llm_output["body"].read()
            llm_response = json.loads(response)["text"]
        elif "command" in self.model_type:
            payload = {
                    "prompt": prompt,
                    "max_tokens": max_new_tokens,
                    }
            llm_output = client.invoke_model(
                    modelId=COHERE_COMMAND_MODEL_ID,
                    body=json.dumps(payload)
                    )
            response = llm_output["body"].read()
            llm_response = json.loads(response)["generations"][0]["text"]
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
        if self.is_bedrock:
            session = aioboto3.Session()
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_TOKEN")
            async with session.client(
                'bedrock-runtime',
                region_name='us-west-2',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key
                ) as client:
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
                            responses = await asyncio.gather(*[result["body"].read() for result in batch_results])
                            #Cohere command-r models
                            batch_results = [json.loads(response)["text"] for response in responses]

                            # Cohere command models
                            # batch_results = [json.loads(response)["generations"][0]["text"] for response in responses]
                            # Claude parsing
                            # batch_results = [json.loads(response)["content"][0]["text"] for response in responses]
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
                                print(e)
                                raise ValueError("Error with LLM batch query")
                end_time = time.time()
                execution_time = end_time - start_time
                message = f"Results took {execution_time} seconds"
                self.logging.info("NUM RESULTS %d", len(results))
                assert len(results) == len(dataset)
                return results
        else:
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
                            print(e)
                            raise ValueError("Error with LLM batch query")
            end_time = time.time()
            execution_time = end_time - start_time
            message = f"Results took {execution_time} seconds"
            self.logging.info("NUM RESULTS %d", len(results))
            assert len(results) == len(dataset)
            return results

    # NOTE: images are only available for open ai and claude models
    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for (payload, image_path) in batch_data:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            if self.is_bedrock:
                payload.append(
                        {"type": "image", 
                         "source": {
                             "type": "base64",
                             "media_type": "image/jpeg", 
                             "data": base64_image
                             }
                         }
                        )
            elif self.is_openai:
                payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            else:
                raise Exception("Images are only available for open ai and claude models")
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
        elif self.is_bedrock:
            payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "system": "You are a helpful assistant",
                    "messages": [{ "role": "user", "content": prompt }]
                    }
                     
            if "haiku-3" in self.model_type:
                llm_output = client.invoke_model(
                        modelId=CLAUDE_HAIKU_V3_MODEL_ID,
                        body=json.dumps(payload)
                        )
            else:
                llm_output = client.invoke_model(
                        modelId=CLAUDE_SONNET_MODEL_ID,
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
        elif self.is_bedrock:
            if "haiku-3" in self.model_type:
                payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                        "system": "You are a helpful assistant",
                        "messages": [
                            {"role": "user", "content": [{ "type": "text", "text": prompt }]}
                            ]
                        }
                llm_output = client.invoke_model(
                        modelId=CLAUDE_HAIKU_V3_MODEL_ID,
                        body=json.dumps(payload)
                        )
            elif "haiku-3.5" in self.model_type:
                payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                        "system": "You are a helpful assistant",
                        "messages": [
                            {"role": "user", "content": [{ "type": "text", "text": prompt }]}
                            ]
                        }
                llm_output = client.invoke_model(
                        modelId=CLAUDE_HAIKU_V3_5_MODEL_ID,
                        body=json.dumps(payload)
                        )
            elif "sonnet" in self.model_type:
                payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                        "system": "You are a helpful assistant",
                        "messages": [
                            {"role": "user", "content": [{ "type": "text", "text": prompt }]}
                            ]
                        }
                llm_output = client.invoke_model(
                        modelId=CLAUDE_SONNET_MODEL_ID,
                        body=json.dumps(payload)
                        )
            elif "command-r" in self.model_type:
                payload = {
                        "message": prompt,
                        "max_tokens": max_new_tokens,
                        }
                llm_output = client.invoke_model(
                        modelId=COHERE_COMMAND_R_MODEL_ID,
                        body=json.dumps(payload)
                        )
            elif "command-light" in self.model_type:
                payload = {
                        "prompt": prompt,
                        "max_tokens": max_new_tokens,
                        }
                llm_output = client.invoke_model(
                        modelId=COHERE_COMMAND_LIGHT_MODEL_ID,
                        body=json.dumps(payload)
                        )
            elif "command" in self.model_type:
                payload = {
                        "prompt": prompt,
                        "max_tokens": max_new_tokens,
                        }
                llm_output = client.invoke_model(
                        modelId=COHERE_COMMAND_MODEL_ID,
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

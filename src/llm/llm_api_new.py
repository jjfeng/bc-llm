from src.llm.llm import LLM
from langchain_openai import ChatOpenAI
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from torch.utils.data import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import os
import asyncio
import base64
import time

from src.llm.llm import LLM
from src.llm.constants import *
from torch.utils.data import DataLoader

"""
Please set your HF, OpenAI, and Versa tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""

class LLMApiNew(LLM):
    def __init__(self,
                 seed: int,
                 model_type: str,
                 logging,
                 timeout=60
                 ):
        super().__init__(seed, model_type, logging)

        load_dotenv()
        self.api_model_name = model_type.replace("versa-", "")
        self.timeout = timeout
        self.is_api = True

    def get_client(self, max_new_tokens=5000, temperature=0):
        if self.model_type in OPENAI_MODELS:
            access_token = os.getenv("OPENAI_ACCESS_TOKEN")
            return ChatOpenAI(
                    api_key=access_token, 
                    model_name=self.model_type,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=self.seed,
                    timeout=self.timeout
                    )
        elif self.model_type in VERSA_MODELS:
            raise Exception
        elif self.model_type in BEDROCK_MODELS:
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_TOKEN")
            return ChatBedrockConverse(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_access_key,
                    region_name=AWS_REGION,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    model_id=self._get_model_id()
                    )



    # Note: if passing in an image here the prompt should contain the base64 encoded image.
    # see _encode_images for an example
    def get_output(self, prompt, max_new_tokens=5000, is_image=False) -> str:
        self.logging.info("LLM (%s) prompt %s", self.model_type, prompt)
        llm = self.get_client(max_new_tokens, temperature)
        if is_image:
            raise Exception
        else:
            response = llm.invoke(prompt)

        self.logging.info("LLM response %s", response)
        return response

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
            ):
        llm = self.get_client(max_new_tokens, temperature)
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
                    batch_results = await llm.abatch(batch_prompts)
                    batch_results = [response.content for response in batch_results]
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
        self.logging.info("Results took {execution_time} seconds")
        self.logging.info("NUM RESULTS %d", len(results))
        assert len(results) == len(dataset)
        return results

    def _get_model_id(self):
        model_id = None
        if self.model_type == "cohere-command-r":
            model_id = COHERE_COMMAND_R_MODEL_ID
        elif self.model_type == "cohere-command":
            model_id = COHERE_COMMAND_MODEL_ID
        elif self.model_type == "cohere-command-light":
            model_id = COHERE_COMMAND_LIGHT_MODEL_ID
        elif self.model_type == "claude-haiku-3":
            model_id = CLAUDE_HAIKU_3_MODEL_ID
        elif self.model_type == "claude-sonnet":
            model_id = CLAUDE_SONNET_MODEL_ID
        else:
            raise Exception("Invalid model type")
        return model_id

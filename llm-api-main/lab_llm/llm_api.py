from typing import Optional, List

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from torch.utils.data import Dataset, DataLoader
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.globals import set_debug

from pydantic_core import from_json
from tqdm import tqdm
import os
import asyncio
import base64
import time
import pandas as pd
import numpy as np

from pydantic import BaseModel, ValidationError

from lab_llm.llm import LLM
from lab_llm.llm_cache import LLMCache
import lab_llm.constants as constants
from lab_llm.error_callback_handler import ErrorCallbackHandler

"""
Please set your HF, OpenAI, and Versa tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""

BLOCK = False

def is_valid(model: BaseModel, data_str: str) -> bool:
    """
    Validates data against a Pydantic model and returns True if valid, False otherwise.
    """
    try:
        model.model_validate_json(data_str)
        return True
    except ValidationError:
        return False

class LLMApi(LLM):
    def __init__(self,
                 cache: LLMCache,
                 seed: int,
                 model_type: constants.LLMModel,
                 error_handler: ErrorCallbackHandler,
                 logging,
                 timeout=60
                 ):
        super().__init__(seed, model_type, logging)
        self.cache = cache
        self.timeout = timeout
        self.is_api = True
        self.error_handler = error_handler
    
    def _serialize_llm_response(self, llm_response, response_model: BaseModel=None):
        if response_model is not None:
            return llm_response.model_dump_json()
        else:
            return llm_response.content

    def get_client(self, max_new_tokens=4000, temperature=0, requests_per_second=None):
        if requests_per_second:
            rate_limiter = InMemoryRateLimiter(
                    requests_per_second=requests_per_second,
                    check_every_n_seconds=0.1,
                    max_bucket_size=10
                    )
        else:
            rate_limiter = None
        print("self.model_type", self.model_type.name)
        if constants.is_openai(self.model_type.name):
            access_token = os.getenv("OPENAI_ACCESS_TOKEN")
            return ChatOpenAI(
                    api_key=access_token, 
                    model_name=self.model_type.name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=self.seed,
                    timeout=self.timeout,
                    rate_limiter=rate_limiter,
                    callbacks=[self.error_handler]
                    )
        elif constants.is_anthropic(self.model_type.name):
            access_token = os.getenv("ANTHROPIC_ACCESS_KEY")
            return ChatAnthropic(
                    api_key=access_token, 
                    model_name=self.model_type.name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    max_retries=0,
                    timeout=self.timeout,
                    rate_limiter=rate_limiter,
                    callbacks=[self.error_handler]
                    )
        elif constants.is_versa(self.model_type.name):
            api_key = os.environ.get('VERSA_API_KEY')
            resource_endpoint = constants.VERSA_ENDPOINT.replace("<model_name>", self.model_type.name)
            return AzureChatOpenAI(
                    api_key=api_key,
                    api_version=constants.VERSA_API_VERSION,
                    azure_endpoint=resource_endpoint,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                    seed=self.seed,
                    rate_limiter=rate_limiter,
                    callbacks=[self.error_handler]
                    )
        elif constants.is_bedrock(self.model_type.name):
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_KEY_SECRET")
            return ChatBedrockConverse(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_access_key,
                    region_name=constants.AWS_REGION,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    model_id=constants.BEDROCK_MAPPINGS[self.model_type.name],
                    rate_limiter=rate_limiter,
                    callbacks=[self.error_handler]
                    )

    # Note: if passing in an image here the prompt should contain the base64 encoded image.
    # see _encode_images for an example
    def get_output(
            self, 
            prompt, 
            max_new_tokens=4000, 
            temperature=0,
            response_model:BaseModel=None,
            ) -> Optional[str | BaseModel]:
        # set_debug(True)
        self.logging.info("LLM (%s) prompt %s", self.model_type, prompt)
        llm_response = self.cache.get_response(
            prompt, 
            self.model_type, 
            self.seed, 
            max_new_tokens, 
            temperature,
        )
        print("CACHED", llm_response)
        if (llm_response is not None) and (response_model is not None):
            llm_response = response_model.model_validate_json(llm_response)
        
        if llm_response is not None:
            self.logging.info("LLM response %s", llm_response)
            return llm_response
        else:
            if BLOCK:
                1/0
            llm = self.get_client(max_new_tokens, temperature)
            if (response_model is not None) and (not constants.is_meta(self.model_type.name)):
                llm = llm.with_structured_output(response_model)
            messages = [
                     SystemMessage(content="You are a helpful assistant"),
                     HumanMessage(content=prompt)
                     ]
            llm_response = llm.invoke(messages)
            if (response_model is not None) and constants.is_meta(self.model_type.name):
                llm_response = response_model.model_validate(from_json(llm_response.content))
            llm_response_content = self._serialize_llm_response(llm_response, response_model=response_model)
            self.cache.save_response(
                prompt, 
                llm_response_content,
                self.model_type, 
                self.seed, 
                max_new_tokens, 
                temperature
            )
            if response_model is None:
                return llm_response_content
            else:
                return llm_response

    async def get_outputs(
            self, 
            dataset: Dataset, 
            batch_size:int = 10, 
            max_new_tokens=4000,
            is_image = False,
            max_retries = 1,
            temperature:float = 0,
            callback = None,
            validation_func = None,
            requests_per_second = None,
            response_model:BaseModel=None,
            ) -> Optional[List[str] | List[BaseModel]]:
        llm = self.get_client(max_new_tokens, temperature, requests_per_second)
        if (response_model is not None) and (not constants.is_meta(self.model_type.name)):
            llm = llm.with_structured_output(response_model)
        
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
            print("BATCH PROMPS", len(batch_prompts))
            df = self.cache.get_responses(
                batch_prompts,
                self.model_type, 
                self.seed, 
                max_new_tokens, 
                temperature
            )
            if response_model:
                json_checks = [not is_valid(response_model, res) for res in df.llm_output]
                rows_to_run = np.where(json_checks)[0].tolist()
            else:
                rows_to_run = np.where(df.llm_output.isna())[0].tolist()
            
            # try:
            if len(rows_to_run):
                if BLOCK:
                    1/0
                batch_results = await self._run_batch(
                        df.iloc[rows_to_run].prompt.tolist(),
                        llm, 
                        max_new_tokens, 
                        temperature,
                        response_model=response_model if not constants.is_meta(self.model_type.name) else None
                        )
                for row_id, response in zip(rows_to_run, batch_results):
                    df.loc[row_id, 'llm_output'] = response

            raw_results = df.llm_output.values.tolist()
            self.logging.info(raw_results)
            if response_model is not None:
                batch_results = []
                for res in raw_results:
                    try:
                        batch_results.append(response_model.model_validate_json(res))
                    except Exception as e:
                        self.logging.info(f"failed to get this result {e}")
                        batch_results.append(None)
            else:
                batch_results = raw_results
            assert len(batch_results) == len(batch_prompts)
            # if validation_func is not None:
            #     validation_func(batch_results)
                
            results += batch_results

            if callback is not None:
                callback(results)
            # got_result = True
            # except Exception as e:
            #     message = f"Failed batch idx {i}. Error {e}"
            #     print(message)
            #     for res in raw_results:
            #         print("parse", res)
            #     self.logging.error(message)

        end_time = time.time()
        execution_time = end_time - start_time
        self.logging.info(f"Results took {execution_time} seconds")
        self.logging.info("NUM RESULTS %d (%d)", len(results), len(dataset))
        assert len(results) == len(dataset)
        return results

    async def _run_batch(
            self, 
            prompts_to_run, 
            llm, 
            max_new_tokens, 
            temperature,
            response_model: BaseModel = None
            ):
        system_prompts = [[
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=prompt)
            ] for prompt in prompts_to_run]
        
        batch_results = await llm.abatch(system_prompts, return_exceptions=True)

        batch_results_strs = [""] * len(prompts_to_run)
        for i, (response, prompt) in enumerate(zip(batch_results, prompts_to_run)):
            try:
                batch_results_strs[i] = self._serialize_llm_response(response, response_model=response_model)

                self.cache.save_response(
                    prompt, 
                    batch_results_strs[i], 
                    self.model_type, 
                    self.seed, 
                    max_new_tokens, 
                    temperature
                )
            except Exception as e:
                self.logging.info(f"run_batch had an exception {e} {response}")
                continue

        return batch_results_strs

    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for (payload, image_paths) in batch_data:
            image_paths = image_paths.split("+")
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        # "detail": "low"
                    }
                })
            updated_payloads.append(payload)
        return updated_payloads

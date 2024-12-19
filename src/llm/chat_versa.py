from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.language_models import BaseChatModel
from openai import AzureOpenAI
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


from src.llm.constants import *


class ChatVersa(BaseChatModel):
    def __init__(
            self, 
            model_name,
            api_key, 
            max_tokens, 
            temperature, 
            timeout, 
            seed
            ):
        assert model_name in VERSA_MODELS
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.seed = seed

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = AzureOpenAI(
                api_key=self.api_key,
                api_version=VERSA_API_VERSION,
                azure_endpoint=VERSA_RESOURCE_ENDPOINT,
                timeout=self.timeout
            )

        breakpoint()
        llm_output = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": messages[-1].content},
                ],
                max_tokens=self.max_new_tokens,
                seed=self.seed,
                model=self.model_name
                )

        llm_response = llm_output.choices[0].message.content
        message = AIMessage(content=llm_response)
        generation = ChatGeneration(message=message)
        breakpoint()
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return self.model_name

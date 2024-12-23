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
    model_name: str 
    api_key: str
    max_tokens: int
    temperature: float
    timeout: int
    seed: int

    def __init__(self, **data):
        assert data['model_name'] in VERSA_MODELS
        data['model_name'] = data['model_name'].replace("versa-", "")
        super().__init__(**data)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        resource_endpoint = VERSA_ENDPOINT.replace("<model_name>", self.model_name)
        client = AzureOpenAI(
                api_key=self.api_key,
                api_version=VERSA_API_VERSION,
                azure_endpoint=resource_endpoint,
                timeout=self.timeout
            )

        llm_output = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": messages[-1].content},
                ],
                max_tokens=self.max_tokens,
                seed=self.seed,
                model=self.model_name
                )

        llm_response = llm_output.choices[0].message.content
        message = AIMessage(content=llm_response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name }

"""
Caches LLM responses
"""
import hashlib
import json
from typing import Optional, List, Tuple, Union
from pydantic import BaseModel
import pandas as pd

from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.constants import LLMModel

class LLMCache:
    def __init__(self, db_handler: DuckDBHandler):
        self.db_handler = db_handler
        self._create_cache()
    
    def get_responses(
            self, 
            batch_prompts: list[Union[str, list]],
            model_type: LLMModel,
            seed: int,
            max_new_tokens: int,
            temperature: float,
            ) -> pd.DataFrame:
        db = self.db_handler.get_connection()
        call_params = json.dumps({
                "model_type": model_type.name.value,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
                })
        cache_df = pd.DataFrame({'prompt': batch_prompts})
        cache_df['prompt_hash'] = cache_df.prompt.apply(lambda c: self.compute_hash(json.dumps(c)))
        cache_df['call_params_hash'] = model_type.name.value #self.compute_hash(call_params)
        query = """
            WITH ranked_outputs AS (
                SELECT 
                    llm_cache.prompt_hash,
                    llm_output,
                    row_number() OVER (PARTITION BY llm_cache.prompt_hash ORDER BY created_at DESC) AS rn
                FROM llm_cache 
                JOIN cache_df
                ON llm_cache.prompt_hash = cache_df.prompt_hash
                AND llm_cache.call_params_hash = cache_df.call_params_hash
                )
            SELECT
                prompt_hash,
                llm_output
            FROM ranked_outputs 
            WHERE rn = 1
            """
        result_df = db.execute(query).df()
        if result_df.empty:
            cache_df['llm_output'] = None
        else:
            print("CACHE HIT")
            cache_df = cache_df.merge(result_df, on="prompt_hash", how="left")
        
        return cache_df

    def get_response(
        self,
        prompt: str,
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        ) -> str:
        return self.get_responses([prompt], model_type, seed, max_new_tokens, temperature).llm_output.iloc[0]

    def save_responses(
            self, 
            batch_prompts: list[str],
            llm_outputs: list[str],
            model_type: LLMModel,
            seed: Optional[int],
            max_new_tokens: int,
            temperature: float
            ):
        db = self.db_handler.get_connection()
        call_params = json.dumps({
                "model_type": model_type.name.value,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
                })
        call_params_hash = self.compute_hash(call_params)
        cache_df = pd.DataFrame({'prompt': batch_prompts})
        cache_df['prompt_hash'] = cache_df.prompt.apply(lambda c: self.compute_hash(json.dumps(c)))
        cache_df['llm_output'] = llm_outputs
        cache_df['call_params_hash'] = model_type.name.value #call_params_hash
        cache_df.drop('prompt', axis=1)
        db.sql("INSERT INTO llm_cache (prompt_hash, llm_output, call_params_hash) SELECT prompt_hash, llm_output, call_params_hash FROM cache_df")

    def save_response(
        self, 
        prompt: str,
        llm_output: str,
        model_type: LLMModel,
        seed: Optional[int],
        max_new_tokens: int,
        temperature: float 
    ):
        self.save_responses(
            [prompt],
            [llm_output],
            model_type,
            seed,
            max_new_tokens,
            temperature
        )

    def compute_hash(self, text: str) -> str:
        text = text.strip()
        text = text.encode('utf-8')
        return hashlib.sha256(text).hexdigest()

    def _create_cache(self):
        db = self.db_handler.get_connection()
        db.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash VARCHAR,
                llm_output TEXT,
                call_params_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import transformers
import time
from transformers import StoppingCriteria, StoppingCriteriaList

from src.llm.llm import LLM
from src.llm.dataset import TextDataset


class LLMLocal(LLM):
    def __init__(self, seed: int, model_type: str, logging):
        super().__init__(seed, model_type, logging)
        transformers.utils.logging.set_verbosity(10)
        transformers.set_seed(seed)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_type,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_type,
            device_map="auto",
            padding_side="left",
            use_fast=True
        )
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logging.info("LLM device %s", self.model.device)
        self.is_api = False

    def get_output(self, prompt: str, max_new_tokens=5000, is_image=False) -> str:
        self.logging.info("LLM prompt %s", prompt)
        input_ids, attn_masks = self._tokenize_prompts([prompt])
        input_ids = input_ids.to('cuda')
        attn_masks = attn_masks.to('cuda')

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_masks,
            max_new_tokens=max_new_tokens,
            length_penalty=0.0,
            temperature=1e-5,
            # do_sample=False,
        )
        decoded_output = self.tokenizer.batch_decode(outputs)[0]
        output = self._clean_output(decoded_output)
        self.logging.info("LLM response %s", output)
        return output

    """
    Parallelizes dataloader by the number of gpus available. Each prompt is returned by the dataset and a batch is
    tokenized with self._tokenize_prompts
    """

    def get_outputs(
            self,
            dataset: TextDataset,
            max_new_tokens: int = 10,
            batch_size: int = 4,
            top_k: int = 10,
            temperature: float = 1.0,
            is_image=False,
            validation_func=None,
            callback=None
    ) -> list[str]:
        assert not is_image
        gpus = torch.cuda.device_count()
        #  The collate_fn is called to transform a batch of text to tokens
        dataloader = DataLoader(
            dataset,
            # num_workers=gpus,
            batch_size=batch_size,
            collate_fn=self._tokenize_prompts
        )
        results = []
        start_time = time.time()
        i = 0
        # terminators = [
        # self.tokenizer.eos_token_id,
        # self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        class StopOnEOTToken(StoppingCriteria):
            def __init__(self, stop_token_id):
                self.stop_token_id = stop_token_id

            def __call__(self, input_ids, scores, **kwargs):
                # Check if the last generated token is the stop token
                return input_ids[0, -1] == self.stop_token_id
        eot_token_id = self.tokenizer.encode("<|eot_id|>")[0]
        stopping_criteria = StoppingCriteriaList(
            [StopOnEOTToken(eot_token_id)])

        for batch_inputs, batch_masks in tqdm(dataloader):
            batch_inputs = batch_inputs.to('cuda')
            batch_masks = batch_masks.to('cuda')
            print(f'i={i} batch_inputs.shape={batch_inputs.shape}')

            outputs = self.model.generate(
                input_ids=batch_inputs,
                attention_mask=batch_masks,
                max_new_tokens=max_new_tokens,
                # top_k=top_k,
                # length_penalty=0.0,
                temperature=1e-5,
                eos_token_id=eot_token_id,
                stopping_criteria=stopping_criteria,
                # do_sample=False,
            )

            decoded_output = self.tokenizer.batch_decode(outputs)
            print(decoded_output)

            results += list(map(self._clean_output, decoded_output))

            if validation_func is not None:
                validation_func(results)

            if callback is not None:
                callback(results)

            # print("Finished generating a batch of outputs")
        end_time = time.time()
        execution_time = end_time - start_time
        message = f"Results took {execution_time} seconds"
        print(message)
        self.logging.info(message)
        return results

    def _clean_output(self, output: str) -> str:
        split_out = output.split(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        cleaned_out = split_out[1].replace('\n\n', '', 1).replace(
            "<|eot_id|>", '',).replace("<|start_header_id|>", "")
        return cleaned_out

    """
    Called for each batch of the dataloader. Returns tensors of tokens for input_ids and attention masks. 
    Padding is automatically added to these.
    """

    def _tokenize_prompts(self, prompts: list[str], truncate_note_chars=None) -> (torch.Tensor, torch.Tensor):
        # tried running this with 4000 for LLaMA 8B
        chat_prompts = []
        if truncate_note_chars is not None:
            prompts = [prompt[:truncate_note_chars] for prompt in prompts]
        for prompt in prompts:
            # we can have apply_chat_template return tensors, but we do this in the next step so padding can be added
            chat_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_prompts.append(chat_prompt)

        inputs = self.tokenizer(
            chat_prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        return input_ids, attn_masks

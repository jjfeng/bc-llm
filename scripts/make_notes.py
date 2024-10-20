"""
Script for making notes with sdoh, dx, gender, age, demographic, etc
"""
import os
import sys
import logging
import time
import argparse
import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())
from src.constants import DIAGNOSES, SDOH_OPTIONS

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the data we want to learn concepts for")
    parser.add_argument("--num-records", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument('--sdoh-probs', type=str, default="0.2,0.15,0.15,0.1,0.4")
    parser.add_argument("--log-file", type=str, help="log file")
    parser.add_argument("--out-file", type=str, help="csv file with notes")
    args = parser.parse_args()
    args.sdoh_probs = np.array(list(map(float, args.sdoh_probs.split(","))), dtype=float)
    return args

# Function to generate patient notes in batches using the LLaMA 3 model
def generate_patient_notes_in_batch(prompts, tokenizer, model, batch_size=8, max_length=20):
    batched_notes = []
    for i in range(0, len(prompts), batch_size):
        print("iter", i)
        is_successful = False
        while not is_successful:
            batch_prompts = [
                [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content":  prompt},
                    {"role": "assistant", "content":  "Patient note:\n"},
                ] for prompt in prompts[i:i + batch_size]]

            inputs = tokenizer.apply_chat_template(
                batch_prompts,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
                padding=True
                )
            outputs = model.generate(inputs["input_ids"][:,:-1], attention_mask=inputs["attention_mask"][:,:-1], max_length=max_length, pad_token_id=tokenizer.eos_token_id)

            is_successful = True
            note_batch = []
            for input_ids, output in zip(inputs["input_ids"], outputs):
                print("input len", len(input_ids))
                print("ORIGINAL", tokenizer.decode(output, skip_special_tokens=True))
                decoded_note = tokenizer.decode(output[len(input_ids):], skip_special_tokens=True)
                if len(decoded_note) < 5:
                    is_successful = False
                    break
                    # regenerate note...
                note_batch.append(decoded_note.replace("assistant", "").strip())
        batched_notes += note_batch
    return batched_notes

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info(args)

    # Load the model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", padding_side='left')
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Generate prompts
    prompts = []
    tabular_data = []
    for i in range(args.num_records):
        sdoh = np.random.choice(SDOH_OPTIONS, p=args.sdoh_probs)
        gender = random.choice(['male', 'female'])
        dx = random.choice(DIAGNOSES).lower()
        age = random.randint(18, 100)
        tabular_data.append(pd.Series({
            "dx": dx,
            "is_cardiac": int(("heart" in dx) or ("cardi" in dx)),
            "housing": int("housing" in sdoh),
            "smoking": int("smoking" in sdoh),
            "food": int("food" in sdoh),
            "gender": int(gender == "male"),
            "age": age,
            "over_60": int(age >= 60),
            # other things we could include: insurance?
        }))
        if sdoh == 'none reported':
            prompt = f"Generate a patient note for a {gender} aged {age} years with a diagnosis of {dx}. Note should be deidentified, so no names or ID numbers. Only include sections 'Demographics', 'Chief Complaint', and 'Medical History'. Each section should only have at most 3 sentences. The note may not exactly copy phrases from this prompt."
        else:
            prompt = f"Generate a patient note for a {gender} aged {age} years with a diagnosis of {dx}. The patient social determinants of health include: {sdoh}. Note should be deidentified, so no names or ID numbers. Only include sections 'Demographics', 'Chief Complaint', 'Medical History', and 'SDOH'. Each section should only have at most 3 sentences. The note may not exactly copy phrases from this prompt, so it cannot use the same exact phrase for describing the patient's SDOH."
        prompt += " Only output the note. Do not output anything else."
        prompts.append(prompt)

    print(prompts)
    tabular_data_df = pd.DataFrame(tabular_data)
    print(tabular_data_df)

    # Generate notes in batches
    st_time = time.time()
    notes = generate_patient_notes_in_batch(prompts, tokenizer, model, batch_size=args.batch_size, max_length=args.max_length)
    tabular_data_df["sentence"] = notes
    tabular_data_df.to_csv(args.out_file, index=False)
    print("time", time.time() - st_time)

    for n in notes:
        print("--------------------")
        print(n)

if __name__ == "__main__":
    main(sys.argv[1:])

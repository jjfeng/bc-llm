# BC-LLM

All experiments are managed using `scons` and `nestly`.
If you want to run a single experiment, specify the experiment's folder name, e.g. `scons exp_mimic`.

## LLM Api
To use the either the OpenAI models or Hugging Face models through the API add a .env file in the root folder of this directory

```
llm-vi $ touch .env
```

Add your token for Open AI and/or Hugging face
```
llm-vi $ echo "OPENAI_ACCESS_TOKEN=<YOUR TOKEN>" >> .env
llm-vi $ echo "HF_ACCESS_TOKEN=<YOUR TOKEN>" >> .env
```

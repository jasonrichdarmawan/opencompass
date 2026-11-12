# configs/eval_qwen36_lcb_v6.py
# flake8: noqa
# isort: skip_file
# yapf: disable
from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import (
        LCBCodeGeneration_dataset,
    )

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='qwen3.6-35b-a3b-nvfp4',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://localhost:8000/v1',
        path='/root/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4',
        tokenizer_path='nvidia/Qwen3.6-35B-A3B-NVFP4',
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=8192,
        max_seq_len=262144,
        temperature=0.0,
        batch_size=1,
        retry=3,
        openai_extra_kwargs=dict(
            extra_body={'chat_template_kwargs': {'enable_thinking': False}}
        ),
    )
]

# Official LiveCodeBench v6 academic code-generation config, with a stable
# output cap for local serving.
LCBCodeGeneration_dataset['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_dataset['n'] = 6
LCBCodeGeneration_dataset['infer_cfg']['inferencer']['max_out_len'] = 4096

datasets = [LCBCodeGeneration_dataset]

# configs/eval_qwen36_local.py
# flake8: noqa
# isort: skip_file
# yapf: disable
import os

from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.bigcodebench.bigcodebench_full_instruct_gen import bigcodebench_full_instruct_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_time_split_gen_6966bc import LCB_datasets
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    from opencompass.configs.datasets.ruler.ruler_256k_gen import ruler_datasets as ruler_256k_datasets

# The OpenCompass-hosted BigCodeBench evaluator is currently unavailable
# (HTTP 403), while the upstream Space is reachable.
for _bigcodebench_dataset in bigcodebench_full_instruct_datasets:
    _bigcodebench_dataset['eval_cfg']['evaluator'][
        'remote_execute_api'] = 'https://bigcode-bigcodebench-evaluator.hf.space/'
del _bigcodebench_dataset

# RULER uses the GPT-4 tokenizer by default, but the difference between
# GPT-4 and Qwen tokenization is large enough to overflow the model context
# at 256k. Use the locally cached Qwen tokenizer when available.
_ruler_tokenizer = os.path.expanduser(
    '~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4')
for _ruler_dataset in ruler_256k_datasets:
    _ruler_dataset['num_samples'] = 10
    if os.path.isdir(_ruler_tokenizer):
        _ruler_dataset['tokenizer_model'] = _ruler_tokenizer
        _ruler_dataset['max_seq_length'] = 250000
    else:
        _ruler_dataset['max_seq_length'] = 250000
del _ruler_dataset

# Do not leak the os module into OpenCompass's serialized config dump.
del os

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

datasets = (
    humaneval_datasets
    + bigcodebench_full_instruct_datasets
    + LCB_datasets
    + ruler_16k_datasets
    + ruler_256k_datasets
)

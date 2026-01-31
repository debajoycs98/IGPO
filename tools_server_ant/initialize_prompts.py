import yaml
from time import strftime, gmtime
from jinja2 import StrictUndefined, Template
from typing import Dict, Any

from tools_server.tools import get_tools

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def initialize_system_prompt(system_prompt_templates, tools, today) -> str:
    system_prompt = populate_template(
        system_prompt_templates,
        variables={
            "tools": tools,
            "today": today
        },
    )
    return system_prompt


# 初始化system prompts
try:
    config_path = "./tools_server/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if not config['codeact_env_disabled']:  # 启用code act
        SYSTEM_PROMPT = initialize_system_prompt(config['system_prompt_codeact'], get_tools(config), strftime("%Y-%m-%d", gmtime()))
    else:
        SYSTEM_PROMPT = initialize_system_prompt(config['system_prompt'], get_tools(config), strftime("%Y-%m-%d", gmtime()))
    print("SYSTEM_PROMPT",SYSTEM_PROMPT)
except Exception as e:
    print(f"initialize system prompts error: {e}")
    SYSTEM_PROMPT = None


"""
IGPO Tool Server - System Prompt Initialization (Web Search Only)
"""

import os
from time import strftime, gmtime
from typing import Dict, Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_system_prompt(config_path: Optional[str] = None) -> str:
    """Load system prompt from config or use default."""
    today = strftime("%Y-%m-%d", gmtime())
    
    if HAS_YAML and config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            template = config.get('system_prompt', '')
            if template:
                return template.replace('{{today}}', today)
        except:
            pass
    
    # Default prompt
    return f"""## Background information 
* Today is {today}
* You are a Deep Research Assistant

You have access to the following tool:
- web_search: Search the web for information

Your output format should be:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
{{"name": "web_search", "arguments": {{"query": ["search query 1", "search query 2"]}}}}
</tool_call>
"""


# Initialize at module import
try:
    SYSTEM_PROMPT = load_system_prompt()
except:
    SYSTEM_PROMPT = load_system_prompt.__doc__

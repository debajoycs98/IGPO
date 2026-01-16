from smolagents import Tool
from tools_server.code.local_python_exec import LocalPythonExecutor

class CodeActTool(Tool):
    name = "code_act"
    description = "Execute python code."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code, and the result must be placed in the '_output' variable.",
        }
    }
    example = {"name": "code_act", "arguments": {"code": "xxxx"}}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        self.additional_authorized_imports = kwargs.get('additional_authorized_imports',[])
        self.tools_add = kwargs.get('tools')
        super().__init__(*args, **kwargs)
        
    def forward(self, code: str) -> str:
        pythonexe = LocalPythonExecutor()
        if self.tools_add and len(self.tools_add)>0:
            pythonexe.send_tool(self.tools_add)
        return pythonexe(code)


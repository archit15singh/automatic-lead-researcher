# Import necessary libraries and modules
import autogen
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
import math
import os
from langchain.tools.file_management.read import ReadFileTool

# Define configuration parameters
config_list_gpt = autogen.config_list_from_json("OAI_CONFIG_LIST")

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list_gpt,
    "timeout": 120,
}


# Define a base tool input model
class CircumferenceToolInput(BaseModel):
    radius: float = Field()


# Define the CircumferenceTool
class CircumferenceTool(BaseTool):
    name = "circumference_calculator"
    description = "Use this tool when you need to calculate a circumference using the radius of a circle"
    args_schema: Type[BaseModel] = CircumferenceToolInput

    def _run(self, radius: float):
        return float(radius) * 2.0 * math.pi


# Function to get the file path of an example
def get_file_path_of_example():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    target_folder = os.path.join(parent_dir, "test")
    file_path = os.path.join(target_folder, "test_files/radius.txt")
    return file_path


# Define a function to generate llm_config from a LangChain tool
def generate_llm_config(tool):
    function_schema = {
        "name": tool.name.lower().replace(" ", "_"),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema


# Instantiate tools
read_file_tool = ReadFileTool()
custom_tool = CircumferenceTool()

# Construct the llm_config
llm_config = {
    "functions": [
        generate_llm_config(custom_tool),
        generate_llm_config(read_file_tool),
    ],
    "config_list": config_list_gpt,
    "timeout": 120,
}

# User proxy setup
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "workspace"},
)

# Register tools and start the conversation
user_proxy.register_function(
    function_map={
        custom_tool.name: custom_tool._run,
        read_file_tool.name: read_file_tool._run,
    }
)

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# Initiate the chat
user_proxy.initiate_chat(
    chatbot,
    message=f"Read the file with the path {get_file_path_of_example()}, then calculate the circumference of a circle that has a radius of that file's contents.",  # 7.81mm in the file
    llm_config=llm_config,
)

import re

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

assistant = AssistantAgent(name="assistant", llm_config={"config_list": config_list})

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: bool(
        re.search(r"TERMINATE\s*$", x.get("content", "").rstrip())
    ),
    code_execution_config={"work_dir": "workspace"},
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
        Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

user_proxy.initiate_chat(
    assistant,
    message="conduct an analysis on the best saas to build based on your own assumptions. do it step by step and show me the reasoning",
)

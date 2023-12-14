import autogen

config_list_gpt = autogen.config_list_from_json("OAI_CONFIG_LIST")

llm_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt,
    "timeout": 120,
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "workspace"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

user_proxy.initiate_chat(
    assistant,
    message="""
Who should read this paper: https://arxiv.org/abs/2308.08155
""",
)

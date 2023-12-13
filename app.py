# 1. # create new .py file with code found below
# 2. # install ollama
# 3. # install model you want “ollama run mistral”
# 4. conda create -n autogen python=3.11
# 5. conda activate autogen
# 6. which python
# 7. python -m pip install pyautogen
# 7. ollama run mistral
# 8. ollama run codellama
# 9. # open new terminal
# 10. conda activate autogen
# 11. python -m pip install litellm
# 12. litellm --model ollama/mistral
# 13. # open new terminal
# 14. conda activate autogen
# 15. litellm --model ollama/codellama

import autogen


def create_config_list(base_url, api_key):
    return [{"base_url": base_url, "api_key": api_key}]


def create_llm_config(config_list):
    return {"config_list": config_list}


def create_assistant_agent(name, llm_config):
    return autogen.AssistantAgent(name=name, llm_config=llm_config)


def create_user_proxy_agent(
    name,
    human_input_mode,
    max_consecutive_auto_reply,
    termination_msg_func,
    code_execution_config,
    llm_config,
    system_message,
):
    return autogen.UserProxyAgent(
        name=name,
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        is_termination_msg=termination_msg_func,
        code_execution_config=code_execution_config,
        llm_config=llm_config,
        system_message=system_message,
    )


def initiate_chat(task):
    # Configuration for Mistral
    config_list_mistral = create_config_list("http://0.0.0.0:8000", "NULL")
    llm_config_mistral = create_llm_config(config_list_mistral)

    # Configuration for Codellama
    config_list_codellama = create_config_list("http://0.0.0.0:3764", "NULL")
    llm_config_codellama = create_llm_config(config_list_codellama)

    # Create Assistant Agent and User Proxy Agent
    coder = create_assistant_agent(name="Coder", llm_config=llm_config_codellama)
    user_proxy = create_user_proxy_agent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        termination_msg_func=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
        code_execution_config={"work_dir": "workspace"},
        llm_config=llm_config_mistral,
        system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
    Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    )

    # Initiate the chat
    user_proxy.initiate_chat(coder, message=task)
    return True

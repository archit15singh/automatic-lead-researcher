from autogen import config_list_from_json
import autogen

# Get api key
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list, "seed": 42, "timeout": 120}

# Create user proxy agent, coder, product manager
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin who will give the idea and run the code provided by Coder.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "workspace",
    },
    human_input_mode="NEVER",
)
coder = autogen.AssistantAgent(
    name="Coder",
    system_message="you are an expert code writer. generate code with no issues according to the requirements.",
    llm_config=llm_config,
)
pm = autogen.AssistantAgent(
    name="product_manager",
    system_message="You will help break down the initial idea into a well scoped requirement for the coder; Do not involve in future conversations or error fixing.  do it step by step",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="critic",
    system_message="You will critique the code written by the coder to make sure all the edge cases and features are being supported. do it step by step",
    llm_config=llm_config,
)

# Create groupchat
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm, critic], messages=[])
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message="Given a binary tree, the task is to find the height of the tree. The height of the tree is the number of vertices in the tree from the root to the deepest node. ",
)

import sys
print(f"Python version: {sys.version}")
print("-" * 30)

try:
    import autogen_core
    try:
        print(f"autogen_core version from __version__: {autogen_core.__version__}")
    except AttributeError:
        print("autogen_core __version__ attribute not found (rely on pip show).")
    print(f"autogen_core location: {autogen_core.__file__}")
except ImportError as e:
    print(f"Failed to import autogen_core: {e}")

print("-" * 30)

try:
    import autogen_agentchat.agents
    print(f"autogen_agentchat.agents location: {autogen_agentchat.agents.__file__}")
    print(f"UserProxyAgent class from autogen_agentchat.agents: {autogen_agentchat.agents.UserProxyAgent}")
    
    print("\nInspecting UserProxyAgent class itself:")
    upa_class = autogen_agentchat.agents.UserProxyAgent
    has_class_a_initiate_chat = hasattr(upa_class, "a_initiate_chat")
    has_class_initiate_chat = hasattr(upa_class, "initiate_chat")
    print(f"UserProxyAgent CLASS has 'a_initiate_chat': {has_class_a_initiate_chat}")
    print(f"UserProxyAgent CLASS has 'initiate_chat': {has_class_initiate_chat}")

    print("\nInspecting an INSTANCE of UserProxyAgent:")
    test_agent_instance = autogen_agentchat.agents.UserProxyAgent(name="test_agent_for_diag_instance")
    has_instance_a_initiate_chat = hasattr(test_agent_instance, "a_initiate_chat")
    has_instance_initiate_chat = hasattr(test_agent_instance, "initiate_chat") 
    print(f"UserProxyAgent INSTANCE has 'a_initiate_chat': {has_instance_a_initiate_chat}")
    print(f"UserProxyAgent INSTANCE has 'initiate_chat': {has_instance_initiate_chat}")

    # In ra các phương thức liên quan đến chat để kiểm tra thêm nếu cần
    # print("\nPotential chat-related methods on UserProxyAgent instance:")
    # for attr in dir(test_agent_instance):
    #    if "initiate" in attr.lower() or "chat" in attr.lower() or "send" in attr.lower() or "receive" in attr.lower():
    #        print(f"  - {attr}")

except ImportError as e:
    print(f"Failed to import autogen_agentchat.agents: {e}")
except Exception as e_general:
    print(f"An unexpected error occurred during diagnosis: {e_general}")
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message
import tools


load_dotenv('.env')

def load_agent():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     streaming=True,
                     verbose=True,
                     temperature=0,
                     client=None)
    tool_list = [tools.GetPttPostsKeywordsByDate(),
                 tools.GetKeywordsVote(),
                 tools.GetKeywordsVoteTrend(),
                 tools.GetUpvoteCommentsByKeyword(),
                 tools.GetDownvoteCommentsByKeyword(),]
    agent = initialize_agent(tool_list,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    return agent

agent = load_agent()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="CLLT Final Project - Demo", page_icon=":robot:")
st.header("CLLT Final Project - Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "請給我 2023/01/08 PTT 貼文的關鍵詞，不需要做其他分析", key="input")
    return input_text


user_input = get_text()

if user_input:
    propmt = f"""你是一個臺灣的 PTT 專家，請使用提供的 tools 完成後面提供給你的工作，並使用臺灣中文回答問題。
    有些提供的 tool 完全不會使用到，但是你可以自己決定要不要使用。
    你的工作是：
    {user_input}"""
    output = agent.run(propmt)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
import streamlit as st
import json
import sqlite3
from datetime import datetime
import asyncio
from streamlit_ace import st_ace
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
import pandas as pd
from openai import OpenAI
import subprocess
import sys

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

st.set_page_config(page_title="NeuroCode Companion", page_icon="🧠💻", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(15deg, #13547a 0%, #80d0c7 100%);
        color: white;
    }
    .stButton > button {
        color: #4F8BF9;
        border-radius: 20px;
        height: 3em;
        background-color: rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .css-1d391kg {
        background-color: rgba(251, 251, 251, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
    }
    .memory-item {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .memory-key {
        font-weight: bold;
        color: #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

for key in ['messages', 'user_memory', 'productivity_data', 'code_snippets']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['messages', 'code_snippets'] else {}

conn = sqlite3.connect('neurocode_companion.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS productivity (date TEXT, focus_time INTEGER, tasks_completed INTEGER)')
conn.commit()

def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_coding = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json')

async def execute_code(code):
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        return result.stdout if result.stdout else result.stderr
    except subprocess.TimeoutExpired:
        return "Execution timed out after 10 seconds."
    except Exception as e:
        return f"Error executing code: {str(e)}"

def update_productivity(focus_time, tasks_completed):
    date = datetime.now().strftime("%Y-%m-%d")
    c.execute("INSERT INTO productivity VALUES (?, ?, ?)", (date, focus_time, tasks_completed))
    conn.commit()

def get_productivity_data(days=7):
    c.execute("SELECT * FROM productivity WHERE date >= date('now', '-7 days')")
    return pd.DataFrame(c.fetchall(), columns=['date', 'focus_time', 'tasks_completed'])

def visualize_productivity():
    data = get_productivity_data()
    fig1 = px.line(data, x='date', y='focus_time', title='Daily Focus Time')
    fig2 = px.bar(data, x='date', y='tasks_completed', title='Tasks Completed')
    for fig in [fig1, fig2]:
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig1, fig2

def update_user_memory(new_info):
    st.session_state.user_memory.update(new_info)

async def chatbot_response(user_input):
    system_prompt = f"""You are NeuroCode, an AI assistant for ADHD full-stack Python developers. Your role is to provide friendly, supportive advice on coding, project management, and ADHD strategies. Always be concise, clear, and supportive in your responses.

User profile: {json.dumps(st.session_state.user_memory)}

Your capabilities include:
1. Executing Python code
2. Updating productivity data
3. Providing coding tips and ADHD management strategies

When using these capabilities, always explain what you're doing and why it's helpful for the user."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python code to execute"},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_productivity",
                "description": "Update productivity data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_time": {"type": "integer", "description": "Focus time in minutes"},
                        "tasks_completed": {"type": "integer", "description": "Number of tasks completed"},
                    },
                    "required": ["focus_time", "tasks_completed"],
                },
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model='llama3.1:8b',
            messages=messages,
            tools=tools,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                new_content = chunk.choices[0].delta.content
                full_response += new_content
                yield new_content

            if chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls
                for tool_call in tool_calls:
                    if tool_call.function.name == 'execute_code':
                        args = json.loads(tool_call.function.arguments)
                        result = await execute_code(args['code'])
                        tool_response = f"\nExecuted code. Result:\n{result}\n"
                        full_response += tool_response
                        yield tool_response
                        messages.append({"role": "function", "name": "execute_code", "content": result})

                    elif tool_call.function.name == 'update_productivity':
                        args = json.loads(tool_call.function.arguments)
                        update_productivity(args['focus_time'], args['tasks_completed'])
                        tool_response = f"\nProductivity updated: Focus time {args['focus_time']} minutes, {args['tasks_completed']} tasks completed.\n"
                        full_response += tool_response
                        yield tool_response
                        messages.append({"role": "function", "name": "update_productivity", "content": tool_response})

                follow_up_response = client.chat.completions.create(
                    model='llama3.1:8b',
                    messages=messages + [{"role": "assistant", "content": full_response}],
                    stream=True
                )
                for chunk in follow_up_response:
                    if chunk.choices[0].delta.content is not None:
                        new_content = chunk.choices[0].delta.content
                        full_response += new_content
                        yield new_content

        update_user_memory({"last_interaction": f"User input: {user_input}\nAssistant response: {full_response}"})

    except Exception as e:
        yield f"Sorry, I'm having trouble connecting to my language model. Error: {str(e)}"

def display_memory():
    st.subheader("🧠 User Profile")
    for key, value in st.session_state.user_memory.items():
        st.markdown(f"""
        <div class="memory-item">
            <span class="memory-key">{key}:</span> {value}
        </div>
        """, unsafe_allow_html=True)

async def main():
    st.title("🧠💻 NeuroCode Companion")
    
    st.header("💬 Chat with NeuroCode")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about coding, project management, or ADHD strategies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            async for response_chunk in chatbot_response(prompt):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    with st.expander("💻 Code Editor"):
        language = st.selectbox("Select language", ["python", "javascript", "html", "css", "sql"])
        code = st_ace(language=language, theme="monokai", font_size=14, key="code_editor")
        if st.button("▶️ Execute Code"):
            with st.spinner("Executing code..."):
                result = await execute_code(code)
            st.code(result)
    
    with st.expander("📊 Productivity Tracker"):
        col1, col2 = st.columns(2)
        with col1:
            focus_time = st.number_input("Today's focus time (minutes)", min_value=0, max_value=1440, value=0, step=15)
        with col2:
            tasks_completed = st.number_input("Tasks completed today", min_value=0, value=0, step=1)
        if st.button("📊 Update Productivity"):
            update_productivity(focus_time, tasks_completed)
            st.success("Productivity data updated!")
        fig1, fig2 = visualize_productivity()
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    
    with st.sidebar:
        st_lottie(lottie_coding, height=200, key="coding")
        st.markdown("## 🛠️ Quick Actions")
        if st.button("💡 Get Coding Tip"):
            with st.spinner("Generating tip..."):
                tip_response = client.chat.completions.create(
                    model='llama3.1:8b',
                    messages=[{"role": "user", "content": "Provide a coding tip for ADHD developers."}],
                    stream=False
                )
                tip = tip_response.choices[0].message.content
            st.info(tip)
        
        st.markdown("## 🧠 User Profile Management")
        first_name = st.text_input("First Name", st.session_state.user_memory.get("first_name", ""))
        last_name = st.text_input("Last Name", st.session_state.user_memory.get("last_name", ""))
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.user_memory.get("age", 0))
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"], index=["Male", "Female", "Non-binary", "Prefer not to say"].index(st.session_state.user_memory.get("gender", "Prefer not to say")))
        description = st.text_area("Short Description", st.session_state.user_memory.get("description", ""))
        
        if st.button("Update Profile"):
            update_user_memory({
                "first_name": first_name,
                "last_name": last_name,
                "age": age,
                "gender": gender,
                "description": description
            })
            st.success("User profile updated successfully!")
        
        if st.checkbox("View User Profile"):
            display_memory()
    
    st.markdown("---")
    st.markdown("NeuroCode Companion: Empowering ADHD developers to reach their full potential! 🚀🧠")

if __name__ == "__main__":
    asyncio.run(main())

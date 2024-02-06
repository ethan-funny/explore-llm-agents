import os
from openai import OpenAI
from gnews import GNews


def llm(system_prompt, user_query, stop=None):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages,
      temperature=0,
      max_tokens=400,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )

    return response.choices[0].message.content


agent_prompt = """解决问答任务时，需要交错进行思考（Thought）、行动（Action）和观察（Observation）这三个步骤。思考环节可以对当前情况进行推理分析，行动环节分为以下四种类型：
1. Search[keywords]: 使用多个关键词进行搜索，多个关键词使用,分隔。
2. TTS[text]: 使用 TTS 技术将 text 转成语音。
3. Summary[text]: 对 text 进行摘要总结。
4. Finish[answer]: 返回答案并完成任务。

以下是一个示例。
Question: 总结最近的aigc热点新闻。
Thought 1: 我需要搜索最近跟aigc相关的热点新闻，具体关键词有aigc,llm,人工智能,gpt,大模型等
Action 1: Search[aigc,llm,人工智能,gpt,大模型]
Observation 1: AI for Science将人工智能与科学研究深度结合，推动科学发现和生活改变。Quora筹集7500万美元用于加速其AI聊天平台Poe的发展，主要投入AI开发者创作货币化。
Thought 2: 我需要对搜索结果进行摘要总结。
Action 2: Summary[AI for Science将人工智能与科学研究深度结合，推动科学发现和生活改变。Quora筹集7500万美元用于加速其AI聊天平台Poe的发展，主要投入AI开发者创作货币化。]
Observation 2: AI for Science 推动科学与生活创新。Quora筹集7500万美元加速AI平台Poe开发。
Thought 3: 我已总结出最近的热点新闻：AI for Science 推动科学与生活创新。Quora筹集7500万美元加速AI平台Poe开发。
Action 3: Finish[AI for Science 推动科学与生活创新。Quora筹集7500万美元加速AI平台Poe开发。]
"""


def react_agent(question, system_prompt):
    user_query = f"Question: {question}\n"
    print(user_query)

    for i in range(1, 5):
        # 设置停止词 Observation，让 LLM 仅生成每一轮的 thought 和 action
        thought_action = llm(system_prompt, user_query + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            thought = thought_action.strip().split('\n')[0]
            action = llm(system_prompt, user_query + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()

        # 根据 action 执行相应的操作，得到 observation
        obs, done = step(action)
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        user_query += step_str  # 将之前的 思考-行动-观察 也加入

        print(f"step {i} \n{user_query}\n")
        if done:
            break
    
    if not done:
        obs, done = step("Finish[]")

    return obs


def step(action):
    done = False
    obs = None
    action = action.strip()

    if action.startswith("Search[") and action.endswith("]"):
        keywords = action[len("Search["):-1]
        obs = search(keywords)

    elif action.startswith("Summary[") and action.endswith("]"):
        text = action[len("Summary["):-1]
        obs = summary(text)

    elif action.startswith("Finish[") and action.endswith("]"):
        answer = action[len("Finish["):-1]
        done = True
        obs = answer
    
    else:
        print(f"Invalid action: {action}")
        obs = "Invalid action"
    
    return obs, done


def search(keywords):
    print(f"Searching for {keywords}...")
    gn = GNews(language='zh-Hans', country='CN', period='7d', max_results=10)
    news_items = gn.get_news(keywords)
    content = ""
    for item in news_items:
        content += f"{item['description']}\n"

    return content


def summary(text):
    system_prompt = "你是一个文本总结专家，使用中文，将 ```text``` 中的 text 进行总结，不超过100个中文字符。"
    user_query = f" ```{text}``` "

    return llm(system_prompt, user_query)


if __name__ == "__main__":
    question = "总结最近商业投资相关的新闻"
    result = react_agent(question, agent_prompt)
    print(f"Result: {result}")

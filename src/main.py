import os
from langchain_google_vertexai import VertexAI
from langchain_prompty import create_chat_prompt


PROJECT_ID = os.environ.get("PROJECT_ID", "")
LOCATION = os.environ.get("LOCATION", "asia-northeast1")

USE_CHAT_MODEL_NAME = os.environ.get(
    "USE_CHAT_MODEL_NAME",
    "gemini-1.5-flash-001"
)


if __name__ == "__main__":

    cwd = os.getcwd()

    try:
        prompt = create_chat_prompt(f'{cwd}/basic.prompty')
        prompt_text = prompt.invoke(
            {
                'question': "LangChainとはなんですか?"
            }
        )
        chat = VertexAI(model_name=USE_CHAT_MODEL_NAME, temperature=0)
        answer = chat.invoke(prompt_text)
        print(answer)
    except Exception as e:
        print(str(e))

from openai import OpenAI
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai

load_dotenv("/home/admincsp/conversational_rag/.env")

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

print(f"Testing Model: {model}")

try:
    client = wrap_openai(OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    ))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    print("Success!")
    print(completion.choices[0].message.content)

except Exception as e:
    print("FAILED")
    print(e)

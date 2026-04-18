import os
from groq import Groq

print("Script is starting...")
client = Groq(api_key='ask-sara-for-key')

query = "Using the string 'hidad', apply the necessary spaces to form the sentence. Only respond with the sentence itself."
try:
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": f"{query}"}],
        model="llama-3.3-70b-versatile",
    )
    sent = completion.choices[0].message.content
    print(f"This is the sentence: {sent}")
except Exception as e:
    print(f"An error occurred: {e}")

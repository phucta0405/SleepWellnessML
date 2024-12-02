import openai

def get_gpt_response(prompt, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[              
            {"role": "system", "content": "You are a helpful health assistant. You provide wellness advice, tips for healthy living, and support for managing health issues."},
            {"role": "user", "content": prompt} 
        ],
        max_tokens=150,         
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()





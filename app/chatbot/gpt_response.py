import openai

def get_gpt_response(prompt, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[              
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt} 
        ],
        max_tokens=150,         
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()

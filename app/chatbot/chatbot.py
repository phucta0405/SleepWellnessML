from gpt_response import get_gpt_response
from database import fetch_health_advice, predict_and_add_to_db
import config

def chat():
    print("Hi, I am your virtual health assistant. How can I help you with your health and wellness today?")
    print("Type 'exit' to end the chat.")
    
    api_key = input("Insert your API key here: ")  # API key for GPT-3/4
    conversation_history = []  # List to store conversation for GPT

    while True:
        user_message = input("You: ")
        
        if user_message.lower() == 'exit':
            print("Have a good day, stay healthy!")
            break
        
        if 'predict' in user_message.lower():  # Trigger prediction with the trained model
            prediction = predict_and_add_to_db(user_message, 'path_to_your_model/model.pkl')
            print(f"Health Assistant: {prediction}")
        
        elif 'advice' in user_message.lower():  # Fetch health advice from the database
            topic = user_message.lower().replace("advice", "").strip()  # Extract topic from user message
            advice = fetch_health_advice(topic)
            
            if advice:
                print("Health Assistant:", advice)
            else:
                print("Health Assistant: Sorry, I don't have advice on that topic right now.")
        
        else:  # Use GPT to generate conversational responses
            conversation_history.append(f"You: {user_message}")
            prompt = "\n".join(conversation_history)
            gpt_response = get_gpt_response(prompt, api_key)
            print("Health Assistant:", gpt_response)
            conversation_history.append(f"Health Assistant: {gpt_response}")

if __name__ == "__main__":
    chat()

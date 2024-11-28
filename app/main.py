from recommendations import predict_and_recommend_sleep, provide_sleep_advice, check_sleep_goal

def get_valid_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0: 
                return value
            else:
                print("Please enter a value greater than zero.")
        except ValueError:
            print("Invalid input. Please input a numeric value.")

def main():
    try:
        user_age = get_valid_input("Input your age: ")
        gender_input = input("Input your gender (Male/Female): ").strip()
        phy_act = get_valid_input("Input your minutes on physical activities: ")
        sleep_duration = get_valid_input("Input your sleep duration in hours: ")

        predicted_sleep_duration = predict_and_recommend_sleep(user_age, gender_input, phy_act)
        
        if predicted_sleep_duration is not None:
            sleep_goal_message = check_sleep_goal(sleep_duration, predicted_sleep_duration)
            print(sleep_goal_message)
        
        advice = provide_sleep_advice(predicted_sleep_duration, user_age, gender_input, phy_act)
        print(advice)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

import json
import os
from datetime import datetime
import numpy as np

from sklearn.linear_model import LogisticRegression

DATA_FILE = "habits_data.json"



def load_data():
    if not os.path.exists(DATA_FILE):
        return {"habits": {}}
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def today_date():
    return datetime.now().strftime("%Y-%m-%d")


def get_day_number(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").weekday()



def add_habit(data):
    habit = input("Enter habit name: ")
    if habit in data["habits"]:
        print("Habit already exists!")
        return

    data["habits"][habit] = {
        "history": {},
        "streak": 0
    }
    save_data(data)
    print("Habit added!")


def mark_habit(data):
    if not data["habits"]:
        print("No habits found!")
        return

    print("\nHabits:")
    for i, habit in enumerate(data["habits"], 1):
        print(f"{i}. {habit}")

    choice = int(input("Select habit number: ")) - 1
    habit_name = list(data["habits"].keys())[choice]

    status = input("Completed? (y/n): ").lower()
    date = today_date()

    if status == 'y':
        data["habits"][habit_name]["history"][date] = 1
        update_streak(data, habit_name, True)
    else:
        data["habits"][habit_name]["history"][date] = 0
        update_streak(data, habit_name, False)

    save_data(data)
    print("Updated!")


def update_streak(data, habit, completed):
    if completed:
        data["habits"][habit]["streak"] += 1
    else:
        data["habits"][habit]["streak"] = 0


def view_progress(data):
    for habit, details in data["habits"].items():
        history = details["history"]
        total = len(history)
        done = sum(history.values())
        percent = (done / total * 100) if total else 0

        print(f"\nHabit: {habit}")
        print(f"Streak: {details['streak']} ")
        print(f"Completion: {percent:.2f}%")
        print("Progress:", "+" * int(percent // 5))


def reminders(data):
    print("\n Reminder:")
    for habit in data["habits"]:
        print("-", habit)


def analytics(data):
    most_missed = None
    max_missed = -1

    for habit, details in data["habits"].items():
        missed = list(details["history"].values()).count(0)
        if missed > max_missed:
            max_missed = missed
            most_missed = habit

    print("\n Most missed habit:", most_missed)



def predict_habit(data):
    if not data["habits"]:
        print("No habits available!")
        return

    print("\nSelect Habit:")
    habits = list(data["habits"].keys())
    for i, h in enumerate(habits, 1):
        print(f"{i}. {h}")

    choice = int(input("Enter choice: ")) - 1
    habit = habits[choice]

    history = data["habits"][habit]["history"]

    if len(history) < 5:
        print("Not enough data for prediction (need at least 5 days)")
        return

    X = []
    y = []

    for date, value in history.items():
        day_num = get_day_number(date)
        X.append([day_num])
        y.append(value)

    X = np.array(X)
    y = np.array(y)

    model = LogisticRegression()
    model.fit(X, y)

    today = get_day_number(today_date())
    prediction = model.predict([[today]])[0]
    prob = model.predict_proba([[today]])[0][prediction]

    print("\n AI Prediction:")
    if prediction == 1:
        print(f"You are likely to COMPLETE '{habit}' today ")
    else:
        print(f"You may MISS '{habit}' today ")

    print(f"Confidence: {prob*100:.2f}%")



def menu():
    data = load_data()

    while True:
        print("\n STUDENT HABIT TRACKER ")
        print("1. Add Habit")
        print("2. Mark Habit")
        print("3. View Progress")
        print("4. Reminders")
        print("5. Analytics")
        print("6. Predict Habit Success")
        print("7. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            add_habit(data)
        elif choice == "2":
            mark_habit(data)
        elif choice == "3":
            view_progress(data)
        elif choice == "4":
            reminders(data)
        elif choice == "5":
            analytics(data)
        elif choice == "6":
            predict_habit(data)
        elif choice == "7":
            print("Goodbye ")
            break
        else:
            print("Invalid choice!!")


if __name__ == "__main__":
    menu()

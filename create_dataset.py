
import pandas as pd

data = {
    "message": [
        # Spam messages (50)
        "Congratulations! You won a free iPhone",
        "Claim your free prize now",
        "Win cash instantly by clicking this link",
        "You have won a lottery worth 50000",
        "Get a free gift card today",
        "Urgent! Claim your reward now",
        "Limited offer buy one get one free",
        "You are our lucky winner",
        "Earn money from home easily",
        "Click here to claim your prize",
        "Congratulations you won a cash reward",
        "Free entry in our lucky draw",
        "Get a loan approved instantly",
        "Win a brand new smartphone today",
        "Exclusive deal just for you",
        "Your account won a special bonus",
        "Claim your free vacation package",
        "You have been selected for a prize",
        "Earn thousands every week",
        "Act now to receive your free gift",
        "Get rich quickly with this offer",
        "Free recharge available claim now",
        "You won a shopping voucher",
        "Congratulations your number has won",
        "Click now for a special discount",
        "Lowest price guaranteed buy today",
        "Get a free laptop by entering now",
        "You are eligible for a cash prize",
        "Win big with our lucky draw",
        "Claim your bonus before it expires",
        "Free membership limited time offer",
        "Make money fast with no investment",
        "Your reward is waiting click here",
        "Get instant cash approval now",
        "Special promotion win exciting gifts",
        "Congratulations claim your free ticket",
        "You have won a shopping spree",
        "Earn extra income from home",
        "Final notice claim your reward",
        "Get free access for a limited time",
        "You are selected for a cash giveaway",
        "Win exciting prizes by registering",
        "Exclusive bonus available today",
        "Claim your free shopping coupon",
        "Instant loan available apply now",
        "Get a free holiday package",
        "You won a gift hamper",
        "Hurry up your prize will expire",
        "Double your money with this offer",
        "Click here to unlock your bonus",

        # Ham / legitimate messages (50)
        "Hey how are you doing today",
        "Are we meeting after college",
        "Please call me when you are free",
        "I will reach home in ten minutes",
        "Can you send me the class notes",
        "Let's have lunch together tomorrow",
        "Your class starts at ten AM",
        "Thank you for helping me yesterday",
        "Please bring my notebook tomorrow",
        "I will call you in the evening",
        "Did you complete the assignment",
        "The lecture has been postponed",
        "I am waiting outside the library",
        "Please share the meeting details",
        "What time are you coming home",
        "Let's prepare for the exam together",
        "I have sent you the document",
        "Can we talk after the class",
        "Please remind me about the meeting",
        "I will be late today",
        "Did you eat your lunch",
        "Happy birthday have a great day",
        "The bus will arrive in fifteen minutes",
        "I am going to the market",
        "Please check your email",
        "We have a test next Monday",
        "Can you explain this question",
        "I will meet you near the gate",
        "The assignment deadline is tomorrow",
        "Please bring your laptop to class",
        "Mom asked you to call her",
        "I have reached the hostel safely",
        "Are you coming to the library",
        "Let's watch a movie this weekend",
        "Your package has arrived at reception",
        "Please send me the project file",
        "I am studying for tomorrow's exam",
        "The teacher shared new notes",
        "Can you help me with this problem",
        "I will be available after five PM",
        "We should start the project today",
        "Dinner is ready come downstairs",
        "I have booked the tickets",
        "Please wait for me near the bus stop",
        "The meeting is scheduled for three PM",
        "I will submit the assignment tonight",
        "Thanks for sharing the information",
        "Let's discuss the project tomorrow",
        "I am on my way to college",
        "See you at the canteen tomorrow"
    ],
    "label": ["spam"] * 50 + ["ham"] * 50
}

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("spam_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total messages:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nFirst five rows:")
print(df.head())

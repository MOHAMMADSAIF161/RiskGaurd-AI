import os
from dotenv import load_dotenv
import razorpay

load_dotenv()

key_id = os.getenv("RAZORPAY_KEY_ID")
key_secret = os.getenv("RAZORPAY_KEY_SECRET")

client = razorpay.Client(auth=(key_id, key_secret))

print("Razorpay client initialized successfully")
print("Key ID loaded:", bool(key_id))
print("Key Secret loaded:", bool(key_secret))
from flask import Flask, render_template, request, send_from_directory
import instaloader
import instaloader.exceptions
import time
import pickle
import numpy as np
import os

app = Flask(__name__)
loader = instaloader.Instaloader()

with open('detector.pkl', 'rb') as file:
    model = pickle.load(file)

def fetch_instagram_details(username):
    start_time = time.time()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        followers_count = profile.followers
        following_count = profile.followees
        posts_count = profile.mediacount
        
        return followers_count, following_count, posts_count
    
    except (instaloader.exceptions.ProfileNotExistsException, instaloader.exceptions.QueryReturnedNotFoundException):
        print(f"The profile '{username}' does not exist or is private.")
        return None, None, None
    
    except instaloader.exceptions.ConnectionException:
        print(f"Failed to connect to Instagram. Please check your internet connection.")
        return None, None, None
    
    finally:
        end_time = time.time()
        if end_time - start_time > 10:
            print(f"Fetching details for profile '{username}' took too long. Assuming it does not exist or is private.")
            return None, None, None

def predict_from_instagram_details(username):
    followers_count, following_count, posts_count = fetch_instagram_details(username)
    if followers_count is None:
        return "The profile does not exist or is private."
    if followers_count is None:
        followers_count = 0
    if following_count is None:
        following_count = 0
    if posts_count is None:
        posts_count = 0
    
    input_data = np.array([[
        0,  #favourites
        followers_count,
        posts_count,
        following_count,
        0,  #listed
        0,  #geo enabled
        0,  #default profile
        0   #profile use background image
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return f"The account is predicted to be fake. Followers: {followers_count}, Following: {following_count}, Posts: {posts_count}"
    else:
        return f"The account is predicted to be real. Followers: {followers_count}, Following: {following_count}, Posts: {posts_count}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('username')
    result = predict_from_instagram_details(username)
    return result

if __name__ == '__main__':
    app.run(debug=True)

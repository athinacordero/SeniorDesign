import json
import requests

def getPosts(subreddit="depression", limit=1000, utcTime=None): 
    url = "https://api.pushshift.io/reddit/submission/search?subreddit="+str(subreddit)+"&limit="+str(limit)
    if utcTime != None:
        url += "&before="+str(utcTime)

    # Request URL and return data
    response = requests.get(url)
    return response.json()

def getDatasetForSubreddit(subreddit="depression", n_batches=1):
    lastTimeRecieved = None # UTC time of last received post
    dataset_complete = [] # Whole dataset

    # Get n_batches of data. Make sure the posts are only those received before t
    # the time of the last received post as to assure all posts are unique
    print("Getting data for subreddit " + subreddit)
    for i in range(n_batches):
        data = getPosts("depression", 1000, lastTimeRecieved)

        # Add posts in this batch to dataset
        for post in data['data']:
            dataset_complete.append(post)
            lastTimeRecieved = post["created_utc"]
        
    return dataset_complete


def checkDatasetForUniqueness(dataset):
    totalRepeatCount = 0 # Number of posts present more than once in dataset 

    # Loop through dataset and find repeats of posts
    for post in dataset:
        identicalPosts = list(filter(lambda x: x['id'] == post['id'], dataset))
        if len(identicalPosts) > 1:
            totalRepeatCount += 1
    
    if totalRepeatCount != 0:
        print("Found " + str(totalRepeatCount) + " duplicates in dataset")

def writeDatasetToFile(dataset, subreddit="depression"):
    with open('datasets/'+subreddit+'_text_samples_extended.json', 'w') as outfile:
        dataset_out = {}
        dataset_out['data'] = dataset
        json.dump(dataset_out, outfile)

def main():
    subreddits = ['depression', 'Anxiety', 'foreveralone', 'socialanxiety', 'SuicideWatch', 
                    'berkeley', 'PowerLedger', 'TalesFromYourServer', 'tifu']

    for subreddit in subreddits:
        dataset_complete = getDatasetForSubreddit(subreddit=subreddit, n_batches=15)
        writeDatasetToFile(dataset_complete, subreddit=subreddit)

if __name__ == "__main__":
    main()
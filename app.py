import pickle

with open('sklearn_objects.pkl', 'rb') as f:
    mlb, p = pickle.load(f)


# predicting the model with samples
classes = ['.net', 'android', 'asp.net', 'c', 'c#', 'c++', 'css', 'html', 'ios', 'iphone',
 'java', 'javascript', 'jquery', 'mysql', 'objective-c', 'php', 'python', 'ruby',
 'ruby-on-rails', 'sql']

def suggestion(question):
    x=[]
    x.append(question)
    mlb_predictions = p.predict([question])
    tags = mlb.inverse_transform(mlb_predictions)    
    suggested_tags = []

    for tag in tags:
        suggested_tags.append(tag)
    
    return suggested_tags


def main():
    # Get the question from the user
    question = input("Enter your question: ")

    # Predict the tags for the question
    predicted_tags = suggestion(question)

    # Print the predicted tags
    print("Suggested tags:", predicted_tags)


if __name__ == "__main__":
    main()

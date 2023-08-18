# from asyncio.windows_events import NULLMultiLabelBinarizer
from tkinter import messagebox  # All imports
from tkinter import *
from PIL import ImageTk, Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import pickle
import ast
import pandas as pd

df = pd.read_csv("empty.csv")


df["Tags"] = df["Tags"].apply(lambda x: ast.literal_eval(x))
y = df["Tags"]
multilabel = MultiLabelBinarizer()  # multiLabelBinarier for Tags
y = multilabel.fit_transform(df["Tags"])
pd.DataFrame(y, columns=multilabel.classes_)

# vectorization
tfidf = TfidfVectorizer(analyzer="word", max_features=10000)
X = tfidf.fit_transform(df["Text"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=10)


# declaration of classifiers
svc = LinearSVC()

clf = OneVsRestClassifier(svc)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pickled_model = pickle.load(open('model.pkl', 'rb'))


# predicting the model with samples

def suggestion(abc):
    x=[]
    x.append(abc)
    # tfidf.fit(x)
    xt=tfidf.transform(x)
    clf.predict(xt)
    answ = multilabel.inverse_transform(clf.predict(xt))

    for i in range(0,len(answ)+2):
      for j in answ:  
        if(i==0):  
            # print(j[i])
            tag_Label1.configure(text = j[i])
            i+1
            continue
        elif(i==1):
            tag_Label2.configure(text=j[i])    
            i+1
            continue
        # else:
        #     # print(j[i])
        #     tag_Label3.configure(text=j[i])





root=Tk()
root.geometry("1920x1080+0+0")
# root.resizable(True,True)
# root.state("zoomed")
root.title("Tag Prediction on StackOverflow")
root.config(bg="white")

icon=PhotoImage(file = 'stack.png')
root.iconphoto(False,icon)

imge = PhotoImage(file =r"stack.png")     

mainFrame=Frame(root,bg="white")
mainFrame.place(x=200,y=50,width="966",height="600")

#Background image fit
img=ImageTk.PhotoImage(Image.open("so2.png")) #bg image
label_img = Label(mainFrame,image=img)
label_img.pack()

#title image
image_label =Label(image=imge).place(x=210,y=60,height=70,width=90)

titleLabel=Label(mainFrame,bg="white",fg="#000080",text="Tag Prediction On Stack Overflow",font=("lato",20,"bold"))
titleLabel.place(x=10,y=10,width="946",height="70")

urlLabel=Label(mainFrame,text="Enter your question   :",font=("tahoma",15))
urlLabel.place(x=10,y=140)

urlText=Text(mainFrame,bg="white",fg="#006666")
urlText.place(x=250,y=105,width="600",height="100")
urlText.configure(font=("courier",15,"italic"))

tagLabel=Label(mainFrame,fg="black",text="Suggested Tags :  ",font=("tahoma",15))
tagLabel.place(x=10,y=360)

#tag_Label1,tag_Label2,tag_Label3 to showcase the predicted tags

tag_Label1=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",14))
tag_Label1.place(x=220,y=360)

tag_Label2=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",15))
tag_Label2.place(x=320,y=360)

tag_Label3=Label(mainFrame,fg="black",bg="yellow",text="",font=("courier",15))
tag_Label3.place(x=425,y=360)


def suggest():
    x =urlText.get(1.0,"end-1c")
    if(not x):
        messagebox.showinfo("showinfo", "Enter question")
    else:
        suggestion(x)


Suggest_tagbutton= Button(mainFrame,text = 'Suggest tags',bd ='5',bg='orange',pady=5,command=suggest)
Suggest_tagbutton.pack(side='top')
Suggest_tagbutton.place(x=400,y=500)


root.mainloop()
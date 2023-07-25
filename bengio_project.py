import mysql.connector
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


#Connection to the database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Umvcn10052002@",
    database="restaurant2",
    port=3307
)

mycursor = mydb.cursor()
print("Connection Established")

#Model for classifying the review as positive or negative
def restaurant_model():
    #reading dataset
    dataset = pd.read_csv('Restaurant_Reviews.csv', delimiter='\t')
    #cleaning the text
    nltk.download('stopwords')
    corpus = []
    #cleaning text of 1000 rows
    for i in range(0, 1000):
        #removing punctuations,numbers etc and converting it to lower case
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        #creating portstemmer object for stemming
        ps = PorterStemmer()
        #collecting all stopwords and removing not from it 
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        #filtering out words other then stopwords and then stemming the words
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        # rejoin all string array elements to create back into a string
        review = ' '.join(review)
        #  append each string to create array of clean text
        corpus.append(review)

       
    # Creating the Bag of Words model
    cv = CountVectorizer(max_features=1500)
    # X contains corpus (dependent variable)
    X = cv.fit_transform(corpus).toarray()    
    # y contains answers if review is positive or negative
    y = dataset.iloc[:, 1].values


    #Splitting train and test data
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
    

    #Traing Model using Nave Baye's algorithm
    model = GaussianNB()
    model.fit(X_train, y_train)
    #Predicting the Test set results
    y_pred = model.predict(X_test)
    #checking the accuracy of model
    st.write("The accuracy of the model is:", accuracy_score(y_test,y_pred))
    
    
    #displaying image at the top of the page
    restaurant_image = Image.open(r'''C:\Users\chait\Desktop\bengio_project\restaurant_image3.jpg''')
    restaurant_resize_image = restaurant_image.resize((700, 270))
    st.image(restaurant_resize_image)

    #Heading of the page
    st.title("Restaurant Review Analysis")
    st.markdown("Analyzing the restaurant review as positive or negative")

    #taking user review as input
    user_review = st.text_area("Please enter the review")

    #defining classify button for classifying user review as positive or negative 
    def classify_click():
        
        if user_review != "":
            #cleaning text of user review
            corpus2 = []
            #removing punctuations,numbers etc from user review and converting it to lower case
            new_review = re.sub("[^a-zA-z]", ' ', user_review)
            new_review = new_review.lower()
            new_review = new_review.split()
            #creating portstemmer object
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            #removing not from the stopwords
            all_stopwords.remove('not')
            #performing stemming on user review after filtering words other than stopwords
            new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
            new_review = " ".join(new_review)

            #creating array of clean text
            corpus2.append(new_review)
            cv2 = CountVectorizer(max_features=1500)
            X2 = cv2.fit_transform(corpus + corpus2).toarray()
            my = X2[-1].reshape(1, -1)

            #predicting result with the help of our trained model
            result = model.predict(my)
            if result == 1:
                answer = "Positive"
            else:
                answer = "Negative"
            
            #displaying result whether the review was positive or negative
            st.write(f"The Review was {answer}")
            if result == 1:
                #displaying thubs up is the review was positive
                thumbsup = Image.open(r'''C:\Users\chait\Desktop\bengio_project\thumbs_up.png''')
                thumbsup_resize_image = thumbsup.resize((200, 200))
                st.image(thumbsup_resize_image)
            else:
                #displaying thumbs down if the review was negative
                thumbsdown = Image.open(r'''C:\Users\chait\Desktop\bengio_project\thumbsdown.png''')
                thumbsdown_resize_image = thumbsdown.resize((200, 200))
                st.image(thumbsdown_resize_image)

    #creating a classify button
    button_c = st.button("classify")
    if button_c:
        #calling classify_click function if button was clicked
        classify_click()

def main():
    
    menu = ["Welcome","Login","SignUp","LogOut"]
    option = st.sidebar.selectbox("Menu",menu)
    #Welcome page
    if option=="Welcome":
        st.title("Hey!!Team Bengio Welcomes You To This page Please Log in")  
        welcome_image=Image.open(r'''C:\Users\chait\Desktop\bengio_project\Welcome_image.jpg''')
        welcome_resize_image=welcome_image.resize((680,400))
        st.image(welcome_resize_image)
    
    #signup page
    elif option == "SignUp":
        st.subheader("Signup")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password")
        try:
            if st.button("Create"):
                sql = "insert into user(username,password) values(%s,%s)"
                val = (new_username, new_password)
                mycursor.execute(sql, val)
                mydb.commit()
                st.success("Record Created Successfully!!!")
        except:
            st.warning("This account already exists")
    #login page
    elif option == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            sql = "select password from user where username = %s and password = %s"
            val = (username,password)
            mycursor.execute(sql, val)
            data = mycursor.fetchone()
            # st.subheader(data)
            mydb.commit()
            if data:
                restaurant_model()
            else:
                st.warning("Incorrect Username or password")
    #logout page
    elif (option == "LogOut"):
        st.header("You have Logged Out Successfully!")
        st.header("THANK YOU FOR VISITING OUR PAGE")


if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle


def create_model(data):
    X = data.drop(columns='diagnosis',axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=42)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred) * 100
    clr = classification_report(y_test,y_pred)
    print(f"Accuracy:{accuracy}%\n")
    print("===============CLASSIFICATION REPORT=============\n")
    print(clr)

    return model,scaler

  

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(columns=['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data.diagnosis.apply(lambda x: 1 if x == 'M' else 0)

    return data





def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    with open('model/model.pkl','wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
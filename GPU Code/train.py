from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from joblib import parallel_backend
from sklearn.model_selection import train_test_split

import load

def main():
    dataset = load.load_dataset()
    y = dataset["species"]
    X = dataset[dataset.columns[1:]]

    # Label Encode Everything
    le = LabelEncoder()
    
    y = le.fit_transform(y)

    X['sex'] = le.fit_transform(X['sex'])
    X['location'] = le.fit_transform(X['location'])
    X.columns = X.columns.astype(str)
    
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.30, random_state=50)

    # CPU Bound
    with parallel_backend('threading', n_jobs=8):
        rfc = RandomForestClassifier()
        gnb = GaussianNB()
        knc = KNeighborsClassifier()
        svc = SVC()
        mlp = MLPClassifier()
        

        print("Training Models:")
        print("\tRFC")
        rfc.fit(Xt, yt)
        print("\tGNB")
        gnb.fit(Xt, yt)
        print("\tKNC")
        knc.fit(Xt, yt)
        print("\tSVC")
        svc.fit(Xt, yt)
        print("\tMLP")
        mlp.fit(Xt, yt)

        print("Testing Models:")
        ypred = rfc.predict(Xv)
        print("\tRFC:")
        print(accuracy_score(yv, ypred))
 
        ypred = gnb.predict(Xv)
        print("\tGNB:")
        print(accuracy_score(yv, ypred))

        ypred = knc.predict(Xv)
        print("\tKNC:")
        print(accuracy_score(yv, ypred))
        
        ypred = svc.predict(Xv)
        print("\tSVC:")
        print(accuracy_score(yv, ypred))       

        ypred = mlp.predict(Xv)
        print("\tMLP:")
        print(accuracy_score(yv, ypred))

if __name__ == "__main__":
    main()

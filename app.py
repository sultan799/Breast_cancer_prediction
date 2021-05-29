from flask import Flask,redirect,url_for,render_template,request
import numpy as np
import pandas as pd
import pickle


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()] 
    feature_values=[np.array(input_features)]

    feature_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']

    breast_cancer_cases=pd.DataFrame(feature_values,columns=feature_names)

    output=model.predict(breast_cancer_cases)

    return render_template('benign.html',data=output)

if __name__ == "__main__":
    app.run(debug=True)



'''@app.route('/greet/<string:person>')
def greeting(person):
    return "hello "+person
@app.route('/admin')
def hello_admin():
    return 'hello admin'
@app.route('/guest/<string:guest>')
def hello_guest(guest):
    return "hello "+guest+' as a guest'
@app.route('/user/<string:name>')
def hello_user(name):
    if name=='admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('hello_guest',guest=name))
'''


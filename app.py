from flask import Flask, render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf=pickle.load(file)
file.close()
@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose= int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        inputFeatures=[fever,pain,age,runnyNose,diffBreath]
        infprob=clf.predict_proba([inputFeatures])[0][1]
        print(infprob)
        return render_template('show.html', inf=(infprob*100))
    return render_template('index.html')
    #return 'Probability =' + str(infprob)
if __name__=="_main_":
    app.run(debug=True)
    
from flask import Flask,render_template,request
import pickle
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas

application = Flask(__name__)
app=application


scaler=pickle.load(open("models/scaler_house.pkl","rb"))
model=pickle.load(open("models/model_house.pkl","rb"))


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/submit",methods=["POST"])
def submit():
    if request.method=="POST":
        year=int(request.form.get("year"))
        age=float(request.form.get("age"))
        mrt=float(request.form.get("nearest_mrt_distance"))
        ct=request.form.get("no_of_convenience_stores")
        latitude=float(request.form.get("latitude"))
        longitude=float(request.form.get("longitude"))

        data = {'Year': [year], 'age': [age], 'Nearest MRT Distance': [mrt], 'no of convience stores': [ct], 'Latitude': [latitude], 'Longitude': [longitude]}
        df = pandas.DataFrame(data)

        scaled=scaler.transform(df)

        result=model.predict(scaled)

        return render_template("index.html",result=result)


if __name__=="__main__":
    app.run(host="0.0.0.0")

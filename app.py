from flask import Flask,request,render_template,jsonify

from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData

app=Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            cap_diameter=float(request.form.get("cap_diameter")),
            cap_shape= request.form.get("cap_shape"),
            cap_surface = request.form.get("cap_surface"),
            cap_color = request.form.get("cap_color"),
            does_bruise_or_bleed = request.form.get("does_bleed"),
            gill_attachment= request.form.get("gill_attachment"),
            gill_color = request.form.get("gill_color"),
            stem_height= float(request.form.get("stem_height")),
            stem_width= float(request.form.get("stem_width")),
            stem_color = request.form.get("stem_color"),
            ring_type= request.form.get("ring_type"),
            habitat = request.form.get("habitat"),
            season = request.form.get("season")
        )

        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictionPipeline()

        pred=predict_pipeline.predict(final_data)
        
        if pred[0] == 0:
            result = "Poisonous"
            out_image = "poisonous_image.webp"
        else:
            result = "Edibile"
            out_image = "healthy_mushroom.jpg"

        return render_template("result.html",final_result=result , image = out_image)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)
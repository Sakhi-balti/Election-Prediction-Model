
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

#------------Initialized Flask App --------->
application = Flask(__name__)

app = application

#------------Home Route-------------------->
@app.route('/')
def index():

    return render_template('index.html')
# ---------Predict Route--------------------->
@app.route('/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            District = str(request.form.get('District')),
            ConstituencyTitle = str(request.form.get('ConstituencyTitle')),
            Votes = int(request.form.get('Votes')),
            TotalValidVotes = int(request.form.get('TotalValidVotes')),
            TotalRejectedVotes = int(request.form.get('TotalRejectedVotes')),
            TotalVotes = int(request.form.get('TotalVotes')),
            TotalRegisteredVoters = int(request.form.get('TotalRegisteredVoters')),
            Turnout = float(request.form.get('Turnout')),
            Year = int(request.form.get('Year'))
           
        )
        # call the dataframe function
        pred_df = data.get_data_frame()
        print( pred_df )

        prediction_pipeline =  PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        return render_template('index.html', result = results[0])
    


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
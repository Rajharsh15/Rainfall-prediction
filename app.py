from flask import Flask , render_template, request
from rainfall_model import load_and_clean_data , balance_data, train_rain_model, predict_rain

app =Flask(__name__)

data = load_and_clean_data()
balance_data = balance_data(data)
model , features , train_acc , test_acc = train_rain_model(balance_data)

@app.route('/')
def home():
    return render_template("index.html", features= features, train_acc= train_acc, test_acc = test_acc)
   
@app.route('/predict',methods=['POST'])
def predict():
    try:
        input_values = [float(request.form[f]) for f in features]
        prediction = predict_rain(model, input_values)
    except Exception as e:
        prediction = f"⚠️ Error: {e}"
    return render_template("index.html", features=features, train_acc=train_acc, test_acc=test_acc,
                           input_values=request.form, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('m.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': 
        print(request.form.getlist('mycheckbox')) 
        return ' done '
    return render_template('h.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.getvalues(mycheckbox )]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('h.html', prediction_text='disease $ {}'.format(prediction))
   

if __name__ == "__main__":
    app.run(debug=True)
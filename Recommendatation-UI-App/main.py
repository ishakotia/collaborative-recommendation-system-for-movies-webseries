from flask import Flask, jsonify, request, render_template
from k_means import *
import time
from flask import Response

app = Flask(__name__)


@app.route('/join', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    print(text1)
    text1 = text1.strip() #WhiteSpaces clean
    df = get_recommendation_by_title(text1)
    print(df)
    return Response(df.to_json(orient="records"), mimetype='application/json')

@app.route('/' , methods=['GET', 'POST'])
def home():
   return render_template('home.html')


if __name__ == '__main__':
	print("Starting the Server.....")
	start_time = time.time()
	run_model()
	print("--- %s seconds ---" % (time.time() - start_time))
	# app.run(debug=True)
	app.run()	
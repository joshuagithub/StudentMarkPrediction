import json 
from flask import Flask, render_template, url_for, request
from run_model import runmodel

app= Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/mark/', methods=['POST'])
def mark():
	inp=[0,0,0,0]
	inp[0]= int(request.form['sem1'])
	inp[1]= int(request.form['sem2'])
	inp[2]= int(request.form['sem3'])
	inp[3]= int(request.form['sem4'])
	value= runmodel(inp)
	return render_template('index.html', value= value[0])	

if __name__ == "__main__":
	app.run(debug=True)


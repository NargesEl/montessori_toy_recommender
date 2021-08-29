"""
This is the *C*ontroller part of the MVC

(the templates are the *V*iew part of the MVC)

If we had a database, that would be the *M*odel part
"""
import numpy as np
from flask import Flask, render_template, request
from toy_recommender import get_recommender

app = Flask(__name__)

  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommender')
def toy_recommender():
    print(request.args)  
    
    age1 = int(request.args.get('age1', 0))
    age2 = int(request.args.get('age2', 0))
    age3 = int(request.args.get('age3', 0))
    motor = int(request.args.get('motor', 0))
    language = int(request.args.get('language', 0))
    stem = int(request.args.get('stem', 0))
    emotional = int(request.args.get('emotional', 0))
    social = int(request.args.get('social', 0))

    user_input = np.array([age1, age2, age3, motor, language, stem, emotional, social]).reshape(1, -1)

    top5 = get_recommender(user_input)
    print (top5)
    return render_template('recommendation.html', toy_recommendations=top5)  



if __name__ == "__main__":
   
    app.run(debug=True, port=5000)
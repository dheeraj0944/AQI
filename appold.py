from flask import *
import sqlite3
import os
import base64
import secrets
import threading

connection = sqlite3.connect('database.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS sensors(time TEXT, voltage TEXT, current TEXT, temp TEXT, humidity TEXT, Ac_voltage TEXT)"""
cursor.execute(command)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            session['name'] = name
            
            return render_template('userlog.html')
        else:
            return render_template('signin.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        data = request.form
        values = []
        keys = []
        for key in data:
            keys.append(key)
            values.append(float(data[key]))

        print(keys, values)

        import pickle
        model=pickle.load(open("new_rf.pkl","rb"))
        out=model.predict([values])[0]
        charge_time=out[0]//200
        rull=out[1]//30
        res=1
        # from mapp import create_map
        # create_map()
        
        return render_template('userlog.html' ,res=res, result=out,ct=charge_time,rull=rull,rdis=out[1])
        


    return render_template('userlog.html')


@app.route('/predictdemand', methods=['GET', 'POST'])
def predictdemand():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        data = request.form
        values = []
        keys = []
        for key in data:
            keys.append(key)
            values.append(float(data[key]))

        print(keys, values)

        import pickle 
        ran_for=pickle.load(open('reg.pkl','rb'))
        out=ran_for.predict([values])[0]
        print(out)
        
        
        return render_template('demand.html', Data=[0,0,0,0,0], result=out)
        
        # return render_template('demand.html' ,res=res, result=out,ct=charge_time,rull=rull,rdis=out[1],Data=[0,0,0,0,0])
    return render_template('demand.html',Data=[0,0,0,0,0])



@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

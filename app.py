from flask import Flask, render_template, request

app = Flask(__name__)
print(app)

@app.route('/',methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        #<insert python script here>
        print("hi")
    #exec(open('main.py').read())
    return render_template('index.html')

@app.route('/foo', methods=['GET', 'POST'])
def foo():
    #print("hi")
    exec(open('main.py').read())
    '''execute whatever code you want when the button gets clicked here'''
    return render_template('index.html')
if __name__ == "__main__":
    #exec(open('main.py').read())
    app.run()

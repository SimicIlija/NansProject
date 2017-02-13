from flask import Flask, render_template, request
import os
from utills import manipulatedata as md
from utills.plotting import create

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv'}


@app.route("/", methods=['GET', 'POST'])
def hello(result = None):
    if request.method == 'POST':
        f = request.files['path']
        if f.filename == '':
            return render_template('welcome.html')
        if not(allowed_file(f.filename)):
            return render_template('welcome.html')
        t = request.form['type']
        i = request.form['PolyExp']
        f.save(os.path.join("data/", f.filename))
        processed_data = md.manipulate_data(f.filename, t, int(i))
        result = create(processed_data)
        return render_template('result.html', result=result, name=f.filename[:-4])
    return render_template('welcome.html')


if __name__ == "__main__":
    app.run()

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np


app = Flask(__name__)
app.config["DEBUG"] = False
path = os.getcwd()
''' ------- [Human-Dog App Config] ------- '''
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_DIR = os.path.join(path, 'mysite/static/uploads')
SERVE_DIR = os.path.join(path, 'mysite/static/serve/')
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['SERVE_DIR'] = SERVE_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 224 * 255

# ----------- Views ------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def human_dog():
    uploaded_imgs = os.listdir(app.config['UPLOAD_DIR'])
    # deleting all previous user-uploaded images if any
    if uploaded_imgs != []:
        for img in uploaded_imgs:
            os.remove(os.path.join(app.config['UPLOAD_DIR'], img))
    return render_template("human-dog.html")

@app.route('/', methods=["POST"])
def human_dog_upload():
    if request.method == "POST":
        if 'files[]' not in request.files:
            flash('No files selected')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        if request.form.get('Submit'):
            flash('File(s) successfully uploaded. Processing...')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_DIR'], filename))

        return redirect(url_for("human_dog_results"))

# @app.route('/tester')
# def tester():
#     uploaded_imgs = ['uploads/'+im for im in os.listdir(app.config['UPLOAD_FOLDER']) if '.DS' not in im]
#     num_files = len(uploaded_imgs)
#     return render_template("tester.html", num_files=num_files, uploaded_imgs=uploaded_imgs)

@app.route('/human-dog-results')
def human_dog_results():
    uploaded_imgs = ['uploads/' + im for im in os.listdir(app.config['UPLOAD_DIR']) if '.DS' not in im]
    file_len = len(uploaded_imgs)
    result = {i: str(np.load(os.path.join(app.config['SERVE_DIR'], 'results.npy'))[i])
            for i in range(file_len)}

    return render_template("human_dog_results.html", file_len=file_len, uploaded_imgs=uploaded_imgs, result=result)

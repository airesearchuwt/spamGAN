import flask
from flask_cors import CORS, cross_origin
import os
import json
from flask import request, jsonify, abort
import spamGAN_train_DCG_gpt2
import gpt2_tokenizer
from werkzeug.utils import secure_filename


app = flask.Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 5
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = '/home/ubuntu/spamGAN-as-a-Service/opspam/files'


@app.route('/ping', methods=['GET'])
def ping():
    """
        Determine if the container is working and healthy. In this sample container, we declare
        it healthy if we can load the model successfully.
    """
    # You can insert a health check here
    # status = 200 if health else 404
    return flask.Response(response='The server is connected!\n', status=200, mimetype='application/json')

@app.route('/singleinference', methods=['POST'])
def predict():
    data = str(request.data)
    if data == None:
        abort(400)
    if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt')):
            os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt'))
    if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt')):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt'))
    if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt')):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt'))
    with open(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt'), 'w') as f:
        f.write(data + '\n')
        f.write('This is a fake review serving as a placeholder.')
    with open(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt'), 'w') as f:
        f.write('0' + '\n')
        f.write('0' + '\n')
    gpt2_tokenizer.make_bpe_file(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt'), os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt'))
    res = spamGAN_train_DCG_gpt2.start()
    os.system('rm /tmp/event*')
    return json.dumps(res)


@app.route('/inference', methods=['POST'])
def predicts():
    """
        Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
        it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
        just means one prediction per line, since there's a single column.
    """
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        print(filename)
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt')):
            os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_review.txt'))
        if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt')):
            os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt'))
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    
    gpt2_tokenizer.make_bpe_file(os.path.join(app.config['UPLOAD_PATH'], filename), os.path.join(app.config['UPLOAD_PATH'], 'test_review_bpe.txt'))
    
    with open(os.path.join(app.config['UPLOAD_PATH'], filename), 'r') as f:
        content = f.readlines()
    
    if os.path.exists(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt')):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt'))
    
    with open(os.path.join(app.config['UPLOAD_PATH'], 'test_label.txt'), 'w') as f:
        for i in range(len(content)):
            f.write('0' + '\n')
    res = spamGAN_train_DCG_gpt2.start()
    os.system('rm /tmp/event*')
    return json.dumps(res)


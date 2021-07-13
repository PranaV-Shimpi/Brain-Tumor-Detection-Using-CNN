import os
from os.path import join, dirname, realpath
import numpy as np

#keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Flask utils
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request,redirect

# Define a flask app
new_model = load_model('brain_tumor_model.h5')
app = Flask(__name__)

# Model saved with Keras model.save()
@app.route('/',methods=['GET'])
def upload():
	# Main page
    return render_template('upload.html')

@app.route('/uploader',methods=['POST','GET'])
def uploader():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		path = os.path.dirname(__file__)
		basepath=os.path.normpath(path)

		file_path=os.path.join(basepath,'uploads',secure_filename(f.filename)).replace("\\","/")
		f.save(file_path)
		img5 = image.load_img(file_path , target_size=(240,240))
		images5 = image.img_to_array(img5)
		images5 = np.expand_dims(images5,axis=0)
		prediction = new_model.predict(images5)

		#result = prediction

		if prediction <=0.5 :
		    #print("Tumor Not Detected")
		    #plt.title('No')
		    #return "The Person has no Brain Tumor"
			return render_template('no.html')
			# url = 'https://www.yahoo.com/'
			# return redirect(url, code=307)
		else:
		    #print("Tumor Detected")
		    #plt.title('Yes')
		    #return 'file uploaded succesfully'
			 #sql = 'https://index-report.netlify.app/'
			 #return redirect(sql, code=300)
			return render_template('Yes.html')

if __name__ == '__main__':
    app.run(debug=False)

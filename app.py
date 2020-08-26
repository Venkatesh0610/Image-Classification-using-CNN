import os
import numpy as np
from flask import Flask,request,render_template
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
#global graph
#graph=tf.compat.v1.get_default_graph()


app=Flask(__name__)
model=load_model("flowers.h5")

@app.route("/")
def home():
    return render_template("index6.html")

@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        pred=model.predict_classes(x)
        #print("prediction",pred)
        index=["daisy","rose","sunflower"]
        result=str(index[pred[0]])
        return result
    return None

#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run()
    #app.run(host='0.0.0.0', port=port)
            
            
            
            

        
             

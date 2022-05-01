from io import StringIO
from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open(r'rf_reg.pkl', 'rb'))

app = Flask(__name__)



# @app.route('/', methods=['POST','GET'])
# def man():
#     # return "hello"
#     


@app.route('/', methods=['POST','GET'])
def home():
    if request.method=="POST":
        input_dict = request.form.to_dict()
        input_values = input_dict.values()
        input_values = list(map(float, list(input_values)))
        input_values = np.array(input_values)
        print(input_values.shape)
        input_values = input_values.reshape(1, -1)
        prediction = model.predict(input_values)[0]
        return str(prediction)
        
    # arr = [ [ np.arr(data1, data2 ........ datan)]]
    # arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14]])
    # arr=arr.reshape(1,-1)
    
    # pred = model.predict(arr)[0]
   # print(type(pred))
    # if pred !='\0':
    #     pred=str(pred)
    #     return  pred
    #   #  return render_template('end2.html')
    #     #return "helloworld"
    else:
        return render_template('index_1.html')  
    

if __name__ == "__main__":
    app.run(debug=False)
    
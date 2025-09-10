import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file,send_from_directory
import io
import os


dataset = None
new_data = None
X_train = None
X_test = None
y_train = None
y_test = None


app = Flask(__name__)

print(app.root_path)
save_dir = os.path.join(app.root_path, "static", "images")
os.makedirs(save_dir, exist_ok=True)


def process_one(df):
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values(by=['Date'], inplace=True, ascending=True)
    df.set_index("Date", inplace = True)

    x = pd.DataFrame(df['Sales'])

    return x


def process_two():
    global dataset
    global X_train
    global X_test
    global y_train
    global y_test

    dataset['Sale_LastMonth'] =dataset['Sales'].shift(+30)
    dataset['Sale_2MonthsBack'] =dataset['Sales'].shift(+60)
    dataset['Sale_3MonthsBack'] =dataset['Sales'].shift(+90)

    dataset.dropna(inplace =True)

    org = dataset

    x1,x2,x3,y = dataset['Sale_LastMonth'],dataset['Sale_2MonthsBack'],dataset['Sale_3MonthsBack'],dataset['Sales']
    x1,x2,x3,y = np.array(x1),np.array(x2),np.array(x3),np.array(y)
    x1,x2,x3,y = x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
    final_x = np.concatenate((x1,x2,x3),axis = 1)

    X_train,X_test,y_train,y_test = train_test_split(final_x,y,test_size =0.12,random_state = 0)


def createpdata(num_days):
    global dataset
    last_index_tail = dataset.index[-1]
    date_range = pd.date_range(start=last_index_tail, periods=num_days, freq='D')

    data = {
        'Sale_LastMonth': None,
        'Sale_2MonthsBack': None,
        'Sale_3MonthsBack': None,
        'Sales' : None
    }

    pred_df = pd.DataFrame(data, index=date_range)

    i =1
    date = last_index_tail
    while i<=num_days:
        lstmnth = date -timedelta(days=30)
        lst2mnth = date -timedelta(days=60)
        lst3mnth = date -timedelta(days=90)
        pred_df.at[date,'Sale_LastMonth']  = dataset.at[lstmnth,'Sales']
        pred_df.at[date,'Sale_2MonthsBack']  = dataset.at[lst2mnth,'Sales']
        pred_df.at[date,'Sale_3MonthsBack']  = dataset.at[lst3mnth,'Sales']

        date = date + timedelta(days = 1)
        i = i+1

    return pred_df


def createfinal_df():

    global dataset
    global X_train
    global X_test
    global y_train
    global y_test
    global yyy

    z1,z2,z3 = yyy['Sale_LastMonth'],yyy['Sale_2MonthsBack'],yyy['Sale_3MonthsBack']
    z1,z2,z3 = np.array(z1),np.array(z2),np.array(z3)
    z1,z2,z3 = z1.reshape(-1,1),z2.reshape(-1,1),z3.reshape(-1,1)
    z_test = np.concatenate((z1,z2,z3),axis = 1)

    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train, y_train)
    knn_future_pred=knn_reg.predict(z_test)

    yyy['Sales'] = knn_future_pred

    yyy = yyy[['Sales','Sale_LastMonth', 'Sale_2MonthsBack', 'Sale_3MonthsBack']]
    final_df = pd.concat([dataset, yyy])

    return final_df


def plot_new_data(df):

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Sales'], label='Sales', color='blue')
    plt.fill_between(df.index, df['Sales'] - 1.96*df['Sales'].std(), df['Sales'] + 1.96*df['Sales'].std(), color='blue', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer


def generate_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='Sales', data=df, color='lightblue')
    sns.stripplot(y='Sales', data=df, color='red', jitter=True, size=4, alpha=0.6)

    plt.xlabel('Sales')
    plt.title('Box Plot with Jittered Data Points')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def generate_trendplot(df):

    rcParams['figure.figsize'] = 14, 10

    decomposition = sm.tsa.seasonal_decompose(df, model='additive')
    decomposition.plot()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def generate_rfplot():
    global dataset
    global X_train
    global X_test
    global y_train
    global y_test

    model = RandomForestRegressor(n_estimators=100,max_features=3, random_state=1)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(pred,label='Random_Forest_Predictions')
    plt.plot(y_test,label='Actual Sales')
    plt.legend(loc="upper left")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def generate_knnplot():

    global dataset
    global X_train
    global X_test
    global y_train
    global y_test

    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train, y_train)
    knn_pred=knn_reg.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(knn_pred,label='KNN_Predictions')
    plt.plot(y_test,label='Actual Sales')
    plt.legend(loc="upper left")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer


def generate_xgbplot():
    global dataset
    global X_train
    global X_test
    global y_train
    global y_test

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    num_round = 100
    bst = xgb.train(params, dtrain, num_round)

    y_pred = bst.predict(dtest)

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred,label='XGB_Predictions')
    plt.plot(y_test,label='Actual Sales')
    plt.legend(loc="upper left")
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer


def generate_finalplot():
    global final_df
    global yyy

    plt.figure(figsize=(20, 8))
    
    start_idx = int(len(final_df) * 0.85)

    x =final_df.iloc[start_idx:]
    x['Sales'].plot(label= 'Past Data',linestyle='-')
    yyy['Sales'].plot(color ='red',label = 'The Future',marker='o', markersize=8, linestyle='-',linewidth=2,)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)  
    plt.tight_layout()  
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            global dataset
            dataset= pd.read_csv(uploaded_file)

            df_desc = dataset.describe()
            df_head = dataset.head()
            processed_data_html = df_desc.to_html(classes='data-table', table_id="summary")
            head_data_html = df_head.to_html(classes='data-table', table_id="head")
            
            dataset = process_one(dataset)

            normalplot = plot_new_data(dataset)
            normalplot_path =f"{save_dir}/normalplot.png"  
            with open(normalplot_path, 'wb') as f:
                f.write(normalplot.getbuffer())
            
            plot1= "/static/images/normalplot.png"

            boxplot = generate_boxplot(dataset)
            boxplot_path = f"{save_dir}/boxplot.png"  
            with open(boxplot_path, 'wb') as f:
                f.write(boxplot.getbuffer())
            plot2="/static/images/boxplot.png"

            dataset =  pd.DataFrame(dataset['Sales'].resample('D').mean())
            dataset = dataset.interpolate(method='linear')

            trendplot = generate_trendplot(dataset)
            trendplot_path = f"{save_dir}/trendplot.png"  
            with open(trendplot_path, 'wb') as f:
                f.write(trendplot.getbuffer())
            plot3="/static/images/trendplot.png"

            process_two()

            return render_template('results.html',tables=[processed_data_html, head_data_html],plot1 =plot1,plot2 = plot2,plot3 =plot3)

    return render_template('index.html')


@app.route('/additional_plots')
def additional_plots():

    rfplot = generate_rfplot()
    rfplot_path =f"{save_dir}/rfplot.png"  
    with open(rfplot_path, 'wb') as f:
        f.write(rfplot.getbuffer())
    plot4 ="/static/images/rfplot.png"

    knnplot = generate_knnplot()
    knnplot_path = f"{save_dir}/knnplot.png"  
    with open(knnplot_path, 'wb') as f:
        f.write(knnplot.getbuffer())
    plot5 ="/static/images/knnplot.png"

    xgbplot = generate_xgbplot()
    xgbplot_path = f"{save_dir}/xgbplot.png"  
    with open(xgbplot_path, 'wb') as f:
        f.write(xgbplot.getbuffer())
    plot6 ="/static/images/xgbplot.png"
    
    return render_template('additional_plots.html',plot4 =plot4,plot5 =plot5,plot6=plot6)


@app.route('/forecast', methods=['POST'])
def forecast():
    global yyy
    global final_df
    number = request.form['number_input']
    yyy = createpdata(int(number))
    final_df = createfinal_df()

    finalplot = generate_finalplot()
    finalplot_path = f"{save_dir}/finalplot.png"  
    with open(finalplot_path, 'wb') as f:
        f.write(finalplot.getbuffer())

    fplot ="/static/images/finalplot.png"
    
    return render_template('forecast.html', fplot=fplot)


@app.route('/process_input', methods=['POST'])
def process_input():
    
    global final_df
    global yyy

    fplot ='/static/images/finalplot.png'
    
    input_data = request.form['input_data']
    processed_output =final_df.at[input_data,'Sales']
    return render_template('forecast.html',processed_output = processed_output,fplot = fplot)


if __name__ == '__main__':
    app.run(debug=True)










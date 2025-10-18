import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")



def load_data(file_path):
    return pd.read_csv(file_path)

def dimensions(df):
    return df.shape

def fill_missing_values(data):
    missing_data = data.isnull().sum()
    Columns_missing = missing_data[missing_data > 0].index
    if len(Columns_missing) == 0:
        print("No missing values found.")
        return data
    else:
        for column in Columns_missing:
            if data[column].dtype in ['float64', 'int64']:
                mean_value = data[column].mean()
                data[column].fillna(mean_value, inplace=True)
            else:
                mode_value = data[column].mode()[0]
                data[column].fillna(mode_value, inplace=True)
        return data
    
def Split_Data(Data):
    X = Data.drop(columns=['Delivery_Time_min', 'Order_ID','Vehicle_Type'], axis=1)
    y = Data['Delivery_Time_min']
    return X, y

def Split_train_test(Data):
    X, y = Split_Data(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def encode_with_onehot(Data):
    Object_Data = Data.select_dtypes(include=['object'])
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_array = encoder.fit_transform(Object_Data)
    encoded_cols = encoder.get_feature_names_out(Object_Data.columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=Data.index)
    Data = pd.concat([Data.drop(columns=Object_Data.columns), encoded_df], axis=1)
    return Data

def Scaler_Standerd(Data):
    Numeric_Data = Data.select_dtypes(include= ['float64', 'int64'])
    scaler = StandardScaler()
    Data[Numeric_Data.columns] = scaler.fit_transform(Numeric_Data)
    return Data



def Train_Model(X_train, X_test, y_train, y_test):
    Final_Result_Score = {}
    Models = ['RandomForestRegressor', 'Support Vector Regressor']
    for mod in Models:

        if mod == 'RandomForestRegressor':
            model = RandomForestRegressor(max_depth=10, min_samples_leaf=2, n_estimators=400)
        elif mod == 'Support Vector Regressor':
            model = SVR(C=10, gamma='scale', kernel='rbf')

        model.fit(X_train, y_train)
        y_pred_RF = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_RF)   
        mae = mean_absolute_error(y_test, y_pred_RF)
        Final_Result_Score[mod] = { 'r2_score': r2, 'MAE': mae }
    return Final_Result_Score

def get_max_score(Final_Score_Dict):
    best_model = None
    best_r2 = float('-inf')
    for model, scores in Final_Score_Dict.items():
        if scores['MAE'] > best_r2:
            best_MAE = scores['MAE']
            best_model = model
    return best_model, best_MAE


My_Data = load_data('MyRealData.csv')
My_Data = fill_missing_values(My_Data)
X,y = Split_Data(My_Data)

X = Scaler_Standerd(X)
X = encode_with_onehot(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Results = Train_Model(X_train, X_test, y_train, y_test)

best_model, best_MAE = get_max_score(Results)



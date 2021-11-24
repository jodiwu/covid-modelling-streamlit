# importing all the required packages
import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm

covid_df = pd.read_excel('data/Covid-19 SG.xlsx', skiprows=range(1, 557))
traffic_df = pd.read_excel("./data/AirTraffic.xlsx")
sgx_df = pd.read_csv('./data/HistoricalPrices.csv', skipfooter=43,engine='python')

# Data Preprocessing of the COVID Data
# processing NaN values
covid_df = covid_df[pd.notnull(covid_df['Phase'])]
covid_df['Cumulative Individuals Vaccinated'] = covid_df['Cumulative Individuals Vaccinated'].fillna(0)

sg_population = 5.686 * 10**6
covid_df['Percentage Vaccinated'] = covid_df['Cumulative Individuals Vaccinated'].divide(sg_population)

# convert to datetime object for easier processing
covid_df['Date'] = pd.to_datetime(covid_df['Date'])

# convert date to numerical value
covid_df['Date'] = covid_df['Date'].map(dt.datetime.toordinal)

# find 7 days Moving Average as another feature
covid_df['7 days Moving Average'] = covid_df['Daily Confirmed'].rolling(window=7).mean()

# replace NaN values
covid_df['7 days Moving Average'].fillna(covid_df['Daily Confirmed'], inplace=True)

# covid_df['Date'] = pd.to_datetime(covid_df['Date'],format="%d/%m/%Y")

# getting useful columns
# new_columns = ['Date','Daily Confirmed', 'Still Hospitalised','Phase','7 days Moving Average','Percentage Vaccinated']
new_columns = ['Date', 'Still Hospitalised','Phase','7 days Moving Average', 'Percentage Vaccinated']
covid_df = covid_df.reindex(columns=new_columns)
covid_df = covid_df[new_columns]

# preprocessing SGX data for the target
# convert to datetime object for easier processing
sgx_df['Date'] = pd.to_datetime(sgx_df['Date'])

# convert date to numerical value and rename column
sgx_df['Date'] = sgx_df['Date'].map(dt.datetime.toordinal)
sgx_df = sgx_df.rename(columns={' Open':'STI Price'})

# narrowing down to just open instead of high, low and close
sgx_df = sgx_df[['Date','STI Price']]

# merge with dataset on date
merged_df = pd.merge(covid_df,sgx_df, how='inner', on='Date')

# Preprocessing Air Traffic data
traffic_df['Date'] = pd.to_datetime(traffic_df['Date'])
traffic_df['Date'] = traffic_df['Date'].map(dt.datetime.toordinal)
traffic_df = traffic_df.rename(columns={'No. of Passengers':'Passengers'})

traffic_df = traffic_df[['Date','Passengers']]

merged_df = pd.merge(merged_df, traffic_df, on='Date')

# Encoding categorical data which is the phase
one_hot_cont = pd.get_dummies(merged_df['Phase'], prefix="Phase")
df = merged_df.join(one_hot_cont)
del df["Phase"]
del df['Phase_Stabilisation Phase']

##### BUILDING MODEL #####
def normalize_z(dfin):
    dfout = (dfin - dfin.mean(axis=0))/dfin.std(axis=0)
    return dfout

def normalize_minmax(dfin):
    dfout = (dfin - dfin.min(axis=0))/(dfin.max(axis=0)-dfin.min(axis=0))     
    return dfout

def get_features_targets(df, feature_names, target_names):
    df_feature = df.loc[:,feature_names]
    df_target = df.loc[:,target_names]
    return df_feature, df_target

def prepare_feature(df_feature):
    n = df_feature.shape[0]
    ones = np.ones(n).reshape(n,1)
    return np.concatenate((ones,df_feature.to_numpy()),axis = 1)

def prepare_target(df_feature):
    return df_feature.to_numpy()

def predict(df_feature, beta):
    df_feature = normalize_z(df_feature)
    X = prepare_feature(df_feature)
    return predict_norm(X, beta)

def predict_norm(X, beta):
    return np.matmul(X,beta)

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    np.random.seed(random_state)
    TestSize = int(test_size*len(df_feature))
    testchoice = np.random.choice(len(df_feature),size = TestSize, replace = False)
    remainder = []
    for i in df_feature.index:
        if i not in testchoice:
            remainder.append(i)
    trainchoice = np.random.choice(remainder, size = len(remainder),replace = False)
    df_feature_train = df_feature.iloc[trainchoice]
    df_target_train = df_target.iloc[trainchoice]
    df_feature_test = df_feature.iloc[testchoice]
    df_target_test = df_target.iloc[testchoice]
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def poly_features(df_feature, colname, colname_transformed, degree=2):
    col = df_feature[colname]
    df_feature[colname_transformed] = np.power(col, degree)
    return df_feature

class LinearRegression:
    def __init__(self):
        self.beta = None
        self.J_storage = None
    
    def fit(self,X,y,iterations,alpha):
        """
        Fit Linear model with the datasets using gradient descent.

        Parameters
            X : Training data.

            y : Target Values
        """
        n,p = X.shape
        self.J_storage = np.zeros(iterations)
        beta = np.random.randn(p,1) / np.sqrt(n) #Weight Initialization
        for i in range(iterations):
            y_pred = self.predict_norm(X,beta)
            error = y_pred - y
            beta = beta - (alpha/n) * np.matmul(X.T,error)
            cost = self.compute_cost(X,y,beta)
            self.J_storage[i] = cost
        self.beta = beta
        return self.beta

    def predict(self,df_feature,normalisation=None):
        """
        Predict the target values
        
        Parameters
            df_feature : Test data.
            normalisation : Normalisation method for the test data before prediction. Default is None.

        Returns
            y_pred : Predicted target values.

        """
        df_feature = self.prepare_feature(df_feature)
        if normalisation == 'standard':
            return self.predict_norm(self.normalize_z(df_feature), self.beta)
        elif normalisation == 'min-max':
            return self.predict_norm(self.normalize_minmax(df_feature), self.beta)
        else:
            return self.predict_norm(df_feature, self.beta)
    
    def predict_norm(self,X, beta):
        return np.matmul(X,beta)
    
    def compute_cost(self,X, y, beta):
        m = X.shape[0]
        y_pred = np.matmul(X,beta)
        error = y_pred - y
        J = (1/(2*m))*np.matmul(error.T,error)
        return J[0][0] 

    def normalize_z(dfin):
        dfout = (dfin - dfin.mean(axis=0))/dfin.std(axis=0)
        return dfout
    
    def normalize_minmax(dfin):
        dfout = (dfin - dfin.min(axis=0))/(dfin.max(axis=0)-dfin.min(axis=0))        
        return dfout
        
    def prepare_feature(self,df_feature):
        n = df_feature.shape[0]
        ones = np.ones(n).reshape(n,1)
        return np.concatenate((ones,df_feature.to_numpy()),axis = 1)

class Evaluate:
    def __init__(self,target,prediction):
        self.target = target
        self.prediction = prediction
    
    def mean_squared_error(self):
        n = self.target.shape[0]
        error = self.target-self.prediction
        mse = np.matmul(error.T,error)/n
        return mse

    def r2_score(self):
        rss = np.sum((self.prediction - self.target) ** 2)
        tss = np.sum((self.target-self.target.mean()) ** 2)
        r2 = 1 - (rss / tss)
        return r2

    def adjusted_r2_score(self):
        r2 = self.r2_score()
        n = self.target.shape[0]
        k = self.target.shape[1]
        adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)       
        return adj_r2
    
    def evaluate(self):
        r2 = self.r2_score()
        adjusted_r2 = self.adjusted_r2_score()
        mse = self.mean_squared_error()
        print(f"mse : {mse}\n")
        print(f"r2 : {r2}")
        print(f"adjusted r2 : {adjusted_r2}")
    
    def __str__(self) -> str:
        return self.evaluate()

##### POLYNOMIAL REGRESSION #####
# Extract feature and target
columns = ['Date', 'Still Hospitalised', '7 days Moving Average',
       'Percentage Vaccinated','Phase_Phase 2 (Heightened Alert)','Passengers', 'Phase_Preparatory Stage']
poly_transform = ["7 days Moving Average","Still Hospitalised","Percentage Vaccinated"]

df_features, df_target = get_features_targets(df,columns,["STI Price"])
df_features = poly_features(df_features, poly_transform, poly_transform,3)

# normalize features
df_features = normalize_z(df_features)

df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)

# change to numpy array and append column for feature
X = prepare_feature(df_features_train) # concatenating for the y intercept
target = prepare_target(df_target_train)

model = LinearRegression()
iterations = 10000
alpha = 0.01
model.fit(X,target,iterations,alpha)
print(f"Beta : {model.beta}")
pred = model.predict(df_features_test)
plt.plot(model.J_storage)
score = Evaluate(df_target_test,pred).evaluate()

# st.write(plt.scatter(df_features_test["Date"], df_target_test))
# st.write(plt.scatter(df_features_test["Date"], pred))


##### DISPLAY #####
st.set_page_config(layout="wide")

col1, col2, col3, col4, col5, col6 = st.columns(6)
# home_link = 'http://github.com'
# with col2:
#     if st.button("Home"):
#         st.markdown(home_link, unsafe_allow_html=True)

# with col5:
#     if st.button("Predict"):
#         st.markdown(home_link, unsafe_allow_html=True)

with col2:
    home_link = '[Home](https://covid-backend-modelling.herokuapp.com/)'
    st.markdown(home_link, unsafe_allow_html=True)
with col6:
    predict_link = '[Predict](https://covid-backend-modelling.herokuapp.com/view)'
    st.markdown(predict_link, unsafe_allow_html=True)

st.title("Predicting Singapore's Straits Times Index (STI) growth amidst COVID-19 using Polynomial Regression")
st.write("")
st.write("**Dataset that we will be using:**")
st.dataframe(df)
st.write("")
st.write("For our final model, we are making use of all the features (excluding STI Price) shown in the table above, and the STI Price is the target. In our model, we perform a polynomial transformation of degree 3 on the variables **7 days Moving Average**, **Still Hospitalised** and **Percentage Vaccinated**.")
st.write("")
st.write("**Visualisation of STI against each feature:**")
# fig = plt.scatter(df_features_test["Date"], df_target_test)
# st.pyplot(fig)
# st.pyplot(plt.scatter(df_features_test["Date"], pred))

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure(figsize=(5, 5))
    # plt.scatter(df_features_train, df_target_test,lw=0.1,color="r",label="validation")
    # plt.scatter(df_features_test, pred,lw=0.1,color="b",label="prediction")
    plt.scatter(df_features_test["Date"], df_target_test, color="r", label="validation")
    plt.scatter(df_features_test["Date"], pred, color="b", label="prediction")
    plt.title("STI against Date", fontsize=10)
    plt.xlabel("Date",fontsize=6)
    plt.ylabel("Predicted STI",fontsize=6)
    # plt.xlim(int(x_min), int(x_max)+5)
    # plt.ylim(int(y_min), int(y_max)+5)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(5, 5))
    plt.scatter(df_features_test["Still Hospitalised"], df_target_test, color="r", label="validation")
    plt.scatter(df_features_test["Still Hospitalised"], pred, color="b", label="prediction")
    plt.title("STI against Hospitalised", fontsize=10)
    plt.xlabel("Still Hospitalised",fontsize=6)
    plt.ylabel("Predicted STI",fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3 = plt.figure(figsize=(5, 5))
    plt.scatter(df_features_test["7 days Moving Average"], df_target_test, color="r", label="validation")
    plt.scatter(df_features_test["7 days Moving Average"], pred, color="b", label="prediction")
    plt.title("STI against Moving Average", fontsize=10)
    plt.xlabel("7 days Moving Average",fontsize=6)
    plt.ylabel("Predicted STI",fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig3)

with col4:
    fig4 = plt.figure(figsize=(5, 5))
    plt.scatter(df_features_test["Percentage Vaccinated"], df_target_test, color="r", label="validation")
    plt.scatter(df_features_test["Percentage Vaccinated"], pred, color="b", label="prediction")
    plt.title("STI against Percentage Vaccinated", fontsize=10)
    plt.xlabel("Percentage Vaccinated",fontsize=6)
    plt.ylabel("Predicted STI",fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig4)

col5, col6 = st.columns(2)

with col5:
    fig5 = plt.figure(figsize=(5, 5))
    plt.scatter(df_features_test["Passengers"], df_target_test, color="r", label="validation")
    plt.scatter(df_features_test["Passengers"], pred, color="b", label="prediction")
    plt.title("STI against No. of Passengers", fontsize=10)
    plt.xlabel("Passengers",fontsize=6)
    plt.ylabel("Predicted STI",fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig5)

# with col6:
#     fig6 = plt.figure(figsize=(5, 5))
#     plt.scatter(df_features_test["Phase_Phase 2 (Heightened Alert)"], df_target_test, color="r", label="validation")
#     plt.scatter(df_features_test["Phase_Phase 2 (Heightened Alert)"], pred, color="b", label="prediction")
#     plt.title("STI against Phase_Phase 2 (Heightened Alert)", fontsize=10)
#     plt.xlabel("Phase_Phase 2 (Heightened Alert)",fontsize=6)
#     plt.ylabel("Predicted STI",fontsize=6)
#     plt.legend(fontsize=6)
#     plt.tick_params(labelsize=6)
#     st.pyplot(fig6)
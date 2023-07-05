import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

Breast_cancer_model = pickle.load(open('Breast_cancer_model.sav',"rb"))

scaler = StandardScaler()

# reading csv file as the pandas dataframe
data = pd.read_csv("cancer.csv")
X1 = data.drop(['id','Unnamed: 32','area_worst','radius_worst','area_mean','radius_mean','perimeter_worst','perimeter_mean','fractal_dimension_mean','fractal_dimension_worst','smoothness_mean','smoothness_worst','diagnosis'], axis=1)
# features are removed using vif
X_scaled = scaler.fit_transform(X1)

# setting page title
st.set_page_config(page_title='Breast Cancer Prediction',layout="wide")

with st.sidebar:
    selected = option_menu("Breast Cancer Prediction",["Breast Cancer Prediction",'INSIGHTS'],icons = ["person","toggles"],default_index=0)
  
if (selected == "Breast Cancer Prediction"):
    st.title("Breast Cancer Prediction using MACHIENE LEARNING")
    col1,col2,col3,col4,col5 = st.columns((1,1,1,1,1))
    with col1:
        radius_mean = st.text_input('radius mean')
    with col2:
        texture_mean = st.text_input('texture mean')
    with col3:
        perimeter_mean = st.text_input('perimeter mean')
    with col4:
        area_mean = st.text_input('area mean')
    with col5:
        smoothness_mean = st.text_input('smoothness mean')
    with col1:
        compactness_mean = st.text_input('compactness mean')
    with col2:
        concavity_mean = st.text_input('concavity_mean')
    with col3:
        concave_points_mean = st.text_input('concave points mean')
    with col4:
        symmetry_mean = st.text_input('symmetry mean')
    with col5:
        fractal_dimension_mean = st.text_input('fractal dimension mean')
    with col1:
        radius_se = st.text_input('radius standarad error')
    with col2:
        texture_se = st.text_input('texture standarad error')
    with col3:
        perimeter_se = st.text_input('perimeter standarad error')
    with col4:
        area_se = st.text_input('area standarad error')
    with col5:
        smoothness_se = st.text_input('smoothness standarad error')
    with col1:
        compactness_se = st.text_input('compactness standarad error')
    with col2:
        concavity_se = st.text_input('concavity standarad error')
    with col3:
        concave_points_se = st.text_input('concave_pts standarad error')
    with col4:
        symmetry_se = st.text_input('symmetry standarad error')
    with col5:
        fractal_dimension_se = st.text_input('fractal_dimn standarad error')
    with col1:
        radius_worst = st.text_input('radius_worst')
    with col2:
        texture_worst = st.text_input('texture_worst ')
    with col3:
        perimeter_worst = st.text_input('perimeter_worst')
    with col4:
        area_worst = st.text_input('area_worst')
    with col5:
        smoothness_worst = st.text_input('smoothness_worst')
    with col1:
        compactness_worst = st.text_input('compactness_worst')
    with col2:
        concavity_worst = st.text_input('concavity_worst')
    with col3:
        concave_points_worst = st.text_input('concave_points_worst')
    with col4:
        symmetry_worst = st.text_input('symmetry_worst')
    with col5:
        fractal_dimension_worst = st.text_input('fractal_dimension_worst')
   
    
    
    
    
    Breast_Cancer_diagnosis = ''
    if st.button('Breast Cancer Test Result'):
        X_scaled = scaler.transform([[float(texture_mean),float(compactness_mean),float(concavity_mean),float(concave_points_mean),
                   float(symmetry_mean), float(radius_se), float(texture_se),float(perimeter_se), float(area_se),
                   float(smoothness_se), float(compactness_se),float(concavity_se), float(concave_points_se), 
                   float(symmetry_se),float(fractal_dimension_se), float(texture_worst), float(compactness_worst),
                   float(concavity_worst), float(concave_points_worst),float(symmetry_worst)]])
        Breast_Cancer_prediction = Breast_cancer_model.predict(X_scaled)
        if (Breast_Cancer_prediction[0] == 0):
            Breast_Cancer_diagnosis = 'THE PERSON CELL IS  BENIGN IS NOT HARMFUL'
        elif (Breast_Cancer_prediction[0] == 1):
            Breast_Cancer_diagnosis = 'THE PERSON CELL IS MALIGNANT IS HARMFUL'
    st.success(Breast_Cancer_diagnosis)  

def heatmap(df):
    plt.subplots(figsize=(20,10))
    sns.heatmap(df, fmt='.1f',annot=True,cbar=True)
    
if (selected == 'INSIGHTS'): 
        
    Data_mean = data.groupby('diagnosis').mean()
    st.write("We can clearly see that for most of the features, the mean values are higher for Malignant(1) cases and lower for Benign(0) cases")
    st.dataframe(Data_mean)
    
    
    # Let's encode target column B  'Benign' as 0[not harmful] M 'Malignant' as 1[harmful]
    diagnosis_dict = {'M':1, 'B':0}
    data['diagnosis'] = data['diagnosis'].map(diagnosis_dict)
    df = data.corr()
    # Display the heatmap
    st.subheader("Heatmap")
    heatmap(df)
    st.pyplot()
    
    st.write("We can clearly see that for most of the features are high correlation multicolinearity  so removing the high corelated features")

    import math

    sizes = data['radius_mean'].apply(lambda x : math.pi*(x*x))

    # Create a scatter plot with varying circle sizes
    sns.scatterplot(x=data['radius_mean'],y=data['texture_mean'] , hue=data['diagnosis'], s=sizes, alpha=0.5)

    # Set the title and labels
    plt.title("Scatter Plot with Varying Circle Sizes")
    plt.xlabel("radius_mean")
    plt.ylabel('texture_mean')

    # Display the plot
    plt.show()
    st.pyplot()

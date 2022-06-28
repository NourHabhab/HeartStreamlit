#Importing Libraries 
import streamlit as st
import pandas as pd
import numpy as np
from tkinter import Button
from turtle import color
from PIL import Image
import plotly.express as px
import time
import hydralit_components as hc
import altair as alt
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


#Layout 
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)
st.expander('Expander')

#creating menu data which will be used in navigation bar specifying the pages of the dashboard
menu_data = [
{'label':"Home", 'icon': "bi bi-house"},
{'label':"EDA", 'icon': "bi bi-clipboard-data"},
{'label':'Overview', 'icon' : "bi bi-graph-up-arrow"},

{'label':'Application', 'icon' : "fa fa-brain"}]
over_theme = {'txc_inactive': 'white','menu_background':'rgb(180,151,231)', 'option_active':'white'}

#Updating layout Design and Layout 
menu_id = hc.nav_bar(menu_definition = menu_data,
                    sticky_mode = 'jumpy',
                    sticky_nav = True,
                    hide_streamlit_markers = False,
                    override_theme = {'txc_inactive': 'white',
                                        'menu_background' : '#000099',
                                        'txc_active':'#0178e4',
                                        'option_active':'white'})

#Editing First Page of the Dashboard 
if menu_id == "Home":
    image = Image.open('LogoMSBA.png')

    row_spacer1, row_1, row_spacer2, row_2 = st.columns((.1, .1, .3, 1.8))
    with row_spacer1:
        st.image(image,width=100)
    with row_spacer2:
        st.caption(' | HealthCare Analytics')

    col1,col2 = st.columns(2)
    
    with col1:
        st.write("")
        st.markdown("#### CardioVascular Disease: A Better Understanding")
        st.write("""
                    Cornary Artery Disease (CAD) is the leading cause of death worldwide, as it increases the occurence of heart failure and ischemic heart disease (IHD) leading to Myocardial Infarctions (MI).
                    The narrowing of the coronary vessels with the weakning cardiac muscles will decrease the oxygen supply to the heart eventually stopping its functional state.
                    Increasing early diagnosis and the ability to predict the occurence of CAD and IHD  will decrease mortality rate and will decrease economic burden on our healthcare system.          
        """)
        m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #FFFFFF;
            background-color: #000099;
            }
        </style>""", unsafe_allow_html=True)
        b = st.button("Start exploring!")
        #button=st.button("Start Exploring!")
        if b:
            with hc.HyLoader('Now doing loading',hc.Loaders.pulse_bars,):
                time.sleep(5)
            st.markdown("<h6 style='text-align: left; font-family:cursive;color: black'>HeartDisease=0 represents No Cardiovascular Disease</h6>", unsafe_allow_html=True)
            st.markdown("<h6 style='text-align: left; font-family:cursive;color: black'>HeartDisease=1 represents Cardiovascular Disease</h6>", unsafe_allow_html=True)
            
            
    with col2:
        image = Image.open('iStock-530199842.jpg')
        st.image(image,width=600)
        st.write("***")
        
#Second Page

#Editing Second Page of the Dashboard   
if menu_id =="EDA":
    #Reading File
    HA=pd.read_csv('heart1.csv')
    
    #Demographical Overview
    #Splitting first row into two columns for plots 
    col3,col4 = st.columns(2)
    with col3:
        fig=px.histogram(HA, x="Sex",
             color='HeartDisease', barmode='group',
             height=400,width=600,title=" Gender Difference in Cardiovasvular disease",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig.update_yaxes( # the y-axis is in dollars
        title=None
        )
        fig.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig.update_layout(
                title={
                    'text': "Gender Difference in Cardiovasvular disease",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})


        st.plotly_chart(fig,use_container_width=True)
        
    with col4:
            
        fig1=px.histogram(HA, x="Age",
                    color='HeartDisease', barmode='group',
                    height=400,width=700,title="Incidence of Cardiovascular disease and Age",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig1.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig1.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig1.update_layout(
                title={
                    'text': "Incidence of Cardiovascular disease and Age",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        st.plotly_chart(fig1,use_container_width=True)
    
    #line dividing Demograohical  Overview from Diagnostical Overview  
    theme_override = {'bgcolor': '#000099','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    'icon_color': 'white', 'icon': 'bi bi-calendar', 'content_text_size' : '10%'}
    hc.progress_bar(content_text= 'RiskFactors&Symptoms', override_theme=theme_override)
        
   
    
    #Splitting first row into three columns for plots 
    col5,col6,col7=st.columns(3)
    
    with col5:
        fig3=px.histogram(HA, x="ChestPainType",
             color='HeartDisease', barmode='group',
             height=500,width=700,title="ChestPain in Cardiovascular Disease",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig3.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig3.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig3.update_layout(
                title={
                    'text': "ChestPain in Cardiovascular Disease",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        st.plotly_chart(fig3,use_container_width=True)
        
    with col6:
        fig4=px.histogram(HA, x="ST_Slope",
             color='HeartDisease', barmode='group',
             height=500,width=700,title="ST-slope in Cardiovascular Disease",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig4.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig4.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig4.update_layout(
                title={
                    'text': "ST-slope in Cardiovascular Disease",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

        st.plotly_chart(fig4,use_container_width=True)
    with col7:
        fig5=px.histogram(HA, x="RestingECG",
             color='HeartDisease', barmode='group',
             height=500,width=700,title="Can RestingEKG be a Diagnositc tool?",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig5.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig5.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig5.update_layout(
                title={
                    'text': "Can RestingEKG be a Diagnositc tool?",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        st.plotly_chart(fig5,use_container_width=True)

#Third Page 
#Editing Second Page of the Dashboard  
if menu_id == "Overview":
    #Reading the csv file 
    HA=pd.read_csv('heart1.csv')
    
    #Splitting first row into two columns for plots 
    col01,col02=st.columns(2)
    with col01:
        fig6 = px.scatter(HA, x="Cholesterol", y="Age", color="Sex",height=400,width=700,title="Cholestrol levels with respect to Age&Gender",color_discrete_map={"M": "midnightblue", "F": "darkgrey"},template="simple_white")
        fig6.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig6.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig6.update_layout(
                title={
                    'text': "Cholestrol levels with respect to Age&Gender",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        st.plotly_chart(fig6,use_container_width=True)
    
    with col02:
        fig7=px.histogram(HA, x="Cholesterol",
             color='HeartDisease', barmode='group',
             height=400,width=700,title="Cholestrol Levels and Cardiovasvular disease",color_discrete_map={1: "midnightblue", 0: "darkgrey"},template="simple_white")

        fig7.update_yaxes( # the y-axis is in dollars
            title=None
        )
        fig7.update_layout( # customize font and legend orientation & position
        font_family="Rockwell",
        legend=dict(
         orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
        )
        fig7.update_layout(
                title={
                    'text': "Cholestrol Levels and Cardiovasvular disease",
                    
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        st.plotly_chart(fig7,use_container_width=True)
        
    #line dividing plots from Customization option   
    theme_override = {'bgcolor': '#000099','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    'icon_color': 'white', 'icon': 'bi bi-calendar', 'content_text_size' : '10%'}
    hc.progress_bar(content_text= 'Customize your Own Graph! ', override_theme=theme_override)

    
    #Splitting first row into four columns for filters 
    col8,col9,col10,col11=st.columns(4)
    with col8:
        y_axis = st.selectbox('Select y-axis', ['Age', 'Cholesterol', 
                                                    'FastingBS',"MaxHR","Sex","ChestPainType","RestingBP",
                                                    "RestingECG","ExerciseAngina","Oldpeak","ST_Slope"])
    with col9:
        
        x_axis = st.selectbox('Select x-axis', ['Age', 'Cholesterol', 
                                                    'FastingBS',"MaxHR","Sex","ChestPainType","RestingBP",
                                                    "RestingECG","ExerciseAngina","Oldpeak","ST_Slope"])
    with col10:
        label = st.selectbox('Select label', ['HeartDisease'])
    
    with col11:
 
        select_graph = st.selectbox('Select Graph', ('point', 'bar', 'area', 'line'))
        
    #Spacing graph to be in the middle 
    col001,col002,col003,col004=st.columns(4)
    with col002:
        chart = alt.Chart(data=HA, mark=select_graph).encode(alt.X(x_axis, scale=alt.Scale(zero=False)), 
                                                                alt.Y(y_axis, scale=alt.Scale(zero=False)),color=label).configure_axis(
        grid=False
        ).properties(
        width=900,
        height=400
        )
        st.write(chart)
#Fourth Page 
#Editing Fourth Page of the Dashboard 



        
if menu_id == "Application":
    #loading file for Data-preprocessing 
    df=pd.read_csv("heart1.csv")
    #seperate title and animation
    col1, col2= st.columns([2,2])
    with col1:
        st.write("# Diagnose With Machine Learning!")
        st.markdown(f"""
        <div>
            <div style="vertical-align:left;font-size:20px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Whether you are a physician or a patient yourself, get a chance to predict the chance of a cardiovascular disease with one click!
            Early Diagnosis is the key for a healthier future!
        </div>""",unsafe_allow_html = True)
    
    with col2:
        
        image = Image.open('h.jpg')
        st.image(image, width=530)
        
    #line dividing animation and machine learning 
    theme_override = {'bgcolor': '#000099','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    'icon_color': 'white', 'icon': 'bi bi-calendar', 'content_text_size' : '10%'}
    hc.progress_bar(content_text= 'Fill the Form! ', override_theme=theme_override)
    
    #defining function to create inputs that will give a prediction
    def user_input_features():
        col3, col4= st.columns(2)
        
        with col3:
            Age = st.number_input("Insert Age", value=100, min_value=1)
            Age=float(Age)
            Gender= st.selectbox('Select the gender', ('F', 'M'))
            RestingBP = st.number_input("Insert Resting BloodPressure", value=500, min_value=1)
            RestingBP=float(RestingBP)
            Cholesterol = st.number_input("Insert Cholesterol Level", value=500, min_value=1)
            Cholesterol=float(Cholesterol)
            FastingBS = st.selectbox("Do you/your patient have FastingBP", ("Yes","No"))
            ChestPainType=st.selectbox("Any ChestPain?Specify Type", ("TA","ATA","NAP","ASY"))
        
        with col4:
            RestingECG= st.selectbox('What is the RestingECG result?', ('Normal', 'ST','LVH'))
            MaxHR=st.number_input("What is your/your patient's Maximum HeartRate?")
            ExerciseAngina=st.selectbox('Do you/your Patient experience Exercise Angina?', ("Y","N"))
            OldPeak=st.number_input('What is the old Peak of ST depression?',value=100, min_value=-10)
            OldPeak=float(OldPeak)
            St_slope=st.selectbox('What is the ST slope in EKG result ?', ('UP', 'Float','Down'))
        
        data = {'Age': Age,
                'Sex': Gender,
                'ChestPainType': ChestPainType,
                'RestingBP ': RestingBP ,
                'RestingECG': RestingECG,
                'MaxHR': MaxHR,
                'ExerciseAngina': ExerciseAngina,
                'Oldpeak': OldPeak,
                'ST_Slope': St_slope,
                'FastingBS':FastingBS,
                'Cholesterol':Cholesterol
                
                }
        features= pd.DataFrame(data, index=[0])
        return features
    

    #Defining Numerical Features
    numerical_features = ['Age','RestingBP','MaxHR','Oldpeak','Cholesterol','HeartDisease']
    df_numerical = df[numerical_features]
    #Definig Categorical Features
    categorical_features = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope','FastingBS']
    df_categorical = df[categorical_features]
    #Combining both Features into one Dataframe
    combined = pd.concat([df_categorical, df_numerical], axis = 1)
    
    #Defining X and Y 
    X = combined.drop(['HeartDisease'], axis=1)
    y = combined['HeartDisease']
    
    #Encoding the Y label 
    encoderlabel = LabelEncoder()
    y = encoderlabel.fit_transform(y)

    #pipeline for all necessary transformations
    cat_pipeline= Pipeline(steps=[
            ('impute', SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'None')),
            
            ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
            ])

    #chose best model based on previous trials
    model = RandomForestClassifier(class_weight='balanced')

    pipeline_model = Pipeline(steps = [('transformer', cat_pipeline),
                             ('model', model)])

    #train the model
    pipeline_model.fit(X, y)
    
    #predciting HeartDisease with the above model with the input Features 
    df1= user_input_features()
    prediction = pipeline_model.predict(df1)
    
    #Button Layout 
    m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #FFFFFF;
            background-color: #000099;
            }
        </style>""", unsafe_allow_html=True)
  
    
    
    pred_button = st.button('Predict')
    
    #Ediiting the last Outcome 
    if pred_button:
        if prediction[0] == 0:
            st.success('Patient is not a candidate to sustain a Cardiovascular Disease (0)')
            
        else:
            st.error('Patient is a candidate to sustain a Cardiovascular Disease(1)')

    

from distutils.command.upload import upload

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
st.sidebar.header("Navigation")

data = pd.read_csv('cars_ds_final.csv') #reading csv file
df = data   #copying the data into one more csv varivale

img = Image.open("2018_Toyota_Camry_(ASV70R)_Ascent_sedan_(2018-08-27)_01.jpg")


with st.sidebar:                #SideBar Options
    selected = option_menu(
            menu_title="Main Menu",  # required
            options=[ "Upload Dataset","General Dataset Info", "Data Cleaning","Plot Visualization","Unsupervised Training","Overall Inference"],  # required
            default_index=0,  # optional
            )

def Upload():        #Upload The File
    st.title("AutoMobile Industry Analysis !")
    st.subheader("Data Analysis of Indian Automotive Industry Using Python & Streamlit")
    upload = st.file_uploader("Upload Your Dataset (In CSV Format)")
    if upload is not None:
        data=pd.read_csv(upload)

def GenDataInfo():           #General Info bou the dataset 
    st.title("General Dataset Info")
    st.subheader(" Dataset Display")
    if upload is not None:
        if st.checkbox("Preview Dataset"):    # displaying the dataset from the beginning ,bottom and Random
            if st.button("Head"):
                st.write(data.head())
            if st.button("Tail"):
                st.write(data.tail())
            if st.button("Random"):
                st.write(data.sample(8))
            if st.checkbox("DataType of Each Column"):    #Column Analysis
                st.text("DataTypes")
                st.write(data.astype(str))
            data_shape=st.radio("What Dimension Do You Want To Check?",('Rows',
                                                                    'Columns'))  #Check the number of Row and Column
            if data_shape=='Rows':
                st.text("Number of Rows")
                st.write(data.shape[0])
            if data_shape=='Columns':
                st.text("Number of Columns")
                st.write(data.shape[1])
            st.subheader("Name of the columns")
            st.write(data.columns)
            st.subheader("Statiscal Analysis of Dataset")   #Numerical Analysis of the dataset(Mean ,Mediam, etc of all the column)
            des=data.describe()
            st.write(des)


            fig_col1, fig_col2 = st.columns(2)  #chart to determine the the count of bservation with respect to the Features
            with fig_col1:
                st.markdown("### First Chart")
                l_D = len(data)
                c_m = len(data.Make.unique())
                c_c = len(data.Model.unique())
                n_f = len(data.columns)
                fig = px.bar(x=['Observations',"Makers",'Models','Features'],y=[l_D,c_m,c_c,n_f], width=800,height=400)
                fig.update_layout(
                    title="Dataset Statistics",
                    xaxis_title="",
                    yaxis_title="Counts",
                    font=dict(
                    size=16,
                    )
                )

                st.write(fig)
                st.write("Inference - Since the dataset is full of features," 
                "we will choose only a subset of useful features to work with," 
                "also we will clean the data to extract useful information")


def cleandata():                             #Function for cleaning the Data and modifying it 

                                       

    df.fillna('')                                                      #Cleaning the df dataset 
    df.replace(' ', '')
    df['Ex-Showroom_Price']=df['Ex-Showroom_Price'].str.replace(',','')
    df['Ex-Showroom_Price']=df['Ex-Showroom_Price'].str.replace('Rs.','')
    df['Displacement']=df['Displacement'].str.replace('cc','')




    global data                                                              #Cleaning the Data variable 
    data['car'] = data['Make'] + ' ' + data['Model']
    c = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
        'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
    data_full = data.copy()
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].str.replace(',','',regex=False)
    data['Ex-Showroom_Price'] = data['Ex-Showroom_Price'].astype(int)
    data = data[c]
    data = data[~data["ARAI_Certified_Mileage"].isnull()]
    data = data[~data["Make"].isnull()]
    data = data[~data["Width"].isnull()]
    data = data[~data["Cylinders"].isnull()]
    data = data[~data["Wheelbase"].isnull()]
    data = data[~data['Fuel_Tank_Capacity'].isnull()]
    data = data[~data['Seating_Capacity'].isnull()]
    data = data[~data['Torque'].isnull()]
    data['Height'] = data['Height'].str.replace(' mm','',regex=False).astype(float)
    data['Length'] = data['Length'].str.replace(' mm','',regex=False).astype(float)
    data['Width'] = data['Width'].str.replace(' mm','',regex=False).astype(float)
    data['Wheelbase'] = data['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
    data['Fuel_Tank_Capacity'] = data['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
    data['Displacement'] = data['Displacement'].str.replace(' cc','',regex=False)
    data.loc[df["ARAI_Certified_Mileage"] == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
    data.loc[df["ARAI_Certified_Mileage"] == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
    data['ARAI_Certified_Mileage'] = data['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
    data["Number_of_Airbags"].fillna(0,inplace= True)
    data['price'] = data['Ex-Showroom_Price'] * 0.014
    data.drop(columns='Ex-Showroom_Price', inplace= True)
    data["price"] = data["price"].astype(int)
    HP = data["Power"].str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
    HP = HP.apply(lambda x: round(x,2))
    TQ = data["Torque"].str.extract(r'(\d{1,4}).*').astype(int)
    TQ = TQ.apply(lambda x: round(x,2))
    data["Torque"] = TQ
    data["Power"] = HP
    data["Doors"] = data['Doors'].astype(int)
    data['Seating_Capacity'] = data['Seating_Capacity'].astype(int)
    data['Number_of_Airbags'] = data['Number_of_Airbags'].astype(int)
    data.Displacement = data['Displacement'].astype(int)
    data['Cylinders'] = data['Cylinders'].astype(int)
    data.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
                    'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']
    
    return df,data

    

def plot():                   #Plot analysis of the dataset
    global data
    df,data =cleandata()
    st.title("Plot Visulization")
    st.header("Correlation Map of different Features of the Dataset")
    df[['Cylinders', 'Valves_Per_Cylinder', 'Doors', 'Seating_Capacity', 'Number_of_Airbags', 'Ex-Showroom_Price', 'Displacement']] = df[['Cylinders', 'Valves_Per_Cylinder', 'Doors', 'Seating_Capacity', 'Number_of_Airbags', 'Ex-Showroom_Price', 'Displacement']].apply(pd.to_numeric)
    f = plt.figure(figsize=(14,10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    st.pyplot(f)
    st.subheader("Inference")
    st.write("1. Ex-Showroom price is positively correlated to Displacement.")
    st.write("2. Ex-Showroom Price is Positively Correlated to the number of Cylinders. This means, more the number of cylinders, more the ex-showroom price.")
    st.write("3. The more the number of cylinders in a car, the more will be its displacement. Generally speaking, the higher an engine’s displacement the more power it can create.")
    st.write("4. The number of doors is highly negatively correlated with Displacement")

    st.header("Analysis of Car With Body Type.")
    counter=plt.figure(figsize=(16,7))
    sns.countplot(data=df, y='Body_Type',alpha=.6,color='blue')
    plt.title('Cars by car body type',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    st.pyplot(counter)
    st.subheader("Inference")
    st.write("Indian market is favourable for SUVs, sedans, and Hatchback.")


    st.header("Brands with most Number of Cars in the Market.")
    fig = plt.figure(figsize = (10,10))
    ax = fig.subplots()
    df.Make.value_counts().plot(ax=ax, kind='pie')
    ax.set_ylabel("")
    ax.set_title("Top Car Making Companies in India")
    plt.show()
    st.pyplot(fig)
    st.subheader("Inference")
    st.write("1. Maruti Suzuki has more car variants than any other company in India.")
    st.write("2. The Top 5 companies with more than car variants in India are Maruti Suzuki, Hyundai, Mahindra, Tata, and Toyota.")
    st.write("3. Sports car variants are very low.")

    st.header("Graph of Body Type and Ex-Showroom Price")
    bar = plt.figure(figsize = (10,10))
    PriceByType = df.groupby('Body_Type').sum().sort_values('Ex-Showroom_Price', ascending=False)
    PriceByType = PriceByType.reset_index()
    bar = px.bar(x='Body_Type', y ="Ex-Showroom_Price", data_frame=PriceByType)
    st.write(bar)
    st.subheader("Inference")
    st.write("1. If we some up all the SUVs Ex-Showroom price present in the Dataset then it will be nearly 2B INR")
    st.write("2. Sports cars a minimal spike in the graph")

    st.header("Cars Count By Engine Type")
    big = plt.figure(figsize = (10,10))
    ax = big.subplots()
    df.Fuel_Type.value_counts().plot(ax=ax, kind='pie')
    ax.set_ylabel("")
    ax.set_title("")
    plt.show()
    st.pyplot(big)
    st.subheader("Inference:")
    st.write("1. Almost 90% of Indian cars run on Petrol or Diesel. This is Scary if we see it from an Environment point of view.")
    st.write("2. This data is going to change because electric vehicles have arrived in India.")
    

    st.header("Relation between Displacement and Power")
    plain = plt.figure(figsize=(10,8))
    sns.scatterplot(data=df, x='Displacement', y='Ex-Showroom_Price',hue='Body_Type',palette='viridis',alpha=.89, s=120 );
    plt.xticks(fontsize=13);
    plt.yticks(fontsize=13)
    plt.xlabel('power',fontsize=14)
    plt.ylabel('price',fontsize=14)
    plt.title('Relation between Displacement and price',fontsize=20);
    st.pyplot(plain)
    st.subheader("Inference:")
    st.write("This data is self-explanatory. The price and power of the sports car are the highest.")

    st.header("Analysis of Price")
    box = plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='price',width=.3,color='blue', hue= 'fuel_type')
    plt.title('Box plot of Price',fontsize=18)
    plt.xticks([i for i in range(0,800000,100000)],[f'{i:,}$' for i in range(0,800000,100000)],fontsize=14)
    plt.xlabel('price',fontsize=14);
    st.pyplot(box)
    st.subheader("Inference:")
    st.write("Seems that there is a lot of outliers that form a very different type(s) of cars or to be mor exact there are very different categories in the automotive market")
     
    st.header("Analysis of Car Price with Body Type of car ")
    boxplot=plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='price', y='body_type', palette='viridis')
    plt.title('Box plot of Price of every body type',fontsize=18)
    plt.ylabel('')
    plt.yticks(fontsize=14)
    plt.xticks([i for i in range(0,800000,100000)],[f'{i:,}$' for i in range(0,800000,100000)],fontsize=14);
    st.pyplot(boxplot)
    st.subheader("Inference:")
    st.write("It's Clear that Car body type strongly affect the price")

    st.header("Analysis of Car Engine")
    Histplot=plt.figure(figsize=(14,6))
    sns.histplot(data=data, x='displacement',alpha=.6, color='darkblue',bins=10)
    plt.title('Cars by engine size (in CC)',fontsize=18)
    plt.xticks(fontsize=13);
    plt.yticks(fontsize=13);
    st.pyplot(Histplot)
    st.subheader("Inference:")
    st.write("Seems like most of cars have engine size in the 1000:2000cc range")

    st.header("Analysis of Horsepower")
    Horse=plt.figure(figsize=(14,6))
    sns.histplot(data=data, x='power',alpha=.6, color='darkblue')
    plt.title('Cars by engine size (in CC)',fontsize=18);
    plt.xticks(fontsize=13);
    plt.yticks(fontsize=13);
    st.pyplot(Horse)
    st.subheader("Inference:")
    st.write("Most Indian Cars have less power ")

    st.header("Analysis of Car Body Type,Power and Car Price")
    c=plt.figure(figsize=(10,8))
    sns.scatterplot(data=data, x='power', y='price',hue='body_type',palette='viridis',alpha=.89, s=120 );
    plt.xticks(fontsize=13);
    plt.yticks(fontsize=13)
    plt.xlabel('power',fontsize=14)
    plt.ylabel('price',fontsize=14)
    plt.title('Relation between power and price',fontsize=20);
    st.pyplot(c)
    st.subheader("Inference:")
    st.write("Horsepower of car seems to be highly related to car price but car body type seems a little bit blurry but hatchbacks seems to be the body type with the least horsepower and price.")


    st.header("Scater plot to understand the important variable in the dataset")

    st.pyplot(sns.pairplot(data,vars=[ 'displacement', 'mileage', 'power', 'price'], hue= 'fuel_type',
        palette=sns.color_palette('magma',n_colors=4),diag_kind='kde',height=2, aspect=1.8))

    st.subheader("Inferencs:")
    st.write("Seems there are a lot of multicollinearity between variables(Independent Variable are highly correlated to each other).")

    st.header("Using more interactive plot to show the previous plot and also adding the car manufacturer")
    gig = px.scatter_3d(data, x='power', z='price', y='mileage',color='make',width=800,height=750)
    gig.update_layout(showlegend=True)
    gig.show();
    st.write(gig)

    st.subheader("Inference:")
    st.write("As shown in the figures clustring the market needs a lot of effort as the separation of clusters is not that obvious")


def training():                                  #Unsupervised Section
    df,data=cleandata()
    st.title("Is That it ?")
    st.write("It's now clear that we have to look to many dimensions in order to cluster the market, as the more features we explore the harder it's to cluster the market. These dimensions affect the decision of the buyers not to mention it also precvied totally different due to the very different mental models of buyers, in other words, price horsepower and mileage are not everything, some buyers would like to have a long wheel base car, some would like to have wider car all of the previous features, and more, strongly affect the buyer' decisions.This means that two car can have very similar price and milage but one is a van with lots of space and the other is just a four doors sedan, these two cars are precieved as two different categories in the automotive industry so space length width and height of the car can also be a vital factor.")
    

    st.image(img)

    st.markdown("**So, a three dimensional representation won't tell everythings, so thats why we will try to consider clustering to use the very different features associated with each car**")
    st.title("Clustering")
    st.write("Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data analysis, and a common technique for statistical data analysis, used in many fields, including pattern recognition, image analysis, information retrieval, bioinformatics, data compression, computer graphics and machine learning.")
    st.markdown("**The type of clustering used here is k-means clustering**")
    st.write("k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances (squared Euclidean distances)")
    st.markdown("**1. We will drop the cars over 60k as they totally not a match for the corolla**")
    data = data[data.price < 60000]


    st.markdown("**2. Now we choose a collection of features to build 8 clusters of cars**")
    num_cols = [ i for i in data.columns if data[i].dtype != 'object']

    st.markdown("**Fitting K-means clustering model with 10 clusters and adding a cluster column to the dataset**")
    km = KMeans(n_clusters=8, n_init=20, max_iter=400, random_state=0)     #K-means Algorithm 
    clusters = km.fit_predict(data[num_cols])
    data['cluster'] = clusters
    data.cluster = (data.cluster + 1).astype('object')
    st.markdown("**Congratulations !! Your Data is clustered")
    st.write("Sample of the dataset after Clustering")
    st.write(data.sample(5))
    st.header("Now we check some scatter plots but with adding cluster.")
    st.subheader("Price vs Power")

    clust=plt.figure(figsize=(10,8))
    sns.scatterplot(data=data, y='price', x='power',s=120,hue='cluster',palette='viridis')
    plt.legend(ncol=4)
    plt.title('Scatter plot of price and horsepower with clusterspclusters predicted', fontsize=18)
    plt.xlabel('power',fontsize=16)
    plt.ylabel('price',fontsize=16);
    st.pyplot(clust)
    st.subheader("Inference:")
    st.write("We can see the the clusters are strongly affected by the price with clear speration between clusters but it's kind of blurry when it comes to power")


    st.subheader("Power vs Mileage")
    clust2=plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x='power', y='mileage',s=120,hue='cluster',palette='viridis')
    plt.legend(ncol=4)
    plt.title('Scatter plot of milage and horsepower with clusters', fontsize=18);
    plt.xlabel('power',fontsize=16)
    plt.ylabel('mileage',fontsize=16);
    st.pyplot(clust2)

    st.subheader("Inference:")
    st.write("But yet we can see that clusters speration in power is stronger than mileage which almost have no separation of clusters")


    st.subheader("Engine size vs Fuel tank")
    clust3=plt.figure(figsize=(8,6))
    sns.scatterplot(data=data, x='fuel_tank', y='displacement',s=120,hue='cluster',palette='viridis')
    plt.legend(ncol=4)
    plt.title('Scatter plot of milage and horsepower with clusters', fontsize=18);
    plt.xlabel('Fuel Tank Capacity ',fontsize=16)
    plt.ylabel('Engine size',fontsize=16);
    st.pyplot(clust3)

    st.header("Now we make an interactive 3D scatter plot of price power, and mileage using also clusters")
    d = px.scatter_3d(data, x='power', z='price', y='mileage',color='cluster',
            height=700, width=800,color_discrete_sequence=sns.color_palette('colorblind',n_colors=8,desat=1).as_hex(),
            title='price power, and mileage')
    d.show()
    st.write(d)


    st.subheader("Now we check the average prices of each cluster")
    clust4=plt.figure(figsize=(14,6))
    sns.barplot(data=data, x= 'cluster', ci= 'sd', y= 'price', palette='viridis',order=data.groupby('cluster')['price'].mean().sort_values(ascending=False).index);
    plt.yticks([i for i in range(0,65000,5000)])
    plt.title('Average price of each cluster',fontsize=20)
    plt.xlabel('Cluster',fontsize=16)
    plt.ylabel('Avg car price', fontsize=16)
    plt.xticks(fontsize=14);
    st.pyplot(clust4)

    st.subheader("Now we check how many cars exists in each cluster")
    clust5=plt.figure(figsize=(14,6))
    sns.countplot(data=data, x= 'cluster', palette='viridis',order=data.cluster.value_counts().index);
    # plt.yticks([i for i in range(0,65000,5000)])
    plt.title('Number of cars in each cluster',fontsize=18)
    plt.xlabel('Cluster',fontsize=16)
    plt.ylabel('Number of cars', fontsize=16)
    plt.xticks(fontsize=14);
    st.pyplot(clust5)
    st.subheader("Inference:")
    st.write("We can generally say that even if clusters generated are not determinant yet we can see that they still can be useful")



    st.subheader("Inference:")
    st.write("As shown in the scatter splits earlier , there is a clear seperation of clusters when it comes to prices")
    st.subheader("Now we check how many cars exists in each cluster")


    clust5=plt.figure(figsize=(14,6))
    sns.countplot(data=data, x= 'cluster', palette='viridis',order=data.cluster.value_counts().index);
    # plt.yticks([i for i in range(0,65000,5000)])
    plt.title('Number of cars in each cluster',fontsize=18)
    plt.xlabel('Cluster',fontsize=16)
    plt.ylabel('Number of cars', fontsize=16)
    plt.xticks(fontsize=14);
    st.pyplot(clust5)
    st.subheader("Inference:")
    st.write("We can generally say that even if clusters generated are not determinant yet we can see that they still can be useful")


def conclusion():                                          #Concluding information about the Dataset
    st.title("Conclusion")
    st.subheader("How is that any useful?")
    st.write("With clustring there are too many variable taken in considration which are hard to be traced by normal methods. The clusters generated by the KMeans model can be used to identify what is the strategic group that form a strong competition to the company products in the market it also show the close clusters to this group which also can be put in considration in some cases.")
        

    st.subheader("Problem with clustring")
    st.write("As tempting as it's to use clustring to produce strategic groups it worth mentioning that the clustring process itself is a little bit ambigous and features contribution to the clustering process can't be easily explained so the overall interpretability of the model forms a challenge")


    st.subheader("So is it worthless?!")
    st.write("Absolutely not!, clustring may be not determinant but it can be used to augment the management decision by using it side by side with human intuition to come out with the right strategic group")
    



if selected=="Upload Dataset":
    Upload()
elif selected=="General Dataset Info":
    GenDataInfo()
elif selected=="Data Cleaning":
    st.title("Data Cleaning")
    st.write("Missing data is always a problem in real life scenarios. Areas like machine learning and data mining face severe issues in the accuracy of their model predictions because of poor quality of data caused by missing values. In these areas, missing value treatment is a major point of focus to make their models more accurate and valid.")
    
    st.header("Null Values")
    null = data.isnull().sum().sort_values(ascending = False)
    st.write(null)
    st.write("As You can see There is lot of null values in the Dataset . In order to perform analysis of the data , we have to clean the dataset (remove and modify the null values) without affecting the dataset ")
    st.write("Press the Below button to clean  and modify the data ")
    if st.button("Clean Dataset"):
        cleandata()
        st.write("Dataset is Cleaned !!")
elif selected=="Plot Visualization":
    plot()
elif selected=="Unsupervised Training":
    training()
elif selected=="Overall Inference":
    conclusion()


# Automobile_analysis.github.io


Website Link - [[Link](https://share.streamlit.io/sap-20/automobile_analysis.github.io/main/app.py)](https://share.streamlit.io/sap-20/automobile_analysis.github.io/main/app.py)
#### Strategic grouping ####


![alt text](https://bernardmarr.com/wp-content/uploads/2021/07/the-10-biggest-strategy-mistakes-companies-make.png)


A strategic group is a concept used in strategic management that groups companies within an industry that have similar business models or similar combinations of strategies. For example, the restaurant industry can be divided into several strategic groups including fast-food and fine-dining based on variables such as preparation time, pricing, and presentation. The number of groups within an industry and their composition depends on the dimensions used to define the groups. Strategic management professors and consultants often make use of a two dimensional grid to position firms along an industry's two most important dimensions in order to distinguish direct rivals (those with similar strategies or business models) from indirect rivals.


### Objective ###

The main objective of this report is to be able to identify the strategic group of a car given a set of features to that car.

The Car we want to identify the strategic group for is Toyota Corolla which is known as Corolla Altis in india


![alt text](https://imgd.aeplcdn.com/1056x594/cw/ec/26588/Toyota-Corolla-Altis-Exterior-114986.jpg?wm=1&q=85)


The Corolla is a world wide best seller and it has been sold for many years, but as the automotive market became full of comptitiors an extensive analysis is needed to identifiy the strategic group competing with the company in the market


 # Approach #

This report will follow a mathematical approach using a method called clustring to achive the objective of identifying strategic group.


### Challenge ###

The automotive industry is one of the largest industries out there it's a 2.6 trillion dollar industry!

But inside the industry there are too many categories and subcategories constructed by too many variables that it almost safe to say that every category is an industry of itself. for instance the car body variable is a vital one, as diffrernt body types are being used for very different reason here is a list of some car body type:

* SEDAN
* COUPE
* STATION WAGON
* HATCHBACK
* CONVERTIBLE
* SPORT-UTILITY VEHICLE (SUV)
* MINIVAN

**And this just to name a few!, here is a photo show some of car types in more details**


![alt text](http://carsonelove.com/wp-content/uploads/2014/01/Type-of-Cars.jpg)


And all of the variety above is only regarding the car body type which is only one variable!, not to mention that there are grey areas where some car body types can be irrelevant to customer decision.

**So for a car company it's really a challenge to identify its strategic group as it really takes a lot of effort to put all variable in consideration.**

**This report will have to main part**
* Exploratory Data Analysis 
* UnSupervised Learning 

## Exploratory Data Analysjs ##
Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.


## UnSupervised Learning ##
Unsupervised learning refers to the use of artificial intelligence (AI) algorithms to identify patterns in data sets containing data points that are neither classified nor labeled.




## Packages Used ##

matplotlib==3.5.2
missingno==0.5.1
numpy==1.22.3
pandas==1.4.2
Pillow==9.1.1
plotly==5.8.0
scikit_learn==1.1.1
seaborn==0.11.2
streamlit==1.9.0
streamlit_option_menu==0.3.2


### Setup the Development Environment ###

**Install**
This project requires Python and the following Python libraries installed:

1. [matplotlib](https://matplotlib.org/)
2. [Numpy](https://numpy.org/)
3. [missingno](https://pypi.org/project/missingno/)
4. [Seaborn](https://pypi.org/project/seaborn/)
5. [Pandas](https://pypi.org/project/pandas/)
6. [pyplot](https://plotly.com/python/plotly-express/)
7. [scikit_learn](https://scikit-learn.org/stable/)
etc,....

**Rest librarry are in the Packages Used Section**


In the Command Prompt, You Can Try **Pip install *Package Name* ** where package name is the name of Python packages That You have to downaload



**Adding the project to the system**

* For Cloneing the project and Entering the Project Directory ,
* ```shell
   git clone 
   cd Automobile_analysis.github.io
   ```
* For Running the Project , You can try : 
* ```shell
   streamlit run app.py
   ```
 
 
 **Note - The Folder already has two CSV FILE (names - cars_ds_final and cars_engage_2022).You can use any of one file for the analysis**
 
Refernce - You can also check the jupyter notebook for the analysis(Attached in the File):
   
 

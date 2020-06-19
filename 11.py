#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from flask import Flask, request, render_template, session, redirect,url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree,linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.style.use('ggplot')


# In[5]:


app = Flask(__name__)

@app.route('/error/<c_name>')
def error(c_name):
    return render_template('my-error.html',error_in=c_name)

@app.route('/')
def home_page():
    return render_template('home_link.html')

@app.route('/about',methods=['POST','GET'])
def About():
    return render_template('about.html',title ='About the project')

@app.route('/Country')
def Country():
    return render_template('Country.html',title ='Country')

@app.route('/comparemedaltally')
def country_menu():
    return render_template('country_menu.html')

@app.route('/particularyear')
def compare_medal_tally():
    return render_template('comparemedaltally_specific_year.html',title = 'Compare medal tally for two countries')

@app.route('/particularyear', methods=['POST'])
def compare_medal_tally_display():
    if request.method == "POST":
        selected_year = eval(request.form.get("Year12", None))
        countryname1 = request.form.get("country11", None)
        countryname2 = request.form.get("country22", None)
    
    plt.figure(figsize=(20,10))

    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Year','Gold','Silver','Bronze','Total']]

    list_of_countries = list(GBS_athlete_data2.region.unique())
    list_of_year = list(GBS_athlete_data2.Year.unique())

    if countryname1 not in list_of_countries:
        return redirect(url_for('error',c_name=countryname1))
    if countryname2 not in list_of_countries:
        return redirect(url_for('error',c_name=countryname2))
    if selected_year not in list_of_year:
        return redirect(url_for('error',c_name=selected_year))
    
    
    GBS_countryname1 = GBS_athlete_data2[GBS_athlete_data2['region'] == countryname1]
    GBS_countryname1 = GBS_countryname1[GBS_countryname1['Year']==selected_year]
    #GBS_countryname1.plot.bar()

    #plt.plot(GBS_countryname1['Year'],GBS_countryname1['Total'],label='Total')
    plt.bar(0,GBS_countryname1['Gold'],width=2)
    plt.bar(2,GBS_countryname1['Silver'],width=2)
    plt.bar(4,GBS_countryname1['Bronze'],width=2)
    plt.bar(6,GBS_countryname1['Total'],width=2)

    GBS_countryname2 = GBS_athlete_data2[GBS_athlete_data2['region'] == countryname2]
    GBS_countryname2 = GBS_countryname2[GBS_countryname2['Year']==selected_year]
    #GBS_countryname2.plot.bar()

    plt.bar(12,GBS_countryname2['Gold'],width=2)
    plt.bar(14,GBS_countryname2['Silver'],width=2)
    plt.bar(16,GBS_countryname2['Bronze'],width=2)
    plt.bar(18,GBS_countryname2['Total'],width=2)

    plt.xticks([3,15],[countryname1,countryname2],fontsize = 40)

 
    plt.savefig('static/'+countryname1+countryname2+'Countryplot.png')
    copy_country_name = 'static/'+countryname1+countryname2+ 'Countryplot.png'
    #print(copy_country_name)
    return render_template('compare_tally.html',country_name =copy_country_name)

@app.route('/historicalcomparison')
def Historicalcomparison():
    return render_template('comparemedaltally_all_time.html')

@app.route('/historicalcomparison', methods=['POST'])
def historicalcomparison_display():
    if request.method == "POST":
        countryname1 = request.form.get("country11", None)
        countryname2 = request.form.get("country22", None)
    
    plt.figure(figsize=(20,10))
    
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Year','Total']]
    
    list_of_countries = list(GBS_athlete_data2.region.unique())

    if countryname1 not in list_of_countries:
        return redirect(url_for('error',c_name=countryname1))
    if countryname2 not in list_of_countries:
        return redirect(url_for('error',c_name=countryname2))
    
    
    GBS_countryname1 = GBS_athlete_data2[GBS_athlete_data2['region'] == countryname1]
    plt.plot(GBS_countryname1['Year'],GBS_countryname1['Total'],label=countryname1)

    GBS_countryname2 = GBS_athlete_data2[GBS_athlete_data2['region'] == countryname2]
    plt.plot(GBS_countryname2['Year'],GBS_countryname2['Total'],label=countryname2)

    plt.xticks(np.arange(1880, 2020, 10),fontsize = 10,rotation=90)
    # above xticks years are made up to be inclusive
    plt.legend(fontsize =20)
     
    plt.savefig('static/'+countryname1+countryname2+'Historicalplot.png')
    copy_country_name = 'static/'+countryname1+countryname2+ 'Historicalplot.png'
    #print(copy_country_name)
    return render_template('historical_display.html',country_name =copy_country_name)

@app.route('/popular')
def popular():
    return render_template('popular.html',title = 'Compare medal tally for two countries')

@app.route('/predictgender')
def Predict_gender():
    return render_template('gender_predict.html',title = 'Predict gender of the player')
@app.route('/predictgender', methods=['POST'])
def predict_gender_processing():
    age = eval(request.form['text69'])
    height = eval(request.form['text70'])
    weight = eval(request.form['text70'])
    
    
    #reading the csv file
    athlete_dataframe = pd.read_csv('athlete_events.csv')
    #print(athlete_dataframe.columns)

    #predicting the gender of the player depending on his\her age,height,weight
    sample_df = athlete_dataframe.dropna(subset=['Sex','Age','Height','Weight'])

    #for prediction we only need age,height,weight
    sample_df = sample_df[['Sex','Age','Height','Weight']]

    sample_df.replace(to_replace=['M','F'],value=[0,1],inplace=True)
    #print(sample_df)

    X = np.array(sample_df[['Age','Height','Weight']])
    y = np.array(sample_df['Sex'])

    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = tree.DecisionTreeClassifier()
    model.fit(X_train,y_train)

    accuracy = model.score(X_test , y_test)
    gender = int(model.predict([[int(age),float(height),float(weight)]]))
    
    plt.figure(figsize=(10,10))
    variable1 = sns.scatterplot(x='Height', y='Weight', hue='Sex', data=sample_df,legend='full')
    plt.title('Height $ Weight gender wise', fontsize=20)
    variable2 = variable1.get_figure()
    variable2.savefig('static/gender_plot.png')
    
    fig_loc ='static/gender_plot.png'
    return redirect(url_for('gender_predict_display_result',acc=accuracy,pred=gender,fig_l=fig_loc))

@app.route('/predictgender/<acc>/<pred>/<path:fig_l>')
def gender_predict_display_result(acc,pred,fig_l):
    if pred == 1:
        predicted = 'Female'
    else:
        predicted = 'Male'
     
    return render_template('gender_display.html',accuracy=acc,predicted_val=predicted,fig_loc=fig_l)

@app.route('/predictmedal')
def Predict_medaltally():
    return render_template('medal_predict.html',title = 'Predict medal tally for the country')

@app.route('/predictmedal', methods=['POST'])
def predict_medaltally_processing():
    noc = request.form['text69']
    participation = eval(request.form['text70'])
    
    
    #reading the csv file
    athlete_dataframe = pd.read_csv('athlete_events.csv')


    #Predicting medal count for next olympics for a particular country
    sample_df = athlete_dataframe[athlete_dataframe['NOC'] == noc]
    #count() doesnt add up NaN values

    #grouping the data year - wise
    sample_df = sample_df.groupby('Year').count()
    sample_df.reset_index(level='Year',inplace=True)
    #print(sample_df)

    #we r predicting here on the basis of number of players participated that year
    X = sample_df['ID'].values.reshape(-1,1)
    y = sample_df['Medal'].values.reshape(-1,1)

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2)


    model = linear_model.LinearRegression()
    model.fit(X_train ,y_train)

    #slope print(model.coef_)
    #y-intercept print(model.intercept_)
    accuracy = model.score(X_test , y_test)

    predicted_value =model.predict([[participation]])
    predicted_integer =int(predicted_value)
    print(predicted_integer)
    
    x_array = np.array(sample_df.ID.unique())
    y_array = np.array([x*model.coef_ + model.intercept_ for x in x_array])
    x_array2 = x_array.reshape(len(sample_df['ID']),-1)
    y_array2 = y_array.reshape(len(sample_df['ID']),-1)

    plt.figure(figsize=(10,10))
    plt.plot(x_array2,y_array2 )
    plt.scatter(sample_df['ID'],sample_df['Medal'])
    plt.title('Medal count of '+noc+' year-wise',fontsize=14)
    plt.xlabel('Participation count',fontsize= 12)
    plt.ylabel('Medal count',fontsize=12)
    plt.savefig('static/predict_medal.png')
    
    fig_loc ='static/predict_medal.png'
    return redirect(url_for('predict_medaltally_display_result',acc=accuracy,pred=predicted_integer,fig_l=fig_loc))

@app.route('/predictmedal/<acc>/<pred>/<path:fig_l>')
def predict_medaltally_display_result(acc,pred,fig_l):
    return render_template('medal_display.html',accuracy=acc,predicted_val=pred,fig_loc=fig_l)

@app.route('/sportswise')
def input_sport_Sportwise():
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    sportlist=GBS_athlete_data2['Sport'].unique()
    sportlist = list(sportlist)
    
    return render_template('all_sports.html',the_listy=sportlist)

@app.route('/selectedsport/<the_sport>')
def sport_operation_display(the_sport):
    
    return render_template('sport_op.html',s_name=the_sport)

@app.route('/selectedsport/<the_sport>/top10_countries_alltime')
def top_10_country_all_operation(the_sport):
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport==the_sport]
    GBS_athlete_data2 = GBS_athlete_data2.groupby(['region']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Gold','Silver','Bronze','Total']]


    GBS_athlete_data2=GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False])
    GBS_athlete_data2.head(10)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    GBS_athlete_data2=GBS_athlete_data2.iloc[:10,:]
    
    
    
    
    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  
    
    bar_width=4  
    plt.figure(figsize=(20,11))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel('region',fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+4,GBS_athlete_data2['region'],fontsize=15,rotation=90)
    plt.legend()
    
   
    copy_country_name = 'static/'+'All Time TOP 10'+the_sport+ 'Countryplot.png'
    plt.savefig(copy_country_name)
    
    print(copy_country_name)   
    
    return render_template('graph_n_table.html',s_name=the_sport,country_name_the_path_graph=copy_country_name,  tables=[GBS_athlete_data2.to_html(classes='data')], titles=GBS_athlete_data2.columns.values)

@app.route('/selectedsport/<the_sport>/top10_countries_year')
def top_10_country_year_operation(the_sport):
    
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport==the_sport]

    GBS_athlete_data2 = GBS_athlete_data2.groupby(['Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['Year','Gold','Silver','Bronze','Total']]



    GBS_athlete_data2=GBS_athlete_data2.sort_values(['Year'], ascending=[True])
    print(GBS_athlete_data2['Year'])
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    
    
    return render_template('year_sport_played.html',the_listy=GBS_athlete_data2['Year'],sporty_name=the_sport)

@app.route('/selectedsport/top10_countries_year/<the_sport>/<the_year>')
def top_10_country_year_operation_display(the_sport,the_year):
      
    the_year = eval(the_year)
    
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Year==the_year]
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport==the_sport]


    GBS_athlete_data2 = GBS_athlete_data2.groupby(['region']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Gold','Silver','Bronze','Total']]


    GBS_athlete_data2=GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False])
    
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    GBS_athlete_data2=GBS_athlete_data2.iloc[:10,:]
    print('hi')
    
    
    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  
    
    bar_width=4  
    plt.figure(figsize=(20,11))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel('region',fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+4,GBS_athlete_data2['region'],fontsize=15,rotation=90)
    plt.legend()  ## PLOTTING legend for empty dataframe gives error
    
    yr= str(the_year)
    copy_country_name = 'static/TOP 10 countries in '+yr+the_sport+ 'Countryplot.png'
    plt.savefig(copy_country_name)
    
    print(copy_country_name)
    
    return render_template('graph_n_table.html',s_name=the_sport,country_name_the_path_graph=copy_country_name,  tables=[GBS_athlete_data2.to_html(classes='data')], titles='MEDAL Tally')




@app.route('/selectedsport/<the_sport>/top10_players_alltime')
def top_10_player_all_operation(the_sport):
    # TOP 10 palyers in particular sport
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport==the_sport]


    GBS_athlete_data2 = GBS_athlete_data2.groupby(['Name','region',]).sum().reset_index( )
    GBS_athlete_data2 = GBS_athlete_data2[['Name','region','Gold','Silver','Bronze','Total']]


    GBS_athlete_data2=GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False])

    medal_won_filter=GBS_athlete_data2['Total'] > 0
    GBS_athlete_data2 = GBS_athlete_data2[medal_won_filter]
    
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    GBS_athlete_data2=GBS_athlete_data2.iloc[:10,:]
    
 
    return render_template('only_table.html',s_name=the_sport , tables=[GBS_athlete_data2.to_html(classes='data')], titles='MEDAL Tally')



@app.route('/selectedsport/<the_sport>/Display_medaltally_yearwise')
def display_the_tally_sport_yearwise(the_sport):
    
    return render_template('sport_op.html',s_name=the_sport)


@app.route('/bestsport')
def input_country_best_sport():
    return render_template('my-form.html')

@app.route('/bestsport', methods=['POST'])
def listing_the_sports_bestsport():    
    fed_country_name = request.form.get("country22,None")
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Sport','Year','Gold','Silver','Bronze','Total']]
    



    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==fed_country_name]
    # this is for all the years
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Sport').sum().reset_index()
   
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    
    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  

    bar_width=4  
    plt.figure(figsize=(20,10))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    #plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel('Sport',fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+3,GBS_athlete_data2['Sport'],fontsize=15,rotation=90)
    plt.legend()
    print("x")

    copy_country_name = 'static/'+fed_country_name+ 'Sports All time medals.png'
    plt.savefig(copy_country_name)
    #GBS_athlete_data2.head()
    #return render_template('disp_list.html',the_listy=GBS_athlete_data2['Year'],c_name=fed_country_name)
    return render_template('graph_n_list.html',the_listy=GBS_athlete_data2['Sport'],graphy_path=copy_country_name,country_name=fed_country_name)



@app.route('/<which_sport>/<country_name111>/<path:country_name_the_path_graph>')
def disp_bestsport(which_sport,country_name111,country_name_the_path_graph):
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Sport','Year','Gold','Silver','Bronze','Total']]

    

   
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==country_name111]
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport == which_sport]
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Year').sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['Year','Gold','Silver','Bronze','Total']]
    print(GBS_athlete_data2.head(10))
    
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    print(GBS_athlete_data2.head(10))
   
    
    return render_template('graph_n_table.html',country_name_the_path_graph =country_name_the_path_graph,  tables=[GBS_athlete_data2.to_html(classes='data')], titles=GBS_athlete_data2.columns.values)



@app.route('/thepath/<country_name>/<which_sport>')
def best_sport_processing(country_name,which_sport):
    #text = request.form['text69']
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Sport','Year','Gold','Silver','Bronze','Total']]

    

    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==country_name]
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Sport == which_sport]
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Year').sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['Year','Gold','Silver','Bronze','Total']]
    print(GBS_athlete_data2.head(10))
    
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    print(which_sport)

    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  
    
    bar_width=4  
    plt.figure(figsize=(20,10))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    #plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel(which_sport,fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+3,GBS_athlete_data2['Year'],fontsize=15,rotation=90)
    ####plt.legend()
    
    
    copy_country_name = 'static/'+country_name+which_sport+ 'listed_yearwise.png'
    plt.savefig(copy_country_name)
    
    print(copy_country_name)    
    return redirect(url_for('disp_bestsport',country_name_the_path_graph =copy_country_name, which_sport=which_sport,country_name111=country_name))

    #return render_template('simple_fig.html',country_name =copy_country_name)
    # you were trying to display figure under route('/thepath/<country_name>/<which_year>') which was resulting in creation
    # of path : /thepath/USA/2004/static/theplottedgraphname.png  OF COURSE this path doesn't have our image
    # so we displayed the image in a route function which was in home directory using  :/<pathvar_for_image>

    ####-------------------------------------------------------------------------------------####
    ####-------------------------------------------------------------------------------------####
    ####-------------------------------------------------------------------------------------####
    
@app.route('/bestyear')
def input_country():
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    

    GBS_athlete_data2['region']=GBS_athlete_data2['region'].astype(str)
    countries = GBS_athlete_data2['region'].unique()
    countries = list((countries))
    countries.sort()
    return render_template('select_country_dropdown.html',country_listy=countries,the_title_of_page='Best Performing Year for a Country')

@app.route('/bestyear', methods=['POST'])
def listing_the_years(): 
    if request.method == "POST":
        #selected_year = eval(request.form.get("Year12", None))
        fed_country_name = request.form.get("country11", None)
        #countryname2 = request.form.get("country22", None)
        
    #fed_country_name = request.form['text69']
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Year','Gold','Silver','Bronze','Total']]
    



    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==fed_country_name]
    # this is for all the years
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Year').sum().reset_index()
   
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    
    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  

    bar_width=4  
    plt.figure(figsize=(20,10))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    #plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel('Year',fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+3,GBS_athlete_data2['Year'],fontsize=15,rotation=90)
    plt.legend()


    copy_country_name = 'static/'+fed_country_name+ 'All time medals.png'
    plt.savefig(copy_country_name)
    #GBS_athlete_data2.head()
    #return render_template('disp_list.html',the_listy=GBS_athlete_data2['Year'],c_name=fed_country_name)
    return render_template('graph_n_list69.html',the_listy=GBS_athlete_data2['Year'],graphy_path=copy_country_name,country_name=fed_country_name)



@app.route('/disp69/<the_year_to_plot>/<country_name111>/<path:country_name_the_path_graph>')
def disp(the_year_to_plot,country_name111,country_name_the_path_graph):
    
    print(' in disp function')
    
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Sport','Year','Gold','Silver','Bronze','Total']]

    year_under_obsv = eval(the_year_to_plot)    

    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==country_name111]
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Year == year_under_obsv]
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Sport').sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['Sport','Gold','Silver','Bronze','Total']]
    
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    print(GBS_athlete_data2.head(10))
    print(year_under_obsv)
    
    return render_template('graph_n_table.html',country_name_the_path_graph =country_name_the_path_graph,  tables=[GBS_athlete_data2.to_html(classes='data')], titles=GBS_athlete_data2.columns.values)



@app.route('/thepath69/<country_name>/<which_year>')
def my_form_post(country_name,which_year):
    #text = request.form['text69']
    pkTallyRead2 = open('GBS_med_number', 'rb')  
    GBS_athlete_data = pickle.load(pkTallyRead2)
    GBS_athlete_data2 = GBS_athlete_data[GBS_athlete_data['Season'] == 'Summer']
    #GBS_athlete_data2 = GBS_athlete_data2.groupby(['region','Year']).sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['region','Sport','Year','Gold','Silver','Bronze','Total']]

    year_under_obsv = eval(which_year)    

    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.region==country_name]
    GBS_athlete_data2 = GBS_athlete_data2[GBS_athlete_data2.Year == year_under_obsv]
    GBS_athlete_data2 = GBS_athlete_data2.groupby('Sport').sum().reset_index()
    GBS_athlete_data2 = GBS_athlete_data2[['Sport','Gold','Silver','Bronze','Total']]
    print(GBS_athlete_data2.head(10))
    
    GBS_athlete_data2.sort_values(['Gold', 'Silver','Bronze'], ascending=[False, False,False],inplace=True)
    GBS_athlete_data2.reset_index(drop=True, inplace=True)
    
    print('year_under_obsv')

    y = (range(len(GBS_athlete_data2)))
    new_y = [20*i for i in y] # for making values to be plotted spaced out (sparsely)
    new_y = np.array(new_y)  
    
    bar_width=4  
    plt.figure(figsize=(20,10))


    plt.bar(new_y,GBS_athlete_data2['Gold'],width=bar_width,label='Gold')
    plt.bar(new_y+bar_width,GBS_athlete_data2['Silver'],width=bar_width,label='Silver')
    plt.bar(new_y+2*bar_width,GBS_athlete_data2['Bronze'],width=bar_width,label='Bronze')
    #plt.bar(new_y+3*bar_width,GBS_athlete_data2['Total'],width=bar_width,label='Total')

    plt.xlabel('Sport',fontsize= 20)
    plt.ylabel('Medal Count',fontsize=20)

    plt.yticks(fontsize=15)
    plt.xticks(new_y+3,GBS_athlete_data2['Sport'],fontsize=15,rotation=90)
    plt.legend()
    
    yr= str(year_under_obsv)
    copy_country_name = 'static/'+country_name+yr+ 'Countryplot.png'
    plt.savefig(copy_country_name)
    
    print(copy_country_name)    
    return redirect(url_for('disp',country_name_the_path_graph =copy_country_name, the_year_to_plot=year_under_obsv,country_name111=country_name))

    #return render_template('simple_fig.html',country_name =copy_country_name)
    # you were trying to display figure under route('/thepath/<country_name>/<which_year>') which was resulting in creation
    # of path : /thepath/USA/2004/static/theplottedgraphname.png  OF COURSE this path doesn't have our image
    # so we displayed the image in a route function which was in home directory using  :/<pathvar_for_image>
    














if __name__ == '__main__':
    app.run()


# In[ ]:





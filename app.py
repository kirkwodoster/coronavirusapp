import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit 
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html 
import datetime as dt

#### Importing Data ###########################################################
sheet_Death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
sheet_Recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
sheet_Confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'

###############################################################################
#Data Pre-Processing
###############################################################################

sheet_Death = pd.read_csv(sheet_Death_url)
sheet_Recovered = pd.read_csv(sheet_Recovered_url)
sheet_Confirmed = pd.read_csv(sheet_Confirmed_url)


data_death = sheet_Death
data_Recovered = sheet_Recovered
data_Confirmed = sheet_Confirmed


df_Death = pd.DataFrame(data_death)
df_Recovered = pd.DataFrame(data_Recovered)
df_Confirmed = pd.DataFrame(data_Confirmed)

city_state_Death = df_Death['Province/State'].str.cat(df_Death['Country/Region'], sep = ", ")
city_state_Recovered = df_Recovered['Province/State'].str.cat(df_Recovered['Country/Region'], sep = ", ")
city_state_Confirmed = df_Confirmed['Province/State'].str.cat(df_Confirmed['Country/Region'], sep = ", ")

df_Death = df_Death.fillna(0)
df_Recovered = df_Recovered.fillna(0)
df_Confirmed = df_Confirmed.fillna(0)


total_Deaths = df_Death.iloc[:,-1].values.sum()
total_Recovered = df_Recovered.iloc[:,-1].values.sum()
total_Confirmed = df_Confirmed.iloc[:,-1].values.sum()
 

confirmed_tab = pd.DataFrame(df_Confirmed['Country/Region'].values, df_Confirmed.iloc[:,-1].values).reset_index()
confirmed_tab.columns = ['Total','Country/Region',]
confirmed_tab = confirmed_tab[['Country/Region','Total']]
confirmed_tab = confirmed_tab.groupby(['Country/Region']).sum().sort_values('Total', ascending=False).reset_index()

recovered_tab = pd.DataFrame(df_Recovered['Country/Region'].values, df_Recovered.iloc[:,-1].values).reset_index()
recovered_tab.columns = ['Total','Country/Region',]
recovered_tab = recovered_tab[['Country/Region','Total']]
recovered_tab = recovered_tab.groupby(['Country/Region']).sum().sort_values('Total', ascending=False).reset_index()

death_tab = pd.DataFrame(df_Death['Country/Region'].values, df_Death.iloc[:,-1].values).reset_index()
death_tab.columns = ['Total','Country/Region',]
death_tab = death_tab[['Country/Region','Total']]
death_tab = death_tab.groupby(['Country/Region']).sum().sort_values('Total', ascending=False).reset_index()

                           
infected_dict = {'Country/Region': confirmed_tab['Country/Region'].values,
                 'Confirmed': confirmed_tab['Total'].values,
                 'Recovered':  recovered_tab['Total'].values,
                 'Deaths': death_tab['Total'].values}

infected_tab = pd.DataFrame(infected_dict)

total_Confirmed = df_Confirmed.iloc[:,-1].values.sum()
total_Recovered = df_Recovered.iloc[:,-1].values.sum()
total_Deaths = df_Death.iloc[:,-1].values.sum()


#####Needed for Size of Dots####################################################
Confirmed_size = df_Confirmed.iloc[:,-1].values
cdr = pd.DataFrame()
cdr['Confirmed'] = df_Confirmed.iloc[:,-1].values
cdr['Recovered'] = df_Recovered.iloc[:,-1].values
cdr['Death'] = df_Death.iloc[:,-1].values
###############################################################################

##### Geo Graphic Plot ########################################################

mapbox_access_token ="pk.eyJ1Ijoia2lya3dvb2RzdGVyIiwiYSI6ImNrNnN0YzRtMTAzaWgzbHFpeDY3NXNmYXEifQ.pL_1KrPfWXY5I-C7XMwesg"

fig_map = go.Figure()

fig_map.add_trace(go.Scattermapbox(
        lat=df_Confirmed.Lat,
        lon=df_Confirmed.Long,
        mode='markers',
         
        marker=dict(
       
        color='#ca261d',#(len(df_Confirmed)*['red']),
        #size=scaler(Confirmed_size),
        opacity = .4,
        size=Confirmed_size,
        sizemode='area',
        sizeref = 2. * max(Confirmed_size) / (120 ** 2),
        sizemin=4

        ),
        hovertemplate = "<b>%{text}</b><br><br>" +
                        "%{hovertext}<br>" +
                        "<extra></extra>",
        hovertext = ['Comfirmed: {}<br>Recovered: {}<br>Death: {}'.format(i, j, k) 
        for i,j,k in zip(cdr['Confirmed'], cdr['Death'], cdr['Recovered'])],
        text=city_state_Death
    ))

fig_map.update_layout(
        
    height = 800,
    hovermode='closest',
    
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=35.8617,
            lon=104.1954
            
    
        ),
        pitch=0,
        zoom=2
    ),

    
)



fig_map.update_layout(mapbox_style="light")
fig_map.show(config={'displayModeBar': False})


box7figsize = 300
##############################################################################
legend = legend=dict(x=0, y=-.1)
##############################################################################

confirmed_totals_all = df_Confirmed.iloc[:,4:].sum(axis=0)
recovered_totals_all = df_Recovered.iloc[:,4:].sum(axis=0)
death_totals_all = df_Death.iloc[:,4:].sum(axis=0)

confirmed_recent = df_Confirmed.iloc[:,4:].sum(axis=0).values[-1]
recovered_recent = df_Recovered.iloc[:,4:].sum(axis=0).values[-1]
death_recent = df_Death.iloc[:,4:].sum(axis=0).values[-1]

x = pd.to_datetime(death_totals_all.index)


fig_total = go.Figure()

fig_total.add_trace(go.Scatter(x=x, y=confirmed_totals_all,
                    mode='lines+markers',
                    name='Confirmed',
                    line = dict(color='#E8AC41')
                    ))

fig_total.add_trace(go.Scatter(x=x, y=death_totals_all,
                    mode='lines+markers',
                    name='Deaths',
                    line = dict(color='red')
                    ))

fig_total.add_trace(go.Scatter(x=x, y=recovered_totals_all,
                    mode='lines+markers',
                    name='Recovered',
                    line = dict(color='green')
                    ))

fig_total.update_layout(height = box7figsize,
                   xaxis_tickformat='%b %d',
                   legend_orientation="h",
                   #title='Total Confirmed, Deaths, Reocvered', 
                   yaxis_type="log",
                   xaxis_title='<b>Totals</b>',
                   template = 'plotly_white',
                  
                   legend=legend,
                   margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=5,
                        pad=0)
                   )


################################################################################################
fig_ratio = go.Figure()

Death_per_Confirm = death_totals_all.values / confirmed_totals_all.values
Recover_per_Confirm = recovered_totals_all.values / confirmed_totals_all.values

#fig.update_xaxes(type="log")
fig_ratio.add_trace(go.Scatter(x=x, y=Death_per_Confirm,
                    mode='lines+markers',
                    name='Deaths',
                    line = dict(color='red')
                    ))
fig_ratio.add_trace(go.Scatter(x=x, y=Recover_per_Confirm,
                    mode='lines+markers',
                    name='Recovered',
                    line = dict(color='green')
                    ))

fig_ratio.update_layout(height = box7figsize,
                    #title='Deaths and Recovered per Confirmed',
                    template = 'plotly_white',
                    xaxis_tickformat='%b %d',
                    yaxis=dict(gridwidth = .1),
                    xaxis=dict(gridwidth = .1),
                    legend_orientation="h",
                    legend=legend,
                    xaxis_title='<b>Death & Recovered per Confirmed</b>',
                    
                    margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=5,
                        pad=0)
                   )


########################################################################################################################



x1 = np.array(range(0,len(x)))
y = confirmed_totals_all 
  
def test(x, a, b,c,d): 
    return a*np.exp(b*x)
  
param, param_cov = curve_fit(test, x1, y) 

ans = (param[0]*(np.exp(param[1]*x1))) 


fig_fit = go.Figure()


fig_fit.add_trace(go.Scatter(x=x, y=confirmed_totals_all,
                    mode='markers',
                    name='Confirmed',
                    line = dict(color='#0057a5')
                    ))


fig_fit.add_trace(go.Scatter(x=x, y=ans,
                    mode='lines',
                    name='Regression',
                    line = dict(color='red')
                    ))
fig_fit.update_layout(height = box7figsize,
                    legend_orientation="h",
                    xaxis_tickformat='%b %d',
                    #title='Total Confirmed',
                    #yaxis_title='Log Totals',
                    template = 'plotly_white',
                    legend=legend,
                    xaxis_title='<b>Prediction of Confirmed</b>',
                    margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=5,
                        pad=0)
                   )

#########################################################################################################################


daySinceinfected = dt.datetime.today() - dt.datetime(2019,12,31)
daySinceinfected = daySinceinfected.days




app = dash.Dash()

external_stylesheets = ['main.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)




tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #0057a5',
    'padding': '20px',
    'fontWeight': 'bold'
     
}

tab_selected_style = {
    'borderTop': '1px solid #0057a5',
    'borderBottom': '1px solid #0057a5',
    'backgroundColor': '#0057a5',
    'color': 'white',
    'padding': '20px'
}

def serve_layout(): 
    return html.Div(className="wrapper",
              children = [

                     html.Div(style={'vertical-align': 'left', 'display':'inline-block'
                                     },
                       
                        className = 'box8',
                 children=[
                         html.Div(
                                 children=[
                          html.Img(style={'border-radius': '8px', 'vertical-align': 'middle', 'width':'60px', 'margin-right':'10px',
                     'height':'60px'}
                     ,src=app.get_asset_url("corona_virus2.png"))], style={'display':'inline-block'}),
                     html.Div(
                             children = [
                    html.P('Coronavirus COVID-19 Global DashBoard')],style={'vertical-align': 'left', 'display':'inline-block', 'color':'white', 'font-size':'20'})
            
            
                     ]),
    
    
               html.Div( 
                       
                        className = 'box1',
                 children=[
                     html.Div(
                         
                              children=[
                                  html.H3(style={
                                                 'fontWeight':'bold','color':'#0057a5'},
                                               children=[
                                                   html.P(style={'color':'#0057a5'}),
                                                   '{:,}'.format(daySinceinfected),
                                               ]),
                                  html.H5(style={'color':'#0057a5'},
                                               children="Days Since First Infected")                                        
                                  
                                                           
])]),
                                     

    
               html.Div( 
                        className = 'box2',
                 children=[
                     html.Div(
                         
                              children=[
                                  html.H3(style={
                                                 'fontWeight':'bold','color':'#E8AC41'},
                                               children=[
                                                   html.P(style={'color':'#E8AC41'}),
                                                   '{:,}'.format(total_Confirmed),
                                               ]),
                                  html.H5(style={'color':'#E8AC41'},
                                               children="Total Infected")                                        
                                  
                                                           
])]),
    
                html.Div( 
                        className = 'box3',
                 children=[
                     html.Div(
                         
                              children=[
                                  html.H3(style={
                                                 'fontWeight':'bold','color':'green'},
                                               children=[
                                                   html.P(style={'color':'green'}),
                                                   '{:,}'.format(total_Recovered),
                                               ]),
                                  html.H5(style={'color':'green'},
                                               children="Total Recovered")                                        
                                  
                                                           
])]),
                  html.Div( 
                        className = 'box4',
                 children=[
                     html.Div(
                         
                              children=[
                                  html.H3(style={
                                                 'fontWeight':'bold','color':'red'},
                                               children=[
                                                   html.P(style={'color':'red'}),
                                                   '{:,}'.format(total_Deaths),
                                               ]),
                                  html.H5(style={'color':'red'},
                                               children="Total Deaths")                                        
                                  
                                                           
])]),
    
                html.Div( 
                        className = 'box5',
                 children=[
                          dcc.Graph(
                    figure = fig_map)
                    ]
                 ),
                    
                html.Div(className = 'box6',
                      children = [
                            dash_table.DataTable(
                            
                            virtualization=True,
                            #sort_action="native",
                            #filter_action="native",
                            #row_deletable=True,
                            #css={
                            #    "rule": "display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;"
                            #},
                            style_table = {'maxHeight': '300px'},
                            style_header= {'fontWeight': 'bold', 'background-color': '#0057a5', 'color':'white' },
                            style_data= {"whiteSpace": "normal"},
                            style_cell= {
                                "padding": "15px",
                                "midWidth": "0px",
                                "width": "25%",
                                "textAlign": "center",
                                "border": "white",
                            },
                            style_cell_conditional=[
                                {'font-family': 'aktiv-grotesk-thin, sans-serif'},
                                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
                                {'if': {'column_id': 'Confirmed'},'color': '#E8AC41', 'font-weight':'bold'},
                                {'if': {'column_id': 'Recovered'},'color': 'Green', 'font-weight':'bold'},
                                {'if': {'column_id': 'Deaths'},'color': 'Red', 'font-weight':'bold'},
                                {'if': {'column_id': 'Country/Region'},'color': '#0057a5', 'fontWeight': 'bold','font-weight':'bold'}
                                
                            ],
                            columns=[{"name": i, "id": i} for i in infected_tab.columns],
                            data=infected_tab.to_dict('records')
                        )
                              ] 
                       ),
                  html.Div(className = 'box7', #https://dash.plot.ly/dash-core-components/tabs #https://www.phillipsj.net/posts/deploying-dash-to-google-app-engine/
                      children = [
                              dcc.Tabs([
                                      
                                                      
                                dcc.Tab(label='Confirmed/Recovered/Deaths', style=tab_style, selected_style=tab_selected_style, children=[
                                    dcc.Graph(
                                        figure=fig_total
                                  
                                    )
                                ]),
                                dcc.Tab(label='Deaths/Recovered per Confirmed', style=tab_style, selected_style=tab_selected_style, children=[
                                    dcc.Graph(
                                        figure=fig_ratio
                                            
                                        
                                    )
                                ]),
                                dcc.Tab(label='Prediction', style=tab_style, selected_style=tab_selected_style, children=[
                                    dcc.Graph(
                                        figure=fig_fit
                    
                
            )
        ]),
    ])
])
                              ] 
                       )
                  
                  

                  
                  
                  
                  
                  
                 
                
app.layout = serve_layout      


if __name__ == '__main__':
    app.run_server(debug=False)






























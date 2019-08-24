
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___

# Choropleth Maps Exercise 

Welcome to the Choropleth Maps Exercise! In this exercise we will give you some simple datasets and ask you to create Choropleth Maps from them. Due to the Nature of Plotly we can't show you examples

[Full Documentation Reference](https://plot.ly/python/reference/#choropleth)

## Plotly Imports


```python
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import pandas as pd
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


**Import pandas and read the csv file: 2014_World_Power_Consumption**


```python
df = pd.read_csv('2014_World_GDP')
```

**Check the head of the DataFrame.**


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COUNTRY</th>
      <th>GDP (BILLIONS)</th>
      <th>CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>21.71</td>
      <td>AFG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>13.40</td>
      <td>ALB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>227.80</td>
      <td>DZA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>American Samoa</td>
      <td>0.75</td>
      <td>ASM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>4.80</td>
      <td>AND</td>
    </tr>
  </tbody>
</table>
</div>



**Referencing the lecture notes, create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary.**


```python
data = dict(
    type = "choropleth",
    locations = df['CODE'],
    text = df['COUNTRY'],
    z = df['GDP (BILLIONS)'],
    colorbar = {'title': 'GDP in Billions USD'}
    )

layout = dict(
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        # see at: https://plot.ly/python/reference/#layout-geo-projection
        projection = {'type': 'stereographic'}
        )
    )
```


```python
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap, validate=False)
```


<div>
        
        
            <div id="426a5904-ac0d-450f-9039-2ca987c508ad" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("426a5904-ac0d-450f-9039-2ca987c508ad")) {
                    Plotly.newPlot(
                        '426a5904-ac0d-450f-9039-2ca987c508ad',
                        [{"colorbar": {"title": {"text": "GDP in Billions USD"}}, "locations": ["AFG", "ALB", "DZA", "ASM", "AND", "AGO", "AIA", "ATG", "ARG", "ARM", "ABW", "AUS", "AUT", "AZE", "BHM", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BMU", "BTN", "BOL", "BIH", "BWA", "BRA", "VGB", "BRN", "BGR", "BFA", "MMR", "BDI", "CPV", "KHM", "CMR", "CAN", "CYM", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COD", "COG", "COK", "CRI", "CIV", "HRV", "CUB", "CUW", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM", "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "ETH", "FLK", "FRO", "FJI", "FIN", "FRA", "PYF", "GAB", "GMB", "GEO", "DEU", "GHA", "GIB", "GRC", "GRL", "GRD", "GUM", "GTM", "GGY", "GNB", "GIN", "GUY", "HTI", "HND", "HKG", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL", "IMN", "ISR", "ITA", "JAM", "JPN", "JEY", "JOR", "KAZ", "KEN", "KIR", "KOR", "PRK", "KSV", "KWT", "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", "MAC", "MKD", "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX", "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "NAM", "NPL", "NLD", "NCL", "NZL", "NIC", "NGA", "NER", "NIU", "MNP", "NOR", "OMN", "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "PRI", "QAT", "ROU", "RUS", "RWA", "KNA", "LCA", "MAF", "SPM", "VCT", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SGP", "SXM", "SVK", "SVN", "SLB", "SOM", "ZAF", "SSD", "ESP", "LKA", "SDN", "SUR", "SWZ", "SWE", "CHE", "SYR", "TWN", "TJK", "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV", "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM", "VGB", "WBG", "YEM", "ZMB", "ZWE"], "text": ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas, The", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "British Virgin Islands", "Brunei", "Bulgaria", "Burkina Faso", "Burma", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Cook Islands", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Curacao", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Falkland Islands (Islas Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "French Polynesia", "Gabon", "Gambia, The", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guam", "Guatemala", "Guernsey", "Guinea-Bissau", "Guinea", "Guyana", "Haiti", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica", "Japan", "Jersey", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Macedonia", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia, Federated States of", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Namibia", "Nepal", "Netherlands", "New Caledonia", "New Zealand", "Nicaragua", "Nigeria", "Niger", "Niue", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Swaziland", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Virgin Islands", "West Bank", "Yemen", "Zambia", "Zimbabwe"], "type": "choropleth", "uid": "d131b797-edc8-470c-944c-b8e27d09b092", "z": [21.71, 13.4, 227.8, 0.75, 4.8, 131.4, 0.18, 1.24, 536.2, 10.88, 2.52, 1483.0, 436.1, 77.91, 8.65, 34.05, 186.6, 4.28, 75.25, 527.8, 1.67, 9.24, 5.2, 2.09, 34.08, 19.55, 16.3, 2244.0, 1.1, 17.43, 55.08, 13.38, 65.29, 3.04, 1.98, 16.9, 32.16, 1794.0, 2.25, 1.73, 15.84, 264.1, 10360.0, 400.1, 0.72, 32.67, 14.11, 0.18, 50.46, 33.96, 57.18, 77.15, 5.6, 21.34, 205.6, 347.2, 1.58, 0.51, 64.05, 100.5, 284.9, 25.14, 15.4, 3.87, 26.36, 49.86, 0.16, 2.32, 4.17, 276.3, 2902.0, 7.15, 20.68, 0.92, 16.13, 3820.0, 35.48, 1.85, 246.4, 2.16, 0.84, 4.6, 58.3, 2.74, 1.04, 6.77, 3.14, 8.92, 19.37, 292.7, 129.7, 16.2, 2048.0, 856.1, 402.7, 232.2, 245.8, 4.08, 305.0, 2129.0, 13.92, 4770.0, 5.77, 36.55, 225.6, 62.72, 0.16, 28.0, 1410.0, 5.99, 179.3, 7.65, 11.71, 32.82, 47.5, 2.46, 2.07, 49.34, 5.11, 48.72, 63.93, 51.68, 10.92, 11.19, 4.41, 336.9, 2.41, 12.04, 10.57, 0.18, 4.29, 12.72, 1296.0, 0.34, 7.74, 6.06, 11.73, 4.66, 112.6, 16.59, 13.11, 19.64, 880.4, 11.1, 201.0, 11.85, 594.3, 8.29, 0.01, 1.23, 511.6, 80.54, 237.5, 0.65, 44.69, 16.1, 31.3, 208.2, 284.6, 552.2, 228.2, 93.52, 212.0, 199.0, 2057.0, 8.0, 0.81, 1.35, 0.56, 0.22, 0.75, 0.83, 1.86, 0.36, 777.9, 15.88, 42.65, 1.47, 5.41, 307.9, 304.1, 99.75, 49.93, 1.16, 2.37, 341.2, 11.89, 1400.0, 71.57, 70.03, 5.27, 3.84, 559.1, 679.0, 64.7, 529.5, 9.16, 36.62, 373.8, 4.51, 4.84, 0.49, 29.63, 49.12, 813.3, 43.5, 0.04, 26.09, 134.9, 416.4, 2848.0, 17420.0, 55.6, 63.08, 0.82, 209.2, 187.8, 5.08, 6.64, 45.45, 25.61, 13.74]}],
                        {"geo": {"projection": {"type": "stereographic"}, "showframe": false}, "title": {"text": "2014 Global GDP"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('426a5904-ac0d-450f-9039-2ca987c508ad');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


## USA Choropleth

**Import the 2012_Election_Data csv file using pandas.**


```python
df = pd.read_csv('2012_Election_Data')
```

**Check the head of the DataFrame.**


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>ICPSR State Code</th>
      <th>Alphanumeric State Code</th>
      <th>State</th>
      <th>VEP Total Ballots Counted</th>
      <th>VEP Highest Office</th>
      <th>VAP Highest Office</th>
      <th>Total Ballots Counted</th>
      <th>Highest Office</th>
      <th>Voting-Eligible Population (VEP)</th>
      <th>Voting-Age Population (VAP)</th>
      <th>% Non-citizen</th>
      <th>Prison</th>
      <th>Probation</th>
      <th>Parole</th>
      <th>Total Ineligible Felon</th>
      <th>State Abv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>41</td>
      <td>1</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>58.6%</td>
      <td>56.0%</td>
      <td>NaN</td>
      <td>2,074,338</td>
      <td>3,539,217</td>
      <td>3707440.0</td>
      <td>2.6%</td>
      <td>32,232</td>
      <td>57,993</td>
      <td>8,616</td>
      <td>71,584</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>81</td>
      <td>2</td>
      <td>Alaska</td>
      <td>58.9%</td>
      <td>58.7%</td>
      <td>55.3%</td>
      <td>301,694</td>
      <td>300,495</td>
      <td>511,792</td>
      <td>543763.0</td>
      <td>3.8%</td>
      <td>5,633</td>
      <td>7,173</td>
      <td>1,882</td>
      <td>11,317</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>61</td>
      <td>3</td>
      <td>Arizona</td>
      <td>53.0%</td>
      <td>52.6%</td>
      <td>46.5%</td>
      <td>2,323,579</td>
      <td>2,306,559</td>
      <td>4,387,900</td>
      <td>4959270.0</td>
      <td>9.9%</td>
      <td>35,188</td>
      <td>72,452</td>
      <td>7,460</td>
      <td>81,048</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>42</td>
      <td>4</td>
      <td>Arkansas</td>
      <td>51.1%</td>
      <td>50.7%</td>
      <td>47.7%</td>
      <td>1,078,548</td>
      <td>1,069,468</td>
      <td>2,109,847</td>
      <td>2242740.0</td>
      <td>3.5%</td>
      <td>14,471</td>
      <td>30,122</td>
      <td>23,372</td>
      <td>53,808</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>71</td>
      <td>5</td>
      <td>California</td>
      <td>55.7%</td>
      <td>55.1%</td>
      <td>45.1%</td>
      <td>13,202,158</td>
      <td>13,038,547</td>
      <td>23,681,837</td>
      <td>28913129.0</td>
      <td>17.4%</td>
      <td>119,455</td>
      <td>0</td>
      <td>89,287</td>
      <td>208,742</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>



**Now create a plot that displays the Voting-Age Population (VAP) per state. If you later want to play around with other columns, make sure you consider their data type. VAP has already been transformed to a float for you.**


```python
data = dict(
    type='choropleth',
    colorscale = 'Viridis',
    reversescale = True,
    locations = df['State Abv'],
    z = df['Voting-Age Population (VAP)'],
    locationmode = 'USA-states',
    text = df['State'],
    marker = dict(
        line = dict(
            color = 'rgb(255,255,255)',
            width = 1
        )
    ),
    colorbar = {
        'title':"Voting-Age Population (VAP)"
    }
) 

layout = dict(
    title = '2012 General Election Voting Data',
    geo = dict(
        scope='usa',
        showlakes = True,
        lakecolor = 'rgb(85,173,240)'
    )
)
```


```python
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
```


<div>
        
        
            <div id="408c1c89-c4e5-4859-98c6-f88f52b381b7" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("408c1c89-c4e5-4859-98c6-f88f52b381b7")) {
                    Plotly.newPlot(
                        '408c1c89-c4e5-4859-98c6-f88f52b381b7',
                        [{"colorbar": {"title": {"text": "Voting-Age Population (VAP)"}}, "colorscale": "Viridis", "locationmode": "USA-states", "locations": ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "District of Columbia", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], "marker": {"line": {"color": "rgb(255,255,255)", "width": 1}}, "reversescale": true, "text": ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"], "type": "choropleth", "uid": "b4dd324b-108d-4406-8669-275d36b01519", "z": [3707440.0, 543763.0, 4959270.0, 2242740.0, 28913129.0, 3981208.0, 2801375.0, 715708.0, 528848.0, 15380947.0, 7452696.0, 1088335.0, 1173727.0, 9827043.0, 4960376.0, 2356209.0, 2162442.0, 3368684.0, 3495847.0, 1064779.0, 4553853.0, 5263550.0, 7625576.0, 4114820.0, 2246931.0, 4628500.0, 785454.0, 1396507.0, 2105976.0, 1047978.0, 6847503.0, 1573400.0, 15344671.0, 7496980.0, 549955.0, 8896930.0, 2885093.0, 3050747.0, 10037099.0, 834983.0, 3662322.0, 631472.0, 4976284.0, 19185395.0, 1978956.0, 502242.0, 6348827.0, 5329782.0, 1472642.0, 4417273.0, 441726.0]}],
                        {"geo": {"lakecolor": "rgb(85,173,240)", "scope": "usa", "showlakes": true}, "title": {"text": "2012 General Election Voting Data"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('408c1c89-c4e5-4859-98c6-f88f52b381b7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Great Job!

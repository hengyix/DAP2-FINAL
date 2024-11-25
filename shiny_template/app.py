# %%
from shiny import App, render, ui
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
alt.renderers.enable("png")
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import plotly.graph_objects as go
import plotly.express as px
import io
from sklearn.linear_model import LinearRegression

# %%
# Read in the GTD dataset
path = '/Users/hengyix/Documents/GitHub/DAP2-FINAL/data/'
file_gtd = 'globalterrorismdb.csv'
df_gtd = pd.read_csv(path + file_gtd, low_memory=False)

gtd_clean = df_gtd[['iyear', 'country', 'country_txt', 'gname',
                 'attacktype1', 'attacktype1_txt', 'nkill', 'nwound', 'motive']]
# Focos on the 21st century
gtd_clean = gtd_clean[gtd_clean['iyear'] > 1999]
gtd_clean['nhurt'] = gtd_clean['nkill'] + gtd_clean['nwound']

# Data cleaning for the two maps
gtd_count = gtd_clean.groupby('country_txt').agg(
    attack_count=('country_txt', 'size'),
    casualties=('nhurt', 'sum')
).reset_index()
gtd_map = gtd_count.copy()

# Data cleaning for the table
gtd_table = df_gtd.copy()
gtd_table['nhurt'] = gtd_table['nkill'] + gtd_table['nwound']
gtd_table = gtd_table.groupby(['iyear', 'country_txt']).agg(
    attack_count=('country_txt', 'size'),
    casualties=('nhurt', 'sum')
).reset_index()

# %%
# Read in the shapefile
file_shape = 'world-administrative-boundaries/world-administrative-boundaries.shp'
world_shapefile = gpd.read_file(path + file_shape)
world_shapefile = world_shapefile[['name', 'geometry']]

# Match the countries with the shapefile
gtd_map.loc[gtd_map['country_txt'] == 'Serbia', 'attack_count'] += 162
gtd_map.loc[gtd_map['country_txt'] == 'Serbia', 'attack_count'] += 11
gtd_map.loc[gtd_map['country_txt'] == 'Serbia', 'casualties'] += 279
gtd_map.loc[gtd_map['country_txt'] == 'Serbia', 'casualties'] += 8
gtd_map.loc[gtd_map['country_txt'] == 'Yugoslavia', 'attack_count'] += 106
gtd_map.loc[gtd_map['country_txt'] == 'Yugoslavia', 'casualties'] += 91

# Adjust country names in the shapefile to handle NAs manually
name_dict_map = {
    'Bosnia & Herzegovina': 'Bosnia-Herzegovina',
    'Brunei Darussalam': 'Brunei',
    'Timor-Leste': 'East Timor',
    'Iran (Islamic Republic of)': 'Iran',
    "Côte d'Ivoire": 'Ivory Coast',
    "Lao People's Democratic Republic": 'Laos',
    'Libyan Arab Jamahiriya': 'Libya',
    'The former Yugoslav Republic of Macedonia': 'Yugoslavia',
    'Moldova, Republic of': 'Moldova',
    'Congo': 'Republic of the Congo',
    'Russian Federation': 'Russia',
    'Slovakia': 'Slovak Republic',
    'Republic of Korea': 'South Korea',
    'Saint Lucia': 'St. Lucia',
    'Syrian Arab Republic': 'Syria',
    'United Republic of Tanzania': 'Tanzania',
    'U.K. of Great Britain and Northern Ireland': 'United Kingdom',
    'United States of America': 'United States',
    'West Bank': 'West Bank and Gaza Strip',
}
world_shapefile['name'] = world_shapefile['name'].map(name_dict_map).fillna(world_shapefile['name'])

gtd_map = pd.merge(world_shapefile, gtd_map, left_on='name', right_on='country_txt', how='left').fillna(0)

# %%
# Function for map function is defined below in server

# Define the function for democracy plot
file_democracy = 'p5v2018.csv'
df_democracy = pd.read_csv(path + file_democracy)
# Clean the dataset
df_democracy = df_democracy.loc[df_democracy['year'] > 1999, [
    'country', 'year', 'polity']]
# Calculate the average democracy score for each country
country_score = df_democracy.groupby(
    'country')['polity'].mean().reset_index(name='avg_score')


# Moderate the country names in country_score to do the matching
# Erase NAs maually
name_dict_demo = {
    'Bosnia': 'Bosnia-Herzegovina',
    'Congo Kinshasa': 'Democratic Republic of the Congo',
    'Dominican Republic': 'Dominica',
    'Timor Leste': 'East Timor',
    'Myanmar (Burma)': 'Myanmar',
    'Congo-Brazzaville': 'Republic of the Congo',
    'Serbia and Montenegro': 'Serbia-Montenegro',
    'Korea South': 'South Korea'
}
country_score['country'] = country_score['country'].map(
    name_dict_demo).fillna(country_score['country'])
gtd_score = pd.merge(gtd_count, country_score, how='inner',
                     left_on='country_txt', right_on='country')
# Add HK count to China count
gtd_score.loc[gtd_score['country_txt'] == 'China', 'attack_count'] += 5
# Filter the dataset to include only rows where attack counts exceed 1000
gtd_score = gtd_score.loc[gtd_score['attack_count'] > 1000]
x = gtd_score['avg_score'].values.reshape(-1, 1)
y = gtd_score['attack_count'].values

# %%
# Make the plot using matplotlib
def plot_attacks_vs_democracy():
    fig, ax = plt.subplots()
    # Regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    ax.scatter(gtd_score['avg_score'], gtd_score['attack_count'], 
           color='steelblue', s=60, alpha=0.6)
    ax.plot(gtd_score['avg_score'], y_pred, color='red')

    ax.set_title('Democracy Performance vs. Number of Terrorist Attacks (over 1000)', 
             fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Democracy Score (2000-2020)', fontsize=10)
    ax.set_ylabel('Number of Attacks', fontsize=10)

    for spine in ax.spines.values():
        spine.set_alpha(0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# %%
# Import the GDP data
file_name = "GDP.csv"
df_gdp = pd.read_csv(path + file_name)
df_gdp = df_gdp[df_gdp["Series Name"] == "GDP (current US$)"]

# Convert the GDP values to numeric type
gdp_columns = [col for col in df_gdp.columns if "YR" in col]
for col in gdp_columns:
    df_gdp[col] = pd.to_numeric(df_gdp[col], errors="coerce")

# Calculate the average GDP
df_gdp["average_gdp"] = df_gdp[gdp_columns].mean(axis=1)

# Calculate the amount of attack from GTD data
gtd_count = gtd_clean.groupby("country_txt").agg(
    attack_count=("country_txt", "size")
).reset_index()

# Match the names
df_gdp["Country Name"] = df_gdp["Country Name"].replace({
    "Bahamas, The": "Bahamas",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czechia": "Czech Republic",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "Timor-Leste": "East Timor",
    "Egypt, Arab Rep.": "Egypt",
    "Gambia, The": "Gambia",
    "Hong Kong SAR, China": "Hong Kong",
    "Iran, Islamic Rep.": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Lao PDR": "Laos",
    "North Macedonia": "Macedonia",
    "Congo, Rep.": "Republic of the Congo",
    "Russian Federation": "Russia",
    "Korea, Rep.": "South Korea",
    "Eswatini": "Swaziland",
    "Syrian Arab Republic": "Syria",
    "Turkiye": "Turkey",
    "Venezuela, RB": "Venezuela",
    "Viet Nam": "Vietnam",
    "West Bank and Gaza": "West Bank and Gaza Strip",
    "Yemen, Rep.": "Yemen"
})

# Clean countries that no longer exist
gtd_count = gtd_count[gtd_count["country_txt"] != "International"]

# Adjust for Yugoslavia and Taiwan
serbia_montenegro_count = gtd_count.loc[gtd_count["country_txt"] == "Serbia-Montenegro", "attack_count"].values[0]
yugoslavia_count = gtd_count.loc[gtd_count["country_txt"] == "Yugoslavia", "attack_count"].values[0]

serbia_count = (serbia_montenegro_count / 2) + (yugoslavia_count / 2)
montenegro_count = (serbia_montenegro_count / 2) + (yugoslavia_count / 2)
gtd_count.loc[gtd_count["country_txt"] == "Serbia", "attack_count"] += serbia_count
gtd_count.loc[gtd_count["country_txt"] == "Montenegro", "attack_count"] += montenegro_count

gtd_count = gtd_count[gtd_count["country_txt"] != "Serbia-Montenegro"]
gtd_count = gtd_count[gtd_count["country_txt"] != "Yugoslavia"]

taiwan_count = gtd_count.loc[gtd_count["country_txt"] == "Taiwan", "attack_count"].values[0]
gtd_count.loc[gtd_count["country_txt"] == "China", "attack_count"] += taiwan_count
gtd_count = gtd_count[gtd_count["country_txt"] != "Taiwan"]

# Merge the dataframes
gtd_gdp = gtd_count.merge(df_gdp[["Country Name", "average_gdp"]], left_on="country_txt", right_on="Country Name", how="left")

# Use log to scale the data
gtd_gdp["log_average_gdp"] = np.log(gtd_gdp["average_gdp"])
gtd_gdp["log_attack_count"] = np.log(gtd_gdp["attack_count"])
gtd_gdp_clean = gtd_gdp.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_average_gdp", "log_attack_count"])

# %%
# Make the plot using matplotlib
def plot_gdp_vs_attacks():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.scatter(gtd_gdp_clean["log_average_gdp"], gtd_gdp_clean["log_attack_count"], color="steelblue", alpha=0.6, s=60)
    
    # Fit and plot a linear regression line
    m, b = np.polyfit(gtd_gdp_clean["log_average_gdp"], gtd_gdp_clean["log_attack_count"], 1)
    ax.plot(gtd_gdp_clean["log_average_gdp"], m * gtd_gdp_clean["log_average_gdp"] + b, color="red")
    
    ax.set_xlabel("Log Average GDP ($)", fontsize=10)
    ax.set_ylabel("Log Number of Attacks", fontsize=10)
    ax.set_title("Average GDP vs Number of Terrorist Attacks", fontsize=12, fontweight='bold')
    return fig

# %%
# Categorize countries into three regions
SA = ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Pakistan', 
      'Maldives', 'Nepal', 'Sri Lanka']

gtd_clean['region'] = gtd_clean['country_txt'].apply(
    lambda x: 'Iraq' if x == 'Iraq' else 
              'South Asia' if x in SA else 
              'Other Regions'
)

gtd_gnames = gtd_clean[['region', 'gname']]
gtd_gnames = gtd_gnames.groupby(['region', 'gname']).size().reset_index(name='count')
gtd_gnames['Percentage'] = gtd_gnames.groupby('region')['count'].transform(lambda x: x / x.sum())
gtd_gnames.sort_values(by=['region', 'Percentage'], ascending=[True, False], inplace=True)
gtd_gnames = gtd_gnames.groupby('region').head(10)

# Define a function to plot the dynamic chart using matplotlib
def plot_group_names(region):
    data_filtered = gtd_gnames[gtd_gnames['region'] == region]

    seen_names = {}
    def unique_truncate(name, max_length=20):
        truncated = name if len(name) <= max_length else name[:17] + '...'
        if truncated in seen_names:
            seen_names[truncated] += 1
            truncated = f"{truncated}({seen_names[truncated]})"
        else:
            seen_names[truncated] = 0
        return truncated
    
    data_filtered['gname_short'] = data_filtered['gname'].apply(unique_truncate)

    n_colors = len(data_filtered)
    cmap = cm.get_cmap('tab20', n_colors)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_colors)]

    fig = go.Figure(data=[go.Bar(
        x=data_filtered['gname_short'],
        y=data_filtered['Percentage'],
        marker=dict(color=colors),
        hovertext=data_filtered['gname'],
        hoverinfo="text+y",
        showlegend=False
        )
    ])
    fig.update_layout(
        title={
            'text': f'Groups Responsible for Most Attacks in {region}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 16,
                'weight': 'bold'
            }
        },
        xaxis_tickangle=-45,
        title_font_size=12,
        xaxis_title_font_size=10,
        yaxis_title_font_size=10,
        plot_bgcolor='#fdfaf4',
        paper_bgcolor='white'
    )
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_content = buffer.getvalue()
    return ui.HTML(html_content)

# %%
# NLP for Taliban motivations
df_gtd_Taliban = gtd_clean[gtd_clean["gname"] == "Taliban"]
motive_texts_Taliban = df_gtd_Taliban["motive"].dropna().tolist()

# Add some meaningless words to the stop words list
custom_stop_words = list(ENGLISH_STOP_WORDS.union({
    "motive", "specific", "attack", "sources", "stated", 
    "claimed", "responsibility", "noted", "carried","unknown", "taliban", "incident", "information", "targeted", "victims", "accused", "victim", "posited", "suspected", "group", "believed"
}))

vectorizer = TfidfVectorizer(stop_words=custom_stop_words, min_df=3)
tfidf_matrix = vectorizer.fit_transform(motive_texts_Taliban)

# Get the key words and corresponding weights
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1

# Transfer to dataframe
tfidf_df = pd.DataFrame({"Word": feature_names, "Score": tfidf_scores})
tfidf_df = tfidf_df.sort_values(by="Score", ascending=False).head(10)

# Make the plot
def Taliban_nlp():
    fig, ax = plt.subplots(figsize=(10, 6))

    words = tfidf_df['Word']
    scores = tfidf_df['Score']

    ax.bar(words, scores, color="steelblue")
    ax.set_title("Top 10 Keywords of Taliban's Motive", fontsize=16, weight='bold')
    ax.set_xlabel("Keywords", fontsize=14)
    ax.set_ylabel("TF-IDF Score", fontsize=14)
    plt.xticks(rotation=-40, ha='left', fontsize=12)
    plt.yticks(fontsize=12)
    return fig

# %%
# NLP for ISIL motivations
df_gtd_ISIL = gtd_clean[gtd_clean["gname"] == "Islamic State of Iraq and the Levant (ISIL)"]
motive_texts_ISIL = df_gtd_ISIL["motive"].dropna().tolist()

# Add some meaningless words to the stop words list
custom_stop_words_ISIL = list(ENGLISH_STOP_WORDS.union({
    "motive", "specific", "attack", "sources", "stated", 
    "claimed", "responsibility", "noted", "carried","unknown", "isil", "incident", "information", "targeted", "victims", "accused", "victim", "posited", "suspected", "group", "believed", "area"
}))

vectorizer_ISIL = TfidfVectorizer(stop_words=custom_stop_words_ISIL, min_df=3)
tfidf_matrix_ISIL = vectorizer_ISIL.fit_transform(motive_texts_ISIL)

feature_names_ISIL = vectorizer_ISIL.get_feature_names_out()
tfidf_scores_ISIL = tfidf_matrix_ISIL.sum(axis=0).A1

# Transfer to dataframe
tfidf_df_ISIL = pd.DataFrame({"Word": feature_names_ISIL, "Score": tfidf_scores_ISIL})
tfidf_df_ISIL = tfidf_df_ISIL.sort_values(by="Score", ascending=False).head(10)

def ISIL_nlp():
    fig, ax = plt.subplots(figsize=(10, 6))
    words = tfidf_df_ISIL['Word']
    scores = tfidf_df_ISIL['Score']

    ax.bar(words, scores, color="steelblue")
    ax.set_title("Top 10 Keywords of ISIL's Motive", fontsize=16, weight='bold', loc='center')
    ax.set_xlabel("Keywords", fontsize=14)
    ax.set_ylabel("TF-IDF Score", fontsize=14)
    plt.xticks(rotation=-40, ha='left', fontsize=12)
    plt.yticks(fontsize=12)
    return fig


# %%
# The contents of the first 'page' is a navset with two 'panels'.
page1 = ui.navset_card_underline(
    ui.nav_panel("Plot", [
        ui.output_ui("dynamic_map"),
        ui.input_select("variable", "Select Variable:", choices=["attack_count", "casualties"])
    ]),
    ui.nav_panel("Table", [
        ui.output_data_frame("data"),
        ui.div(
            ui.input_select("year", "Choose a Year:", choices=sorted(gtd_table['iyear'].unique().astype(str))),
            ui.input_select("sort_by", "Sort by:", choices=["attack_count", "casualties"]),
            style="display: flex; gap: 20px; margin-top: 15px;"
        )
    ]),
    title="Global Terrorism Data",
)

# Define the layout for the second page with two plots side by side
page2 = ui.div(
    ui.div(
        ui.output_plot("democracy_plot"),
        style="flex: 1; padding: 20px; border: 1px solid #d3d3d3; margin-right: 10px; background-color: #ffffff;"
    ),
    ui.div(
        ui.output_plot("economics_plot"),
        style="flex: 1; padding: 20px; border: 1px solid #d3d3d3; background-color: #ffffff;"
    ),
    style="display: flex; justify-content: space-between; align-items: center; padding: 20px; background-color: #f9f9f9;"
)

# Define the layout for the third page with dynamic plot
page3 = ui.div(
    ui.div(
        ui.input_select("region", "Select Region:", choices=["South Asia", "Iraq", "Other Regions"]),
        style="margin-bottom: 20px;"
    ),
    ui.output_ui("group_names_plot"),
    ui.div(
        ui.output_plot('Taliban_motivation'),
        style="flex: 1; padding: 20px; border: 1px solid #d3d3d3; margin-right: 10px; background-color: #ffffff;"
    ),
    ui.div(
        ui.output_plot('ISIL_motivation'),
        style="flex: 1; padding: 20px; border: 1px solid #d3d3d3; margin-right: 10px; background-color: #ffffff;"
    ),
    style="padding: 20px; background-color: #f9f9f9;"
)

app_ui = ui.div(
    ui.page_navbar(
        ui.nav_spacer(),  # Push the navbar items to the right
        ui.nav_panel("Page 1", page1),
        ui.nav_panel("Page 2", page2),
        ui.nav_panel("Page 3", page3),
        title="Global Terrorism Analysis (2000-2020)"
    ),
    style="padding: 20px; font-family: Arial, sans-serif; background-color: #f3f3f3; color: #333;"
)

# %%
def server(input, output, session):
    @output
    @render.ui
    def dynamic_map():
        variable = input.variable()
        print(f"Selected variable: {variable}")  # Debug
        if variable not in gtd_map.columns:
            return ui.div(f"Variable '{variable}' not found in the data.")
        
        fig = px.choropleth(
            gtd_map,
            geojson=gtd_map.set_geometry('geometry').__geo_interface__,
            locations='name',
            featureidkey='properties.name',
            color=variable,
            hover_name='name',
            hover_data=[variable],
            color_continuous_scale='Reds'
        )
        fig.update_geos(showcoastlines=False, visible=False, bgcolor='lightblue')
        fig.update_layout(height=500, margin=dict(r=0, t=40, l=0, b=0))

        # Generate HTML plot
        buffer = io.StringIO()
        fig.write_html(buffer)
        return ui.HTML(buffer.getvalue())


    @render.data_frame
    def data():
        year = int(input.year())
        sort_by = input.sort_by()
        filtered_data = gtd_table[gtd_table['iyear'] == year][["country_txt", "iyear", "attack_count", "casualties"]]
        sorted_data = filtered_data.sort_values(by=sort_by, ascending=False)
        return sorted_data
    
    @render.plot
    def democracy_plot():
        fig = plot_attacks_vs_democracy()
        return fig
    
    @render.plot
    def economics_plot():
        fig = plot_gdp_vs_attacks()
        return fig
    
    @render.ui
    def group_names_plot():
        region = input.region()
        fig_html = plot_group_names(region)
        return fig_html
    
    @render.plot
    def Taliban_motivation():
        fig = Taliban_nlp()
        return fig
    
    @render.plot
    def ISIL_motivation():
        fig = ISIL_nlp()
        return fig



# %%
app = App(app_ui, server)



import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
import seaborn.objects as so
import matplotlib.colors as mcolors  

st.set_page_config(layout="wide")

blue = sns.color_palette("Blues", n_colors=5)
orange = sns.color_palette("Oranges", n_colors=5)
yellow = sns.color_palette("YlOrBr", n_colors=5)
green = sns.color_palette("Greens", n_colors=5)
reds = sns.color_palette("Reds", n_colors=5)

custom_colors = {
    'cat_A': mcolors.rgb2hex(green[2]),   
    'cat_A_0': mcolors.rgb2hex(green[1]),  
    'cat_A_1': mcolors.rgb2hex(green[3]),  
    
    'cat_B': mcolors.rgb2hex(yellow[1]), 
    'cat_B_0': mcolors.rgb2hex(yellow[0]),  
    'cat_B_1': mcolors.rgb2hex(yellow[2]),  
    
    'cat_C': mcolors.rgb2hex(orange[3]),  
    'cat_C_0': mcolors.rgb2hex(orange[2]),  
    'cat_C_1': mcolors.rgb2hex(orange[4]),  
    
    'cat_D': mcolors.rgb2hex(reds[3]),  
    'cat_D_0': mcolors.rgb2hex(reds[2]),  
    'cat_D_1': mcolors.rgb2hex(reds[4]),  

    'cat_todos': mcolors.rgb2hex(blue[2]),  
    'cat_todos_0': mcolors.rgb2hex(blue[1]),  
    'cat_todos_1': mcolors.rgb2hex(blue[3])   
}

def adjust_palette(palette, is_lighter=True):
    if is_lighter:
        return sns.light_palette(palette[1], n_colors=len(palette), reverse=False)
    else:
        return sns.dark_palette(palette[1], n_colors=len(palette), reverse=True)

blue_0 = adjust_palette(blue, is_lighter=True)  # Lighter blue
blue_1 = adjust_palette(blue, is_lighter=False)  # Darker blue

orange_0 = adjust_palette(orange, is_lighter=True)  # Lighter orange
orange_1 = adjust_palette(orange, is_lighter=False)  # Darker orange

yellow_0 = adjust_palette(yellow, is_lighter=True)  # Lighter yellow
yellow_1 = adjust_palette(yellow, is_lighter=False)  # Darker yellow

green_0 = adjust_palette(green, is_lighter=True)    
green_1 = adjust_palette(green, is_lighter=False)  

red_0 = adjust_palette(reds, is_lighter=True)    
red_1 = adjust_palette(reds, is_lighter=False)

data_dictionary = {
    'Gráficos':'graficos',
    'Fecha de recepción de datos': 'date_recepcion_data',
    'Edad': 'age',
    'Provincia': 'provincia',
    'Géneros': 'genero',
    'Percepción de cambio': 'RECOMENDACIONES_AJUSTE',
    'Exposición luz natural': 'FOTICO_luz_natural_8_15_integrada',
    'Exposición luz artificial': 'FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada',
    'Estudios no foticos integrados': 'NOFOTICO_estudios_integrada',
    'Trabajo no fotico integrado': 'NOFOTICO_trabajo_integrada',
    'Otra actividad habitual no fotica': 'NOFOTICO_otra_actividad_habitual_si_no',
    'Cena no fotica integrada': 'NOFOTICO_cena_integrada',
    'Horario de acostarse - Hábiles': 'HAB_Hora_acostar',
    'Horario decidir dormir - Hábiles': 'HAB_Hora_decidir',
    'Minutos dormir - Hábiles': 'HAB_min_dormir',
    'Hora despertar - Hábiles': 'HAB_Soffw',
    'Alarma - Hábiles': 'NOFOTICO_HAB_alarma_si_no',
    'Siesta habitual integrada': 'HAB_siesta_integrada',
    'Calidad de sueño - Hábiles': 'HAB_calidad',
    'Horario de acostarse - Libres': 'LIB_Hora_acostar',
    'Horario decidir dormir - Libres': 'LIB_Hora_decidir',
    'Minutos dormir - Libres': 'LIB_min_dormir',
    'Hora despertar - Libres': 'LIB_Offf',
    'Alarma - Libres': 'LIB_alarma_si_no',
    'Recomendación - Alarma no fotica (sí/no)': 'rec_NOFOTICO_HAB_alarma_si_no',
    'Recomendación - Luz natural (8-15)': 'rec_FOTICO_luz_natural_8_15_integrada',
    'Recomendación - Luz artificial (8-15)': 'rec_FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada',
    'Recomendación - Estudios no foticos integrados': 'rec_NOFOTICO_estudios_integrada',
    'Recomendación - Trabajo no fotico integrado': 'rec_NOFOTICO_trabajo_integrada',
    'Recomendación - Otra actividad habitual no fotica (sí/no)': 'rec_NOFOTICO_otra_actividad_habitual_si_no',
    'Recomendación - Cena no fotica integrada': 'rec_NOFOTICO_cena_integrada',
    'Recomendación - Siesta habitual integrada': 'rec_HAB_siesta_integrada',
    'MEQ Puntaje total': 'MEQ_score_total',
    'MSFsc': 'MSFsc',
    'Duración Del Sueño - Hábiles': 'HAB_SDw',
    'Desviación Jet Lag Social': 'SJL',
    'Hora de inicio de sueño no laboral centrada': 'HAB_SOnw_centrado'
}
age_categories = ['A', 'B', 'C', 'D']
category_colors = {'A': custom_colors['cat_A'],'B': custom_colors['cat_B'],'C': custom_colors['cat_C'], 'D': custom_colors['cat_D']}
category_colors_gender = {'A': [custom_colors['cat_A_0'], custom_colors['cat_A_1']],'B': [custom_colors['cat_B_0'], custom_colors['cat_B_1']],'C': [custom_colors['cat_C_0'], custom_colors['cat_C_1']], 'D': [custom_colors['cat_D_0'], custom_colors['cat_D_1']]}

class Authentication:
    def __init__(self):
        self.credentials = {
            "MRI": "MRI-TABLERO-2025"  # Replace with your desired username and password
        }

    def validate_user(self, username, password):
        return self.credentials.get(username) == password

class DatabaseUploader:
    def __init__(self):
        self.uploaded_before = None
        self.uploaded_after = None

    def upload_data(self):
        # Manually upload CSV files using the file uploader
        self.uploaded_before = st.file_uploader("Cargar la base de datos previo al crash en formato CSV")
        self.uploaded_after = st.file_uploader("Cargar la base de datos posterior al crash en formato CSV")

        return self.uploaded_before, self.uploaded_after

    def load_data(self, data_loader):
        # Load the data using DataLoader
        if self.uploaded_before is not None and self.uploaded_after is not None:
            with st.spinner("Loading data..."):
                return data_loader.load_data(self.uploaded_before, self.uploaded_after, 'Geo.csv')

        return None

class InstructivoApp:
    def display(self):
        st.title("Instructivo para usar la Aplicación")
        st.markdown("[Link para sugerir cambios](https://docs.google.com/document/d/1oy7gBG45nn5Netl34mc3leJ3IgIqcNyHN-72geDS6W4/edit?usp=drive_link)")

        st.write("""
            Bienvenido a la aplicación de análisis de datos. Aquí te explicamos cómo usar cada sección:

            1. **Carga de Datos**: 
               - Sube dos archivos CSV, uno con datos previos al 'Crash' y otro con los datos posteriores en formato CSV.
            
            2. **Seleccione la cantidad de gráficos que desea ver**:
               - Define la cantidad de gráficos que podrás ver en simultáneo.
               - Para cada gráfico se puede cambiar lo que se desea ver y los filtros.

            3. **Gráficos**:
                - Selecciona el gráfico que deseas ver.
                - **Entradas - Usuarios**:
                    - **Entradas**: Muestra todos los ID que usaron la aplicación; pueden aparecer ID repetidos, ya que un usuario puede usar muchas veces la aplicación.
                    - **Usuarios**: Muestra los ID únicos, conservando solo una entrada por cada usuario que haya usado la aplicación más de una vez. Por defecto, toma la última entrada.

                - **Recomendaciones**:
                    - **Siguieron recomendaciones**:
                        - **Sí**: Usuarios que siguieron las recomendaciones.
                        - **No**: Usuarios que no siguieron las recomendaciones.
                        - **Ambas**: Muestra tanto los usuarios que siguieron las recomendaciones como los que no.

                    - **Diferencia de días mínimo**: La cantidad mínima de días entre entradas de un mismo usuario.
                    - **Diferencia de días máximo**: La cantidad máxima de días entre entradas de un mismo usuario.

                    - **Antes - Después**:
                        - **Antes**: La primera entrada de un usuario que usó más de una vez la aplicación.
                        - **Después**: La segunda entrada de un usuario que usó más de una vez la aplicación.
                        - **Ambas**: Muestra tanto la primera entrada como la segunda entrada.
                        - **Antes vs Después**: Muestra en los gráficos, de manera superpuesta, la diferencia entre las entradas antes y después.

                - **Fechas**: Filtra por intervalo de fechas de las entradas de datos.
                - **Género**: Filtra por género.
                - **Edades**:
                    - Permite seleccionar el rango de edades a visualizar.
                    - Permite seleccionar el rango etario a visualizar.

                - **Configurar rango etario**:
                    - Configura los grupos de rango etario, hasta un máximo de 4 grupos.

                - **Mostrar datos**:
                    - Muestra el dataset con los filtros seleccionados.

                - **Filtrar por usuarios**:
                    - Permite filtrar usuarios por ID y seleccionar la cantidad de entradas que se desean ver de ese usuario.
        """)

class DataLoader: 
    def __init__(self):
        self.df = pd.DataFrame()
        
    def load_data(self, before_path, after_path, geo_path):
        df_before = pd.read_csv(before_path)
        df_after = pd.read_csv(after_path)
        df_geo = pd.read_csv(geo_path,sep=';')
        self.df = pd.concat([df_before, df_after], ignore_index=True)
        self.df['date_recepcion_data'] = pd.to_datetime(self.df['date_recepcion_data'])
        self.df.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['days_diff'] = self.df.groupby('user_id')['date_recepcion_data'].diff().dt.days.fillna(0)
        self.df = pd.merge(self.df, df_geo, how='left', on='provincia')
       
       # columns_to_fix = ['rec_NOFOTICO_HAB_alarma_si_no','rec_FOTICO_luz_natural_8_15_integrada','rec_FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada','rec_NOFOTICO_estudios_integrada','rec_NOFOTICO_trabajo_integrada','rec_NOFOTICO_otra_actividad_habitual_si_no','rec_NOFOTICO_cena_integrada','rec_HAB_siesta_integrada']
       #self.df[columns_to_fix] = self.df[columns_to_fix].fillna('None').astype(str)
        
        time = pd.to_datetime(self.df['HAB_Hora_acostar'], format='%H:%M')
        self.df['HAB_Hora_acostar'] = time.dt.hour + time.dt.minute / 60
        
        time = pd.to_datetime(self.df['HAB_Hora_decidir'], format='%H:%M')
        self.df['HAB_Hora_decidir'] = time.dt.hour + time.dt.minute / 60
        
        time = pd.to_datetime(self.df['HAB_Soffw'], format='%H:%M')
        self.df['HAB_Soffw'] = time.dt.hour + time.dt.minute / 60
        
        time = pd.to_datetime(self.df['LIB_Hora_acostar'], format='%H:%M')
        self.df['LIB_Hora_acostar'] = time.dt.hour + time.dt.minute / 60
        
        time = pd.to_datetime(self.df['LIB_Hora_decidir'], format='%H:%M')
        self.df['LIB_Hora_decidir'] = time.dt.hour + time.dt.minute / 60
        
        time = pd.to_datetime(self.df['LIB_Offf'], format='%H:%M')
        self.df['LIB_Offf'] = time.dt.hour + time.dt.minute / 60

        time = self.df['HAB_SDw']
        self.df['HAB_SDw'] = time / 60
        
        columns_int = [
            'RECOMENDACIONES_AJUSTE',
            'rec_NOFOTICO_HAB_alarma_si_no',
            'rec_FOTICO_luz_natural_8_15_integrada',
            'rec_FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada',
            'rec_NOFOTICO_estudios_integrada',
            'rec_NOFOTICO_otra_actividad_habitual_si_no',
            'rec_NOFOTICO_cena_integrada',
            'rec_HAB_siesta_integrada'
        ]

        self.convert_columns_to_int(columns_int)    
            
        self.df = self.categorize_age(self.df,20,60,80)
        
        return self.df
    
   
    def categorize_age(self, df, age_b_min, age_c_min, age_d_min):
        def age_category(age):
            if age < age_b_min:
                return 'A'  # Youngest
            elif age < age_c_min:
                return 'B'
            elif age < age_d_min:
                return 'C'
            else:
                return 'D'  # Oldest
        df['age_category'] = df['age'].apply(age_category)
        return df

    def convert_columns_to_int(self, columns_to_convert):
        def convert_value(value):
            if pd.isna(value):  # Check for NaN values
                return 'None'  # Return the string 'None' for NaN values
            elif value is None:
                return 'None'  # Return the string 'None' for None values
            elif value == 'None':
                return 'None'  # Return the string 'None' if it is already that
            elif value == 'XX':
                return value  # Keep 'XX' unchanged
            else:
                return int(float(value))  # Convert valid string/float values to int
        for column in columns_to_convert:
            if column != 'RECOMENDACIONES_AJUSTE':
                # Apply conversion to all columns except 'RECOMENDACIONES_AJUSTE'
                self.df[column] = self.df[column].apply(convert_value)
            else:
                # Handle 'RECOMENDACIONES_AJUSTE' specifically
                self.df[column] = self.df[column].apply(lambda value: int(float(value)) if not pd.isna(value) and value != 'None' else value)
                     
class StreamLit:
    def __init__(self, df, plot_id):
        self.df = df
        self.plot_id = plot_id
        self.initialize_filters()

    def initialize_filters(self):
        if 'datos_' + self.plot_id not in st.session_state:
            st.session_state['datos_' + self.plot_id] = False
        
        if 'selected_gender_' + self.plot_id not in st.session_state:
            st.session_state['selected_gender_' + self.plot_id] = 0
       
        if 'age_category_selectbox_' + self.plot_id not in st.session_state:
            st.session_state['age_category_selectbox_' + self.plot_id] = 'Todos'

        if 'age_range_slider_' + self.plot_id not in st.session_state:
            st.session_state['age_range_slider_' + self.plot_id] = [self.df['age'].min(), self.df['age'].max()]

        if 'df_selected_' + self.plot_id not in st.session_state:
            st.session_state['df_selected_' + self.plot_id] = self.df

        if 'age_a_min_' + self.plot_id not in st.session_state:
            st.session_state['age_a_min_' + self.plot_id] = 18

        if 'age_b_min_' + self.plot_id not in st.session_state:
            st.session_state['age_b_min_' + self.plot_id] = 30

        if 'age_c_min_' + self.plot_id not in st.session_state:
            st.session_state['age_c_min_' + self.plot_id] = 60

        if 'age_d_min_' + self.plot_id not in st.session_state:
            st.session_state['age_d_min_' + self.plot_id] = 80
            
        if 'recommendations_selectbox_' + self.plot_id not in st.session_state:
            st.session_state['recommendations_selectbox_' + self.plot_id] = 'Si'

        if 'ambas_antes_despues_' + self.plot_id not in st.session_state:
            st.session_state['ambas_antes_despues_' + self.plot_id] = 'Antes'

        if 'entradas_usuarios_filter_' + self.plot_id not in st.session_state:
            st.session_state['entradas_usuarios_filter_' + self.plot_id] = 'Entradas'

        if 'all_dates_checkbox_' + self.plot_id not in st.session_state:
            st.session_state['all_dates_checkbox_' + self.plot_id] = False

        if 'all_ages_checkbox_' + self.plot_id not in st.session_state:
            st.session_state['all_ages_checkbox_' + self.plot_id] = False

        if 'all_genders_checkbox_' + self.plot_id not in st.session_state:
            st.session_state['all_genders_checkbox_' + self.plot_id] = False

        if 'all_recommendations_checkbox_' + self.plot_id not in st.session_state:
            st.session_state['all_recommendations_checkbox_' + self.plot_id] = False

        if 'min_days_diff_input_' + self.plot_id not in st.session_state:
            st.session_state['min_days_diff_input_' + self.plot_id] = 10

        if 'max_days_diff_input_' + self.plot_id not in st.session_state:
            st.session_state['max_days_diff_input_' + self.plot_id] = 30

        if 'rango_etario_' + self.plot_id not in st.session_state:
            st.session_state['rango_etario_' + self.plot_id] = False

        if 'define_age_category_' + self.plot_id not in st.session_state:
            st.session_state['define_age_category_' + self.plot_id] = False
        
        if 'entradas_usuarios_checkbox_' + self.plot_id not in st.session_state:
            st.session_state['entradas_usuarios_checkbox_' + self.plot_id] = False
        
        if 'plot_' + self.plot_id not in st.session_state:
            st.session_state['plot_' + self.plot_id] = 'Gráficos'
        
        if 'filtrar_usuarios_checkbox' + self.plot_id  not in st.session_state:
            st.session_state['filtrar_usuarios_checkbox' + self.plot_id ] = False
        
        if 'filtrar_entradas_checkbox' + self.plot_id  not in st.session_state:
            st.session_state['filtrar_entradas_checkbox' + self.plot_id ] = False
        
        if 'persepcion_checkbox_' + self.plot_id  not in st.session_state:
            st.session_state['persepcion_checkbox_' + self.plot_id ] = False
            
        if 'persepcion_selectbox_' + self.plot_id  not in st.session_state:
            st.session_state['persepcion_selectbox_' + self.plot_id ] = 0
    
    def sidebar(self):
      
        # Obtener el gráfico seleccionado
        selected_plot = st.sidebar.selectbox('Gráfico', list(data_dictionary.keys()), key='plot_' + self.plot_id)

        # Claves para el estado en session_state
        checkbox_key = 'entradas_usuarios_checkbox_' + self.plot_id
        filter_key = 'entradas_usuarios_filter_' + self.plot_id
        reset_key = 'reset_checkbox_' + self.plot_id

        # Inicializar el estado de la bandera si no existe
        if reset_key not in st.session_state:
            st.session_state[reset_key] = False

        # Comportamiento cuando se selecciona 'age'
        if data_dictionary.get(selected_plot) == 'age' or data_dictionary.get(selected_plot) == 'genero' :
            st.session_state[checkbox_key] = True
            st.session_state[filter_key] = "Usuarios"
            st.session_state[reset_key] = False  # Resetear la bandera
        else:
            # Si no es 'age', cerrar el checkbox una vez, luego permitir interacción normal
            if not st.session_state[reset_key]:
                st.session_state[checkbox_key] = False  # Cerrar el checkbox automáticamente
                st.session_state[reset_key] = True     # Marcar que ya se reseteó

        # Mostrar el checkbox y selectbox si está activado
        if st.sidebar.checkbox('Entradas - Usuarios', key=checkbox_key):
            st.sidebar.selectbox("Entrada Usuarios", options=["Entradas", "Usuarios"], key=filter_key)


        st.sidebar.checkbox("Recomendaciones", key='all_recommendations_checkbox_' + self.plot_id)
        if st.session_state['all_recommendations_checkbox_' + self.plot_id]:
            st.sidebar.selectbox("Siguieron recomendaciones", options=['Si', 'No',"Ambas"], key='recommendations_selectbox_'  + self.plot_id)
            st.sidebar.number_input("Min days difference", min_value=0, max_value=1000, value=10,  key='min_days_diff_input_'  + self.plot_id)
            st.sidebar.number_input("Max days difference", min_value=0, max_value=1000, value = 30,  key='max_days_diff_input_' + self.plot_id)
            st.sidebar.selectbox("Antes Después", options=["Antes", "Después",  'Antes vs Después'  ], key='ambas_antes_despues_' + self.plot_id)

        st.sidebar.checkbox(f'Fechas', key='all_dates_checkbox_' + self.plot_id)
        if  st.session_state['all_dates_checkbox_' + self.plot_id]:
            st.sidebar.date_input(f"Start Date", value=self.df['date_recepcion_data'].min(), key='start_date_input_' + self.plot_id)
            st.sidebar.date_input(f"End Date", value=self.df['date_recepcion_data'].max(), key='end_date_input_' + self.plot_id)
        
        st.sidebar.checkbox(f'Persepción del cambio', key='persepcion_checkbox_' + self.plot_id)
        if  st.session_state['persepcion_checkbox_' + self.plot_id]:
            st.sidebar.selectbox(f"Persepción del cambio", options=[0,1,2,3,4,5] , key='persepcion_selectbox_' + self.plot_id)

        st.sidebar.checkbox(f"Géneros", key='all_genders_checkbox_' + self.plot_id)
        if st.session_state['all_genders_checkbox_' + self.plot_id]:
            st.sidebar.selectbox(f"Seleccione el genero", options=self.df['genero'].unique().tolist() , key='selected_gender_' + self.plot_id)
        
        st.sidebar.checkbox(f"Distribución de edades", key='all_ages_checkbox_' + self.plot_id)
        if st.session_state['all_ages_checkbox_' + self.plot_id]:
            st.sidebar.slider(f"Seleccione el rango etario", min_value=int(self.df['age'].min()), max_value=int(self.df['age'].max()), value=(int(self.df['age'].min()), int(self.df['age'].max())), key='age_range_slider_' + self.plot_id)
            #st.sidebar.selectbox(f"Seleccionar Rango Etario", ['Todos', 'A', 'B', 'C', 'D'], key='age_category_selectbox_' + self.plot_id)
        
        # st.sidebar.checkbox("Configurar Rangos Etarios", key='define_age_category_' + self.plot_id )
        # if st.session_state['define_age_category_'+ self.plot_id] :
        #     st.sidebar.number_input("Edad mínimia para A", min_value=0, max_value=100 ,value = 18, key='age_a_min_'+ self.plot_id)
        #     st.sidebar.number_input("Edad mínimia para B", min_value=0, max_value=100,value = 30, key='age_b_min_'+ self.plot_id)
        #     st.sidebar.number_input("Edad mínimia para C", min_value=0, max_value=100,value = 60,  key='age_c_min_'+ self.plot_id)
        #     st.sidebar.number_input("Edad mínimia para D", min_value=0, max_value=100,value = 80,  key='age_d_min_'+ self.plot_id)
       
        st.sidebar.checkbox("Mostrar datos", key='datos_' + self.plot_id)

        if not st.session_state.get('filtrar_entradas_checkbox' + self.plot_id, False):
            filtrar_por_usuarios = st.sidebar.checkbox("Filtrar por usuarios", key='filtrar_usuarios_checkbox' + self.plot_id)
            if filtrar_por_usuarios:
                st.sidebar.text_input('Ingrese el ID del usuario', key='filtrar_usuarios_texto' + self.plot_id)

        if not st.session_state.get('filtrar_usuarios_checkbox' + self.plot_id, False):
            filtrar_por_entradas = st.sidebar.checkbox("Filtrar por cantidad de entradas", key='filtrar_entradas_checkbox' + self.plot_id)
            if filtrar_por_entradas:
                st.sidebar.number_input('Ingrese cantidad de entradas', key='filtrar_usuarios_cantidad' + self.plot_id, min_value=1, step=1, format="%d")

class Filters:
    def __init__(self, df, plot_id):
        self.df = df
        self.result = pd.DataFrame()
        self.result_antes = pd.DataFrame()
        self.result_despues = pd.DataFrame()
        self.plot_id = plot_id

    def entries_users(self, df):
        return df.drop_duplicates(subset='user_id', keep='last')

    def dates(self, df):
        date_min = pd.to_datetime(st.session_state[f'start_date_input_{self.plot_id}'])
        date_max = pd.to_datetime(st.session_state[f'end_date_input_{self.plot_id}'])
        return df[(df['date_recepcion_data'] >= date_min) & (df['date_recepcion_data'] <= date_max + pd.Timedelta(days=1))]

    def ages(self, df):
        age_min, age_max = st.session_state[f'age_range_slider_{self.plot_id}']
        return df[(df['age'] >= age_min) & (df['age'] <= age_max)]

    def genders(self, df, gender):
        return df[df['genero'] == gender ]
    
    def select_age_category(self, df, age_category):
        return df[df['age_category'] == age_category]

    def recomendations(self, df, days_min, days_max, rec_filter, when_filter):
        df = df.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True])
        df = df.reset_index(drop=True)
        final_indices = set()  # Usamos un conjunto para evitar duplicados
        if st.session_state['persepcion_checkbox_' + self.plot_id]==True:
            for idx in range(1, len(df)):
                if df.loc[idx - 1, 'user_id'] == df.loc[idx, 'user_id']:
                    if df.loc[idx, 'RECOMENDACIONES_AJUSTE'] == st.session_state['persepcion_selectbox_' + self.plot_id]:
                        if rec_filter == 'Ambas':
                            if df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] in ['si', 'no']:
                                if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                                        if when_filter == 'Ambas':
                                            final_indices.update([idx - 1, idx])  # Añadimos ambos índices al conjunto
                                        elif when_filter == 'Antes':
                                            final_indices.add(idx - 1)
                                        elif when_filter == 'Después':
                                            final_indices.add(idx)
                        elif rec_filter == 'Si' and df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] == 'si':
                            if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                                if when_filter == 'Ambas':
                                    final_indices.update([idx - 1, idx])
                                elif when_filter == 'Antes':
                                    final_indices.add(idx - 1)
                                elif when_filter == 'Después':
                                    final_indices.add(idx)
                        elif rec_filter == 'No' and df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] == 'no':
                            if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                                if when_filter == 'Ambas':
                                    final_indices.update([idx - 1, idx])
                                elif when_filter == 'Antes':
                                    final_indices.add(idx - 1)
                                elif when_filter == 'Después':
                                    final_indices.add(idx)
        if st.session_state['persepcion_checkbox_' + self.plot_id]==False:
            for idx in range(1, len(df)):
                if df.loc[idx - 1, 'user_id'] == df.loc[idx, 'user_id']:
                    if rec_filter == 'Ambas':
                        if df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] in ['si', 'no']:
                            if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                                if df.loc[idx, 'RECOMEDACIONES_AJUSTE'] == st.session_state['persepcion_selectbox_' + self.plot_id]:
                                    if when_filter == 'Ambas':
                                        final_indices.update([idx - 1, idx])  # Añadimos ambos índices al conjunto
                                    elif when_filter == 'Antes':
                                        final_indices.add(idx - 1)
                                    elif when_filter == 'Después':
                                        final_indices.add(idx)
                    elif rec_filter == 'Si' and df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] == 'si':
                        if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                            if when_filter == 'Ambas':
                                final_indices.update([idx - 1, idx])
                            elif when_filter == 'Antes':
                                final_indices.add(idx - 1)
                            elif when_filter == 'Después':
                                final_indices.add(idx)
                    elif rec_filter == 'No' and df.loc[idx, 'SEGUISTE_RECOMENDACIONES'] == 'no':
                        if days_min <= df.loc[idx, 'days_diff'] <= days_max:
                            if when_filter == 'Ambas':
                                final_indices.update([idx - 1, idx])
                            elif when_filter == 'Antes':
                                final_indices.add(idx - 1)
                            elif when_filter == 'Después':
                                final_indices.add(idx)
        final_indices = sorted(final_indices)
        return df.loc[final_indices].reset_index(drop=True)
    
    def persepcion(self, df):
        df = df.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True])
        df = df.reset_index(drop=True)
        final_indices = set()
        if st.session_state['all_recommendations_checkbox_' + self.plot_id] == False:
            for idx in range(1, len(df)):
                if  df.loc[idx - 1, 'user_id'] == df.loc[idx, 'user_id']:
                    if df.loc[idx, 'RECOMENDACIONES_AJUSTE'] == st.session_state['persepcion_selectbox_' + self.plot_id]:
                        final_indices.update([idx , idx-1])
        final_indices = sorted(final_indices)
        return df.loc[final_indices].reset_index(drop=True)


    def categorize_age(self, df, age_b_min, age_c_min, age_d_min):
        def age_category(age):
            if age < age_b_min:
                return 'A'  # Youngest
            elif age < age_c_min:
                return 'B'
            elif age < age_d_min:
                return 'C'
            else:
                return 'D'  # Oldest

        df['age_category'] = df['age'].apply(age_category)
        return df

    def users(self, df, user_id=None):
        if user_id is None:
            return df  
        
        filtered_df = df[df['user_id'] == user_id]
        if filtered_df.empty:
            return df  # Devuelve el DataFrame original si no se encuentra el usuario
        
        return filtered_df  # Devuelve el DataFrame filtrado si se encuentra el usuario

    def users_count(self, df, n):
        user_counts = df['user_id'].value_counts()
        repeated_users = user_counts[user_counts == n].index
        df = df[df['user_id'].isin(repeated_users)]
        return df

    def choose_filter(self):
        self.result = self.df
        self.result_antes = self.df
        self.result_despues = self.df
    
        if  st.session_state['filtrar_entradas_checkbox' + self.plot_id ] == True: 
            if 'filtrar_usuarios_cantidad' + self.plot_id in st.session_state:
                self.result = self.users_count(self.result, st.session_state['filtrar_usuarios_cantidad' + self.plot_id])
                self.result_antes = self.users_count(self.result_antes, st.session_state['filtrar_usuarios_cantidad' + self.plot_id])
                self.result_despues = self.users_count(self.result_despues, st.session_state['filtrar_usuarios_cantidad' + self.plot_id])

        if st.session_state[f'all_dates_checkbox_{self.plot_id}']:
            self.result = self.dates(self.result)
            self.result_antes = self.dates(self.result_antes)
            self.result_despues = self.dates(self.result_despues)
            
        if st.session_state[f'persepcion_selectbox_{self.plot_id}'] and not st.session_state[f'all_recommendations_checkbox_{self.plot_id}']:
            self.result = self.persepcion(self.result)
            self.result_antes = self.persepcion(self.result_antes)
            self.result_despues = self.persepcion(self.result_despues)
            
        if st.session_state[f'all_ages_checkbox_{self.plot_id}']:
            self.result = self.ages(self.result)
            self.result_antes = self.ages(self.result_antes)
            self.result_despues = self.ages(self.result_despues)

        if st.session_state[f'all_genders_checkbox_{self.plot_id}']:
            if  st.session_state[f'selected_gender_{self.plot_id}'] != '0 vs 1':
                genero = st.session_state[f'selected_gender_{self.plot_id}']
                self.result = self.genders(self.result, genero)
                self.result_antes = self.genders(self.result_antes, genero)
                self.result_despues = self.genders(self.result_despues, genero)

        if  st.session_state[f'all_recommendations_checkbox_{self.plot_id}']:
            if st.session_state[f'ambas_antes_despues_{self.plot_id}'] != 'Antes vs Después':
                 self.result = self.recomendations(self.result, days_min=st.session_state[f'min_days_diff_input_{self.plot_id}'],days_max=st.session_state[f'max_days_diff_input_{self.plot_id}'], rec_filter=st.session_state[f'recommendations_selectbox_{self.plot_id}'], when_filter=st.session_state[f'ambas_antes_despues_{self.plot_id}'])
            else:
                self.result_antes = self.recomendations(self.result_antes, st.session_state[f'min_days_diff_input_{self.plot_id}'],days_max=st.session_state[f'max_days_diff_input_{self.plot_id}'],rec_filter=st.session_state[f'recommendations_selectbox_{self.plot_id}'], when_filter='Antes')
                self.result_despues = self.recomendations(self.result_despues, st.session_state[f'min_days_diff_input_{self.plot_id}'],days_max=st.session_state[f'max_days_diff_input_{self.plot_id}'],rec_filter=st.session_state[f'recommendations_selectbox_{self.plot_id}'], when_filter='Después')    

        # if  st.session_state[f'rango_etario_{self.plot_id}']:
        #     self.result = self.select_age_category(self.result)
        #     self.result_antes = self.select_age_category(self.result_antes)
        #     self.result_despues = self.select_age_category(self.result_despues)

        # if st.session_state[f'age_category_selectbox_{self.plot_id}'] != 'Todos':
        #     self.result = self.select_age_category(self.result, st.session_state[f'age_category_selectbox_{self.plot_id}'])
        #     self.result_antes = self.select_age_category(self.result_antes, st.session_state[f'age_category_selectbox_{self.plot_id}'])
        #     self.result_despues = self.select_age_category(self.result_despues, st.session_state[f'age_category_selectbox_{self.plot_id}'])

        # if  st.session_state['define_age_category_' + self.plot_id]:  
        #     if (st.session_state['age_b_min_' + self.plot_id] or st.session_state['age_c_min_' + self.plot_id] or st.session_state['age_d_min_' + self.plot_id]):  
        #         self.result = self.categorize_age(self.result, st.session_state['age_b_min_' + self.plot_id],st.session_state['age_c_min_' + self.plot_id],st.session_state['age_d_min_' + self.plot_id])
        #         self.result_antes = self.categorize_age(self.result_antes,st.session_state['age_b_min_' + self.plot_id],st.session_state['age_c_min_' + self.plot_id], st.session_state['age_d_min_' + self.plot_id])
        #         self.result_despues = self.categorize_age(self.result_despues,st.session_state['age_b_min_' + self.plot_id],st.session_state['age_c_min_' + self.plot_id],st.session_state['age_d_min_' + self.plot_id])

        if  st.session_state['filtrar_usuarios_checkbox' + self.plot_id ] == True: 
            if 'filtrar_usuarios_texto' + self.plot_id in st.session_state:
                self.result = self.users(self.result, st.session_state['filtrar_usuarios_texto' + self.plot_id])
                self.result_antes = self.users(self.result_antes, st.session_state['filtrar_usuarios_texto' + self.plot_id])
                self.result_despues = self.users(self.result_despues, st.session_state['filtrar_usuarios_texto' + self.plot_id])
        
        if  st.session_state['entradas_usuarios_checkbox_' + self.plot_id] == True:
            if st.session_state[f'entradas_usuarios_filter_{self.plot_id}'] == 'Usuarios':
                self.result = self.entries_users(self.result)
                self.result_antes = self.entries_users(self.result_antes)
                self.result_despues = self.entries_users(self.result_despues)
            
class PlotGenerator:
    def __init__(self, df, df_combinado, plot_id):
        self.df = df
        self.plot_id = plot_id
        self.count = None
        self.bins = None
        self.color_pie = blue
        self.x = None
        self.y = None
        self.x_label = None
        self.y_label = None
        self.rotation = None
        self.rotation2 = None
        self.title = None
        self.y_visible = True
        self.order = None
        self.hue = None
        self.df_combinado = df_combinado
        self.pie = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
        self.fontsize2 = 10
        self.default_palette = sns.color_palette("Blues", 6)  # Escala continua de azules
        self.color = self.default_palette[3]  # Primer color en la escala (el más oscuro)


    
    # def colors(self):
    #     age_category = st.session_state['age_category_selectbox_' + self.plot_id]
    #     gender = st.session_state['selected_gender_' + self.plot_id]

    #     if st.session_state['all_genders_checkbox_' + self.plot_id] and st.session_state['all_ages_checkbox_' + self.plot_id]:
    #         self.color_pie = blue
    #         self.color = custom_colors['cat_todos']
        
    #     elif  st.session_state['all_genders_checkbox_' + self.plot_id] and st.session_state['all_ages_checkbox_' + self.plot_id]:
    #         if gender == 0:
    #             self.color_pie = blue_0
    #             self.color = custom_colors['cat_todos_0']
    #         elif gender == 1:
    #             self.color_pie = blue_1
    #             self.color = custom_colors['cat_todos_1']

    #     elif  st.session_state['all_genders_checkbox_' + self.plot_id] and not st.session_state['all_ages_checkbox_' + self.plot_id]:
            
    #         if age_category == 'Todos':
    #             if gender == 0:
    #                 self.color = custom_colors['cat_todos_0']
    #                 self.color_pie = blue_0
    #             elif gender == 1:
    #                 self.color = custom_colors['cat_todos_1']
    #                 self.color_pie = blue_1
            
    #         elif age_category == 'A':
    #             if gender == 0:
    #                 self.color = custom_colors['cat_A_0']
    #                 self.color_pie = green_0
    #             elif gender == 1:
    #                 self.color = custom_colors['cat_A_1']
    #                 self.color_pie = green_1

    #         elif age_category == 'B':
    #             if gender == 0:
    #                 self.color = custom_colors['cat_B_0']
    #                 self.color_pie = yellow_0
    #             elif gender == 1:
    #                 self.color = custom_colors['cat_B_1']
    #                 self.color_pie = yellow_1

    #         elif age_category == 'C':
    #             if gender == 0:
    #                 self.color = custom_colors['cat_C_0']
    #                 self.color_pie = orange_0
    #             elif gender == 1:
    #                 self.color = custom_colors['cat_C_1']
    #                 self.color_pie = orange_1
            
    #         elif age_category == 'D':
    #             if gender == 0:
    #                 self.color = custom_colors['cat_D_0']
    #                 self.color_pie = red_0
    #             elif gender == 1:
    #                 self.color = custom_colors['cat_D_1']
    #                 self.color_pie = red_1

    #     elif st.session_state['all_genders_checkbox_' + self.plot_id] and not st.session_state['all_ages_checkbox_' + self.plot_id]:
    #         if age_category == 'Todos':
    #                 self.color = custom_colors['cat_todos']
    #                 self.color_pie = blue
    #         elif age_category == 'A':
    #                 self.color = custom_colors['cat_A_0']
    #                 self.color_pie = green
    #         elif age_category == 'B':
    #                 self.color = custom_colors['cat_B']
    #                 self.color_pie = yellow
    #         elif age_category == 'C':
    #                 self.color = custom_colors['cat_C']
    #                 self.color_pie = orange
    #         elif age_category == 'D':
    #                 self.color = custom_colors['cat_D']
    #                 self.color_pie = reds
        
    def choose_plot(self):
        if st.session_state[f'plot_{self.plot_id}'] == 'Gráficos':
            st.title('Seleccione un gráfico')
        if st.session_state[f'plot_{self.plot_id}'] == 'Fecha de recepción de datos':
            st.title("Fecha de recepeción de Datos")            
            #self.colors()
            if st.session_state['ambas_antes_despues_' + self.plot_id] != 'Antes vs Después':
                self.df['date_recepcion_data'] = pd.to_datetime(self.df['date_recepcion_data'], format='%Y-%m-%d %H:%M:%S')
                self.df['month'] = self.df['date_recepcion_data'].dt.to_period('M')
                grouped_data = self.df.groupby('month').size().reset_index(name='count')
                grouped_data['month'] = grouped_data['month'].dt.to_timestamp()
                self.df = grouped_data
            else:
                print(self.df_combinado.head())
                self.df_combinado['date_recepcion_data'] = pd.to_datetime(self.df_combinado['date_recepcion_data'], format='%Y-%m-%d %H:%M:%S')
                self.df_combinado['month'] = self.df_combinado['date_recepcion_data'].dt.to_period('M')
                grouped_data = self.df_combinado.groupby(['month', 'Periodo']).size().reset_index(name='count')
                grouped_data['month'] = grouped_data['month'].dt.to_timestamp()
                self.df_combinado = grouped_data
            

                
            self.title = 'Uso de la aplicación por mes'
            self.x = 'month'
            self.y = 'count'
            self.x_label = 'Meses'
            self.y_label = 'Frecuencia'
            self.lineplot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Edad':
            st.title("Rangos etarios") 
            #self.colors()
            self.bins = 20
            self.count = 'age_category'
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.title = 'Distribución de edades'
            self.x_label= 'Edades'
            self.y_label = 'Frecuencia'
            self.histo_plot()
            self.count = 'age_category'
            self.title = 'Porcentaje de rangos etarios'
            #self.pie_edad()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Géneros':
            st.title("Géneros") 
            #self.colors()
            self.bins = 2
            self.count = 'genero'
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label= 'Generos'
            self.y_label = 'Frecuencia'
            self.count_plot()
            
        elif st.session_state[f'plot_{self.plot_id}'] == "Provincia":
            st.title("Distribución de localidades") 
            self.map()
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.rotation = 45
            self.rotation2 = 80
            self.fontsize2 = 6
            self.count_plot()
            self.rotation = None   
        elif st.session_state[f'plot_{self.plot_id}'] == 'Percepción de cambio':

            # Título principal
            st.write("# Percepción de cambio")

            # Pregunta al usuario
            st.write("## ¿Cuánto crees que cambiaste tus hábitos por las recomendaciones?")

            # Escala de evaluación
            st.write("### 0: Nada")
            st.write("### 5: Completamente")

            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
            print(data_dictionary[st.session_state[f'plot_{self.plot_id}']])

            if st.session_state['ambas_antes_despues_' + self.plot_id] != 'Antes vs Después':
                # Calcular media y cantidad total
                mean_value = self.df[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].mean()
                data_count = self.df[self.df[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].notna()]['user_id'].nunique()
                # Mostrar resultados generales
                st.write("""
                ## Análisis
                ### Cantidad de usuarios: {}
                ### Media: {:.2f}
                """.format(data_count, mean_value))

            else:
                # Filtrar los datos para los períodos 'Antes' y 'Después'
                df_antes = self.df_combinado[self.df_combinado['Periodo'] == 'Antes']
                df_despues = self.df_combinado[self.df_combinado['Periodo'] == 'Después']

                # Calcular media y cantidad para el período 'Antes'
                mean_value_antes = df_antes[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].mean()
                data_count_antes  = df_antes[df_antes[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].notna()]['user_id'].nunique()

                # Calcular media y cantidad para el período 'Después'
                mean_value_despues = df_despues[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].mean()
                data_count_despues  = df_despues[df_despues[data_dictionary[st.session_state[f'plot_{self.plot_id}']]].notna()]['user_id'].nunique()

                st.write("""
                ## Análisis antes
                ### Cantidad de datos: {}
                ### Media: {:.2f}
                """.format(data_count_antes, mean_value_antes))

                st.write("""
                ## Análisis después
                ### Cantidad de datos: {}
                ### Media: {:.2f}
                """.format(data_count_despues, mean_value_despues))

            
        elif st.session_state[f'plot_{self.plot_id}'] == 'Exposición luz natural':
            st.title('Exposición a la luz natural')
            st.subheader('¿Antes de las 15:00, estás en espacios descubiertos?')
            st.subheader('0: No')
            st.subheader('1: Sí, al menos 3 días a la semana')
            st.subheader('2: Sí, 3 días o más por semana')  
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Exposición luz artificial":
            st.title("Exposición a la luz artificial")
            st.subheader('¿Antes de las 15:00 ¿necesitás encender la luz en el ambiente en el que más estás?')
            st.subheader('0: sí, casi siempre')
            st.subheader('1: No, casi nunca')

            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        
        elif st.session_state[f'plot_{self.plot_id}'] == "Estudios no foticos integrados":
            st.title("Estudios no foticos integrados")
            st.subheader('Si estás estudiando, ¿tenés clases?')
            st.subheader('-1: No estudio y/o no tengo clases')
            st.subheader('0: Sí, menos de 3 días por semana')
            st.subheader('1: Sí, 3 días o más por semana')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        
        elif st.session_state[f'plot_{self.plot_id}'] == "Trabajo no fotico integrado":
            st.title("Trabajo no fotico integrado")
            st.subheader('Estás trabajando?')
            st.subheader('XX: ??')
            st.subheader('0: No')
            st.subheader('-1: Sí, menos de 3 días por semana')
            st.subheader('1: Sí, 3 días o más por semana')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.order = ['xx', '-1', '0', '1']
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Otra actividad habitual no fotica":
            st.title("Otra actividad habitual no fotica")
            st.subheader('¿Hacés alguna otra actividad al menos 3 veces por semana en horarios fijos?')
            st.subheader('0: No')
            st.subheader('1: Sí')
        
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Cena no fotica integrada":
            st.title("Cena no fotica integrada")
            st.subheader('¿Cenas habitualmente en el mismo horario?')
            st.subheader('0: Si')
            st.subheader('-1: No')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Horario de acostarse - Hábiles":
            st.title("Horario de acostarse en días Hábiles")
            st.subheader('¿A qué hora te acostás?')
            st.subheader('Los diás hábiles me acuesto: HH:MM AM/PM')
            self.bins = 24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
            #self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Horario decidir dormir - Hábiles':
            st.title('Horario decidir dormir en días hábiles')
            st.subheader('Una vez que me acosté, decido dormirme: HH:MM AM/PM')

            self.bins = 24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        
        elif st.session_state[f'plot_{self.plot_id}'] == 'Minutos dormir - Hábiles':
            st.title('Minutos dormir en días Hábiles')
            st.subheader('¿Cuántos minutos tardaás en dormirte?')
            st.subheader('Tardo ... minutos en dormirme: Entero')
            self.bins = 24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Hora despertar - Hábiles':
            st.title('Hora despertar en días Hábiles')
            st.subheader('¿A qué hora te despertás?')
            st.subheader('Me despierto: HH:MM AM/PM')
            self.bins = 24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        
        elif st.session_state[f'plot_{self.plot_id}'] == 'Alarma - Hábiles':
            st.title('Alarma en días Hábiles')
            st.subheader('¿Usas alarma o despertador?')
            st.subheader('0: No (Chequear si el 0 es NO)')
            st.subheader('1: Si')
            
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            #self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Siesta habitual integrada':
            st.header('Siesta en días hábiles')
            st.write('¿En general, dormís siesta en tus diás hábiles?')
            st.subheader('0: No')
            st.subheader('1: Sí, menos de 30 minutos')
            st.subheader('2: Sí, más de 30 minutos')
            st.write('Chequear si estan bien las respuestas')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
           # #self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Calidad de sueño - Hábiles':
            st.header('Calidad de sueño en días hábiles')
            st.subheader('1: Muy mal')
            st.subheader('10: Excelente')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.rotation2 = 45
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Horario de acostarse - Libres':
            st.title('Horario de acostarse en días libres')
            st.subheader('¿A qué hora te acostás?')
            st.subheader('Los días libres me acuesto a las: HH:MM AM/PM')
            
            #self.colors()
            self.bins = 24
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Horario decidir dormir - Libres':
            st.title('Horario de decidir dormir en días libres')
            st.subheader('¿A qué hora decidis dormirte?')
            st.subheader('Una vez que me acosté, decido dormirme: HH:MM AM/PM')
            #self.colors()
            self.bins = 24
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Minutos dormir - Libres':
            st.title('Minutos en dormir en días libres')
            st.subheader('¿Cuántos minutos tardás en dormirte?')
            st.subheader('Tardo ... minutos en dormirme')
            
            #self.colors()
            self.bins = 24
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Hora despertar - Libres':
            st.title('Hora de despertar en días libres')
            st.subheader('¿A qué hora te despertás?')
            st.subheader('Me despierto: HH:MM (AM/PM)')
            #self.colors()
            self.bins = 24
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Alarma - Libres':
            st.title('Alarma en días libres')
            st.subheader('¿Usás alarma o despertador?')
            st.subheader('0: No')
            st.subheader('1: Si')
            
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Alarma no fotica (sí/no)":
            st.title("Recomendación - Alarma no fotica (sí/no)")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Luz natural (8-15)":
            st.title("Recomendación - Luz natural (8-15)")
            self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Luz artificial (8-15)":
            st.title("Recomendación - Luz artificial (8-15)")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Estudios no foticos integrados":
            st.title('"Recomendación - Estudios no foticos integrados"')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Trabajo no fotico integrado":
            st.title("Recomendación - Trabajo no fotico integrado")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Otra actividad habitual no fotica (sí/no)":
            st.title('"Recomendación - Otra actividad habitual no fotica (sí/no)"')
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Cena no fotica integrada":
            st.title("Recomendación - Cena no fotica integrada")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == "Recomendación - Siesta habitual integrada":
            st.title("Recomendación - Siesta habitual integrada")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.count_plot()
            ##self.pie_plot()

        elif st.session_state[f'plot_{self.plot_id}'] == "MEQ Puntaje total":
            st.title("MEQ Puntaje total")
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.rotation = 45
            ##self.pie_plot()
            self.fontsize2 = 6
            self.bins = 20
            self.histo_plot()
            self.rotation = 45
        elif st.session_state[f'plot_{self.plot_id}'] == 'MSFsc':
            st.title('Mid-Sleep on Free Days, Sleep-Corrected')
            #self.colors()
            self.x = 'MEQ_score_total_tipo'
            self.y_label = 'Frecuencia'
            self.rotation = 45
            self.fontsize2 = 6
            self.count_plot()
            data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            ##self.pie_plot()
            self.rotation = None
            self.x ='age'
            self.x_label = 'Edad'
            self.y = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.y_label = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.lineplot()
            self.y = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.y_label = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x = 'NOFOTICO_cena_integrada'
            self.x_label = 'Cena Integrada'
            self.bar_plot()
            
        elif st.session_state[f'plot_{self.plot_id}'] == 'Duración Del Sueño - Hábiles':
            st.title('Duración del sueño en días hábiles')
            self.bins = 24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.fontsize2 = 6
            self.histo_plot()
        elif st.session_state[f'plot_{self.plot_id}'] == 'Desviación Jet Lag Social':
            st.title('Desviación Jet Lag Social')
            #Scatter Plot
            #self.colors()
            self.y = 'user_id'
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = None
            self.y_visible = False
            self.scatter_plot()
            # Box Plot
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_visible = False
            self.y_label = None
            self.y = 'user_id'
            self.box_plot()
            #Line Plot
            self.y = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.y_label = st.session_state[f'plot_{self.plot_id}']
            self.x = 'age'
            self.x_label = 'Edad'
            self.lineplot()
            #Scatter Plot 'Desviación Jet Lag Social' vs MSFsc
            self.y = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.y_visible = True
            self.y_label = st.session_state[f'plot_{self.plot_id}']
            self.x = 'MSFsc'
            self.x_label = 'MSFsc'
            self.scatter_plot()
            #Histoplot
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.y = 'user_id'
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.bins = 24
            self.histo_plot()
            
        elif st.session_state[f'plot_{self.plot_id}'] == 'Hora de inicio de sueño no laboral centrada':
            st.title('Hora de inicio de sueño no laboral centrada')
            self.bins=24
            #self.colors()
            self.x = data_dictionary[st.session_state[f'plot_{self.plot_id}']]
            self.x_label = st.session_state[f'plot_{self.plot_id}']
            self.y_label = 'Frecuencia'
            self.histo_plot()
            
    def pie_plot(self):    
        fig, ax = plt.subplots(figsize=(8, 6))
        if st.session_state['ambas_antes_despues_' + self.plot_id] != 'Antes vs Después':
            value_counts = self.df[self.pie].value_counts() 
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=self.color_pie,labeldistance=1.1, pctdistance=0.5)
            st.pyplot(fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig,ax= plt.subplots(figsize=(8, 6))
                value_counts_antes = self.df_combinado[self.df_combinado['Periodo'] == 'Antes'][self.pie].value_counts()
                colors_antes = self.color_pie
                ax.pie(value_counts_antes, labels=value_counts_antes.index, autopct='%1.1f%%', startangle=90, colors=colors_antes)
                ax.set_title('Antes', fontsize=15)
                st.pyplot(fig)
            with col2:
                fig,ax= plt.subplots(figsize=(8, 6))
                value_counts_despues = self.df_combinado[self.df_combinado['Periodo'] == 'Después'][self.pie].value_counts()
                colors_despues = self.color_pie
                ax.pie(value_counts_despues, labels=value_counts_despues.index, autopct='%1.1f%%', startangle=90, colors=colors_despues)
                ax.set_title('Después', fontsize=15)
                st.pyplot(fig)

    def pie_edad(self): 
        fig, ax = plt.subplots(figsize=(8, 6))
        if  st.session_state['all_genders_checkbox_' + self.plot_id]:
            if st.session_state['selected_gender_' + self.plot_id] == 0:
                colors = [ custom_colors['cat_B_0'], custom_colors['cat_C_0'], custom_colors['cat_A_0'], custom_colors['cat_D_0']]
            elif st.session_state['selected_gender_' + self.plot_id] == 1:
                colors = [ custom_colors['cat_B_1'], custom_colors['cat_C_1'], custom_colors['cat_A_1'], custom_colors['cat_D_1']]
        else:
            colors = [ custom_colors['cat_B'], custom_colors['cat_C'], custom_colors['cat_A'], custom_colors['cat_D']]
        value_counts = self.df[self.count].value_counts()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=colors )
        ax.set_title('Rangos Etarios', fontsize=15)
        st.pyplot(fig)

    def lineplot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':  
            sns.lineplot(data=self.df_combinado, x=self.x, y=self.y, palette=sns.light_palette(self.color, n_colors=2), ax=ax, hue='Periodo', errorbar=None)
        else:
            sns.lineplot(data=self.df, x=self.x, y=self.y, color=self.color, ax=ax, errorbar=None)
        ax.set_title(self.title, fontsize=20)
        ax.set_xlabel(self.x_label, fontsize=15)
        ax.set_ylabel(self.y_label, fontsize=15)
        plt.xticks(rotation=self.rotation)
        st.pyplot(fig)

    def histo_plot(self): 
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot the histogram
        if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':
            sns.histplot(data=self.df_combinado, x=self.x, kde=False, bins=self.bins, ax=ax, palette=sns.light_palette(self.color, n_colors=2), hue='Periodo', ) 
        else:
            sns.histplot(data=self.df, x=self.x, kde=False, bins=self.bins, ax=ax, color=self.color)
        # Add value annotations to the bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=self.fontsize2, color='grey', rotation=self.rotation2)
        # Set the title and axis labels
        ax.set_title(self.title, fontsize=20)
        ax.set_xlabel(self.x_label, fontsize=15)
        ax.set_ylabel(self.y_label, fontsize=15)
        # Adjust other plot settings
        plt.xticks(rotation=self.rotation)
        ax.yaxis.set_visible(self.y_visible)
        # Display the plot
        st.pyplot(fig)

    def count_plot(self):
            fig, ax = plt.subplots(figsize=(8, 6))
            if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':
                sns.countplot(data=self.df_combinado,x=self.x, ax=ax, palette=sns.light_palette(self.color, n_colors=2), dodge=True, order=self.order, hue='Periodo')
            else:
                sns.countplot(data=self.df, x=self.x, ax=ax, color=self.color, order=self.order)
            total = sum([p.get_height() for p in ax.patches])
            for p in ax.patches:
                if p.get_height() > 0:
                    value = int(p.get_height())
                    percentage = 100 * p.get_height() / total
                    ax.annotate(f'{value} ({percentage:.1f}%)', (p.get_x() + p.get_width() / 2, p.get_height() ), ha='center', va='bottom', fontsize=self.fontsize2, color='grey', rotation = self.rotation2)
            
            ax.set_title(self.title, fontsize=20)
            ax.set_xlabel(self.x_label, fontsize=15)
            ax.set_ylabel('Frecuencia', fontsize=15)
            plt.xticks(rotation=self.rotation, ha='right')
            st.pyplot(fig)
    
    def bar_plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':
            sns.barplot(data=self.df_combinado, x=self.x, y=self.y, ax=ax, palette=sns.light_palette(self.color, n_colors=2), order=self.order, hue='Periodo', ci=None)
        else:
            sns.barplot(data=self.df, x=self.x, y=self.y, ax=ax, color=self.color, order=self.order, ci=None)
        total = sum([p.get_height() for p in ax.patches])
        for p in ax.patches:
            if p.get_height() > 0:
                value = int(p.get_height())
                percentage = 100 * p.get_height() / total
                ax.annotate(f'{value} ({percentage:.1f}%)', (p.get_x() + p.get_width() / 2, p.get_height() ) , ha='center', va='bottom', fontsize=self.fontsize2, color='grey', rotation = self.rotation2)
       
        ax.set_title(self.title, fontsize=20)
        #ax.text(0.5, 1.02, f"Género: {st.session_state.get(f'selected_gender_{self.plot_id}', 'N/A')}", ha='center', va='bottom', transform=ax.transAxes, fontsize=14, color='grey')
        ax.set_xlabel(self.x_label, fontsize=15)
        ax.set_ylabel(self.y_label, fontsize=15)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    def scatter_plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':
            sns.scatterplot(data=self.df_combinado, x=self.x, y=self.y, ax=ax, hue='Periodo', palette=sns.light_palette(self.color, n_colors=2))
        else:
            sns.scatterplot(data=self.df, x=self.x, y=self.y, ax=ax, color=self.color)
        ax.set_title(self.title, fontsize=20)
        ax.set_xlabel(self.x_label, fontsize=15)
        ax.set_ylabel(self.y_label, fontsize=15)
        ax.yaxis.set_visible(self.y_visible)
        if self.rotation:
            plt.xticks(rotation=45)
        st.pyplot(fig)

    def box_plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if st.session_state['ambas_antes_despues_' + self.plot_id] == 'Antes vs Después':
            sns.boxplot(data=self.df_combinado, x=self.x, ax=ax, palette=sns.light_palette(self.color, n_colors=2), hue='Periodo')
        else:
            sns.boxplot(data=self.df, x=self.x, ax=ax, color=self.color)
        ax.set_title(self.title, fontsize=20)
        ax.set_xlabel(self.x_label, fontsize=15)
        ax.set_ylabel(self.y_label, fontsize=15)
        ax.yaxis.set_visible(self.y_visible)
        plt.xticks(rotation=self.rotation)
        st.pyplot(fig)
        
    def map(self): 
        layer = pdk.Layer("HeatmapLayer",data=self.df,  get_position='[Longitude, Latitude]',  opacity=0.9,  radius_pixels=100,  intensity=1,  )
        view_state = pdk.ViewState(latitude=self.df['Latitude'].mean(),  longitude=self.df['Longitude'].mean(),  zoom=5,  pitch=50  )
        tooltip = {"html": "<b>Province:</b> {provincia}<br><b>Quantity:</b> {quantity}","style": {"backgroundColor": "steelblue","color": "white"}}
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
        st.pydeck_chart(deck)

def main():
    # Initialize the Authentication
    auth = Authentication()
    
    # Check for login session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  

    # Login Form
    if not st.session_state.logged_in:
        st.title("Inicia sesión para acceder a la aplicación")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")

        if st.button("Iniciar sesión"):
            if auth.validate_user(username, password):
                st.session_state.logged_in = True
                st.rerun()  # Re-run the app after successful login
            else:
                st.error("Usuario o contraseña inválidos")
    else:
        # Display application tabs
        tabs = st.tabs(["Instructivo", "Cargar Base de Datos", "Gráficos"])
        
        # Instructivo tab
        with tabs[0]:
            instructivo_app = InstructivoApp()
            instructivo_app.display()

        # Data Upload tab
        with tabs[1]:
            st.title("Cargar base de datos")
            data_loader = DatabaseUploader()
            uploaded_before, uploaded_after = data_loader.upload_data()

            if uploaded_before and uploaded_after:
                with st.spinner("Cargando datos..."):
                    data_loader_app = DataLoader()
                    df_all = data_loader_app.load_data(uploaded_before, uploaded_after, 'Geo.csv')
                st.success("Base de datos cargada correctamente!")
            else:
                st.warning("Por favor, sube ambos archivos CSV para continuar.")
                df_all = None  # Ensure df_all is set to None

        # Graphs tab
        with tabs[2]:
            if df_all is None:
                st.header("Primero cargue la base de datos")
                return

            # Select number of plots
            st.sidebar.header("Seleccione la cantidad de gráficos que desea ver")
            num_plots = st.sidebar.slider("Cantidad de gráficos:", min_value=1, max_value=9, value=1, step=1)
            plots_per_row = 1 if num_plots == 1 else (2 if num_plots == 2 else 3)

            # Generate plots
            plot_count = 0  
            while plot_count < num_plots:
                columns = st.columns(plots_per_row)
                for col in columns:
                    if plot_count < num_plots:
                        plot_id = f'plot_{plot_count + 1}'  
                        st.sidebar.header(f"Gráfico - {plot_count + 1}")  
                        with st.spinner("Cargando datos y aplicando filtros, por favor espere..."):
                            streamlit_app = StreamLit(df_all, plot_id)
                            streamlit_app.sidebar()
                            filters = Filters(df_all, plot_id)
                            filters.choose_filter()
                            df_filtered = filters.result
                            df_filtered_antes = filters.result_antes
                            df_filtered_despues = filters.result_despues
                            column_order =['date_recepcion_data', 'user_id', 'SEGUISTE_RECOMENDACIONES', 'days_diff', 'age', 'age_category', 'genero', 'provincia', 'localidad', 'Latitude', 'Longitude', 'RECOMENDACIONES_AJUSTE', 'date_generacion_recomendacion', 'FOTICO_luz_natural_8_15_integrada', 'FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada', 'NOFOTICO_estudios_integrada', 'NOFOTICO_trabajo_integrada', 'NOFOTICO_otra_actividad_habitual_si_no', 'NOFOTICO_cena_integrada', 'HAB_Hora_acostar', 'HAB_Hora_decidir', 'HAB_min_dormir', 'HAB_Soffw', 'NOFOTICO_HAB_alarma_si_no', 'HAB_siesta_integrada', 'HAB_calidad', 'LIB_Hora_acostar', 'LIB_Hora_decidir', 'LIB_min_dormir', 'LIB_Offf', 'LIB_alarma_si_no', 'MEQ_score_total','rec_NOFOTICO_HAB_alarma_si_no', 'rec_FOTICO_luz_natural_8_15_integrada' ,'rec_FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada',	'rec_NOFOTICO_estudios_integrada', 'rec_NOFOTICO_trabajo_integrada', 'rec_NOFOTICO_otra_actividad_habitual_si_no',	'rec_NOFOTICO_cena_integrada',	'rec_HAB_siesta_integrada',  'MSFsc', 'HAB_SDw', 'SJL', 'HAB_SOnw_centrado']
                            column_order_combinado = ['date_recepcion_data', 'user_id', 'SEGUISTE_RECOMENDACIONES', 'days_diff', 'Periodo' ,'age', 'age_category', 'genero', 'provincia', 'localidad', 'Latitude', 'Longitude', 'RECOMENDACIONES_AJUSTE', 'date_generacion_recomendacion', 'FOTICO_luz_natural_8_15_integrada', 'FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada', 'NOFOTICO_estudios_integrada', 'NOFOTICO_trabajo_integrada', 'NOFOTICO_otra_actividad_habitual_si_no', 'NOFOTICO_cena_integrada', 'HAB_Hora_acostar', 'HAB_Hora_decidir', 'HAB_min_dormir', 'HAB_Soffw', 'NOFOTICO_HAB_alarma_si_no', 'HAB_siesta_integrada', 'HAB_calidad', 'LIB_Hora_acostar', 'LIB_Hora_decidir', 'LIB_min_dormir', 'LIB_Offf', 'LIB_alarma_si_no', 'MEQ_score_total','rec_NOFOTICO_HAB_alarma_si_no', 'rec_FOTICO_luz_natural_8_15_integrada' ,'rec_FOTICO_luz_ambiente_8_15_luzelect_si_no_integrada',	'rec_NOFOTICO_estudios_integrada', 'rec_NOFOTICO_trabajo_integrada', 'rec_NOFOTICO_otra_actividad_habitual_si_no',	'rec_NOFOTICO_cena_integrada',	'rec_HAB_siesta_integrada',  'MSFsc', 'HAB_SDw', 'SJL', 'HAB_SOnw_centrado']
                            df_all = df_all[column_order]
                            df_filtered = df_filtered[column_order]
                
                            df_all = df_all.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True])
                            df_filtered = df_filtered.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True])
                            
                            df_filtered_antes.loc[:, 'Periodo'] = 'Antes'
                            df_filtered_despues.loc[:, 'Periodo'] = 'Después'
                            df_filtered_antes = df_filtered_antes.reset_index(drop=True)
                            df_filtered_despues = df_filtered_despues.reset_index(drop=True)
                            df_combinado = pd.concat([df_filtered_antes, df_filtered_despues], ignore_index=False)
                            df_combinado = df_combinado.reset_index(drop=True)
                            df_combinado = df_combinado[column_order_combinado]
                            df_combinado = df_combinado.sort_values(by=['user_id', 'date_recepcion_data'], ascending=[True, True])
                            with col:  
                                if st.session_state['datos_' + plot_id] == True:                    
                                    
                                    st.title('Datos')
                                    if st.session_state['ambas_antes_despues_' + plot_id] == 'Antes vs Después':   
                                        st.write(f'Cantidad : {len(df_combinado)}')  
                                        st.write(df_combinado)
                                    else:
                                        st.write(f'Cantidad : {len(df_filtered)}')  
                                        st.write(df_filtered)  
                                plot_generator = PlotGenerator(df_filtered, df_combinado, plot_id) 
                                plot_generator.choose_plot()  

                            plot_count += 1  

main()


#streamlit run '/Users/tomasmendietarios/Library/Mobile Documents/com~apple~CloudDocs/I.T.B.A/MiRelojInterno/main.py'


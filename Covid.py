import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from pycaret.regression import *
import datetime

st.subheader('Corona Virus Vaccination No. Prediction')

st.sidebar.header('User input Parameters')

def user_input_feature():
    Country=st.sidebar.selectbox('Select Country',['Austria','Belgium','Bulgaria','Chile','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany',
                             'Hungary','Iceland','Ireland','Italy','Japan','Latvia','Liechtenstein','Lithuania','Luxembourg','Malta','Netherlands',
                             'Poland','Portugal','Romania','Slovakia','Slovenia','Spain','Sweden','Switzerland','United States','Uruguay'])

    Vaccine=st.sidebar.selectbox('Select Vaccines',['Abdala, Soberana02','CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac', 'CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V',
                                'CanSino, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V', 'CanSino, Sinopharm/Beijing, Sinopharm/Wuhan, Sinovac', 
                                'Covaxin, Oxford/AstraZeneca', 'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing', 'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V',
                                'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V', 'Covaxin, Oxford/AstraZeneca, Sputnik V', 'EpiVacCorona, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V',
                                'EpiVacCorona, Sputnik V', 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech', 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V',
                                'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V', 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V',
                                'Johnson&Johnson, Moderna, Pfizer/BioNTech', 'Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing','Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac',
                                'Johnson&Johnson, Pfizer/BioNTech', 'Moderna','Moderna, Oxford/AstraZeneca', 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech', 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V',
                                'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V', 'Moderna, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V', 'Moderna, Pfizer/BioNTech',
                                'Moderna, Pfizer/BioNTech, Sinovac', 'Oxford/AstraZeneca', 'Oxford/AstraZeneca, Pfizer/BioNTech', 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing',
                                'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V', 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac',
                                'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V', 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V',
                                'Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac', 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V', 'Oxford/AstraZeneca, RBD-Dimer, Sputnik V',
                                'Oxford/AstraZeneca, Sinopharm/Beijing', 'Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac', 'Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V', 
                                'Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V', 'Oxford/AstraZeneca, Sinovac', 'Oxford/AstraZeneca, Sinovac, Sputnik V', 'Oxford/AstraZeneca, Sputnik V',
                                'Pfizer/BioNTech', 'Pfizer/BioNTech, Sinopharm/Beijing', 'Pfizer/BioNTech, Sinovac', 'Pfizer/BioNTech, Sputnik V', 'QazVac, Sinopharm/HayatVax, Sputnik V',
                                'Sinopharm/Beijing', 'Sinopharm/Beijing, Sputnik V', 'Sputnik V' ])
    Date=st.sidebar.date_input('Date input')
    #Date=Date.strftime("%m/%d/%Y")

    #Vaccine=st.sidebar.selectbox('Select Vaccine',['Johnson&Johnson','Moderna','Oxford/AstraZeneca','Pfizer/BioNTech','Sinovac','CanSino','Sputnik V','Sinopharm/Beijing'])

    Algorithm=st.sidebar.selectbox('Select Algorithm',['Random Forest Regressor','Decision Tree Regressor','K Neighbors Regressor','Linear Regression'])

    #dt={'country':Country,
    #    'date':Date,
    #    'vaccine':Vaccine}
    le_name_mapping_Country={'Afghanistan': 0, 'Albania': 1, 'Algeria': 2, 'Andorra': 3, 'Angola': 4, 'Anguilla': 5, 'Antigua and Barbuda': 6, 'Argentina': 7, 'Armenia': 8, 'Aruba': 9, 'Australia': 10, 'Austria': 11, 'Azerbaijan': 12, 'Bahamas': 13, 'Bahrain': 14, 'Bangladesh': 15, 'Barbados': 16, 'Belarus': 17, 'Belgium': 18, 'Belize': 19, 'Benin': 20, 'Bermuda': 21, 'Bhutan': 22, 'Bolivia': 23, 'Bonaire Sint Eustatius and Saba': 24, 'Bosnia and Herzegovina': 25, 'Botswana': 26, 'Brazil': 27, 'British Virgin Islands': 28, 'Brunei': 29, 'Bulgaria': 30, 'Burkina Faso': 31, 'Cambodia': 32, 'Cameroon': 33, 'Canada': 34, 'Cape Verde': 35, 'Cayman Islands': 36, 'Central African Republic': 37, 'Chad': 38, 'Chile': 39, 'China': 40, 'Colombia': 41, 'Comoros': 42, 'Congo': 43, 'Cook Islands': 44, 'Costa Rica': 45, "Cote d'Ivoire": 46, 'Croatia': 47, 'Cuba': 48, 'Curacao': 49, 'Cyprus': 50, 'Czechia': 51, 'Democratic Republic of Congo': 52, 'Denmark': 53, 'Djibouti': 54, 'Dominica': 55, 'Dominican Republic': 56, 'Ecuador': 57, 'Egypt': 58, 'El Salvador': 59, 'England': 60, 'Equatorial Guinea': 61, 'Estonia': 62, 'Eswatini': 63, 'Ethiopia': 64, 'Faeroe Islands': 65, 'Falkland Islands': 66, 'Fiji': 67, 'Finland': 68, 'France': 69, 'French Polynesia': 70, 'Gabon': 71, 'Gambia': 72, 'Georgia': 73, 'Germany': 74, 'Ghana': 75, 'Gibraltar': 76, 'Greece': 77, 'Greenland': 78, 'Grenada': 79, 'Guatemala': 80, 'Guernsey': 81, 'Guinea': 82, 'Guinea-Bissau': 83, 'Guyana': 84, 'Honduras': 85, 'Hong Kong': 86, 'Hungary': 87, 'Iceland': 88, 'India': 89, 'Indonesia': 90, 'Iran': 91, 'Iraq': 92, 'Ireland': 93, 'Isle of Man': 94, 'Israel': 95, 'Italy': 96, 'Jamaica': 97, 'Japan': 98, 'Jersey': 99, 'Jordan': 100, 'Kazakhstan': 101, 'Kenya': 102, 'Kosovo': 103, 'Kuwait': 104, 'Kyrgyzstan': 105, 'Laos': 106, 'Latvia': 107, 'Lebanon': 108, 'Lesotho': 109, 'Liberia': 110, 'Libya': 111, 'Liechtenstein': 112, 'Lithuania': 113, 'Luxembourg': 114, 'Macao': 115, 'Madagascar': 116, 'Malawi': 117, 'Malaysia': 118, 'Maldives': 119, 'Mali': 120, 'Malta': 121, 'Mauritania': 122, 'Mauritius': 123, 'Mexico': 124, 'Moldova': 125, 'Monaco': 126, 'Mongolia': 127, 'Montenegro': 128, 'Montserrat': 129, 'Morocco': 130, 'Mozambique': 131, 'Myanmar': 132, 'Namibia': 133, 'Nauru': 134, 'Nepal': 135, 'Netherlands': 136, 'New Caledonia': 137, 'New Zealand': 138, 'Nicaragua': 139, 'Niger': 140, 'Nigeria': 141, 'Niue': 142, 'North Macedonia': 143, 'Northern Cyprus': 144, 'Northern Ireland': 145, 'Norway': 146, 'Oman': 147, 'Pakistan': 148, 'Palestine': 149, 'Panama': 150, 'Papua New Guinea': 151, 'Paraguay': 152, 'Peru': 153, 'Philippines': 154, 'Pitcairn': 155, 'Poland': 156, 'Portugal': 157, 'Qatar': 158, 'Romania': 159, 'Russia': 160, 'Rwanda': 161, 'Saint Helena': 162, 'Saint Kitts and Nevis': 163, 'Saint Lucia': 164, 'Saint Vincent and the Grenadines': 165, 'Samoa': 166, 'San Marino': 167, 'Sao Tome and Principe': 168, 'Saudi Arabia': 169, 'Scotland': 170, 'Senegal': 171, 'Serbia': 172, 'Seychelles': 173, 'Sierra Leone': 174, 'Singapore': 175, 'Sint Maarten (Dutch part)': 176, 'Slovakia': 177, 'Slovenia': 178, 'Solomon Islands': 179, 'Somalia': 180, 'South Africa': 181, 'South Korea': 182, 'South Sudan': 183, 'Spain': 184, 'Sri Lanka': 185, 'Sudan': 186, 'Suriname': 187, 'Sweden': 188, 'Switzerland': 189, 'Syria': 190, 'Taiwan': 191, 'Tajikistan': 192, 'Thailand': 193, 'Timor': 194, 'Togo': 195, 'Tonga': 196, 'Trinidad and Tobago': 197, 'Tunisia': 198, 'Turkey': 199, 'Turkmenistan': 200, 'Turks and Caicos Islands': 201, 'Tuvalu': 202, 'Uganda': 203, 'Ukraine': 204, 'United Arab Emirates': 205, 'United Kingdom': 206, 'United States': 207, 'Uruguay': 208, 'Uzbekistan': 209, 'Vanuatu': 210, 'Venezuela': 211, 'Vietnam': 212, 'Wales': 213, 'Wallis and Futuna': 214, 'Yemen': 215, 'Zambia': 216, 'Zimbabwe': 217}
    le_name_mapping_vaccine={'Abdala, Soberana02': 0, 'CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac': 1, 'CanSino, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V': 2, 'CanSino, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V': 3, 'CanSino, Sinopharm/Beijing, Sinopharm/Wuhan, Sinovac': 4, 'Covaxin, Oxford/AstraZeneca': 5, 'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing': 6, 'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V': 7, 'Covaxin, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V': 8, 'Covaxin, Oxford/AstraZeneca, Sputnik V': 9, 'EpiVacCorona, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V': 10, 'EpiVacCorona, Sputnik V': 11, 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech': 12, 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V': 13, 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V': 14, 'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V': 15, 'Johnson&Johnson, Moderna, Pfizer/BioNTech': 16, 'Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing': 17, 'Johnson&Johnson, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac': 18, 'Johnson&Johnson, Pfizer/BioNTech': 19, 'Moderna': 20, 'Moderna, Oxford/AstraZeneca': 21, 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech': 22, 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V': 23, 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech, Sputnik V': 24, 'Moderna, Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V': 25, 'Moderna, Pfizer/BioNTech': 26, 'Moderna, Pfizer/BioNTech, Sinovac': 27, 'Oxford/AstraZeneca': 28, 'Oxford/AstraZeneca, Pfizer/BioNTech': 29, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing': 30, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinopharm/Wuhan, Sputnik V': 31, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac': 32, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sinovac, Sputnik V': 33, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V': 34, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac': 35, 'Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac, Sputnik V': 36, 'Oxford/AstraZeneca, RBD-Dimer, Sputnik V': 37, 'Oxford/AstraZeneca, Sinopharm/Beijing': 38, 'Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac': 39, 'Oxford/AstraZeneca, Sinopharm/Beijing, Sinovac, Sputnik V': 40, 'Oxford/AstraZeneca, Sinopharm/Beijing, Sputnik V': 41, 'Oxford/AstraZeneca, Sinovac': 42, 'Oxford/AstraZeneca, Sinovac, Sputnik V': 43, 'Oxford/AstraZeneca, Sputnik V': 44, 'Pfizer/BioNTech': 45, 'Pfizer/BioNTech, Sinopharm/Beijing': 46, 'Pfizer/BioNTech, Sinovac': 47, 'Pfizer/BioNTech, Sputnik V': 48, 'QazVac, Sinopharm/HayatVax, Sputnik V': 49, 'Sinopharm/Beijing': 50, 'Sinopharm/Beijing, Sputnik V': 51, 'Sputnik V': 52}

    dt={'country':le_name_mapping_Country[Country],
        'vaccines':le_name_mapping_vaccine[Vaccine],
        'Day':Date.day,
        'Month':Date.month,
        'Year':Date.year}

    lb={'country':Country,
        'vaccines':Vaccine,
        'Day':Date.day,
        'Month':Date.month,
        'Year':Date.year}

    features=pd.DataFrame(dt,index=[0])
    labels=pd.DataFrame(lb,index=[0])

    return features,labels,Algorithm



df,lb,Algo=user_input_feature()

st.subheader('User input Parameters :')
st.write(lb)

loaded_Model=load_model(Algo)
prediction = predict_model(loaded_Model, data = df)

st.subheader(Algo +' prediction')
st.subheader('People Vaccinated Per Hundred is :  ' + str( np.round(prediction['Label'].item(),2)) )
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:05:08 2020

@author: kelly
"""

import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv');

base.describe() #dados gerais sobre a base de dados
base.loc[base['age']  < 0]
base.loc[base.age < 0,'age']


#correção de dados inconsistentes (idades negativas)
base['age'].mean() # media de todas as datas

base['age'][base.age > 0].mean() # media das datas positivas
base.loc[base.age < 0, 'age'] = 40.92 #setar media das idades nos campos errados


#correção de idades faltantes
pd.isnull(base['age']) 
base.loc[pd.isnull(base['age'])] #localizar idades faltantes 

#separaar os previsores das classes
previsores = base.iloc[:, 1:4].values
classes = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#escalonamento de dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
previsores = scaler.fit_transform(previsores)


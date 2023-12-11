import pandas as pd

doctors_info = pd.read_csv('./resources/doctor_info.csv')

a0 = doctors_info.iloc[0, 0]
a1 = doctors_info.iloc[0, 1]
a2 = doctors_info.iloc[0, 2]
a3 = doctors_info.iloc[0, 3]
a4 = doctors_info.iloc[0, 4]



doctors_info['jianjie'] = doctors_info['jianjie'].str.replace('\n', '')
doctors_info['jianjie'] = '<医生简介>' + doctors_info['jianjie'].str.strip() + '</医生简介>'

doctors_info['shangchang'] = doctors_info['shangchang'].str.replace('\n', '')
doctors_info['shangchang'] = '</医生擅长>' + doctors_info['shangchang'].str.strip() + '</医生擅长>'

doctors_info['zg_xm'] = '<姓名>' + doctors_info['zg_xm'].str.strip() + '</姓名>'

doctors_info['zc_mc'] = '<职称>' + doctors_info['zc_mc'].str.strip() + '</职称>'

doctors_info['ksmc'] = '<科室门诊>' + doctors_info['ksmc'].str.strip() + '</科室门诊>'

b0 = doctors_info.iloc[0, 0]
b1 = doctors_info.iloc[0, 1]
b2 = doctors_info.iloc[0, 2]
b3 = doctors_info.iloc[0, 3]
b4 = doctors_info.iloc[0, 4]


doctors_info.to_csv('./resources/doctor_info_clear.txt',sep=';', index=False, header=False)
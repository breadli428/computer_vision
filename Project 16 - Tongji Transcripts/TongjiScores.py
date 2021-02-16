#!/usr/bin/env python
# coding: utf-8

# # 使用说明
# 
# - 登录xuanke网，打开有成绩表的页面，然后保存这个页面到 xuanke_tongji.html。
# - 登录4m3后，从浏览器的 cookie 里复制出登录产生的 JSESSION_ID 和 SERVERNAME，填入下方代码中。
# 
# 
# Cookie 查看方法是在 Chrome 里打开4m3页面，然后网页上右键，并点“检查” (Inspect)。然后选“应用” (Application)，再选“4m3.tongji.edu.cn”，如图。
# 
# 注意：JSESSION_ID 一般以 '.62' 结尾，SERVERNAME 以 's' 开头。一个 Session 的有效时间大概在15分钟左右，过期后需要重做上面的步骤以获取新的 Cookie。
# 
# - 登录4m3后，查找courseTableForStd!courseTable.action项ids信息，填入下方代码中。

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

JSESSION_ID = '0231586A9F3444A447107BA854CA9797.60'
SERVERNAME = 's111'
IDS = 883255806 # IDS与用户选课相关

xuanke_tongji = BeautifulSoup(open(r'c:\Users\Chenhao\Desktop\xuanke_tongji.html', 'r'), 'html.parser')

table = xuanke_tongji
xuanke_data = pd.DataFrame(columns=['Course_No', 'Course_Title', 'Course_Title_English', 'Year', 'Semester', 'Score', 'Credit', 'Mark', 'Pass', 'Required', 'Update_Time'])

rows = table.find_all('tr')
year = 0
sem = 0


for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    if len(cols) == 1:
        m = re.search(r'(\d+)-(\d+)学年第(\d)', cols[0])
        if m is not None:
            year, _, sem = m.groups(0)
            if sem == '2':
                year = int(year) + 1
        
    if len(cols) == 9 and cols[0] != "课号":
        xuanke_data = xuanke_data.append({
            'Course_No': cols[0],
            'Course_Title': cols[1],
            'Course_Title_English': '',
            'Year': int(year),
            'Semester': 'Fall' if sem == '1' else 'Spring',
            'Score': cols[2],
            'Credit': float(cols[3]),
            'Mark': float(cols[4]),
            'Pass': cols[5],
            'Required': cols[7],
            'Update_Time': cols[8]
        }, ignore_index=True)

print("Found courses in xuanke:", len(xuanke_data))


course_data = []

print("Finding courses in 4m3...")

for year in ['2015', '2016', '2017', '2018', '2019']: # 这里修改需要爬的年份
    for semester in ['0', '1']:   # 0 = Fall 秋季, 1 = Spring 春季
        
        tongji_semester_id = 100 + 2 * (int(year) - 2015) + int(semester)
        
        page_4m3 = requests.post('http://4m3.tongji.edu.cn/eams/courseTableForStd!courseTable.action',
                      data={'ignoreHead': 1,
                            'setting.kind': 'std',
                            'startWeek': 1,
                            'semester.id': tongji_semester_id,
                            'ids': IDS},
                      cookies={
                        'JSESSIONID': JSESSION_ID,
                        'SERVERNAME': SERVERNAME,
                        'session_locale': 'en_US'
                    })

        soup = BeautifulSoup(page_4m3.content, 'html.parser')
        
        table = soup.find_all('table', attrs={'class':'gridtable'})[1]
        table_body = table.find('tbody')

        rows = table_body.find_all('tr')
        sem_counter = 0
        for row in rows:
            cols = row.find_all('td')

            cols = [ele.text.strip() for ele in cols]
            if cols == []:
                continue
            
            course_no = cols[1][:-2]
            course_name = cols[2].replace('\n', ' ').replace('\t', '').replace('s(', 's (').replace(')(', ') (')
            credit = cols[6]
            course_data.append([year, semester, course_no, course_name, credit])
            sem_counter += 1
        
        print(f"{'Fall' if semester == '0' else 'Spring'} {year}, {sem_counter} courses founded.")


for row in course_data:
    course_no = row[2]
    course_name = row[3]
    xuanke_data.loc[xuanke_data.Course_No == course_no, 'Course_Title_English'] = course_name


xuanke_data.to_csv("Transcript.csv", index=False)

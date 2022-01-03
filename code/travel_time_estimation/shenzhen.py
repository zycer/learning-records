import math
import time

import requests

url = "https://opendata.sz.gov.cn/api/29200_00403602/1/service.xhtml"
appKey = "8d583dddd1034a85924d01cf0465a5fb"
params = {
    "appKey": "8d583dddd1034a85924d01cf0465a5fb",
    "page": 1,
    "rows": 10000
}
# res = requests.get(url, params=params)
# data = res.json()

total_data = 2052783
count_page = math.ceil(total_data / 10000)
flag = False

for current_page in range(count_page):
    print(f"正在复制第{current_page}数据...")
    time.sleep(2)
    row = 10000 if current_page < count_page - 1 else total_data - current_page * 10000
    params = {
        "appKey": "8d583dddd1034a85924d01cf0465a5fb",
        "page": current_page,
        "rows": row
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        with open("shenzhen.csv", "a+", encoding="utf-8") as f:
            if not flag:
                titles = ','.join(data["data"][0].keys()) + "\n"
                f.write(titles)
                flag = True

            for item in data["data"]:
                values = ','.join(item.values()) + "\n"
                f.write(values)
    except Exception as e:
        print(e)


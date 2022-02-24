import math
import time

import requests


class GPSData:
    def __init__(self):
        self.url = "https://opendata.sz.gov.cn/api/29200_00403602/1/service.xhtml"
        self.appKey = "8d583dddd1034a85924d01cf0465a5fb"
        self.params = {
            "appKey": "8d583dddd1034a85924d01cf0465a5fb",
            "page": 1,
            "rows": 1000
        }
        result = requests.get(self.url, params=self.params)
        self.data = result.json()
        self.count_page = math.ceil(self.data["total"] / 10000)

    def get_data(self, current_page):
        self.params["page"] = current_page
        result = requests.get(self.url, params=self.params)
        return result.json()

    def save_all_data(self):
        flag = False

        for current_page in range(self.count_page):
            print(f"正在复制第{current_page}数据...")
            time.sleep(2)
            row = 10000 if current_page < self.count_page - 1 else self.data["total"] - current_page * 10000
            self.params["page"] = current_page
            self.params["rows"] = row

            try:
                res = requests.get(self.url, params=self.params)
                data = res.json()
                with open("shenzhen0.csv", "a+", encoding="utf-8") as f:
                    if not flag:
                        titles = ','.join(data["data"][0].keys()) + "\n"
                        f.write(titles)
                        flag = True

                    for item in data["data"]:
                        values = ','.join(item.values()) + "\n"
                        f.write(values)
            except Exception as e:
                print(e)

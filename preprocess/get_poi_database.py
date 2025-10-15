import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
def get_nyc_info(latitude, longitude, language="en"):

    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&accept-language={language}"
    response = requests.get(url, headers={"User-Agent": "OpenStreetMap Python API Example"})

    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        # print(address)
        country = address.get("country", "")
        city = address.get("city", "")
        province = address.get("province", "")
        state = address.get("state", "")

        quarter = address.get("quarter", "")
        suburb = address.get("suburb", "")
        neighbourhood = address.get("neighbourhood", "")

        road = address.get("road", "")
        first_key, location = next(iter(address.items()))

        country = country if country else "United States"
        if city:
            city = city
        elif province:
            city = province
        elif state:
            city = state
        else:
            city = "New York"
        if quarter:
            district = quarter
        elif suburb:
            district = suburb
        elif neighbourhood:
            district = neighbourhood
        else:
            district = ""
        street = road if road else location
        address = [city, district, street]
        res = ','.join(i for i in address if i)

        return res
        
    else:
        return "Error: Unable to fetch data from OpenStreetMap"

def get_tky_info(latitude, longitude, language="en"):
    # 构造请求的 URL
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&accept-language={language}"

    # 发送请求
    response = requests.get(url, headers={"User-Agent": "OpenStreetMap Python API Example"})

    # 检查响应状态码
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        # print(address)
        # 提取并返回感兴趣的地理信息
        # japan address location level: country, city, prefecture, district, street
        country = address.get("country", "")
        city = address.get("city", "")
        province = address.get("province", "")

        quarter = address.get("quarter", "")
        suburb = address.get("suburb", "")
        neighbourhood = address.get("neighbourhood", "")

        road = address.get("road", "")
        # location 是返回的address 中第一个字段的内容
        first_key, location = next(iter(address.items()))

        country = country if country else "Japan"
        if city:
            city = city
        elif province:
            city = province
        else:
            city = "Tokyo"

        if quarter:
            district = quarter
        elif suburb:
            district = suburb
        elif neighbourhood:
            district = neighbourhood
        else:
            district = ""

        street = road if road else location

        address = ['Tokyo', city, district]
        res = ','.join(i for i in address if i)
        
        # return in str
        return res

    else:
        return "Error: Unable to fetch data from OpenStreetMap"

def readTrain(filePath):
    # londs: {u: [(i, time)]}
    longs = dict()
    pois = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in tqdm(lines[1:]):
        data = line.split(',')
        # u is user id, i is POI id
        time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]

        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category}
        if u not in longs:
            longs[u] = list()
        longs[u].append((i, time))
    return longs, pois

def readTest(filePath):
    # recents: {trajectory id: [(i, time)]}
    recents = dict()
    pois = dict()
    targets = dict()
    traj2u = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in tqdm(lines[1:]):
        data = line.split(',')
        time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[10]
        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category}
        if trajectory not in traj2u:
            traj2u[trajectory] = u
        if trajectory not in recents:
            recents[trajectory] = list()
        recents[trajectory].append((i, time))
        targets[trajectory] = (i, time)
    return recents, pois, targets, traj2u

def addAddressInfo(pois, dataset_name):
    """
    统一为 POI 增加 address 信息。
    """
    for poi_id, poi_data in tqdm(pois.items(), desc="Adding address info"):
        lati, longi = poi_data["latitude"], poi_data["longitude"]
        if dataset_name == 'tky':
            poi_data["address"] = get_tky_info(lati, longi)
        elif dataset_name == 'nyc':
            poi_data["address"] = get_nyc_info(lati, longi)
        elif dataset_name == 'ca':
            poi_data["address"] = get_nyc_info(lati, longi)
        else:
            poi_data["address"] = "Unknown Address"
    return pois

def getData(datasetName, save_path):
    if datasetName == 'nyc':
        filePath = save_path + '/nyc/{}_sample.csv'
    elif datasetName == 'tky':
        filePath = save_path + '/tky/{}_sample.csv'
    elif datasetName == 'ca':
        filePath = save_path + '/ca/{}_sample.csv'
    else:
        raise NotImplementedError

    trainPath = filePath.format('train')
    testPath = filePath.format('test')
    validatePath = filePath.format('validate')

    # Step 1: 读取训练集和测试集的 POI 信息
    longs, poiInfos = readTrain(trainPath)
    recents, testPoi, targets, traj2u = readTest(testPath)
    poiInfos.update(testPoi)

    # Step 2: 如果有验证集，也提取 POI 信息
    if validatePath:
        recents_, testPoi_, targets_, traj2u_ = readTest(validatePath)
        poiInfos.update(testPoi_)

    # Step 3: 统一为所有 POI 增加 address 信息
    poiInfos = addAddressInfo(poiInfos, datasetName)

    # Step 4: 返回结果
    return longs, recents, targets, poiInfos, traj2u


if __name__ == "__main__":

    dataset_name = "ca"
    # data/tky/test_sample.csv

    # get the data
    longs, recents, targets, poiInfos, traj2u = getData(dataset_name, 'data')

    # write poiInfos to a file
    save_path = f"data/{dataset_name}/POI_database.csv"
    # df = pd.DataFrame(poiInfos)
    # df = df.T
    # df.to_csv(save_path, index=False)

    df = pd.DataFrame.from_dict(poiInfos, orient='index')

    # Step 2: 将 poi id 作为一列
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'poi_id'}, inplace=True)

    # Step 3: 保存为 CSV 文件
    save_path = f"data/{dataset_name}/POI_database.csv"
    df.to_csv(save_path, index=False)

    # 可选：打印输出检查
    print(df.head())


    # read the data
    # df = pd.read_csv(save_path)
    # print(df.head())

    # # transform the column and row
    # df = df.T
    # print(df.head())
    # # index = poi_id, columns: latitude, longitude, category, address
    # df = df.reset_index().rename(columns={"index": "poi_id"})
    # df.columns = ['poi_id', 'latitude', 'longitude', 'category', 'address']
    # print(df.head())

    # write_path = f"data/{dataset_name}/POI_database.csv"
    # df.to_csv(write_path, index=False)

    def map_address(row,POI_database):
        poi_id = row["PoiId"]
        # if poi_id not in POI_database:
        #     run the get address function

        address = POI_database[POI_database["poi_id"]==poi_id]["address"].values[0]
        # split and change the, to space
        address = address.split(',')
        address = ' '.join(address)
        return address
    
    # POI_database = pd.read_csv(save_path)

    # 假设 POI_database 已经被定义
    # POI_database = pd.DataFrame(columns=['poi_id', 'latitude', 'longitude', 'category', 'address'])

    # 新建一个POI_database 的 empty DataFrame
    POI_database = pd.DataFrame(columns=['poi_id', 'latitude', 'longitude', 'category', 'address'])
    

    data_list = []  # 用来存储每行的数据字典

    sample_df = pd.read_csv(f"data/{dataset_name}/train_sample.csv")
    sample_df = pd.concat([sample_df, pd.read_csv(f"data/{dataset_name}/test_sample.csv")], ignore_index=True)
    sample_df = pd.concat([sample_df, pd.read_csv(f"data/{dataset_name}/validate_sample.csv")], ignore_index=True)

    poi_list = sample_df["PoiId"].unique()

    existing_poi_list = POI_database["poi_id"].values

    require_poi_list = [poi for poi in poi_list if poi not in existing_poi_list]
    print(len(require_poi_list))

    for poi in require_poi_list:
        # from sample_df get the latitude and longitude where poi_id == poi
        row = sample_df[sample_df["PoiId"] == poi].iloc[0]
        address = get_nyc_info(row["Latitude"], row["Longitude"])
        data_dict = {
            "poi_id": poi,
            "latitude": row["Latitude"],
            "longitude": row["Longitude"],
            "category": row["PoiCategoryName"],
            "address": address
        }
        data_list.append(data_dict)  # 将字典添加到列表中

    # 使用列表创建DataFrame
    newdf = pd.DataFrame(data_list)
    print(newdf.shape)

    # 将新的DataFrame合并到POI_database
    POI_database = pd.concat([POI_database, newdf], ignore_index=True)

    # sort the poi_id
    POI_database = POI_database.sort_values(by='poi_id').reset_index(drop=True)

    print(POI_database.shape)
    # 将POI_database写入文件
    POI_database.to_csv(save_path, index=False)



    # # rewrite the csv files
    # for i in ['train', 'test', 'validate']:
    #     csv_path = f"data/{dataset_name}/{i}_sample.csv"
    #     csv_save_path = f"data/{dataset_name}/{i}_sample_address.csv"

    #     df = pd.read_csv(csv_path)
    #     # print(df.head()["PoiId"])

    #     POI_database = pd.read_csv(save_path)
    #     df['address']=df.apply(lambda row: map_address(row, POI_database), axis=1)

    #     df.to_csv(csv_save_path, index=False)






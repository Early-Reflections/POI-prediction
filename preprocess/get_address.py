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
    # Construct the request URL
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&accept-language={language}"

    # Send request
    response = requests.get(url, headers={"User-Agent": "OpenStreetMap Python API Example"})

    # Check the response status code
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        # print(address)
        # Extract and return the interested geographical information
        # japan address location level: country, city, prefecture, district, street
        country = address.get("country", "")
        city = address.get("city", "")
        province = address.get("province", "")

        quarter = address.get("quarter", "")
        suburb = address.get("suburb", "")
        neighbourhood = address.get("neighbourhood", "")

        road = address.get("road", "")
        # location is the content of the first field in the returned address
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
            if dataset_name == 'tky':
                pois[i]["address"] = get_tky_info(lati, longi)
            elif dataset_name == 'nyc':
                pois[i]["address"] = get_nyc_info(lati, longi)
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
        time, trajectory, u, lati, longi, i, category = data[1], data[3],data[5], data[6], data[7], data[8], data[10]
        if i not in pois:
            pois[i] = dict()
            pois[i]["latitude"] = lati
            pois[i]["longitude"] = longi
            pois[i]["category"] = category
            if dataset_name == 'tky':
                address = get_tky_info(lati, longi)
            elif dataset_name == 'nyc':
                address = get_nyc_info(lati, longi)
            pois[i]["address"] = address
        if trajectory not in traj2u:
            traj2u[trajectory] = u
        if trajectory not in recents:
            recents[trajectory] = list()
            recents[trajectory].append((i, time))
        else:
            if trajectory in targets:
                recents[trajectory].append(targets[trajectory])
            targets[trajectory] = (i, time)
    return recents, pois, targets, traj2u

def getData(datasetName, save_path):
    if datasetName == 'nyc':
        filePath = save_path + '/nyc/{}_sample.csv'
        # filePath = '../data/nyc/{}_sample.csv'
    elif datasetName == 'tky':
        filePath = save_path + '/tky/{}_sample.csv'
        # filePath = '../data/tky/{}_sample.csv'
    else:
        raise NotImplementedError
    trainPath = filePath.format('train')
    testPath = filePath.format('test')
    validatePath = filePath.format('validate')

    longs, poiInfos = readTrain(trainPath)
    recents, testPoi, targets, traj2u = readTest(testPath)
    recentsVal, valPoi, targetsVal, traj2uVal = readTest(validatePath)
    poiInfos.update(testPoi)
    poiInfos.update(valPoi)

    targets = dict(list(targets.items()))

    return longs, recents, targets, poiInfos, traj2u

if __name__ == "__main__":

    dataset_name = "nyc"
    # data/tky/test_sample.csv

    # # get the data
    # longs, recents, targets, poiInfos, traj2u = getData(dataset_name, 'data')

    # write poiInfos to a file
    save_path = f"data/{dataset_name}/preprocessed/POI_database.csv"
    # df = pd.DataFrame(poiInfos)
    # df = df.T
    # df.to_csv(save_path, index=False)


    # # read the data
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
    
    POI_database = pd.read_csv(save_path)

    # Assume POI_database has been defined
    # POI_database = pd.DataFrame(columns=['poi_id', 'latitude', 'longitude', 'category', 'address'])

    data_list = []  # Used to store the data dictionary for each row

    sample_df = pd.read_csv(f"data/{dataset_name}/preprocessed/sample.csv")

    poi_list = sample_df["PoiId"].unique()

    existing_poi_list = POI_database["poi_id"].values

    require_poi_list = [poi for poi in poi_list if poi not in existing_poi_list]
    print(len(require_poi_list))

    if require_poi_list:

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

        # Create DataFrame using list
        newdf = pd.DataFrame(data_list)
        print(newdf.shape)

        # Merge the new DataFrame into POI_database
        POI_database = pd.concat([POI_database, newdf], ignore_index=True)

        print(POI_database.shape)
        # Write POI_database to file
        POI_database.to_csv(save_path, index=False)

    # rewrite the csv files
    for i in ['train', 'test', 'validate']:
        csv_path = f"data/{dataset_name}/preprocessed/{i}_sample.csv"
        csv_save_path = f"data/{dataset_name}/preprocessed/{i}_sample_address.csv"

        df = pd.read_csv(csv_path)
        # print(df.head()["PoiId"])
        df['address']=df.apply(lambda row: map_address(row, POI_database), axis=1)

        df.to_csv(csv_save_path, index=False)


        
    # csv_path = f"data/{dataset_name}/preprocessed/sample.csv"
    # csv_save_path = f"data/{dataset_name}/preprocessed/sample_address.csv"

    # df = pd.read_csv(csv_path)
    # # print(df.head()["PoiId"])

    # df['address']=df.apply(lambda row: map_address(row, POI_database), axis=1)

    # df.to_csv(csv_save_path, index=False)

    csv_path = f"data/{dataset_name}/preprocessed/sample.csv"
    df = pd.read_csv(csv_path)
    poi_list = df["PoiId"].unique()
    # Check if POI is continuous
    print(len(poi_list))
    print(max(poi_list))
    print(min(poi_list))
    


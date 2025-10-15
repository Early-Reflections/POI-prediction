import torch
import pandas as pd
import os.path as osp
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
import torch.nn as nn

class BertCheckinEmbedding(nn.Module):
    def __init__(self, dataset_name):
        super(BertCheckinEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # read the POI database
        # root = osp.join('D:/Projects/Spatio-Temporal-Hypergraph-Model/data', dataset_name, 'preprocessed')
        root = osp.join('D:/Projects/STCGCN/data', dataset_name, 'preprocessed')
        poi_database_path = osp.join(root, 'POI_database.csv')
        sampling_path = osp.join(root, 'sample.csv')
        self.poi_sample = pd.read_csv(sampling_path)
        # print(self.poi_sample[['Latitude', 'Longitude']].drop_duplicates().shape)
        print(len(self.poi_sample['PoiId'].unique()))

        # 按 Latitude 和 Longitude 分组，统计每个经纬度对应的唯一 PoiId 数量
        # grouped = self.poi_sample.groupby(['Latitude', 'Longitude'])['PoiId']

        # # 统计每个经纬度对应的独一无二的 PoiId 数量
        # unique_poi_counts = grouped.nunique()

        # # 找出对应唯一 PoiId 数量大于 1 的经纬度组合
        # conflicts = unique_poi_counts[unique_poi_counts > 1]

        # # 打印每个冲突的 Latitude, Longitude 及其对应的 PoiId 和每个 PoiId 的重复次数
        # for (lat, lon), _ in conflicts.items():
        #     print(f"Latitude: {lat}, Longitude: {lon}")
        #     poi_ids = grouped.get_group((lat, lon))
        #     poi_counts = poi_ids.value_counts()  # 统计每个 PoiId 的重复数量
        #     for poi_id, count in poi_counts.items():
        #         print(f"  PoiId: {poi_id}, Count: {count}")

        self.poi_database = pd.read_csv(poi_database_path)
        self.sort_poi_database()

    def sort_poi_database(self):
        # Step 1: 提取唯一的 PoiId 和其对应的 Latitude, Longitude
        unique_pois = self.poi_sample[['Latitude', 'Longitude', 'PoiId']].drop_duplicates(subset=['PoiId'])

        # Step 2: 去除 poi_database 中重复的经纬度，只保留第一个 address
        unique_database = self.poi_database.drop_duplicates(subset=['latitude', 'longitude'])

        # Step 3: 合并数据，补充 address
        merged_pois = pd.merge(unique_pois,
                            unique_database[['latitude', 'longitude', 'address']],
                            left_on=['Latitude', 'Longitude'], 
                            right_on=['latitude', 'longitude'], 
                            how='left')

        # Step 4: 如果某些 PoiId 的 address 缺失，用默认值填充
        merged_pois['address'] = merged_pois['address'].fillna('Unknown Address')

        # Step 5: 确保最终每个 PoiId 只保留一个结果
        final_pois = merged_pois[['PoiId', 'address']].drop_duplicates(subset=['PoiId'])

        # Step 6: 打印结果
        print("Final data shape:", final_pois.shape)
        print(final_pois.head())

        # 筛选出 address 为 "Unknown Address" 的数据
        unknown_address_records = final_pois[final_pois['address'] == 'Unknown Address']

        # 打印结果
        print("Records with 'Unknown Address':")
        print(unknown_address_records)

        self.poi_database = final_pois.sort_values(by='PoiId').reset_index(drop=True)

    def bert_embedding(self, text):
        # mapping poi id to address text
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**encoded_input) # The embeddings of the last layer
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

    def apply_bert_embedding(self):
        # Use tqdm to show the progress bar
        embeddings = []
        for address in tqdm(self.poi_database['address'], desc="Encoding addresses"):
            embedding = self.bert_embedding(address)
            embeddings.append(embedding)
        
        # Convert list of tensors to a single tensor
        embeddings_tensor = torch.cat(embeddings, dim=0)  # [num_addresses, hidden_dim]
        return embeddings_tensor


# Test the code
dataset_name = 'ca'
bert_checkin_embedding = BertCheckinEmbedding(dataset_name)
poi_embeddings = bert_checkin_embedding.apply_bert_embedding()

# Save the embeddings to a .pt file
root = osp.join('D:/Projects/STCGCN/data', dataset_name, 'preprocessed')
torch.save(poi_embeddings, osp.join(root, 'bert_address_embedding.pt'))

from google.cloud import vision
import os 
from tqdm import tqdm
import time
import sys 
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *
import argparse
from urllib.parse import urlparse
import json
import requests
import pytineye
import urllib.parse
key_file_path = Path("dataset/key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= key_file_path.as_posix()

print(dir(pytineye.TinEyeAPIRequest))

def detect_web(path,how_many_queries=30):
    """
    Detects web annotations given an image.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection

    page_urls = []
    matching_image_urls = {}
    visual_entities = {}

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )
        
        for page in annotations.pages_with_matching_images:
            page_urls.append(page.url)
            if page.full_matching_images:
                #List of image URLs for that webpage (the image can appear more than once)
                matching_image_urls[page.url] = [image.url for image in page.full_matching_images]
            else:
                matching_image_urls[page.url] = []
            if page.partial_matching_images: 
                matching_image_urls[page.url] += [image.url for image in page.partial_matching_images] 
    else:
        print('No matching images found for ' + path)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            #Collect web entities as entity-score dictionary pairs
            visual_entities[entity.description] = entity.score

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return page_urls, matching_image_urls, visual_entities



def detect_tineye(path, max_results=30):
    # 请替换为你自己的API URL和API密钥
    TINEYE_API_URL = "https://api.tineye.com/rest/"
    TINEYE_API_KEY = "8_S=+7YmvS7wyk*sNaTQkx_0osIn,flM=MNw6W6B"  # 替换为你自己的API密钥

    # 初始化 TinEye API 请求对象
    api = pytineye.TinEyeAPIRequest(api_url=TINEYE_API_URL, api_key=TINEYE_API_KEY)
    
    page_urls = []
    matching_image_urls = {}

    # 检查是否是图片文件
    if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):        
        with open(path, "rb") as img_file:
            image_data = img_file.read()

        # 使用 search_data 进行反向搜索
        try:
            response = api.search_data(image_data, limit=max_results)

            # 直接通过 response.matches 访问匹配项
            if len(response.matches) > 0:  # matches 是一个列表
                for result in response.matches:  # 访问每个匹配项
                    page_url = result.image_url  # 获取图片 URL
                    if page_url not in page_urls:
                        page_urls.append(page_url)
                    
                    if page_url not in matching_image_urls:
                        matching_image_urls[page_url] = []
                    matching_image_urls[page_url].append(result.image_url)  # 获取匹配图片的 URL

            else:
                print(f"No matches found for {path}.")
        except Exception as e:
            print(f"发生错误: {e}")
    else:
        print(f"跳过非图片文件: {path}")
    
    return page_urls, matching_image_urls


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Collect evidence using Google Reverse Image Search.')
    parser.add_argument('--collect_google', type=int, default=0, 
                        help='Whether to collect evidence URLs with the google API. If 0, it is assumed that a file containing URLs already exists.')
    parser.add_argument('--evidence_urls', type=str, default='dataset/retrieval_results/evidence_urls.json',
                        help='Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.')
    parser.add_argument('--google_vision_api_key', type=str,  default= "", #Provide your own key here as default value
                        help='Your key to access the Google Vision services, including the web detection API. Only needed if collect_google is set to 1.')  
    parser.add_argument('--image_path', type=str, default='dataset/processed_img/',
                        help='The folder where the images are stored.') 
    parser.add_argument('--raw_ris_urls_path', type=str, default='dataset/retrieval_results/ris_results.json',
                        help='The json file to store the raw RIS results.') 
    parser.add_argument('--scrape_with_trafilatura', type=int, default=1, 
                        help='Whether to scrape the evidence URLs with trafilatura. If 0, it is assumed that a file containing the scraped webpages already exists.') 
    parser.add_argument('--trafilatura_path', type=str, default='dataset/retrieval_results/trafilatura_data.json',
                        help='The json file to store the scraped trafilatura  content as a json file.')
    parser.add_argument('--json_path', type=str, default='dataset/retrieval_results/evidence.json',
                        help='The json file to store the text evidence as a json file.')
    parser.add_argument('--max_results', type=int, default=30,
                        help='The maximum number of web-pages to collect with the web detection API.') 
    parser.add_argument('--sleep', type=int, default=3,
                        help='The waiting time between two web detection API calls') 
    parser.add_argument('--collect_tineye', type=int, default=1, 
                        help='Whether to collect evidence URLs with TinEye API. If 0, assumes a file with URLs already exists.')
    parser.add_argument('--tineye_api_key', type=str, default="8_S=+7YmvS7wyk*sNaTQkx_0osIn,flM=MNw6W6B",  # Provide your TinEye API key
                        help='Your key to access the TinEye API. Required if collect_tineye is set to 1.')

    

    args = parser.parse_args()
    key = os.getenv(args.google_vision_api_key)

    #Create directories if they do not exist yet
    if not 'retrieval_results'  in os.listdir('dataset/'):
        os.mkdir('dataset/retrieval_results/')
    
    # 加载 MBFC 数据库
    with open("dataset/MBFC Bias Database 12-12-24.json", "r") as file:
        mbfc_data = json.load(file)

    # 创建 {domain: credibility} 快速查找表
    credibility_lookup = {entry["Domain"]: entry["Credibility"] for entry in mbfc_data}

    def get_credibility(url):
        """从 URL 获取主域名，并查询 MBFC 可信度"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")  # 移除 'www.'
        return credibility_lookup.get(domain, "Unknown")  # 若找不到，则返回 Unknown


    all_filtered_results = []  # 用于存储 Google + TinEye 的 RIS 结果
    # Google RIS
    if args.collect_google:
        raw_ris_results = []
        filtered_results = []  # 存储筛选后的结果

        for path in tqdm(os.listdir(args.image_path)):
            urls, image_urls, vis_entities = detect_web(args.image_path + path, args.max_results)

            for url in urls:
                credibility = get_credibility(url)  # 计算单个 URL 可信度

                 # 如果 URL 可信度是 "Unknown"，跳过
                if credibility in ["Unknown", "Low"]:
                    continue  

                # 处理当前 URL 相关的 image_urls
                url_image_urls = {}  # 存储 URL 对应的图片链接

                if url in image_urls:
                    url_image_urls = image_urls[url]  # 直接获取该 URL 关联的所有图片链接
            

                # 存储结果，每个 URL 作为一个单独的条目
                filtered_results.append({
                    'image path': args.image_path + path,
                    'url': url,
                    'credibility': credibility,
                    'image urls': url_image_urls,  # 单独存储每个 URL 相关的图片和可信度
                    'visual entities': vis_entities  # 视觉实体数据仍然和图片相关联
                })

                time.sleep(args.sleep)
        all_filtered_results.extend(filtered_results)  # ✅ 合并 TinEye 结果
    
    # Tineye RIS
    if args.collect_tineye:
        raw_ris_results = []
        filtered_results = []  # 存储筛选后的结果

        for path in tqdm(os.listdir(args.image_path)):
            urls, image_urls = detect_tineye(args.image_path + path, args.max_results)

            for url in urls:
                credibility = get_credibility(url)  # 计算单个 URL 可信度

                 # 如果 URL 可信度是 "Unknown"，跳过
                if credibility in ["Unknown", "Low"]:
                    continue  

                # 处理当前 URL 相关的 image_urls
                url_image_urls = {}  # 存储 URL 对应的图片链接

                if url in image_urls:
                    url_image_urls = image_urls[url]  # 直接获取该 URL 关联的所有图片链接
            

                # 存储结果，每个 URL 作为一个单独的条目
                filtered_results.append({
                    'image path': args.image_path + path,
                    'url': url,
                    'credibility': credibility,
                    'image urls': url_image_urls,  # 单独存储每个 URL 相关的图片和可信度
                })

                time.sleep(args.sleep)
        all_filtered_results.extend(filtered_results)  # ✅ 合并 TinEye 结果
        # 保存筛选后的结果
    with open(args.raw_ris_urls_path, 'w') as file:
        json.dump(all_filtered_results, file, indent=4)
        # 进一步过滤 URL，移除不适合抓取的内容
    selected_data = get_filtered_retrieval_results(args.raw_ris_urls_path)
        # print("🔹 selected_data 预览:", selected_data[:5])  # 只打印前5个数据，避免太多输出
    print(f" selected_data的数量: {len(selected_data)}")
      
        
    # else:
    #     # Load evidence that has already been collected
    #     selected_data = [
    #         d for d in load_json(args.evidence_urls) 
    #         if d['image path'].split('/')[-1] in os.listdir('dataset/processed_img/')
    #     ]
    
    urls = [d['raw url'] for d in selected_data]
    images = [d['image urls'] for d in selected_data]

    if args.scrape_with_trafilatura:
        #Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            output.append(extract_info_trafilatura(urls[u],images[u]))
            #Only store in json file every 50 evidence
            if u%1==0:
                save_result(output,args.trafilatura_path) 
                output = []
   

    #Save all results in a Pandas Dataframe
    evidence_trafilatura = load_json(args.trafilatura_path)
    # print("🔹 `trafilatura_data.json` 即将写入的数据预览:")
    # print(evidence_trafilatura)  # 这里可以检查数据是不是空的
    dataset = load_json('dataset/train.json') + load_json('dataset/val.json')  + load_json('dataset/test.json')
    
    evidence = merge_data(evidence_trafilatura, selected_data, dataset).fillna('').to_dict(orient='records')
    print(f"🔍 Trafilatura 解析成功的数量: {len(evidence_trafilatura)}")
    # print(evidence_trafilatura[:3])
    # Save the list of dictionaries as a JSON file
    with open(args.json_path, 'w') as file:
        json.dump(evidence, file, indent=4)

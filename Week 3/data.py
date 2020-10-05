import os
import time
import requests
import shutil
import gzip

from typing import Dict, List, Tuple

from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split


class Downloader:

    URLS = {
        'single': [
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Appliances.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Arts_Crafts_and_Sewing.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Automotive.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Baby.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Beauty.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Cell_Phones_and_Accessories.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Electronics.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Home_and_Kitchen.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Industrial_and_Scientific.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Musical_Instruments.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Office_Products.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Patio_Lawn_and_Garden.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Pet_Supplies.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Software.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Sports_and_Outdoors.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Tools_and_Home_Improvement.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Toys_and_Games.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz'
        ],
        'multiple': [
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Automotive.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Baby.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Beauty.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Cell_Phones_and_Accessories.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Clothing_Shoes_and_Jewelry.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Electronics.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Grocery_and_Gourmet_Food.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Health_and_Personal_Care.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Home_and_Kitchen.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Musical_Instruments.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Office_Products.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Patio_Lawn_and_Garden.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Pet_Supplies.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Sports_and_Outdoors.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Tools_and_Home_Improvement.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Toys_and_Games.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Video_Games.json.gz'
        ]
    }

    def __init__(self,
                 data_path: str,
                 override: bool = False,
                 sleep_time: float = 0.05,
                 verbose: bool = True):

        self.data_path = data_path
        self.override = override
        self.sleep_time = sleep_time
        self.verbose = verbose

    @staticmethod
    def make_dir(path: str, override: bool = False):
        if override:
            shutil.rmtree(path, ignore_errors=True)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @staticmethod
    def download_file(url: str, save_path: str, verbose: bool = False, chunk_size: int = 8192):
        try:
            filename = save_path.split('/')[-1]
            with requests.get(url, stream=True) as request:
                request.raise_for_status()
                with open(save_path, 'wb') as file_object:
                    for chunk in tqdm(request.iter_content(chunk_size=chunk_size),
                                      desc=f'Download {filename}',
                                      disable=not verbose):
                        if chunk:
                            file_object.write(chunk)
        except KeyboardInterrupt as exception:
            os.remove(save_path)
            raise exception
        except Exception as exception:
            os.remove(save_path)
            raise exception

    def run(self):

        self.make_dir(self.data_path, override=self.override)

        for key in self.URLS:
            for url in tqdm(self.URLS[key], desc=key, disable=not self.verbose):
                filename = key + '_' + url.split('/')[-1]

                time.sleep(self.sleep_time)

                file_path = os.path.join(self.data_path, filename)

                if not os.path.isfile(file_path):
                    self.download_file(url=url, save_path=file_path, verbose=False)


class Parser:

    BAD_CATEGORIES = [
        'electronics',
        'software',
        'appliances',
        'industrial and scientific',
        'musical instruments',
        'arts crafts and sewing',
        'video games',
        'toys and games',
        'clothing shoes and jewelry',
        'home and kitchen',
        'health and personal care',
        'tools and home improvement',
        'patio lawn and garden'
    ]

    def __init__(self,
                 data_path: str,
                 is_question_classification: bool = True,
                 train_samples: int = 250_000,
                 valid_samples: int = 50_000,
                 lower_threshold: int = 5,
                 upper_threshold: int = 512):

        self.data_path = data_path
        self.is_question_classification = is_question_classification

        self.train_samples = train_samples
        self.valid_samples = valid_samples

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    @staticmethod
    def parse_filename(file_path: str) -> str:
        category = file_path.split('/')[-1].split('.')[0]

        category = category.lower().replace('qa_', '')
        category = category.replace('single_', '').replace('multiple_', '')
        category = category.replace('_', ' ')

        return category

    @staticmethod
    def prepare_text(text: str) -> str:
        text = text.lower().encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        return text

    def filter_sample(self, sample: Dict[str, str]) -> bool:

        if not (self.lower_threshold <= len(sample['question']) <= self.upper_threshold):
            return False

        if not (self.lower_threshold <= len(sample['response']) <= self.upper_threshold):
            return False

        if sample['category'] in self.BAD_CATEGORIES:
            return False

        return True

    def parse_single_sample(self,
                            sample: Dict,
                            category: str,
                            with_types: bool = False) -> List[Dict[str, str]]:

        result_sample = {
            'question': self.prepare_text(sample['question']),
            'response': self.prepare_text(sample['answer']),
            'category': category
        }

        if with_types:
            result_sample['question_type'] = sample['questionType']
            result_sample['answer_type'] = sample['answerType']

        result_sample = [result_sample] if self.filter_sample(result_sample) else list()

        return result_sample

    def parse_multiple_sample(self,
                              sample: Dict,
                              category: str) -> List[Dict[str, str]]:

        result_samples: List[Dict[str, str]] = list()

        for question_data in sample['questions']:
            for answer_data in question_data['answers']:
                parsed_sample = {
                    'question': self.prepare_text(question_data['questionText']),
                    'response': self.prepare_text(answer_data['answerText']),
                    'category': category,
                }
                if self.filter_sample(parsed_sample):
                    result_samples.append(parsed_sample)

        return result_samples

    def get_file_paths(self):

        return [os.path.join(self.data_path, file)
                for file in os.listdir(self.data_path)
                if file.endswith('.json.gz')]

    def read_data(self) -> List[Dict[str, str]]:

        file_paths = self.get_file_paths()

        data: List[Dict[str, str]] = list()

        for file_path in tqdm(file_paths, desc='Reading'):
            with gzip.open(file_path) as file_object:
                for sample in file_object:
                    sample = eval(sample)
                    category = self.parse_filename(file_path)

                    if 'single' in file_path:
                        data.extend(self.parse_single_sample(sample, category))
                    elif 'multiple' in file_path:
                        data.extend(self.parse_multiple_sample(sample, category))

        return data

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        data = pd.DataFrame(self.read_data())

        subset = ['question']

        if not self.is_question_classification:
            subset.append('response')

        data = data.drop_duplicates(subset=subset)

        data = data.sample(frac=1)

        if self.train_samples + self.valid_samples >= data.shape[0]:
            raise ValueError('Sum of train_samples and valid_samples must be less than length of data')

        data, valid = train_test_split(data, stratify=data.category, test_size=self.valid_samples)

        unlabeled, train = train_test_split(data, stratify=data.category, test_size=self.train_samples)

        unlabeled = unlabeled[['question', 'response']]

        unlabeled.reset_index(inplace=True, drop=True)
        train.reset_index(inplace=True, drop=True)
        valid.reset_index(inplace=True, drop=True)

        return unlabeled, train, valid

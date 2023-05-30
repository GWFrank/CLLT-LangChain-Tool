import datetime
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from enum import Enum
from pprint import pprint
from typing import Optional, Type

from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import crawler


STOPWORDS = open('hit_stopwords.txt', 'r').read().split('\n')


class CommentType(Enum):
    Upvote = "pos"
    Downvote = "neg"
    Arrow = "neu"


def filenameParser(filename: str) -> tuple[str, str, str]:
    filename = filename[:-4]
    date, time, post_id = filename.split('_')
    return date, time, post_id


def findPostByID(post_id: str) -> str:
    for year_dir in os.listdir("./HatePolitics/"):
        for filename in os.listdir(os.path.join("./HatePolitics/", year_dir)):
            if f"{post_id}" in filename:
                return os.path.join("./HatePolitics/", year_dir, filename)


def getSummaryByContent(contents: list[str]) -> list[str]:
    LANGUAGE = 'chinese'
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    stop_words = set(get_stop_words(LANGUAGE))
    summarizer.stop_words = stop_words.union(set(STOPWORDS))
    tokenizer = Tokenizer(LANGUAGE)
    summaries = []
    for content in contents:
        parser = PlaintextParser.from_string(
            content, tokenizer,
        )
        summaries.append(
            summarizer(parser.document, 1)[0]._text
        )
    return summaries


def getCommentsOfType(tree: ET, type: CommentType) -> str:
    root = tree.getroot()
    selected_comments = root.findall(f"./text/comment[@c_type='{type.value}']")
    return selected_comments

def getKeywords(strings: list[str]):
    concat = ' '.join(strings)
    keywords = jieba.analyse.extract_tags(concat, topK=10, withWeight=False, allowPOS=())
    return keywords


# XML file format:
#
# ```xml
# <?xml version='1.0' encoding='utf-8'?>
# <TEI.2>
#    <teiHeader>
#       <metadata name="media"></metadata>
#       <metadata name="author"></metadata>
#       <metadata name="post_id"></metadata>
#       <metadata name="year"></metadata>
#       <metadata name="board"></metadata>
#       <metadata name="title"></metadata>
#    </teiHeader>
#    <text>
#       <body author="">
#          <s>
#             <w type=""></w> <!-- type=詞性標記 -->
#          </s>
#       </body>
#       <title author=""></title>
#       <!-- c_type {pos: 推, neu: 箭頭, neg: 噓} -->
#       <comment author="" c_type=""></comment>
#    </text>
# </TEI.2>
# ```
#
# Template for a tool ([Reference](https://python.langchain.com/en/latest/modules/agents/tools/custom_tools.html))
#
# ```python
# class TemplateTool(BaseTool):
#     name = "custom_search"
#     description = "useful for when you need to answer questions about current events"
#
#     def _run(self,
#              query: str,
#              run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
#         """Use the tool."""
#         return "query"
#
#     async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("custom_search does not support async")
# ```

class TempTool(BaseTool):
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class GetPostIDsByDate(TempTool):
    """
    Get ptt posts in the database, by date
    """
    name = "get_posts_by_date"
    description = """
    Input: string of date, in format YYYYMMDD, (e.g. 20200101)
    Output: list of post ids, serialized in json
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        year, month, day = query[:4], query[4:6], query[6:]
        assert len(year) == 4 and len(month) == 2 and len(day) == 2
        directory = f"./HatePolitics/{year}"
        in_range_post_ids = []
        files = os.listdir(directory)
        files.sort()
        for filename in files:
            date, time, post_id = filenameParser(filename)
            if date == year + month + day:
                in_range_post_ids.append(post_id)
        return json.dumps(in_range_post_ids, ensure_ascii=False)



class GetArrowCount(TempTool):
    """
    Get arrow count of a post
    """
    name = "get_arrow_count"
    description = """
    Input: post_id returned by get_posts_by_date (e.g. M.1672914887.A.04F)
    Output: arrow count
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Arrow)))



class GetDownvoteCount(TempTool):
    """
    Get upvote count of a post
    """
    name = "get_downvote_count"
    description = """
    Input: post_id returned by get_posts_by_date (e.g. M.1672914887.A.04F)
    Output: downvote count
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Downvote)))



class GetUpvoteCount(TempTool):
    """
    Get upvote count of a post
    """
    name = "get_upvote_count"
    description = """
    Input: post_id returned by get_posts_by_date (e.g. M.1672914887.A.04F)
    Output: upvote count
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Upvote)))



class GetPostTitle(TempTool):
    """
    Get the title of a post by post_id
    """
    name = "get_post_title"
    description = """
    Input: post_id returned by get_posts_by_date (e.g. M.1672914887.A.04F)
    Output: title of the post
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        tree = ET.parse(filename)
        root = tree.getroot()
        title_node = root.find("./text/title")
        title_text = "".join(
            [word.text for word in title_node.findall("./s/w")])
        return title_text



class GetPostBody(TempTool):
    """
    Get the body of a post by post_id
    """
    name = "get_post_body"
    description = """
    Input: post_id returned by get_posts_by_date (e.g. M.1672914887.A.04F)
    Output: body of the post
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        tree = ET.parse(filename)
        root = tree.getroot()
        body_node = root.find("./text/body")
        body_text = [
            "".join([word.text for word in sentence.findall("w")]) for sentence in body_node.findall("s")
        ]
        return json.dumps(body_text, ensure_ascii=False)



class GetPostsTitlesByCrawler(TempTool):
    """
    Get latest news posts titles from crawler
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_posts_titles_by_crawler"
    description = "獲得近期政治新聞標題"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=100)
        titles = [post['title'] for post in posts]
        return json.dumps(titles)



class GetPostsSummaryByCrawler(TempTool):
    """
    Get latest news posts summary content from crawler
    Summarize by package sumy using LsaSummarizer
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_posts_titles_by_crawler"
    description = "獲得近期政治新聞內文概述"
    LANGUAGE = "chinese"
    tokenizer = Tokenizer(LANGUAGE)

    def summarize(self, contents):
        parser = PlaintextParser.from_string(
            contents,
            self.tokenizer,
        )
        summaries = summarizer
        return summaries

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=100)
        contents = [post['content'] for post in posts]
        summaries = summaries(contents)
        return json.dumps(summaries)



class GetPostsKeywordsByCrawler(TempTool):
    """
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_posts_titles_by_crawler"
    description = "獲得近期政治新聞內文關鍵字"
    LANGUAGE = "chinese"
    tokenizer = Tokenizer(LANGUAGE)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=100)
        contents = [post['content'] for post in posts]
        summaries = summaries(contents)
        return json.dumps(summaries)

class GetPttPostsKeywordsByDate(TempTool):
    """
    Get keywords of ptt posts by date
    """
    name = "get_ptt_posts_keywords_by_date"
    description = """獲得近期PTT政治文章內文關鍵字
    input: date (e.g. 20210101)
    output: keywords"""
    LANGUAGE = "chinese"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        post_ids = json.loads(GetPostIDsByDate().run(query))
        contents = []
        for post_id in post_ids:
            contents.append(json.loads(GetPostBody().run(post_id)))
        keywords = getKeywords(contents)
        return json.dumps(keywords)


if __name__ == "__main__":
    # directory = './HatePolitics/2023'
    # files = os.listdir(directory)
    # files.sort()

    # start_date = 20230101
    # end_date = 20230131
    # in_range_post_ids = []
    # for filename in files:
    #     date = int(filename[:8])
    #     if date >= start_date and date <= end_date:
    #         in_range_post_ids.append(filename)

    # # print(inrange_files)
    # print(len(in_range_post_ids))

    # # parse each xml file and get the author name
    # author_count = Counter()
    # min_score = 0
    # max_score = 0
    # min_author = ''
    # max_author = ''
    # min_title = ''
    # max_title = ''

    # for filename in in_range_post_ids:
    #     tree = ET.parse(directory + '/' + filename)
    #     root = tree.getroot()
    #     author = root[0][1].text
    #     title = root[0][5].text
    #     author_count[author] += 1

    #     # loop through all comments
    #     score = 0
    #     for comment in root[1][2:]:
    #         # print(comment)
    #         # print(comment.attrib)
    #         comment_type = comment.attrib['c_type']
    #         if comment_type == 'pos':
    #             score += 1
    #         elif comment_type == 'neg':
    #             score -= 1

    #     if score < min_score:
    #         min_score = score
    #         min_author = author
    #         min_title = title
    #     elif score > max_score:
    #         max_score = score
    #         max_author = author
    #         max_title = title

    # print(author_count.most_common(10))
    # print(min_score, min_author, min_title)
    # print(max_score, max_author, max_title)

    # filename = in_range_post_ids[0]
    # tree = ET.parse(directory + '/' + filename)
    # root = tree.getroot()
    # print(root[1][2].attrib)

    post_ids = json.loads(GetPostIDsByDate().run("20200102"))[:3]
    pprint(post_ids)
    for post_id in post_ids:
        print("="*10)
        print(f"{GetUpvoteCount().run(post_id)} 推 - {GetDownvoteCount().run(post_id)} 噓 - {GetArrowCount().run(post_id)} 箭頭")
        print(GetPostTitle().run(post_id))
        print("\n".join(json.loads(GetPostBody().run(post_id))[:2]))

    # dataset = [
    #     '''
    #     Extremely Severe Cyclonic Storm Mocha was a powerful and deadly tropical cyclone in the North Indian Ocean which affected Myanmar and parts of Bangladesh in May 2023. The second depression and the first cyclonic storm of the 2023 North Indian Ocean cyclone season, Mocha originated from a low-pressure area that was first noted by the India Meteorological Department (IMD) on 8 May. After consolidating into a depression, the storm tracked slowly north-northwestward over the Bay of Bengal, and reached extremely severe cyclonic storm intensity. After undergoing an eyewall replacement cycle, Mocha rapidly strengthened, peaking at Category 5-equivalent intensity on 14 May with winds of 280 km/h (175 mph), tying with Cyclone Fani as the strongest storm on record in the north Indian Ocean, in terms of 1-minute sustained winds. Mocha slightly weakened before making landfall, and its conditions quickly became unfavorable. Mocha rapidly weakened once inland and dissipated shortly thereafter.
    # Thousands of volunteers assisted citizens of Myanmar and Bangladesh in evacuating as the cyclone approached the international border.[6] Evacuations were also ordered for low-lying areas in Sittwe, Pauktaw, Myebon, Maungdaw, and Buthidaung. In Bangladesh, over 500,000 individuals were ordered to be relocated to coastal areas of the country due to the storm's approach. Officials from the military declared the state of Rakhine a natural disaster area. Several villages in Rakhine State were also damaged by the cyclone.
    # Cyclone Mocha killed at least 463 people, including three indirect deaths in Bangladesh. It also injured 719 people, and left 101 others missing.[7][5] The storm caused about US$1.07 million of damage in Bangladesh.[8] '''
    #     'Kumquat plants have thornless branches and extremely glossy leaves. They bear dainty white flowers that occur in clusters or individually inside the leaf axils. The plants can reach a height from 2.5 to 4.5 metres (8.2 to 14.8 ft), with dense branches, sometimes bearing small thorns.[5] They bear yellowish-orange fruits that are oval or round in shape. The fruits can be 1 inch (2.5 cm) in diameter and have a sweet, pulpy skin and slightly acidic inner pulp. All the kumquat trees are self-pollinating. Kumquats can tolerate both frigid and hot temperatures',
    #     '''The photo portrays fourteen Israeli soldiers in an abandoned barracks with traditional army dinnerware. Unlike the original painting, Nes' version lacks tension and shows the soldiers in private conversations, while the central figure (Jesus) "stares vacantly into space". The artist does not provide a specific interpretation, but expresses sympathy and hope that it is not their last meal together. One extra person is added to avoid "direct quotation" of Leonardo da Vinci.[7] The fourteenth man (standing at the left) is the only one, apart for the central figure, who is not engaged in a conversations and looks apart, and the only one whose uniform shows the Israeli Defense Forces patch''',
    #     'Is this the first document?',
    
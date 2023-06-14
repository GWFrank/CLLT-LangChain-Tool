import glob
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from enum import Enum
from pprint import pprint
from typing import Optional

import pandas as pd
import jieba.analyse
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from dotenv import load_dotenv

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import crawler
import weviate_tool

with open("hit_stopwords.txt", "r") as f:
    STOPWORDS = f.read().split('\n')
with open("post_index.json", "r") as f:
    POST_INDEX = json.load(f)
load_dotenv('.env')


class CommentType(Enum):
    Upvote = "pos"
    Downvote = "neg"
    Arrow = "neu"


def filenameParser(filename: str) -> tuple[str, str, str]:
    filename = filename[:-4]
    date, time, post_id = filename.split('_')
    return date, time, post_id


def findPostByID(post_id: str) -> str:
    if post_id in POST_INDEX:
        return POST_INDEX[post_id]

    raise FileNotFoundError(f"Post {post_id} not found")


def getSummaryByContent(contents: list[str]) -> list[str]:
    LANGUAGE = 'chinese'
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    stop_words = set(get_stop_words(LANGUAGE))
    summarizer.stop_words = stop_words.union(set(STOPWORDS))
    tokenizer = Tokenizer(LANGUAGE)
    summaries = []
    for content in contents:
        try:
            parser = PlaintextParser.from_string(
                content, tokenizer,
            )
            summaries.append(
                summarizer(parser.document, 1)[0]._text
            )
        except:
            summaries.append(content)
    return summaries


def getCommentsOfType(tree: ET, type: CommentType) -> str:
    root = tree.getroot()
    selected_comments = root.findall(f"./text/comment[@c_type='{type.value}']")
    return selected_comments


def zhTWSanitizer(strings: list[str]):
    return [re.sub(r"[^\u4e00-\u9fff]", "", s) for s in strings]


def getKeywords(strings: list[str]):
    strings = zhTWSanitizer(strings)
    concat = ' '.join(strings)
    print(f'Keyword: Total content length {len(concat)}')
    jieba.analyse.set_stop_words('hit_stopwords.txt')
    keywords = jieba.analyse.extract_tags(concat, topK=10, withWeight=False,
                                          allowPOS=('ns', 'n', 'vn', 'v'))
    return keywords


def getCommentStringsOfType(tree: ET, type: CommentType) -> list[str]:
    comments = getCommentsOfType(tree, type)
    sentences = []
    for comment in comments:
        for s in comment.findall("./s"):
            sentences.append("".join([w.text for w in s.findall("./w")]))
    return sentences


def getUpvoteComments(post_id):
    return getCommentStringsOfType(ET.parse(findPostByID(post_id)), CommentType.Upvote)


def getDownvoteComments(post_id):
    return getCommentStringsOfType(ET.parse(findPostByID(post_id)), CommentType.Downvote)


def getArrowvoteComments(post_id):
    return getCommentStringsOfType(ET.parse(findPostByID(post_id)), CommentType.Arrow)


def getPostIdByKeyword(keyword: str, count=5) -> list[str]:
    docs = weviate_tool.retrieve_docs(keyword, count=100)
    return [doc.metadata['post_id'] for doc in docs]


def _getPostIdByKeyword(keyword: str, count=5) -> list[str]:
    """
    Pseudo function for getPostIdByKeyword
    """
    post_id_list = [f.split("/")[-1].split("_")[-1][:-4]
                    for f in glob.glob("./HatePolitics/*/*.xml")]
    return post_id_list[:count]


class TempTool(BaseTool):
    LANGUAGE = 'chinese'

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class GetPostIDsByDate(TempTool):
    """
    Get ptt posts in the database, by date
    """
    name = "get_post_ids_by_date"
    description = """獲取指定日期的文章ID
    Input: date in format YYYYMMDD (e.g. 20200101)
    Output: JSON 格式的 list of post ids 
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        query = query.strip()[-8:]
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


class GetArrowCountByID(TempTool):
    """
    Get arrow count of a post
    """
    name = "get_arrow_count_by_id"
    description = """獲得指定文章ID的普通留言數
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: 普通留言數
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Arrow)))


class GetDownvoteCountByID(TempTool):
    """
    Get upvote count of a post
    """
    name = "get_downvote_count_by_id"
    description = """獲得指定文章ID的噓文數
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: 噓文數
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Downvote)))


class GetUpvoteCountByID(TempTool):
    """
    Get upvote count of a post
    """
    name = "get_upvote_count_by_id"
    description = """獲得指定文章ID的推文數
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: 推文數
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        return str(len(getCommentsOfType(ET.parse(filename), CommentType.Upvote)))


class GetPostTitleByID(TempTool):
    """
    Get the title of a post by post_id
    """
    name = "get_post_title_by_id"
    description = """獲得指定文章ID的標題
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: 標題
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        tree = ET.parse(filename)
        root = tree.getroot()
        title_node = root.find("./text/title")
        # title_text = "".join(
        #     [word.text for word in title_node.findall("./s/w")])
        # print(query, title_node)
        title_text = ""
        if title_node is None:
            return title_text
        for sentence in title_node.findall("./s"):
            if sentence is None:
                continue
            for word in sentence.findall("./w"):
                if word is None:
                    continue
                if word.text is not None:
                    title_text += word.text
        return title_text


class GetPostContentByID(TempTool):
    """
    Get the content of a post by post_id
    """
    name = "get_post_content_by_id"
    description = """獲取指定文章ID的內文
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: list of sentences in the body
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        filename = findPostByID(query)
        # print(query, filename)
        tree = ET.parse(filename)
        root = tree.getroot()
        body_node = root.find("./text/body")
        # body_text = [
        #     "".join([word.text for word in sentence.findall("w")]) for sentence in body_node.findall("s")
        # ]
        body_text = []
        for sentence in body_node.findall("s"):
            sentence_text = []
            for word in sentence.findall("w"):
                if word.text is not None:
                    sentence_text.append(word.text)
            body_text.append("".join(sentence_text))
        return json.dumps(body_text, ensure_ascii=False)


class GetPostKeywordsByID(TempTool):
    """
    Get the keywords of a post by post_id
    """
    name = "get_post_keywords_by_id"
    description = """獲取指定文章ID的內文關鍵字
    Input: post_id (e.g. M.1672914887.A.04F)
    Output: list of sentences
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        body_text = json.loads(GetPostContentByID().run(query))
        keywords = getKeywords(body_text)
        return json.dumps(keywords, ensure_ascii=False)


class GetNewsTitlesWithCrawler(TempTool):
    """
    Get latest news titles from crawler
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_news_titles_with_crawler"
    description = """獲得近期政治新聞標題
    Input: empty string
    Output: list of titles
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=10)
        titles = [post['title'] for post in posts]
        return json.dumps(titles, ensure_ascii=False)


class GetNewsSummaryWithCrawler(TempTool):
    """
    Get latest news posts summary content from crawler
    Summarize by package sumy using LsaSummarizer
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_news_summary_with_crawler"
    description = """獲得近期政治新聞內文概述
    Input: empty string
    Output: list of summaries
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=10)
        contents = [post['content'] for post in posts]
        summaries = getSummaryByContent(contents)
        return json.dumps(summaries, ensure_ascii=False)


class GetNewsKeywordsWithCrawler(TempTool):
    """
    Support website: 
        (default): https://news.pts.org.tw/category/1
        https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB
    """
    name = "get_news_keywords_with_crawler"
    description = """獲得近期政治新聞內文關鍵字
    Input: empty string
    Output: list of keywords
    """

    LANGUAGE = "chinese"
    tokenizer = Tokenizer(LANGUAGE)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        posts = crawler.politic_news_crawler('pts', cnt=10)
        contents = [post['content'] for post in posts]
        keywords = getKeywords(contents)
        return json.dumps(keywords, ensure_ascii=False)


class GetPttPostsKeywordsOnDate(TempTool):
    """
    Get keywords of ptt posts of a specific date
    """
    name = "get_ptt_posts_keywords_of_date"
    description = """獲得指定日期PTT政治板文章內文關鍵字
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
            contents += json.loads(GetPostContentByID().run(post_id))
        keywords = getKeywords(contents)
        return json.dumps(keywords, ensure_ascii=False)


def getVoteByFilenames(args):
    file_name, query = args
    date, time, post_id = filenameParser(file_name)
    content = GetPostContentByID().run(post_id)
    if query in content:
        return int(GetUpvoteCountByID().run(post_id)), int(GetDownvoteCountByID().run(post_id))
    return 0, 0


def getVoteDateByFilenames(args):
    file_name, query = args
    date, time, post_id = filenameParser(file_name)
    content = GetPostContentByID().run(post_id)
    if query in content:
        return date, int(GetUpvoteCountByID().run(post_id)), int(GetDownvoteCountByID().run(post_id))
    return date, 0, 0


class GetKeywordsVote(TempTool):
    name = "get_keywords_vote"
    description = """獲得指定關鍵詞的推噓文數量
    Input: 關鍵詞（e.g. 總統）
    Output: 推噓文數量
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        up_vote_count = 0
        down_vote_count = 0
        file_name_list = [(f.split("/")[-1], query)
                          for f in glob.glob("./HatePolitics/*/*.xml")]
        vote_counts = process_map(
            getVoteByFilenames,
            file_name_list,
            max_workers=4,
            ncols=60,
            chunksize=1,
        )
        for vote_count in vote_counts:
            up_vote_count += vote_count[0]
            down_vote_count += vote_count[1]
        return f'關鍵字 {query} | 推：{up_vote_count}、噓：{down_vote_count}、推/噓比：{up_vote_count/(down_vote_count+1e-12)}'


class GetKeywordsVoteTrend(TempTool):
    name = "get_keywords_vote_trend"
    description = """獲得關鍵字在不同時間的推噓文數量
    Input: 關鍵詞（e.g. 總統）
    Output: 不同時間的推噓文數量
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        up_vote_counts = {}
        down_vote_counts = {}
        file_name_list = [(f.split("/")[-1], query)
                          for f in glob.glob("./HatePolitics/*/*.xml")]
        vote_counts = process_map(
            getVoteDateByFilenames,
            file_name_list,
            max_workers=4,
            ncols=60,
            chunksize=1,
        )
        for date, up_vote_count, down_vote_count in vote_counts:
            year = date[:4]
            if year not in up_vote_counts.keys():
                up_vote_counts[year] = 0
                down_vote_counts[year] = 0
            up_vote_counts[year] += up_vote_count
            down_vote_counts[year] += down_vote_count
        fmt = f'關鍵字 {query}：'
        for year in sorted(list(up_vote_counts.keys())):
            up_vote_count = up_vote_counts[year]
            down_vote_count = down_vote_counts[year]
            if down_vote_count == 0:
                fmt += f'\n{year}年 | 推：{up_vote_count}、噓：{down_vote_count}、推/噓比：N/A |'
            else:
                fmt += f'\n{year}年 | 推：{up_vote_count}、噓：{down_vote_count}、推/噓比：{up_vote_count/(down_vote_count):.2f} |'
        return fmt


class GetUpvoteCommentsByKeyword(TempTool):
    name = "get_upvote_comments_by_keyword"
    description = """獲得和指定關鍵詞相關的推文
    Input: 關鍵詞（e.g. 總統）
    Output: 數個推文（以 "," 分隔）
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        post_ids = getPostIdByKeyword(query)
        # post_ids = _getPostIdByKeyword(query)
        up_vote_comments = []
        for post_id in post_ids:
            try:
                comments = getUpvoteComments(post_id)
                up_vote_comments.extend(comments)
            except:
                continue
        return ', '.join(random.sample(up_vote_comments, k=min(5, len(up_vote_comments))))


class GetDownvoteCommentsByKeyword(TempTool):
    name = "get_downvote_comments_by_keyword"
    description = """獲得和指定關鍵詞相關的噓文
    Input: 關鍵詞（e.g. 總統）
    Output: 數個推文（以 "," 分隔）
    """

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # post_ids = getPostIdByKeyword(query)
        post_ids = _getPostIdByKeyword(query)
        up_vote_comments = []
        for post_id in post_ids:
            comments = getDownvoteComments(post_id)
            up_vote_comments.extend(comments)
        return ', '.join(random.sample(up_vote_comments, k=min(5, len(up_vote_comments))))


if __name__ == "__main__":
    # getKeywordsVote = GetKeywordsVote().run("蔡英文")
    # print(GetKeywordsVote().run("蔣萬安"))
    # print(GetKeywordsVote().run("高虹安"))
    # print(GetKeywordsVote().run("侯友宜"))
    # print(GetKeywordsVoteTrend().run(""))
    # print(GetKeywordsVote().run("蔡英文"))
    # print(GetKeywordsVote().run("柯文哲"))
    # print(GetKeywordsVoteTrend().run("噁心"))
    # print(GetKeywordsVoteTrend().run("無恥"))
    # print(GetKeywordsVoteTrend().run("下限"))
    # print(GetKeywordsVoteTrend().run("無能"))
    # print(GetUpvoteCommentsByKeyword().run('水桶'))
    # print(GetUpvoteCommentsByKeyword().run('柯文哲'))
    print(GetNewsSummaryWithCrawler().run(''))
    pass

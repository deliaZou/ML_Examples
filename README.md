#### [1-001 豆瓣电影数据抓取]()





##### 1-001 豆瓣电影数据抓取

接口文档

|        描述        |                            URL                            |                             参数                             |
| :----------------: | :-------------------------------------------------------: | :----------------------------------------------------------: |
|      电影信息      |        https://api.douban.com/v2/movie/subject/:id        |                         id: 电影 id                          |
|      演员信息      |       http://api.douban.com/v2/movie/celebrity/:id        |                         id: 演员 id                          |
|      TOP 250       |           http://api.douban.com/v2/movie/top250           |             start: 数据的开始项 count: 单页条数              |
|        搜索        |          https://api.douban.com/v2/movie/search           | start: 数据的开始项 count: 单页条数 q: 电影名称 tag: 电影标签 |
| 获取电影评论和评分 |   https://api.douban.com/v2/movie/subject/:id/comments    | start: 数据的开始项 count: 单页条数 apikey: 申请的 api key id: 电影 id |
|      电影标签      | https://movie.douban.com/j/search_tags?type=movie&source= |                                                              |
|    根据主题搜索    |        https://movie.douban.com/j/search_subjects         | type: 搜索类型，movie 电影 sort: 如何排序，recommend，time，rank playable: 是否在线播放，on，可以没有  page_limit: 每页的数量 page_start: 起始页 |

- ###### 	豆瓣电影TOP250数据抓取

![image-20191227150649788](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191227150649788.png)

一些备注：

*使用tqdm_notebook加在循环语句中可以查看循环进度

*地址拼接：json_file_path = os.path.join('top250', '{}.json'.format(i))

*访问接口使用while True，抓requests.exceptions.Timeout，请求中要加headers（加入User-Agent参数，不然报错茶壶）

*写文件f.write(json.dumps(json.loads(response.content.decode('utf-8')),indent=4,ensure_ascii=False))

*读写文件加上encoing=utf-8

*按平均分降序排序movies = sorted(movies,key = lambda movie: movie['rating']['average'],reverse= True)

- ###### 演员及相关演员信息爬取





- 




import scrapy
from scrapy.crawler import CrawlerProcess
import json

# https://tldrlegal.com/verified Currently Verified Licenses




class TldrlegalSpider(scrapy.Spider):
    name = "tldrlegal"
    allowed_domains = ['tldrlegal.com']
    start_urls = ['https://tldrlegal.com/search?reverse=true&']

    def parse(self, response):
        # hxs = scrapy.Selector(response)
        # # extract all links from page
        # all_links = hxs.xpath('//a[re:test(@class, "modal-link")]//@href').getall()
        # # iterate over links
        # all_links = list(set(all_links))

        # if response.url.endswith(".json"):
        #     yield {response.url: json.loads(response.body_as_unicode())}
        #
        #
        for href in response.xpath('//*[@id="footer"]/div/span/a[3]/@href').getall():
            url = response.urljoin(href)
            yield response.follow(url, self.parse_json)
            # print("JASON? --> ", href)
            # if href.startswith("/"):
            # yield response.follow(href, callback=self.parse)
            # for link in all_links://*[@id="footer"]/div/span/a[2]

        for href in response.css('a::attr(href)').getall():
            url = response.urljoin(href)
            # print("HREF --> ", href)
            # if href.startswith("/"):
            # if 'api' in href:
            #     yield response.follow(url, self.parse_json)
            # else:
            yield response.follow(url, callback=self.parse)
        # for link in all_links:
        #     if link.startswith("#") and len(link)>1:
        #         url = "https://tosdr.org/api/1/service/{link}.json".format(link=link[1:])
        #         yield response.follow(url, self.parse_json)


    def parse_json(self, response):


        yield {response.url: json.loads(response.body_as_unicode())}

        pass

if __name__ == '__main__':

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',

        'FEED_FORMAT': 'json',
        'FEED_URI': 'tldrlegal.json',
        'LOG_LEVEL': 'ERROR'
    })

    process.crawl(TldrlegalSpider)
    process.start()  # the script will block here until the crawling is finished

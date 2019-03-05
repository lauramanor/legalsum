import scrapy
from scrapy.crawler import CrawlerProcess
import json

# As of 2/7/2019, there are 138 TOS agreements covereted by TOSDR

class TosdrSpider(scrapy.Spider):
    name = "tosdr"
    allowed_domains = ['tosdr.org']
    start_urls = ['https://tosdr.org/']

    def parse(self, response):
        hxs = scrapy.Selector(response)
        # extract all links from page
        all_links = hxs.xpath('//a[re:test(@class, "modal-link")]//@href').getall()
        # iterate over links
        all_links = list(set(all_links))
        for link in all_links:
            if link.startswith("#") and len(link)>1:
                url = "https://tosdr.org/api/1/service/{link}.json".format(link=link[1:])
                yield response.follow(url, self.parse_json)


    def parse_json(self, response):


        yield {response.url: json.loads(response.body_as_unicode())}

        pass

if __name__ == '__main__':
    print('Executing as standalone script')

    # process = CrawlerProcess({
    #     'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    #     'FEED_FORMAT': 'json',
    #     'FEED_URI': 'tosdr.json'
    # })
    #
    # process.crawl(TosdrSpider)
    # process.start()  # the script will block here until the crawling is finished
    # pass
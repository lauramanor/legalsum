import scrapy
from scrapy.crawler import CrawlerProcess
import json



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
            if link.startswith("#"):
                url = "https://tosdr.org/api/1/service/{link}.json".format(link=link[1:])
                yield scrapy.http.Request(url=url, callback=self.get_stuff(url))


    def get_stuff(self, link):
        # yield {
        #     'link': quote.css('span.text::text').get(),
        #     'author': quote.xpath('span/small/text()').get(),
        # }
        print(link)


if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(TosdrSpider)
    process.start()  # the script will block here until the crawling is finished
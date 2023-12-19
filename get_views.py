__author__      = "Marisa Vasconcelos"


import os
import sys
import requests
import json
import datetime

import pandas as pd
import wikipedia as wp

from mwviews.api import PageviewsClient

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class WikipediaAPI(object):

    def __init__(self,directory,lang,title=None, pageid=None):
        self.directory = directory # directory to save jsons
        self.title = title
        self.pageid = pageid
        self.lang = lang # language of the Wikipedia titles
        print(self.title)
        # self.page = wp.page(title=self.title)

        self.wikipedia_url = f"https://{self.lang}.wikipedia.org/w/api.php"
        self.RATE_LIMIT = False
        self.RATE_LIMIT_MIN_WAIT = None
        self.RATE_LIMIT_LAST_CALL = None

        #To collect page _views
        self.pageviews = PageviewsClient(user_agent="marisavas@yahoo.com> Movies pages analysis", parallelism=4)



    def get_creation_date_page(self):
        results = self.request_page_property_once('revisions')
        pageid = list(results['query']['pages'].keys())[0]
        creation = pd.to_datetime(results['query']['pages'][str(pageid)]['revisions'][0]['timestamp'])
        return "%d%02d%02d00" %(creation.year,creation.month,creation.day)



    def request_page_property_once(self,property):

        params = {
        "action":'query',
        "format": "json",
        'prop': property,
        "titles": self.title,
        }

        if property == 'revisions':
            params['rvlimit'] = 1
            params['rvprop'] = 'timestamp'
            params['rvdir'] = 'newer'

        return requests.Session().get(url=self.wikipedia_url, params=params).json()


    def request_page_property(self,property):

        all_results = list()

        params = {
        "action":'query',
        "format": "json",
        'prop': property,
        "titles": self.title,
        }


        if property == 'langlinks':
            params['lllimit'] = 500
        elif property == 'revisions':
            params['rvlimit'] = 'max'
            params['rvprop'] = 'size|ids|comment|timestamp|user'
            params['rvdir'] = 'older'

        else:
            params["pclimit"] = 500

        while True:
            if self.RATE_LIMIT and self.RATE_LIMIT_LAST_CALL and self.RATE_LIMIT_LAST_CALL + self.RATE_LIMIT_MIN_WAIT > datetime.now():
                wait_time = (self.RATE_LIMIT_LAST_CALL + self.RATE_LIMIT_MIN_WAIT) - datetime.now()
                print("Sleep for ", wait_time)
                datetime.time.sleep(int(wait_time.total_seconds()))

            try:
                results = requests.Session().get(url=self.wikipedia_url, params=params).json()

            except ConnectionResetError:
                print("Connection reset")
                with open('log_error.txt','w') as f:
                    wp.f_write("%s\n" % wp.query)
                    sys.exit()
            if 'error' in results:
                if raw_results['error']['info'] in ('HTTP request timed out.', 'Pool queue is full'):
                    print(raw_results['error']['info'])
                    raise HTTPTimeoutError(query)
                else:
                    print(raw_results['error']['info'])
                    raise wp.WikipediaException(raw_results['error']['info'])

            pageid = list(results['query']['pages'].keys())[0]
            # print("Page id ", pageid)

            if property in results['query']['pages'][str(pageid)]:
                all_results.append(results['query']['pages'][str(pageid)][property])
                print(len(results['query']['pages'][str(pageid)][property]))

            if 'continue' in results:
                params.update(results['continue'])
                if self.RATE_LIMIT:
                    self.RATE_LIMIT_LAST_CALL = datetime.now()
            else:
                break

        print(f"Collected {property}", len(all_results))

        #Save results in a json file
        self.save_json(all_results,property)



    def save_json(self,all_results,property):
        filename = self.title.replace(' ','_')
        with open(f'/Users/cecilia/Documents/IC/results_{property}_{filename}.json', 'w') as fout:
            json.dump(all_results, fout)


    def collect_pageviews(self):
        end_time = datetime.datetime.now().strftime('%Y%m%d00')
        start_time = self.get_creation_date_page()
        print(start_time,end_time)

        titles = [self.title]
        views = self.pageviews.article_views(project=f'{self.lang}.wikipedia', articles=titles,
        granularity='monthly', start=start_time, end=end_time)

        df = pd.DataFrame(views)
        df = df.transpose().reset_index()
        folder = self.title.replace(' ','_')
        df = df.rename(columns={f'{folder}':'views','index':'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        filename = self.title.replace(' ','_')
        df.to_csv(f'{self.directory}/results_views_{filename}.csv',sep='\t',index=False)
        print(f"Collected page views for {self.title}")


def main():

    lang = "en"

    properties =  ["langlinks","revisions"]
    directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + f"/IC/data/articles_{lang}/"

    #Read titles filename
    # df = pd.read_csv(f'{directory}/titles.csv' % (tipo,lang),sep='\t')
    df = pd.DataFrame([{'title':'Everything_Everywhere_All_At_Once'}])
    print(df.shape)

    for title in df.title:
        for property in properties:
            wiki = WikipediaAPI(directory,lang,title=title)
            wiki.request_page_property(property)

        wiki.collect_pageviews()


if __name__=="__main__":
    main()
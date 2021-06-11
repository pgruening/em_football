import re
import warnings
from time import sleep

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

URL = 'https://s5.sir.sportradar.com/bwin/en/1/season/58428/fixtures/full'

DRIVER_PATH = '/Users/grueningp/Downloads/chromedriver'

HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
}


def parse(url):
    """
    As other answers mentioned this is basically because the content of page is
    being loaded by help of JavaScript and getting source code with help of
    urlopener or request will not load that dynamic part.
    So here I have a way around of it, actually you can make use of selenium to
    let the dynamic content load and then get the source code from there and
    find for the table. Here is the code that actually give the result you
    expected.
    """
    response = webdriver.Chrome(DRIVER_PATH)
    response.get(url)
    sleep(3)
    sourceCode = response.page_source
    return sourceCode, response


def run():
    src, driver = parse(URL)
    rgx = r'(\d\d:\d\d)(.*)'

    num_pointers = len(driver.find_elements_by_class_name('cursor-pointer'))

    df = MyDataFrame()
    meta = MyDataFrame()

    # find all match buttons
    for index in range(num_pointers):
        pointers = driver.find_elements_by_class_name('cursor-pointer')
        ptr = pointers[index]

        if bool(re.match(rgx, ptr.text)):
            # go to match page
            ptr.click()
            sleep(2.)

            show_more_buttons = get_show_more_buttons(driver)
            while show_more_buttons:
                for button in show_more_buttons:
                    try:
                        button.click()
                    except:
                        continue
                    sleep(.5)
                show_more_buttons = get_show_more_buttons(driver)

            soup = BeautifulSoup(driver.page_source)
            tables_ = soup.findAll('table')[:3]

            page_empty = True
            for table in tables_:
                df, page_empty = process(df, table)
                page_empty = page_empty and page_empty

            meta = read_meta(meta, soup)

            # move back to original screen
            driver.execute_script("window.history.go(-1)")

            df.get_df().to_csv('b_and_w_games.csv')
            meta.get_df().to_csv('b_and_w_meta.csv')


def read_meta(meta, soup):
    panels = soup.findAll(
        'div', attrs={'class': 'panel margin-bottom'})
    content_team = [x.contents for x in panels[0].findAll('div')]
    content_team = [x for x in content_team if len(x) == 1]
    content_team = [
        x[0] for x in content_team if isinstance(x[0], str)
    ]

    tmp = {'team1': content_team[1], 'team2': content_team[5]}

    probabilities = panels[1]
    rgx_prob = r'HomeAway(\d+)\%(\d+)\%(\d+)\%Draw'
    match_prob = re.match(rgx_prob, probabilities.text)
    tmp.update({
        'prob1': match_prob.group(1),
        'prob_draw': match_prob.group(2),
        'prob2': match_prob.group(3),
    })

    form = panels[2].text
    numbers = re.findall(r'\d+', form)
    tmp.update({
        'current_form1': numbers[1],
        'current_form2': numbers[2],
    })

    meta.update(tmp)
    return meta


def process(df, table):
    def _process(x):
        def get_team(team):
            # for some reason, the teams look like "ItalyItaly"
            x = team[:len(team) // 2]
            if team == ''.join(2 * [x]):
                return x
            else:
                return team

        rgx_date = r'(\d\d)/(\d\d)/(\d\d)'
        match_date = re.match(rgx_date, x)
        if bool(match_date):
            return {
                'day': match_date.group(1),
                'month': match_date.group(2),
                'year': match_date.group(3),
            }

        rgx_result = r'(.*)(\d+):(\d+)(.*)'
        match_result = re.match(rgx_result, x)
        if bool(match_result):
            return{
                'team1': get_team(match_result.group(1)),
                'goal1': int(match_result.group(2)),
                'goal2': int(match_result.group(3)),
                'team2': get_team(match_result.group(4)),
            }

        else:
            return None

    is_empty = True
    for entry in table.findAll('tr'):
        row = entry.findAll('td')

        if not row:
            continue

        tmp = dict()
        for x in row:
            z = _process(x.text)
            if z is not None:
                tmp.update(z)

        if 'team1' in tmp.keys():
            is_empty = False
            df.update(tmp, add_missing_values=True, missing_val='N.A.')

    return df, is_empty


def get_show_more_buttons(driver):
    all_buttons = driver.find_elements_by_tag_name('button')
    return [x for x in all_buttons if x.text == 'Show more']

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class MyDataFrame():
    def __init__(self, verbose=0):
        self.x = dict()
        self.max_num_items = 0
        self.verbose = verbose

    def update(self, in_dict, add_missing_values=False, missing_val=np.nan):
        for k, v in in_dict.items():

            if isinstance(v, list):
                warnings.warn(f'Input for {k} is list, consider add_col.')

            if k not in list(self.x.keys()):
                if self.verbose > 0:
                    print(f'added {k}')
                # case 1: df just intialized
                if self.max_num_items == 0:
                    self.x[k] = [v]
                else:
                    # case 2: entire new key is added
                    if add_missing_values:
                        # fill with missing values to current num items
                        self.x[k] = [missing_val] * self.max_num_items
                        self.x[k].append(v)

            else:
                self.x[k].append(v)

        if add_missing_values:
            self._add_missing(missing_val)

    def _add_missing(self, missing_val):
        self._update()
        for k in self.x.keys():
            if self.verbose > 1 and len(self.x[k]) < self.max_num_items:
                print(f'add missing: {k}')

            while len(self.x[k]) < self.max_num_items:
                self.x[k].append(missing_val)

    def _update(self):
        self.max_num_items = max([len(v) for v in self.x.values()])

    def add_col(self, key, col):
        self.x[key] = col

    def get_df(self, cols=None):
        assert self._check_same_lenghts()
        return pd.DataFrame(self.x, columns=cols)

    def _check_same_lenghts(self):
        len_vals = {k: len(v) for k, v in self.x.items()}
        if len(set(len_vals.values())) > 1:
            print(len_vals)
            return False

        return True


if __name__ == '__main__':
    run()
